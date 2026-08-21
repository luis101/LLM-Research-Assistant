"""
Simple RAG pipeline: LangChain orchestration + local Chroma vector store +
Claude or GPT or Gemini for final answer generation.

Setup:
    pip install langchain langchain-chroma langchain-huggingface \
        langchain-text-splitters chromadb pypdf sentence-transformers python-dotenv
    pip install langchain-anthropic      # for --provider anthropic
    pip install langchain-openai         # for --provider openai
    pip install langchain-google-genai   # for --provider google
    pip install langchain-xai            # for --provider xai
    pip install langchain-ollama         # for --provider ollama
    Put your key(s) in a .env file next to this script (it is gitignored):
        ANTHROPIC_API_KEY=sk-ant-...
        OPENAI_API_KEY=sk-...
        GOOGLE_API_KEY=sk-...
        XAI_API_KEY=sk-...
    Only the key for the provider you actually use is required.

Usage:
    Put your PDFs in ./pdfs, then run:
        python rag_pipeline.py                       # Claude (default)
        python rag_pipeline.py --provider openai     # GPT
        python rag_pipeline.py --provider google     # Gemini
        python rag_pipeline.py --provider xai        # xAI
        python rag_pipeline.py --provider openai --model gpt-5.6-terra  # override default model for a provider
        python rag_pipeline.py --provider ollama --model "gpt-oss:20b"

    The vector store is built once and persisted to ./chroma_db. To re-index
    after adding or changing PDFs, delete that folder and run again. Switching
    providers does NOT require re-indexing: embeddings are computed locally and
    are independent of which chat model answers.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# Read API keys (and anything else) from a local .env file if present.
# .env is gitignored, so keys stay out of version control.
load_dotenv()

PDF_DIR = "./pdfs"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 4096
NUM_PREDICT = 2048
RETRIEVE_K = 4

# Chat providers for final answer generation. The default is Anthropic's Claude, the alternative is OpenAI's GPT models.
# Embeddings stay local, so switching provider never invalidates the vector store.
DEFAULT_PROVIDER = "Ollama"
PROVIDERS = {
    "Anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "package": "langchain-anthropic",
        "default_model": "claude-opus-5",
    },
    "OpenAI": {
        "env_var": "OPENAI_API_KEY",
        "package": "langchain-openai",
        "default_model": "gpt-5.6-terra",
    },
    "Google": {
        "env_var": "GOOGLE_API_KEY",
        "package": "langchain-google-genai",
        "default_model": "gemini-2.0-flash",
    },
    "XAI": {
        "env_var": "XAI_API_KEY",
        "package": "langchain-xai",
        "default_model": "grok-4",
    },
    # Ollama runs models locally (default http://localhost:11434) and needs no
    # API key, so env_var is None. Set OLLAMA_HOST if your server is elsewhere.
    # Model ids are name:tag (colon, not hyphen) and must be pulled first.
    # Run `ollama list` to see what you have, or `ollama pull <id>` to add one.
    "Ollama": {
        "env_var": None,
        "package": "langchain-ollama",
        "default_model": "qwen3.5:4b-q8_0",
    },
    # Override the default model for a provider by passing --model MODEL_ID on the command line.
}

# Built-in prompt templates, chosen with --prompt-style. Both must contain the
# {context} and {question} placeholders; so must any --prompt-file you supply.
#
# "strict" keeps the model inside the retrieved excerpts, best when you need
# every sentence traceable to a paper. "creative" lets it reason beyond them,
# but insists that added material is labelled, so you can still tell which
# claims came from your PDFs and which came from the model.
DEFAULT_PROMPT_STYLE = "Strict"
PROMPT_STYLES = {
    "Strict": (
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say politely that you don't know - don't guess. "
        "Cite the source file and page for each claim.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
    "Creative": (
        "You are a research assistant. Treat the excerpts below as your primary evidence, "
        "but you may also draw on your own knowledge to explain, connect, compare, and extend what they say, "
        "including ideas, methods, or literature the excerpts do not mention.\n\n"
        "Keep the two sources of information distinguishable:\n"
        "- Cite the source file and page for anything taken from the excerpts.\n"
        "- Mark anything from your own knowledge with 'Beyond the sources:', "
        "so the reader never mistakes it for something the papers said.\n"
        "- Say so explicitly when you are speculating or when the papers disagree with each other "
        "or with the wider literature.\n\n"
        "Excerpts:\n{context}\n\n"
        "Question: {question}"
    ),
}

def load_and_split(pdf_dir: str):
    """Load PDFs from the given directory and split them into chunks."""
    docs = []
    skipped = []

    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # corrupt/encrypted file - report and continue
            skipped.append(f"{pdf_path.name}: unreadable ({type(exc).__name__}: {exc})")
            continue

        pages_with_text = 0
        # Pages are numbered from 1 so citations match what a PDF viewer shows
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text.strip():
                continue  # image-only page: nothing to embed
            pages_with_text += 1
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": page_number},
                )
            )

        if pages_with_text == 0: # Scanned or unreadable pdf
            skipped.append(f"{pdf_path.name}: No extractable text (scanned? needs OCR)")

    if skipped:
        print("Warning: some PDFs contributed nothing:", file=sys.stderr)
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)

    if not docs:
        raise SystemExit(
            f"No extractable text found in any PDF in {pdf_dir!r}. "
            "Check that the folder contains text-based (non-scanned) PDFs."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks


def get_vectorstore():
    """
    Open the persisted Chroma collection, building it first if needed.

    Chroma.from_documents() appends to an existing collection, so calling it
    on every run would add a duplicate copy of every chunk each time. 
    Build only when the store doesn't exist yet. Otherwise just open it.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(CHROMA_DIR).exists():
        print(f"Reusing existing vector store in {CHROMA_DIR} ...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        if count == 0:
            print(
                f"  {CHROMA_DIR} exists but is empty — delete it and re-run to rebuild.",
                file=sys.stderr,
            )
        else:
            print(f"  {count} chunks indexed")
        return vectorstore

    print("Loading and splitting PDFs ...")
    chunks = load_and_split(PDF_DIR)
    print(f"  {len(chunks)} chunks")

    print("Building vector store (embedding chunks) ...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        return vectorstore
    except BaseException:
        # If the build fails, delete the partially-built store so we don't try to reuse it next time
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        raise


def make_llm(provider: str, model: str | None = None):
    """
    Build the chat model for the chosen provider.

    Only the package for the provider you actually use is needed, and the key is checked here.
    """
    # argparse validates --provider, but not the default taken from LLM_PROVIDER,
    # so an unknown value is still possible.
    if provider not in PROVIDERS:
        raise SystemExit(
            f"Unknown provider {provider!r}. Choose from: {', '.join(sorted(PROVIDERS))}"
        )

    cfg = PROVIDERS[provider]
    model = model or cfg["default_model"]

    # env_var is None for providers that need no key (Ollama runs locally).
    if cfg["env_var"] and not os.environ.get(cfg["env_var"]):
        raise SystemExit(
            f"{cfg['env_var']} not found. Put it in a .env file next to this script or set it in your environment."
        )

    try:
        if provider == "Anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model, max_tokens=MAX_TOKENS, timeout=60, max_retries=2
            )
        elif provider == "OpenAI":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model, max_tokens=MAX_TOKENS, timeout=60, max_retries=2
            )
        elif provider == "Google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model, max_output_tokens=MAX_TOKENS, timeout=60, max_retries=2
            )
        elif provider == "XAI":
            from langchain_xai import ChatXAI

            return ChatXAI(
                model=model, max_tokens=MAX_TOKENS, timeout=60, max_retries=2
            )
        elif provider == "Ollama":
            from langchain_ollama import ChatOllama

            # Ollama calls the output cap num_predict, not max_tokens.
            # reasoning=False matters for thinking models (qwen3, deepseek-r1, ...): 
            # with it on, the model can spend the whole context window on internal reasoning, 
            # hit the cap, and return an EMPTY answer.
            return ChatOllama(model=model, num_predict=NUM_PREDICT, reasoning=False)
    except ImportError as exc:
        raise SystemExit(
            f"Provider {provider!r} could not be loaded from {cfg['package']}: {exc}\n"
            f"If the package is missing: uv add {cfg['package']}"
        )

    raise SystemExit(f"Provider {provider!r} has no constructor in make_llm().")


def load_template(style: str, prompt_file: str | None, prompt_text: str | None = None) -> str:
    """Return the prompt template: raw text, a file, or a built-in style."""
    if prompt_text:
        template = prompt_text
        source = "the custom prompt"
    elif prompt_file:
        path = Path(prompt_file)
        try:
            template = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"Could not read prompt file {prompt_file!r}: {exc}")
        source = f"file {path.name}"
    else:
        template = PROMPT_STYLES[style]
        source = f"style {style!r}"

    # The chain expects both {context} and {question} placeholders to be present in the prompt template, 
    # a template missing either would fail at invoke time with a much less obvious error.
    missing = [v for v in ("{context}", "{question}") if v not in template]
    if missing:
        raise SystemExit(
            f"Prompt from {source} is missing the placeholder(s): {', '.join(missing)}. "
            "A prompt template must contain both {context} and {question}."
        )
    return template


def build_chain(vectorstore, llm, template: str):
    """Build a RAG chain that retrieves relevant chunks and generates an answer."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})

    prompt = ChatPromptTemplate.from_template(template)

    # Format the retrieved documents into a string that includes the source and page number for each chunk
    def format_docs(docs):
        parts = []
        for d in docs:
            source = Path(d.metadata.get("source", "unknown")).name
            page = d.metadata.get("page", "?")
            parts.append(f"[{source}, page {page}]\n{d.page_content}")
        return "\n\n---\n\n".join(parts)
      
    # Build a chain that retrieves relevant chunks, formats them, and passes them to the LLM for answer generation  
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs.")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default=os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER),
        help="Which chat model answers the question (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the provider's default model id.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=sorted(PROMPT_STYLES),
        default=os.environ.get("PROMPT_STYLE", DEFAULT_PROMPT_STYLE),
        help=(
            "strict: answer only from the PDFs. creative: also use the model's "
            "own knowledge, labelled separately (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        metavar="PATH",
        help=(
            "Use a custom prompt template from a text file instead of a built-in "
            "style. Must contain the {context} and {question} placeholders."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fail on a missing key or a broken prompt before spending time embedding.
    llm = make_llm(args.provider, args.model)
    template = load_template(args.prompt_style, args.prompt_file)
    prompt_desc = args.prompt_file or args.prompt_style
    print(f"Provider: {args.provider} ({llm.model}) | prompt: {prompt_desc}")

    vectorstore = get_vectorstore()
    chain = build_chain(vectorstore, llm, template)

    print("\nReady. Ask questions about your PDFs (type 'quit' to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"\n{answer}")


if __name__ == "__main__":
    main()