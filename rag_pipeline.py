"""
Simple RAG pipeline: LangChain orchestration + local Chroma vector store +
Claude or GPT or Gemini for final answer generation.

Setup:
    pip install langchain langchain-chroma langchain-huggingface \
        langchain-text-splitters chromadb pypdf sentence-transformers python-dotenv
    pip install langchain-anthropic      # for --provider anthropic
    pip install langchain-openai         # for --provider openai
    pip install langchain-google-genai   # for --provider google
    Put your key(s) in a .env file next to this script (it is gitignored):
        ANTHROPIC_API_KEY=sk-ant-...
        OPENAI_API_KEY=sk-...
        GOOGLE_API_KEY=sk-...
    Only the key for the provider you actually use is required.

Usage:
    Put your PDFs in ./pdfs, then run:
        python rag_pipeline.py                       # Claude (default)
        python rag_pipeline.py --provider openai     # GPT
        python rag_pipeline.py --provider google     # Gemini
        python rag_pipeline.py --provider openai --model gpt-5.6-terra  # override default model for a provider

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
RETRIEVE_K = 4

# Chat providers for final answer generation. The default is Anthropic's Claude, the alternative is OpenAI's GPT models.
# Embeddings stay local, so switching provider never invalidates the vector store.
DEFAULT_PROVIDER = "anthropic"
PROVIDERS = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "package": "langchain-anthropic",
        "default_model": "claude-opus-5",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "package": "langchain-openai",
        "default_model": "gpt-5.6-terra",
    },
    "google": {
        "env_var": "GOOGLE_API_KEY",
        "package": "langchain-google-genai",
        "default_model": "gemini-pro",
    },
    # Override the default model for a provider by passing --model MODEL_ID on the command line.
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
        print("Warning: some PDFs contributed nothing to the index:", file=sys.stderr)
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
    cfg = PROVIDERS[provider]
    model = model or cfg["default_model"]

    if not os.environ.get(cfg["env_var"]):
        raise SystemExit(
            f"{cfg['env_var']} not found. Put it in a .env file next to this script or set it in your environment."
        )

    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model, max_tokens=MAX_TOKENS, timeout=60, max_retries=2
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model, max_tokens=MAX_TOKENS, timeout=60, max_retries=2
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGemini

            return ChatGoogleGemini(
                model=model, max_output_tokens=MAX_TOKENS, timeout=60, max_retries=2
            )
    except ImportError as exc:
        raise SystemExit(
            f"Provider {provider!r} needs the {cfg['package']} package."
        )


def build_chain(vectorstore, llm):
    """Build a RAG chain that retrieves relevant chunks and generates an answer."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say politely that you don't know - don't guess. "
        "Cite the source file and page for each claim.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

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
    return parser.parse_args()


def main():
    args = parse_args()

    # Fail on a missing key before spending time embedding or loading models.
    llm = make_llm(args.provider, args.model)
    print(f"Provider: {args.provider} ({llm.model})")

    vectorstore = get_vectorstore()
    chain = build_chain(vectorstore, llm)

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