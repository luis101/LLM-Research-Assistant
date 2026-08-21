# LLM Research Assistant

This research assistant aims at generating research answers and ideas based on a collection of PDFs (supposed to be research papers) grounded in their contents.

It is a small RAG (retrieval-augmented generation) pipeline: LangChain for
orchestration, a local Chroma vector database for retrieval, and your choice of
chat model - Claude, GPT, Gemini, Grok, or a local model via Ollama - for obtaining the answer.

## How it works

```
PDFs ──► text per page ──► ~1000-char chunks ──► local embeddings ──► Chroma
                                                                        │
question ──► embed ──► cosine similarity search ────────────────────────┘
                                    │
                          top-k chunks + question
                                    │
                                    ▼
                          chat model ──► answer + citations
```

Two properties are worth understanding, because they explain most of the
design:

**Embeddings are computed locally** by `all-MiniLM-L6-v2` (a small
sentence-transformers model, CPU-only, no API calls, free). Only the final
answer-writing step might need a paid API. Indexing therefore costs nothing.

**Retrieval is what makes this affordable.** Only the handful of chunks
relevant to the question are sent to the model, roughly 900 input tokens per
query, rather than the whole corpus. A ~30-paper library runs close to a
million tokens, so stuffing everything into each request would be both far more
expensive and, past the context limit, impossible.

A consequence: **the chat model and the vector store are independent**. You can
switch providers freely without re-indexing.

## Requirements

- Python 3.13 or larger (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) for dependency management (or set up environment)
- An API key for whichever provider you use, or [Ollama](https://ollama.com)
  running locally, which needs no key

## Setup

```bash
uv sync
```

Then create a `.env` file next to `rag_pipeline.py` with the key(s) you need.
Only the one for the provider you actually use is required:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
```

`.env` is gitignored. Create it in an editor.

Put your PDFs in `./pdfs/`. They must be text-based; scanned/image-only PDFs
contribute nothing (the script warns you by filename when this happens).

## Usage

There are two front ends over the same pipeline: a web dashboard and a CLI.

### Dashboard (recommended)

```bash
uv run streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Choose the provider, model, and prompt style
at the top of the page, then ask questions in the chat box. The conversation is
kept until you press **Clear chat**, and a custom prompt can be typed directly
into a text area. Add `--server.headless true` to suppress the auto-opened browser.

The vector store and the model connection are cached across interactions, so
only the first question pays the ~15-20s setup cost. Changing provider or model reconnects without re-indexing.

### CLI

```bash
uv run python rag_pipeline.py
```

The first run builds the vector store (a few minutes for a few dozen papers);
later runs reuse it and start immediately. Then ask questions at the prompt,
and type `quit` to exit.

The options below apply to the CLI; the dashboard exposes the same choices as
widgets.

### Choosing a model

```bash
uv run python rag_pipeline.py --provider OpenAI
uv run python rag_pipeline.py --provider Ollama
uv run python rag_pipeline.py --provider OpenAI --model gpt-5.6-terra
```

Provider names are case-sensitive and match the keys below.

| `--provider` | Key | Package |
|---|---|---|
| `Anthropic` | `ANTHROPIC_API_KEY` | `langchain-anthropic` |
| `OpenAI` | `OPENAI_API_KEY` | `langchain-openai` |
| `Google` | `GOOGLE_API_KEY` | `langchain-google-genai` |
| `XAI` | `XAI_API_KEY` | `langchain-xai` |
| `Ollama` | *(none — runs locally)* | `langchain-ollama` |

Each provider's default model id is noted in the `PROVIDERS` dict in
`rag_pipeline.py`. Model line-ups change often, so check the provider's current
model list and either edit that default or pass `--model`. An invalid id is not
caught at startup, it results in an error on the first question.

### Choosing how the model answers

```bash
uv run python rag_pipeline.py --prompt-style Creative
uv run python rag_pipeline.py --prompt-file my_prompt.txt
```

| Style | Behaviour |
|---|---|
| `Strict` (default) | Answers **only** from the retrieved excerpts. Says that it doesn't know rather than to guess. Every claim can be traced back to a paper. |
| `Creative` | Also draws on the model's own knowledge to explain, compare, and extend, but labels those additions `Beyond the sources:` and flags speculation, so it is always clear what came from the PDFs. |

`--prompt-file` takes any text file as the template. It must contain both
`{context}` and `{question}` placeholders; this is checked at startup, before
any slow work happens.

Both `--provider` and `--prompt-style` can also be set in `.env` as
`LLM_PROVIDER` and `PROMPT_STYLE`.

## Configuration

Constants at the top of `rag_pipeline.py`:

| Name | Purpose |
|---|---|
| `RETRIEVE_K` | How many chunks to retrieve per question. Increase if answers miss material you know is in the corpus; input cost scales roughly linearly. |
| `MAX_TOKENS` | Output cap for the cloud providers. |
| `NUM_PREDICT` | Output cap for Ollama (its own parameter name). |
| `EMBEDDING_MODEL` | Sentence-transformers model used for retrieval. **Changing this invalidates the vector store** - delete `chroma_db/` and re-index. |
| `PDF_DIR`, `CHROMA_DIR` | Input and index locations. |

Chunking (`chunk_size=1000`, `chunk_overlap=200`) is set in `load_and_split`.

## Re-indexing

The vector store is built once and persisted to `chroma_db/`. To pick up added,
removed, or changed PDFs:

```bash
rm -rf chroma_db    # Remove-Item -Recurse -Force chroma_db
uv run python rag_pipeline.py
```

Deleting is required rather than optional: the build path *appends* to an
existing collection, so re-running the build over an existing store would duplicate
every chunk and let one document be retrieved several times.

## Troubleshooting

**`model 'x' not found (status code: 404)` from Ollama.** Run
`ollama list` to see what you have and `ollama pull <id>` to add one.

**Answers cut off mid-sentence.** Raise `MAX_TOKENS` (cloud) or `NUM_PREDICT`
(Ollama). Note Ollama's default context window is 4096 tokens total and must
hold the prompt *and* the answer, so raising the output cap alone may not be
enough, `num_ctx` controls the window.

**A PDF seems to be missing from answers.** Watch the startup warnings: files
with no extractable text are listed by name.

**`chroma_db exists but is empty`.** Delete the folder and re-run to rebuild.

## Cost

Only the answer-writing step is billed; indexing and retrieval are local and
free. At `RETRIEVE_K=4` a question sends roughly 900 input tokens plus the
answer's output tokens, on the order of a cent or two per question on a
frontier model, and free with `--provider ollama`. Output tokens dominate, so
the model choice and the output cap matter far more than corpus size.

## Layout

```
pdfs/            source PDFs (gitignored)
chroma_db/       persisted vector store (gitignored, rebuildable)
rag_pipeline.py  the pipeline + CLI
dashboard.py     Streamlit UI over the same pipeline
pyproject.toml   dependencies
.env             API keys (gitignored)
```