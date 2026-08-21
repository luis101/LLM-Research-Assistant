"""
Streamlit dashboard for the RAG pipeline.

Run with:
    uv run streamlit run dashboard.py

The pipeline itself is in rag_pipeline.py. This file connects it to a UI.
"""

import streamlit as st
from rag_pipeline import (
    DEFAULT_PROMPT_STYLE, PROMPT_STYLES, PROVIDERS,
    build_chain, get_vectorstore, load_template, make_llm
)

CUSTOM_PROMPT = "Custom"

### Interactive Page Configuration ###
    
st.set_page_config(page_title="LLM Research Assistant", layout="wide")
st.title("Research Assistant: Ask Questions About Research and Get Practical Ideas")

# Streamlit re-runs the whole script on every interaction, so anything expensive must be cached 
# or it is repeated with every question. Loading the embedding model alone takes approximately 15-20s.
@st.cache_resource(show_spinner="Loading vector store...")
def cached_vectorstore():
    return get_vectorstore()

@st.cache_resource(show_spinner="Connecting to model...")
def cached_llm(provider: str, model: str | None):
    return make_llm(provider, model)

st.subheader("LLM Configuration")

# Fail on a missing key or a broken prompt before spending time embedding the PDFs. 
# The user can change the provider, model, or prompt style and re-run to try again.
providers = sorted(PROVIDERS)
provider = st.selectbox("Provider", providers, 
                        index=providers.index("Ollama"))

model = st.text_input("Model",
                      placeholder=f"default: {PROVIDERS[provider]['default_model']}",
                      help="Leave blank to use the provider's default."
                      ).strip()    

styles = sorted(PROMPT_STYLES) + [CUSTOM_PROMPT]
prompt_style = st.selectbox("Prompt style", styles, 
                            index=styles.index(DEFAULT_PROMPT_STYLE))

prompt_text = None
if prompt_style == CUSTOM_PROMPT:
    prompt_text = st.text_area("Input prompt: ",
                               value=PROMPT_STYLES[DEFAULT_PROMPT_STYLE], height=200,
                               help="Must contain the {context} and {question} placeholders."
                               ).strip() or None
    if not prompt_text:
        st.info("Enter a prompt template.")
        st.stop()
else:
    st.text_area("Prompt in use", 
                 value=PROMPT_STYLES[prompt_style],
                 height=200, disabled=True)

### Chat Interface ###

st.subheader("Chat")

if st.button("Clear chat"):
    st.session_state.history = []

# make_llm() and load_template() can raise SystemExit on missing API key or broken prompt,
# so it is called here to fail fast.
try:
    llm = cached_llm(provider, model or None)
    template = load_template(DEFAULT_PROMPT_STYLE if prompt_style == CUSTOM_PROMPT else prompt_style,
                             None, prompt_text)
except SystemExit as exc:
    st.error(str(exc))
    st.stop()

vectorstore = cached_vectorstore()
chain = build_chain(vectorstore, llm, template)

st.write(f"Provider (Model): {provider} ({llm.model}) | Prompt Type: {prompt_style}")

# Keep the conversation instead of replacing the previous answer each time
if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

if question := st.chat_input("Ready. Ask a question about the Research PDFs."):
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Generation can take minutes on a local model - show progress
        with st.spinner("Searching the papers and generating an answer..."):
            try:
                answer = chain.invoke(question)
            except Exception as exc:
                answer = f"Error: {type(exc).__name__}: {exc}"
        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})

