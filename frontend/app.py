import os

import requests
import streamlit as st

# In Docker Compose this is http://backend:8000 (set in docker-compose.yml);
# when running locally it falls back to localhost.
DEFAULT_API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "documents")
QUERY_TIMEOUT = 180   # seconds; the agent may call the LLM several times
SHORT_TIMEOUT = 15

st.set_page_config(page_title="RAG Agentic Q&A", page_icon="🤖", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "collection" not in st.session_state:
    st.session_state.collection = DEFAULT_COLLECTION
if "top_k" not in st.session_state:
    st.session_state.top_k = 5


def api(method: str, path: str, timeout: int = SHORT_TIMEOUT, **kwargs):
    return requests.request(method, f"{st.session_state.api_url}{path}", timeout=timeout, **kwargs)


def error_text(response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def render_details(reasoning, sources, query_time):
    if reasoning:
        with st.expander("🔍 Agent Reasoning"):
            for step in reasoning:
                st.write(f"- {step}")
    if sources:
        with st.expander("📄 Sources"):
            for idx, src in enumerate(sources, 1):
                page = src.get("page")
                where = f"{src.get('source', 'Unknown')}" + (f", page {page}" if page is not None else "")
                st.markdown(f"**Source {idx}** — {where} (relevance: {src.get('relevance_score', 0):.3f})")
                st.text(src.get("content", ""))
    if query_time:
        st.caption(f"Query time: {query_time:.2f}s")


# ---------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Settings")
st.session_state.api_url = st.sidebar.text_input("API URL", st.session_state.api_url)
st.session_state.collection = st.sidebar.text_input(
    "Collection", st.session_state.collection,
    help="Documents are uploaded to, and questions answered from, this collection.",
)
st.session_state.top_k = st.sidebar.slider("Top K results per retrieval", 1, 20, st.session_state.top_k)

st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
if st.sidebar.button("Upload"):
    if uploaded_file is None:
        st.sidebar.warning("Please select a file first.")
    else:
        try:
            with st.spinner("Indexing document..."):
                response = api(
                    "POST", "/upload", timeout=QUERY_TIMEOUT,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"collection_name": st.session_state.collection},
                )
            if response.ok:
                data = response.json()
                st.sidebar.success(
                    f"Indexed {data['num_chunks']} chunks into '{data['collection_name']}'"
                )
            else:
                st.sidebar.error(f"Error: {error_text(response)}")
        except requests.RequestException as e:
            st.sidebar.error(f"Failed to connect: {e}")

st.sidebar.header("📚 Collections")
if st.sidebar.button("Refresh Collections"):
    try:
        response = api("GET", "/collections")
        if response.ok:
            collections = response.json().get("collections", [])
            if not collections:
                st.sidebar.info("No collections found.")
            for col in collections:
                st.sidebar.write(f"- {col['name']} ({col['num_chunks']} chunks)")
        else:
            st.sidebar.error(f"Error: {error_text(response)}")
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to connect: {e}")

if st.sidebar.button(f"Delete collection '{st.session_state.collection}'"):
    try:
        response = api("DELETE", f"/collections/{st.session_state.collection}")
        if response.ok:
            st.sidebar.success("Collection deleted.")
        else:
            st.sidebar.error(f"Error: {error_text(response)}")
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to connect: {e}")

st.sidebar.header("❤️ Health")
if st.sidebar.button("Check Backend Health"):
    try:
        response = api("GET", "/health")
        if response.ok:
            body = response.json()
            st.sidebar.success(f"Status: {body.get('status')} (v{body.get('version')})")
        else:
            st.sidebar.error("Backend not healthy.")
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to connect: {e}")

# ---------------------------------------------------------------- chat
st.title("🤖 RAG Agentic Q&A System")
st.subheader("Ask questions about your documents — powered by AI agents")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_details(msg.get("reasoning"), msg.get("sources"), msg.get("query_time"))

if question := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = api(
                    "POST", "/query", timeout=QUERY_TIMEOUT,
                    json={
                        "question": question,
                        "collection_name": st.session_state.collection,
                        "top_k": st.session_state.top_k,
                    },
                )
                if response.ok:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    st.markdown(answer)
                    render_details(data.get("agent_reasoning"), data.get("sources"),
                                   data.get("query_time_seconds"))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "reasoning": data.get("agent_reasoning", []),
                        "sources": data.get("sources", []),
                        "query_time": data.get("query_time_seconds"),
                    })
                else:
                    st.error(f"Error: {error_text(response)}")
            except requests.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")
