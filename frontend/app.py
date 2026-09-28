import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG Agentic Q&A", page_icon="🤖", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5

st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
if st.sidebar.button("Upload"):
    if uploaded_file is not None:
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{st.session_state.api_url}/upload", files=files)
            if response.status_code == 200:
                data = response.json()
                st.sidebar.success(f"Success! Chunks created: {data.get('num_chunks', 0)}")
            else:
                st.sidebar.error(f"Error: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Failed to connect: {str(e)}")
    else:
        st.sidebar.warning("Please select a file first.")

st.sidebar.header("📚 Collections")
if st.sidebar.button("Refresh Collections"):
    try:
        response = requests.get(f"{st.session_state.api_url}/collections")
        if response.status_code == 200:
            collections = response.json().get("collections", [])
            if not collections:
                st.sidebar.info("No collections found.")
            for col in collections:
                col_name = col.get('name')
                st.sidebar.write(f"- {col_name} ({col.get('num_documents', 0)} docs)")
                # Delete functionality is simplified for display, standard buttons might reload state.
        else:
            st.sidebar.error(f"Error: {response.text}")
    except Exception as e:
         st.sidebar.error(f"Failed to connect: {str(e)}")

st.sidebar.header("⚙️ Settings")
st.session_state.api_url = st.sidebar.text_input("API URL", st.session_state.api_url)
st.session_state.top_k = st.sidebar.slider("Top K Results", 1, 20, st.session_state.top_k)

st.sidebar.header("❤️ Health")
if st.sidebar.button("Check Backend Health"):
    try:
        response = requests.get(f"{st.session_state.api_url}/health")
        if response.status_code == 200:
             st.sidebar.success(f"Status: {response.json().get('status')} (v{response.json().get('version', '1.0')})")
        else:
             st.sidebar.error("Backend not healthy.")
    except Exception as e:
         st.sidebar.error(f"Failed to connect: {str(e)}")


st.title("🤖 RAG Agentic Q&A System")
st.subheader("Ask questions about your documents — powered by AI agents")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("reasoning"):
            with st.expander("🔍 Agent Reasoning"):
                for step in msg["reasoning"]:
                    st.write(f"- {step}")
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source {idx+1}** (Relevance: {src.get('relevance_score', 'N/A')}, Page: {src.get('page', 'N/A')})")
                    st.text(src.get('content', ''))
        if msg.get("query_time"):
            st.caption(f"Query time: {msg['query_time']:.2f}s")

if question := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                payload = {
                    "question": question,
                    "collection_name": "default",
                    "top_k": st.session_state.top_k
                }
                # Default behavior for now, collections could be dynamic
                response = requests.post(f"{st.session_state.api_url}/query", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    reasoning = data.get("agent_reasoning", [])
                    sources = data.get("sources", [])
                    query_time = data.get("query_time_seconds", 0.0)
                    
                    st.markdown(answer)
                    if reasoning:
                        with st.expander("🔍 Agent Reasoning"):
                            for step in reasoning:
                                st.write(f"- {step}")
                    if sources:
                        with st.expander("📄 Sources"):
                            for idx, src in enumerate(sources):
                                st.markdown(f"**Source {idx+1}** (Relevance: {src.get('relevance_score', 'N/A')}, Page: {src.get('page', 'N/A')})")
                                st.text(src.get('content', ''))
                    st.caption(f"Query time: {query_time:.2f}s")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "reasoning": reasoning,
                        "sources": sources,
                        "query_time": query_time
                    })
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")
