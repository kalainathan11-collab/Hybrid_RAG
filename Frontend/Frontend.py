import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Hybrid RAG Architecture Assistant",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown(
    """
    <style>
        .your-class-name {
            border-right: 1px solid #1e293b;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Hybrid RAG assistant. Ask me anything about the architecture document.",
            "sources": None
        }
    ]

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Hybrid RAG Controls")
    st.caption("Vector Database + Knowledge Graph")
    st.divider()

    # Retrieval Mode Selector
    st.subheader("Retrieval Strategy")
    retrieval_mode = st.radio(
        "Select Mode:",
        ["hybrid", "vector", "graph"],
        format_func=lambda x: {
            "hybrid": "🔀 Hybrid (ChromaDB + Neo4j)",
            "vector": "📚 Vector Only (ChromaDB)",
            "graph": "🕸️ Graph Only (Neo4j)"
        }[x]
    )

    st.divider()

    # Database Status
    st.subheader("Connected Stores")
    st.success("🟢 **ChromaDB**: Connected (Vector)")
    st.success("🟢 **Neo4j Aura**: Connected (Graph)")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------
st.title("💬 Architecture Knowledge Assistant")
st.caption(f"Currently querying in **{retrieval_mode.upper()}** mode")

# Render Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display Sources if available
        if message.get("sources"):
            sources = message["sources"]
            with st.expander("🔍 View Retrieved Source Context"):
                if sources.get("vector_sources"):
                    st.markdown("**📚 Chroma Vector Chunks:**")
                    for idx, chunk in enumerate(sources["vector_sources"], 1):
                        st.info(f"**Chunk {idx}:**\n\n{chunk}")
                
                if sources.get("graph_sources"):
                    st.markdown("**🕸️ Neo4j Graph Context:**")
                    for idx, triple in enumerate(sources["graph_sources"], 1):
                        st.code(triple, language="text")

# Input Box for User Prompt
if prompt := st.chat_input("Ask a question about the Spotify architecture..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Query with Backend API
    with st.chat_message("assistant"):
        with st.spinner(f"Retrieving context via {retrieval_mode.upper()} strategy..."):
            try:
                # Fixed: Corrected FastAPI endpoint URL
                backend_url = "http://127.0.0.1:8000/query"
                
                # Fixed: Payload key matched to 'question' expected by FastAPI
                payload = {
                    "question": prompt
                }
                
                response = requests.post(backend_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    
                    # Process string or list responses for vector and graph contexts
                    raw_vector = data.get("vector_context") or data.get("vector_sources") or ""
                    vector_sources = [raw_vector] if isinstance(raw_vector, str) and raw_vector else (raw_vector if isinstance(raw_vector, list) else [])
                    
                    raw_graph = data.get("graph_context") or data.get("graph_sources") or ""
                    graph_sources = [raw_graph] if isinstance(raw_graph, str) and raw_graph else (raw_graph if isinstance(raw_graph, list) else [])

                    sources = {
                        "vector_sources": vector_sources,
                        "graph_sources": graph_sources
                    }

                    # Display Answer
                    st.markdown(answer)

                    # Display Sources Expander
                    if sources["vector_sources"] or sources["graph_sources"]:
                        with st.expander("🔍 View Retrieved Source Context"):
                            if sources["vector_sources"]:
                                st.markdown("**📚 Chroma Vector Chunks:**")
                                for idx, chunk in enumerate(sources["vector_sources"], 1):
                                    st.info(f"**Chunk {idx}:**\n\n{chunk}")
                            
                            if sources["graph_sources"]:
                                st.markdown("**🕸️ Neo4j Graph Context:**")
                                for idx, triple in enumerate(sources["graph_sources"], 1):
                                    st.code(triple, language="text")

                    # Append Assistant Response to Chat History
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = f"Backend returned status code {response.status_code}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": None})

            except requests.exceptions.ConnectionError:
                error_msg = "🚨 Could not connect to backend server. Make sure your FastAPI backend (`http://127.0.0.1:8000`) is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": None})
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": None})