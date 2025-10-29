# app.py

import streamlit as st
from src.logic import create_rag_chain, get_answer

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with any YouTube Video",
    page_icon="📺",
    layout="centered",
)

# --- State Management ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = ""

# --- UI Rendering ---
st.title("📺 Chat with any YouTube Video")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Setup")
    url_input = st.text_input("Enter YouTube URL", key="url_input")
    st.caption("Example: `https://www.youtube.com/watch?v=your_video_id`")

    # Use st.secrets to get the API key
    try:
        hf_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except FileNotFoundError:
        st.error("`secrets.toml` not found. Please create it in the `.streamlit` directory.")
        st.stop()
    except KeyError:
        st.error("`HUGGINGFACEHUB_API_TOKEN` not found in secrets. Please add it.")
        st.stop()

    if st.button("Process Video", type="primary"):
        if url_input:
            if url_input != st.session_state.youtube_url:
                with st.spinner("Processing video... This may take a moment."):
                    st.session_state.youtube_url = url_input
                    # Create and store the RAG chain in session state
                    st.session_state.rag_chain = create_rag_chain(st.session_state.youtube_url, hf_api_token)
                    # Reset chat history for the new video
                    st.session_state.messages = []
                    st.success("Video processed! You can now ask questions.")
        else:
            st.warning("Please enter a YouTube URL.")

# --- Chat Interface ---
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the video"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the video has been processed
    if st.session_state.rag_chain:
        with st.spinner("Thinking..."):
            response = get_answer(st.session_state.rag_chain, prompt)
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please process a YouTube video first.")