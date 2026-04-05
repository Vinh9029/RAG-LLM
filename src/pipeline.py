import streamlit as st
import time
from src.config import config
from src.ingestion import DocumentIngestor
from src.retrieval import AdvancedRetriever
from src.generation import ResponseGenerator
from opening_questions import get_opening_questions

st.set_page_config(page_title="Mental Health Assistant", page_icon="🧠", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f7f9fa;}
    .stChatMessage {background: #e3f2fd; border-radius: 10px; margin-bottom: 10px;}
    .stUserMessage {background: #fff3e0; border-radius: 10px; margin-bottom: 10px;}
    .log-box {background: #f1f8e9; border-radius: 8px; padding: 10px; font-size: 0.9em; color: #333;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Mental Health Virtual Assistant (CBT)")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    severe_level = st.selectbox("Severe Level", ["mild", "moderate", "severe", "normal"], index=1)
    mental_status = st.selectbox("Mental Status", [
        "anxiety", "depression", "normal", "bipolar", "personality disorder", "suicidal", "other"
    ], index=0)
    show_logs = st.checkbox("Show logs (advanced)", value=False)

# Opening questions
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.logs = []
    st.session_state.startup = True

if st.session_state.get("startup", True):
    st.info("**Opening questions:**")
    for q in get_opening_questions(severe_level, mental_status):
        st.write(f"- {q}")
    st.session_state.startup = False

# Init pipeline
@st.cache_resource
def get_pipeline():
    ingest = DocumentIngestor(config.embeddings, config.chunk_size, config.chunk_overlap)
    vectorstore = ingest.get_pinecone_vectorstore(namespace="cbt")
    retriever = AdvancedRetriever(vectorstore, config.cross_encoder_model, config.search_k, config.top_n).get_retriever()
    generator = ResponseGenerator(config.llm)
    return retriever, generator

retriever, generator = get_pipeline()

# Chat UI
st.markdown("---")
st.subheader("💬 Chat with the Assistant")
user_input = st.text_input("Your question:", key="user_input")
if st.button("Send", use_container_width=True) or user_input:
    if user_input:
        log_msgs = []
        t0 = time.time()
        # Language detection & translation
        lang = generator.detect_language(user_input)
        log_msgs.append(f"[Log] Detected language: {lang}")
        translated_query = user_input
        if lang == 'vi':
            translated_query = generator.translate_query_to_english(user_input)
            log_msgs.append(f"[Log] Translated query: {translated_query}")
        else:
            log_msgs.append("[Log] No translation needed.")
        # Query expansion
        expanded_query = generator.expand_query(translated_query)
        log_msgs.append(f"[Log] Expanded query: {expanded_query}")
        # Vector search
        t1 = time.time()
        docs = retriever.invoke(expanded_query)
        t2 = time.time()
        log_msgs.append(f"[Log] Vector DB search time: {t2-t1:.3f} seconds")
        context = "\n\n".join([doc.page_content for doc in docs])
        log_msgs.append(f"[Log] Contextual chunks selected: {len(docs)}")
        # Compose prompt
        prompt = f"System prompt: Virtual Assistant for Mental Health Support.\nSevere level: {severe_level}\nMental status: {mental_status}\nContext: {context}\nUser: {user_input}"
        log_msgs.append(f"[Log] Final prompt to LLM:\n{prompt}")
        # Generate answer
        answer = generator.generate_response(user_input, expanded_query, retriever, severe_level, mental_status)
        t3 = time.time()
        log_msgs.append(f"[Log] LLM generation time: {t3-t2:.3f} seconds")
        # Update chat history
        st.session_state.chat_history.append((user_input, answer))
        st.session_state.logs.append("\n".join(log_msgs))
        # Show answer
        with st.spinner("Thinking..."):
            st.success(answer)

# Show chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Conversation History")
    for user, bot in st.session_state.chat_history:
        st.markdown(f"<div class='stUserMessage'><b>You:</b> {user}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stChatMessage'><b>Assistant:</b> {bot}</div>", unsafe_allow_html=True)

# Show logs
if show_logs and st.session_state.logs:
    st.markdown("---")
    with st.expander("Show/Hide Logs", expanded=False):
        for log in st.session_state.logs:
            st.markdown(f"<div class='log-box'>{log}</div>", unsafe_allow_html=True)
