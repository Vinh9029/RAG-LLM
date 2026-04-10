import streamlit as st
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_config import config
from ingestion import DocumentIngestor
from retrieval import AdvancedRetriever
from generation import ResponseGenerator
from opening_questions import get_opening_questions

st.set_page_config(page_title="Mental Health Assistant", page_icon="🧠", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f7f9fa;}
    .log-box {background: #f1f8e9; border-radius: 8px; padding: 10px; font-size: 0.9em; color: #333; margin-top: 10px; white-space: pre-wrap;}
    
    /* Căn lề phải cho tin nhắn của User (đưa Icon sang phải, Text căn phải) */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
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
    ingest = DocumentIngestor(config.embeddings, config.chunk_size, config.chunk_overlap, getattr(config, 'embedding_dimension', 768))
    vectorstore = ingest.get_pinecone_vectorstore(namespace="cbt")
    retriever = AdvancedRetriever(vectorstore, config.cross_encoder_model, config.search_k, config.top_n).get_retriever()
    generator = ResponseGenerator(config.llm)
    return retriever, generator

retriever, generator = get_pipeline()

# Chat UI
st.markdown("---")

# Show chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, dict):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Hiển thị log đi kèm dưới mỗi tin nhắn của assistant nếu bật show_logs
            if msg["role"] == "assistant" and show_logs and "logs" in msg:
                with st.expander("Advanced Logs", expanded=False):
                    st.markdown(f"<div class='log-box'>{msg['logs']}</div>", unsafe_allow_html=True)

# Chat Input
if user_input := st.chat_input("How can I help you today?"):
    # Add user message to state & render instantly
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Create thinking block for assistant
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                log_msgs = []
                t0 = time.time()
                
                lang = generator.detect_language(user_input)
                log_msgs.append(f"[Log] Detected language: {lang}")
                translated_query = user_input
                if lang == 'vi':
                    translated_query = generator.translate_query_to_english(user_input)
                    log_msgs.append(f"[Log] Translated query: {translated_query}")
                else:
                    log_msgs.append("[Log] No translation needed.")
                    
                expanded_query = generator.expand_query(translated_query)
                log_msgs.append(f"[Log] Expanded query: {expanded_query}")
                
                t1 = time.time()
                docs = retriever.invoke(expanded_query)
                t2 = time.time()
                log_msgs.append(f"[Log] Vector DB search time: {t2-t1:.3f} seconds")
                log_msgs.append(f"[Log] Retrieved {len(docs)} documents")
                
                answer = generator.generate_response(user_input, expanded_query, retriever, severe_level, mental_status)
                t3 = time.time()
                log_msgs.append(f"[Log] LLM generation time: {t3-t2:.3f} seconds")
                log_msgs.append(f"[Log] Response length: {len(str(answer)) if answer else 0} chars")
                
                # Display final answer
                if answer and answer.strip():
                    st.markdown(answer)
                else:
                    st.warning("⚠️ Empty response received from the LLM. Check logs for details.")
                
                joined_logs = "\n".join(log_msgs)
                if show_logs:
                    with st.expander("Advanced Logs", expanded=False):
                        st.markdown(f"<div class='log-box'>{joined_logs}</div>", unsafe_allow_html=True)
                
                # Save to state
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "logs": joined_logs})
            except Exception as e:
                error_msg = str(e)
                import traceback
                print("Full traceback:", traceback.format_exc())
                
                if "Model reloaded" in error_msg:
                    st.error("🔄 **LM Studio is reloading the model.** This usually means it ran out of memory.\n\n**Suggested fixes:**\n1. Lower the **Context Size** in LM Studio settings (try 2048)\n2. Disable long prompts and try again\n3. Consider using a smaller model")
                elif "insufficient_quota" in error_msg or "429" in error_msg:
                    st.error("⚠️ OpenAI API quota exceeded. Please check your billing details and add credits to your OpenAI account (platform.openai.com).")
                else:
                    st.error(f"⚠️ An error occurred during response generation:\n\n{error_msg}\n\nCheck terminal for full traceback.")
