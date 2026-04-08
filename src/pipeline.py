import streamlit as st
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app_config import config
from src.ingestion import DocumentIngestor
from src.retrieval import AdvancedRetriever
from src.generation import ResponseGenerator
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
                
                # OPTIMIZED: Single LLM call for language detection + query expansion
                lang = generator.detect_language(user_input)
                is_vietnamese = lang == 'vi'
                log_msgs.append(f"[Log] Detected language: {lang}")
                
                # LLM Call #1: Translate (if VI) + Expand query
                log_msgs.append("[Log] ✓ LLM Call #1: Translate + Expand query")
                translated_query, expanded_query = generator.translate_and_expand_query(user_input)
                if is_vietnamese:
                    log_msgs.append(f"[Log]   └─ Translated: {user_input[:50]}...")
                    log_msgs.append(f"[Log]   └─ Expanded: {expanded_query[:50]}...")
                else:
                    log_msgs.append(f"[Log]   └─ Expanded: {expanded_query[:50]}...")
                
                t1 = time.time()
                log_msgs.append(f"[Log] Query processing time: {t1-t0:.3f}s")
                
                # Vector search (no LLM call)
                docs = retriever.invoke(expanded_query)
                t2 = time.time()
                log_msgs.append(f"[Log] Vector search time: {t2-t1:.3f}s")
                
                # LLM Call #2: Generate response (+ translate if VI)
                log_msgs.append("[Log] ✓ LLM Call #2: Generate response" + (" + Translate to VI" if is_vietnamese else ""))
                answer = generator.generate_response(user_input, expanded_query, retriever, severe_level, mental_status, is_vietnamese=is_vietnamese)
                t3 = time.time()
                log_msgs.append(f"[Log] Response generation time: {t3-t2:.3f}s")
                log_msgs.append(f"[Log] ───────────────────────────────────")
                log_msgs.append(f"[Log] Total LLM calls: {2 if is_vietnamese else 2} (Optimized ⚡)")
                log_msgs.append(f"[Log] Total time: {t3-t0:.3f}s")
                
                # Display final answer
                st.markdown(answer)
                
                joined_logs = "\n".join(log_msgs)
                if show_logs:
                    with st.expander("Advanced Logs", expanded=False):
                        st.markdown(f"<div class='log-box'>{joined_logs}</div>", unsafe_allow_html=True)
                
                # Save to state
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "logs": joined_logs})
            except Exception as e:
                error_msg = str(e)
                import re
                
                # Detailed error handling for different error types
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    st.error(f"""⚠️ **Request Timeout Error**: The LM Studio API took too long to respond.
                    
**Troubleshooting:**
1. ✓ Is LM Studio running? (Check http://127.0.0.1:1234 in your browser)
2. ✓ Increase timeout: Set `LM_STUDIO_TIMEOUT=600` in your .env (currently 300s)
3. ✓ Reduce model size: Use a smaller model (gpt2, mistral instead of 20B)
4. ✓ Check GPU/CPU usage: Local LLM inference can be slow on CPU
5. ✓ Restart LM Studio: Sometimes it needs to be restarted

**Technical Details**: {error_msg}""")
                elif "insufficient_quota" in error_msg or "429" in error_msg:
                    st.error("⚠️ API quota exceeded. Please check your billing details.")
                elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                    st.error(f"""⚠️ **Connection Error**: Cannot reach the LM Studio API.
                    
**Make sure:**
1. ✓ LM Studio is running (http://127.0.0.1:1234)
2. ✓ API endpoint: LM_STUDIO_API_URL in .env is correct (default: http://127.0.0.1:1234/v1)
3. ✓ No firewall blocking localhost:1234

**Error Details**: {error_msg}""")
                else:
                    st.error(f"⚠️ An error occurred: {error_msg}")
                
                # Log the error for debugging
                log_msgs.append(f"[ERROR] {error_msg}")
                log_msgs.append(f"[ERROR] Error type: {type(e).__name__}")
