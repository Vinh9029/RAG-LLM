from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
import langdetect
from langdetect import detect, LangDetectException
from functools import lru_cache
import json
import os

class ResponseGenerator:
    """Responsible for prompt creation, memory management, and LLM calls (OPTIMIZED - 2 calls max)."""
    # Initialize with LLM and memory for conversation history. Memory keeps the last 5 interactions for context.
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history"
        )
        
        # LRU Cache for language detection
        self._lang_cache = {}
        self.MAX_CACHE_SIZE = 256

    @lru_cache(maxsize=128)
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text with caching.
        Returns 'vi' for Vietnamese, 'en' for English, or defaults to 'en'.
        """
        try:
            detected_lang = detect(text[:100])  # Use first 100 chars to speed up detection
            return detected_lang
        except LangDetectException:
            return 'en'

    def _translate_with_llm(self, text: str, target_lang: str, source_lang: str = 'vi') -> str:
        """
        Translate using LLM with minimal token usage (extreme optimization).
        Returns ONLY translated text, no explanation.
        """
        translation_prompt = PromptTemplate.from_template(
            "Translate {source_lang} to {target_lang}:\n{text}"
        )
        chain = translation_prompt | self.llm | StrOutputParser()
        translated = chain.invoke({"text": text, "source_lang": source_lang, "target_lang": target_lang})
        return translated.strip()

    def translate_and_expand_query(self, user_query: str) -> tuple:
        """
        VIETNAMESE OPTIMIZATION (LLM Call #1):
        - Translate query to English
        - Identify mental health keywords in SAME call
        - Returns: (translated_query, english_query)
        
        This replaces 2 separate calls (translate + expand) into 1 call.
        """
        lang = self.detect_language(user_query)
        
        if lang == 'vi':
            # Single LLM call: translate + identify keywords
            combined_prompt = PromptTemplate.from_template(
                """Translate Vietnamese to English and add mental health keywords.
                Return ONLY: "TRANSLATION: [english text] | KEYWORDS: [comma-separated keywords]"
                
                Vietnamese query:
                {query}"""
            )
            chain = combined_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": user_query})
            
            # Parse result format: "TRANSLATION: ... | KEYWORDS: ..."
            try:
                parts = result.split(" | KEYWORDS: ")
                translated = parts[0].replace("TRANSLATION: ", "").strip()
                keywords = parts[1].strip() if len(parts) > 1 else ""
                expanded = f"{translated}, {keywords}" if keywords else translated
                return translated, expanded
            except:
                return result.strip(), result.strip()
        
        # For English, just identify keywords
        keyword_prompt = PromptTemplate.from_template(
            """Identify mental health keywords and expand user query.
            Return ONLY expanded query with keywords, no explanation.
            
            User query:
            {query}"""
        )
        chain = keyword_prompt | self.llm | StrOutputParser()
        expanded = chain.invoke({"query": user_query})
        return user_query, expanded.strip()

    def generate_response(self, user_query: str, expanded_query: str, retriever, severe_level: str, mental_status: str, is_vietnamese: bool = False) -> str:
        """
        VIETNAMESE OPTIMIZATION (LLM Call #2):
        - Generate mental health response using context
        - If Vietnamese: ALSO translate response back to Vietnamese in SAME call
        - This replaces 2 separate calls (generate + translate) into 1 call
        
        Args:
            user_query: Original user query (for memory)
            expanded_query: Expanded/translated query for retrieval
            retriever: Vector store retriever
            severe_level: Mental health severity level
            mental_status: Mental health condition type
            is_vietnamese: Whether to translate response to Vietnamese
        """
        docs = retriever.invoke(expanded_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        if is_vietnamese:
            # COMBINED: Generate response EN + Return in VI (single LLM call)
            system_prompt = """You are a compassionate, empathetic mental health assistant.
User's severity: {severe_level}
User's mental status: {mental_status}

CRITICAL:
1. If the user shows signs of suicidal ideation or self-harm, IMMEDIATELY provide crisis helpline numbers (e.g., National Suicide Prevention Lifeline: 988 in the US, or equivalent in the user's country) BEFORE any other response.
2. Always remind users that you are an AI assistant and cannot replace professional mental health care from licensed therapists or psychiatrists.
3. Respond with warmth, positivity, and hope. If the context lacks relevant information, acknowledge this and recommend consulting a mental health professional.
4. Respect cultural and individual differences in mental health experiences.
5. Never provide medical diagnoses; instead, suggest symptoms to discuss with a healthcare provider.

Retrieved context:
{context}

---AFTER YOUR RESPONSE, TRANSLATE IT TO VIETNAMESE---
Format: [RESPONSE_EN]: [English response] | [RESPONSE_VI]: [Vietnamese response only]"""
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}")
            ])
            
            combined_response = (prompt_template | self.llm | StrOutputParser()).invoke({
                "context": context,
                "chat_history": chat_history,
                "query": expanded_query,
                "severe_level": severe_level,
                "mental_status": mental_status
            })
            
            # Extract Vietnamese response
            try:
                parts = combined_response.split(" | [RESPONSE_VI]: ")
                response = parts[1].strip() if len(parts) > 1 else parts[0]
            except:
                response = combined_response.strip()
        
        else:
            # English only (single LLM call)
            system_prompt = """You are a compassionate, empathetic, and non-judgmental Virtual Assistant for Mental Health Support.
Your role is to provide evidence-based advice, therapeutic exercises, and coping strategies based on the provided context.

User's severe level: {severe_level}
User's mental health status: {mental_status}

CRITICAL SAFETY RULES:
1. If the user shows signs of suicidal ideation or self-harm, IMMEDIATELY provide crisis helpline numbers (e.g., National Suicide Prevention Lifeline: 988 in the US, or equivalent in the user's country) BEFORE any other response.
2. Always remind users that you are an AI assistant and cannot replace professional mental health care from licensed therapists or psychiatrists.
3. Respond with warmth, positivity, and hope. If the context lacks relevant information, acknowledge this and recommend consulting a mental health professional.
4. Respect cultural and individual differences in mental health experiences.
5. Never provide medical diagnoses; instead, suggest symptoms to discuss with a healthcare provider.

Retrieved context for reference:
{context}"""
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "User: {query}")
            ])
            
            response = (prompt_template | self.llm | StrOutputParser()).invoke({
                "context": context,
                "chat_history": chat_history,
                "query": expanded_query,
                "severe_level": severe_level,
                "mental_status": mental_status
            })
        
        # Save original user query to memory for consistency
        self.memory.save_context({"input": user_query}, {"output": response})
        return response



