try:
    from langchain_core.prompts import PromptTemplate
except (ImportError, AttributeError):
    try:
        from langchain.prompts import PromptTemplate
    except (ImportError, AttributeError):
        from langchain_core.prompts.template import PromptTemplate
        
try:
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    from langchain.output_parsers import StrOutputParser
    
import langdetect
from langdetect import detect, LangDetectException
import time

class ResponseGenerator:
    """Responsible for prompt creation, memory management, and LLM calls."""
    # Initialize with LLM and memory for conversation history. Memory keeps the last 5 interactions for context.
    def __init__(self, llm):
        self.llm = llm
        self.chat_history = []  # Simple list to store conversation history
        self.k = 5  # Keep last 5 interactions
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def _invoke_with_retry(self, chain, input_data, max_retries=None):
        """Invoke LLM chain with retry logic for model reload errors."""
        if max_retries is None:
            max_retries = self.max_retries
            
        for attempt in range(max_retries):
            try:
                response = chain.invoke(input_data)
                if response and response.strip():  # Check response is not empty
                    return response
                print(f"[Attempt {attempt+1}] Empty response from LLM, retrying...")
            except Exception as e:
                error_str = str(e)
                if "Model reloaded" in error_str or "error" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"[Attempt {attempt+1}] Model error: {error_str[:100]}... Retrying in {self.retry_delay}s")
                        time.sleep(self.retry_delay)
                        continue
                raise
        
        return ""  # Return empty string if all retries failed
    
    def _add_to_history(self, message: str):
        """Add a message to chat history, keeping only last k interactions."""
        self.chat_history.append(message)
        if len(self.chat_history) > self.k * 2:  # k conversations = k*2 messages (user + assistant)
            self.chat_history = self.chat_history[-self.k*2:]
    
    def _get_recent_history(self) -> list:
        """Get recent chat history for context."""
        return self.chat_history[-self.k*2:] if self.chat_history else []

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        Returns 'vi' for Vietnamese, 'en' for English, or defaults to 'en'.
        """
        try:
            detected_lang = detect(text)
            return detected_lang
        except LangDetectException:
            return 'en'  # Default to English if detection fails

    def translate_query_to_english(self, query: str) -> str:
        """
        If query is in Vietnamese, translate it to English.
        Uses LLM as translator to maintain context awareness.
        """
        lang = self.detect_language(query)
        
        if lang == 'vi':
            translation_prompt = PromptTemplate.from_template(
                """You are a professional translator. 
                Translate the following Vietnamese text to English while preserving the original meaning and medical/psychological context.
                
                Vietnamese text: {query}
                English translation:"""
            )
            chain = translation_prompt | self.llm | StrOutputParser()
            translated_query = chain.invoke({"query": query})
            return translated_query
        
        # Already in English, return as is
        return query

    def expand_query(self, user_query: str) -> str:
        """
        Expand and clarify the user's query using LLM.
        Identifies mental health conditions (Depression, Anxiety, Bipolar, Suicidal, etc.)
        and adds relevant psychological keywords to optimize vector search.
        
        This uses the translated (English) version for consistency with English PDF documents.
        """
        # First, translate Vietnamese query to English
        english_query = self.translate_query_to_english(user_query)
        
        rewrite_prompt = PromptTemplate.from_template(
            """You are an expert mental health professional and psychologist. 
            Your task is to expand and clarify the user's question to optimize searching in a medical/psychological database.
            
            Identify which mental health condition the user might be experiencing (e.g., Depression, Anxiety, PTSD, Bipolar Disorder, Personality Disorder, Suicidal Ideation, Stress, etc.)
            and add relevant psychological keywords accordingly.
            
            Examples:
            - "I feel sad and hopeless" -> "depression, low mood, hopelessness, anhedonia, negative thoughts, persistent sadness"
            - "I'm worried about everything" -> "anxiety, generalized anxiety disorder, worry, nervousness, panic, stress"
            - "I want to hurt myself" -> "suicidal ideation, self-harm, self-injury, suicide risk, mental health crisis"
            
            Return ONLY the expanded query without explanations.
            
            Original user query: {query}
            Expanded query:"""
        )
        
        chain = rewrite_prompt | self.llm | StrOutputParser()
        expanded_query = chain.invoke({"query": english_query})
        return expanded_query

    def generate_response(self, user_query: str, expanded_query: str, retriever, severe_level: str, mental_status: str) -> str:
        """
        Generate the final response using retrieved context and conversation history.
        Converts response to original language if user queried in Vietnamese.
        """
        docs = retriever.invoke(expanded_query)
        # Limit context to first 1 doc only to save memory
        docs = docs[:1]
        context = docs[0].page_content if docs else "No relevant information found."
        
        # Get recent chat history (limit to just last exchange)
        chat_history_list = self._get_recent_history()[-2:] if self._get_recent_history() else []
        chat_history_text = "\n".join(chat_history_list) if chat_history_list else ""
        
        # Ultra-minimal system prompt
        system_prompt = """You are a helpful mental health assistant. Give brief, supportive advice based on the context.

Context: {context}

Chat: {chat_history}"""
        
        prompt_template = PromptTemplate.from_template(system_prompt + "\n\nUser: {query}\nAssistant:")
        
        response = self._invoke_with_retry(
            (prompt_template | self.llm | StrOutputParser()),
            {
                "context": context[:500],  # Truncate context to 500 chars max
                "chat_history": chat_history_text[:200],  # Truncate history to 200 chars
                "query": user_query[:100]  # Truncate query to 100 chars
            }
        )
        
        # Save to chat history
        self._add_to_history(f"User: {user_query}")
        if response:
            self._add_to_history(f"Assistant: {response[:200]}")  # Store only first 200 chars
        
        # Translate response back to Vietnamese if user queried in Vietnamese
        user_lang = self.detect_language(user_query)
        if user_lang == 'vi' and response:
            response = self.translate_response_to_vietnamese(response)
        
        return response

    def translate_response_to_vietnamese(self, response: str) -> str:
        """
        Translate English response back to Vietnamese for Vietnamese-speaking users.
        """
        # Truncate very long responses to save memory
        response = response[:300]
        
        translation_prompt = PromptTemplate.from_template(
            """Translate to Vietnamese:
            
English: {response}
Vietnamese:"""
        )
        chain = translation_prompt | self.llm | StrOutputParser()
        translated_response = self._invoke_with_retry(
            chain,
            {"response": response},
            max_retries=2
        )
        return translated_response if translated_response else response
