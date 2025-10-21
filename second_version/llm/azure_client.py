from langchain_openai import AzureChatOpenAI
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class AzureLLMClient:
    """Wrapper for Azure OpenAI via LangChain."""
    
    def __init__(self):
        self.client = AzureChatOpenAI(
            deployment_name=settings.deployment_name,
            api_version=settings.api_version, 
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.azure_api_key,
            temperature=0,
            max_tokens=4000
        )
    
    def generate(self, system_prompt: str, user_content: str) -> str:
        """
        Generate completion with system + user messages.
        
        Returns: Raw text response.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            response = self.client.invoke(messages)
            return response.content
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise