"""
Azure OpenAI LLM client using langchain.
"""

from langchain_openai import AzureChatOpenAI
from config.settings import Settings


class LLMClient:
    """Azure OpenAI client wrapper."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AzureChatOpenAI(
            azure_deployment=settings.DEPLOYMENT_NAME,
            api_version=settings.API_VERSION,
            azure_endpoint=settings.AZURE_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            # Note: gpt-5-mini doesn't support temperature parameter
        )
    
    def invoke(self, prompt: str) -> str:
        """Send prompt to LLM and return response."""
        response = self.client.invoke(prompt)
        return response.content
    
    def invoke_with_json(self, prompt: str, max_retries: int = 3) -> dict:
        """
        Send prompt expecting JSON response with retry logic.
        TODO: Implement JSON schema validation with retries.
        """
        # Stub implementation
        return {}
