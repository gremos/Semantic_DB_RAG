"""
Unified LLM Client - Supports Azure OpenAI and Azure-hosted Claude

Provides a consistent interface for both LLM providers:
- Azure OpenAI (GPT-4, GPT-5, etc.)
- Azure Claude (Anthropic models hosted on Azure AI Services)

Usage:
    from src.llm.client import get_llm_client

    client = get_llm_client()  # Uses LLM_PROVIDER from .env
    response = client.invoke("Your prompt here")
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

import httpx
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Azure Foundry Claude SDK
try:
    from anthropic import AnthropicFoundry
    HAS_ANTHROPIC_FOUNDRY = True
except ImportError:
    HAS_ANTHROPIC_FOUNDRY = False

from config.settings import (
    Settings,
    LLMConfig,
    AzureOpenAIConfig,
    AzureClaudeConfig,
    get_settings,
    get_llm_config
)

logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT BASE CLIENT
# ============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to LLM and return response text"""
        pass

    @abstractmethod
    def invoke_with_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Send prompt expecting JSON response with retry logic"""
        pass

    @abstractmethod
    def invoke_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Send a list of messages (chat format) and return response"""
        pass


# ============================================================================
# AZURE OPENAI CLIENT
# ============================================================================

class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI client using LangChain"""

    def __init__(self, config: AzureOpenAIConfig):
        self.config = config
        self.client = AzureChatOpenAI(
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )
        logger.info(f"Initialized Azure OpenAI client: {config.deployment_name}")

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to Azure OpenAI and return response"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.client.invoke(messages)
        return response.content

    def invoke_with_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Send prompt expecting JSON response with retry logic"""
        json_system = (system_prompt or "") + "\n\nRespond with valid JSON only."

        for attempt in range(max_retries):
            try:
                response = self.invoke(prompt, system_prompt=json_system)

                # Extract JSON from response (handle markdown code blocks)
                json_str = self._extract_json(response)
                return json.loads(json_str)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

        return {}

    def invoke_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Send chat messages to Azure OpenAI"""
        langchain_messages = []

        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        response = self.client.invoke(langchain_messages)
        return response.content

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text (handles markdown code blocks)"""
        text = text.strip()

        # Handle ```json ... ``` blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Handle ``` ... ``` blocks
        if text.startswith("```") and text.endswith("```"):
            return text[3:-3].strip()

        # Handle raw JSON
        if text.startswith("{") or text.startswith("["):
            return text

        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

        return text


# ============================================================================
# AZURE CLAUDE CLIENT (using AnthropicFoundry SDK)
# ============================================================================

class AzureClaudeClient(BaseLLMClient):
    """
    Azure-hosted Claude (Anthropic) client using AnthropicFoundry SDK

    Uses the official Anthropic SDK for Azure Foundry integration.
    Endpoint format: https://<resource>.services.ai.azure.com/anthropic/
    """

    def __init__(self, config: AzureClaudeConfig):
        self.config = config

        if not HAS_ANTHROPIC_FOUNDRY:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        # Extract resource name from endpoint URL
        # e.g., https://gremo-miu3pl8n-eastus2.services.ai.azure.com -> gremo-miu3pl8n-eastus2
        import os
        import re

        endpoint = config.endpoint.rstrip('/')
        resource_match = re.match(r'https://([^.]+)\.services\.ai\.azure\.com', endpoint)

        if resource_match:
            # Use resource parameter (preferred for anthropic>=0.75)
            resource_name = resource_match.group(1)
            # Clear env var if set to avoid conflict (library bug workaround)
            os.environ.pop('ANTHROPIC_FOUNDRY_RESOURCE', None)
            self.client = AnthropicFoundry(
                api_key=config.api_key,
                resource=resource_name
            )
        else:
            # Fallback to base_url for non-standard endpoints
            base_url = endpoint
            if not base_url.endswith('/anthropic'):
                base_url = f"{base_url}/anthropic/"
            else:
                base_url = f"{base_url}/"
            os.environ.pop('ANTHROPIC_FOUNDRY_RESOURCE', None)
            self.client = AnthropicFoundry(
                api_key=config.api_key,
                base_url=base_url
            )
        logger.info(f"Initialized Azure Claude (Foundry) client: {config.model}")

    def invoke(self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 3) -> str:
        """Send prompt to Azure Claude and return response with retry on transient errors"""
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,  # Use configured temperature for determinism
            "messages": [{"role": "user", "content": prompt}]
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(**kwargs)
                return self._extract_response_text(message)
            except Exception as e:
                error_str = str(e).lower()
                # Retry on 500/502/503/529 errors (transient Azure issues)
                is_transient = any(code in error_str for code in ['500', '502', '503', '529', 'overloaded', 'rate'])
                if is_transient and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1  # 2, 3, 5 seconds
                    logger.warning(f"Transient API error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise

    def invoke_with_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Send prompt expecting JSON response with retry logic"""
        json_system = (system_prompt or "") + "\n\nRespond with valid JSON only. No markdown formatting."

        for attempt in range(max_retries):
            try:
                response = self.invoke(prompt, system_prompt=json_system)

                # Extract JSON from response
                json_str = self._extract_json(response)

                # Try to parse, with repair fallback
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as parse_error:
                    # Log malformed JSON for debugging
                    logger.debug(f"=== MALFORMED JSON DEBUG (attempt {attempt + 1}) ===")
                    logger.debug(f"Parse error: {parse_error}")
                    logger.debug(f"Raw response length: {len(response)} chars")
                    logger.debug(f"Extracted JSON length: {len(json_str)} chars")
                    logger.debug(f"First 500 chars: {json_str[:500]}")
                    logger.debug(f"Last 500 chars: {json_str[-500:] if len(json_str) > 500 else json_str}")
                    logger.debug(f"=== END MALFORMED JSON DEBUG ===")

                    # Try to repair common JSON issues
                    repaired = self._repair_json(json_str)
                    logger.debug(f"Repaired JSON length: {len(repaired)} chars")
                    return json.loads(repaired)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
                # Log the problematic content to file for analysis
                self._log_malformed_json(response if 'response' in dir() else "N/A", e, attempt)
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        return {}

    def _log_malformed_json(self, response: str, error: Exception, attempt: int):
        """Log malformed JSON to a debug file for analysis"""
        try:
            from pathlib import Path
            from datetime import datetime

            debug_dir = Path("cache/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"malformed_json_{timestamp}_attempt{attempt}.txt"

            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"=== MALFORMED JSON DEBUG ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Attempt: {attempt + 1}\n")
                f.write(f"Error: {error}\n")
                f.write(f"Response length: {len(response)} chars\n")
                f.write(f"\n=== RAW RESPONSE ===\n")
                f.write(response)
                f.write(f"\n\n=== EXTRACTED JSON ===\n")
                f.write(self._extract_json(response))

            logger.info(f"Malformed JSON logged to: {debug_file}")
        except Exception as log_error:
            logger.debug(f"Could not log malformed JSON: {log_error}")

    def _repair_json(self, text: str) -> str:
        """Attempt to repair common JSON formatting issues including truncation"""
        import re

        logger.debug(f"Attempting JSON repair on {len(text)} chars")

        # Remove trailing commas before } or ]
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Fix missing commas between array elements: ]["  or }{"
        text = re.sub(r'(\])\s*(\[)', r'\1,\2', text)
        text = re.sub(r'(\})\s*(\{)', r'\1,\2', text)

        # Fix missing commas between string values: "value"  "key"
        text = re.sub(r'(")\s+(")', r'\1,\2', text)

        # Check for truncation (unclosed structures)
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')

        # Also check for unclosed strings (odd number of unescaped quotes)
        in_string = False
        last_quote_pos = -1
        i = 0
        while i < len(text):
            c = text[i]
            if c == '\\' and in_string:
                i += 2  # Skip escaped character
                continue
            if c == '"':
                in_string = not in_string
                last_quote_pos = i
            i += 1

        # If we're still in a string, the JSON is truncated mid-string
        if in_string:
            logger.debug(f"JSON truncated mid-string at position {last_quote_pos}")
            # Find the last complete object/array element (before the unclosed string)
            # Look backwards for the last complete structure
            truncate_pos = text.rfind('",', 0, last_quote_pos)
            if truncate_pos == -1:
                truncate_pos = text.rfind('"}', 0, last_quote_pos)
            if truncate_pos == -1:
                truncate_pos = text.rfind('"]', 0, last_quote_pos)

            if truncate_pos > 0:
                # Keep up to the last complete element
                text = text[:truncate_pos + 2]  # Include the closing quote and comma/bracket
                logger.debug(f"Truncated to last complete element at position {truncate_pos + 2}")

                # Recalculate unclosed structures
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')

        # Close unclosed structures
        if open_braces > 0 or open_brackets > 0:
            logger.debug(f"Closing {open_brackets} brackets and {open_braces} braces")

            # Find last complete structure by looking for patterns
            # If there's trailing incomplete content after last comma, remove it
            last_comma = text.rfind(',')
            if last_comma > 0:
                remainder = text[last_comma+1:].strip()
                # Check if remainder looks incomplete (has unclosed braces/brackets/quotes)
                rem_braces = remainder.count('{') - remainder.count('}')
                rem_brackets = remainder.count('[') - remainder.count(']')
                rem_quotes = remainder.count('"') % 2

                if rem_braces > 0 or rem_brackets > 0 or rem_quotes > 0:
                    # Remainder is incomplete, truncate at last comma
                    text = text[:last_comma]
                    logger.debug(f"Truncated incomplete remainder at comma position {last_comma}")

                    # Recalculate
                    open_braces = text.count('{') - text.count('}')
                    open_brackets = text.count('[') - text.count(']')

            # Close remaining open structures
            text += ']' * open_brackets
            text += '}' * open_braces
            logger.debug(f"Added closing chars: {']' * open_brackets}{'}' * open_braces}")

        return text

    def invoke_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Send chat messages to Azure Claude"""
        # Convert to Anthropic format
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic only supports "user" and "assistant" roles in messages
            if role == "system":
                # Prepend to system prompt
                system_prompt = (system_prompt or "") + "\n" + content
            else:
                anthropic_messages.append({
                    "role": role if role in ["user", "assistant"] else "user",
                    "content": content
                })

        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,  # Use configured temperature for determinism
            "messages": anthropic_messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt.strip()

        message = self.client.messages.create(**kwargs)
        return self._extract_response_text(message)

    def _extract_response_text(self, message) -> str:
        """Extract text content from Claude API response"""
        # SDK returns a Message object with content list
        if hasattr(message, 'content'):
            text_parts = []
            for block in message.content:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
                elif hasattr(block, 'type') and block.type == 'text':
                    text_parts.append(getattr(block, 'text', ''))
            return "\n".join(text_parts)

        return str(message)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text, handling extra data after JSON"""
        text = text.strip()

        # Handle ```json ... ``` blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Handle ``` ... ``` blocks
        if text.startswith("```") and text.endswith("```"):
            return text[3:-3].strip()

        # Find the JSON object/array by matching braces
        start_char = None
        end_char = None

        for i, c in enumerate(text):
            if c == '{':
                start_char = '{'
                end_char = '}'
                start = i
                break
            elif c == '[':
                start_char = '['
                end_char = ']'
                start = i
                break

        if start_char is None:
            return text

        # Find matching closing brace by counting nesting
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            c = text[i]

            if escape_next:
                escape_next = False
                continue

            if c == '\\' and in_string:
                escape_next = True
                continue

            if c == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

        # Fallback: return from start to last occurrence of end char
        end = text.rfind(end_char) + 1
        if end > start:
            return text[start:end]

        return text


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """
    Create appropriate LLM client based on configuration

    Args:
        config: LLMConfig with provider and credentials

    Returns:
        BaseLLMClient instance (AzureOpenAIClient or AzureClaudeClient)
    """
    if config.provider == "azure_claude":
        return AzureClaudeClient(config.azure_claude)
    else:
        return AzureOpenAIClient(config.azure_openai)


def get_llm_client() -> BaseLLMClient:
    """
    Get LLM client using current settings

    Convenience function that reads LLM_PROVIDER from environment
    and returns the appropriate client.

    Returns:
        BaseLLMClient instance
    """
    config = get_llm_config()
    return create_llm_client(config)


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

class LLMClient:
    """
    Legacy LLM client wrapper for backward compatibility

    Deprecated: Use get_llm_client() instead
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = create_llm_client(settings.llm)
        logger.warning(
            "LLMClient is deprecated. Use get_llm_client() or create_llm_client() instead."
        )

    def invoke(self, prompt: str) -> str:
        """Send prompt to LLM and return response."""
        return self._client.invoke(prompt)

    def invoke_with_json(self, prompt: str, max_retries: int = 3) -> dict:
        """Send prompt expecting JSON response with retry logic."""
        return self._client.invoke_with_json(prompt, max_retries=max_retries)
