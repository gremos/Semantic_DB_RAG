import json
import re
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class JSONExtractor:
    """Robust JSON extraction from LLM responses with multiple strategies."""
    
    @staticmethod
    def extract(response: str, log_failures: bool = True) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Extract JSON from LLM response using multiple strategies.
        
        Returns:
            (parsed_json, extraction_method_used)
        """
        if not response or not response.strip():
            if log_failures:
                logger.error("Empty response received")
            return None, "empty_response"
        
        # Strategy 1: Direct JSON parse
        result = JSONExtractor._try_direct_parse(response)
        if result:
            return result, "direct_parse"
        
        # Strategy 2: Extract from markdown code blocks
        result = JSONExtractor._extract_from_markdown(response)
        if result:
            return result, "markdown_extraction"
        
        # Strategy 3: Find JSON object boundaries
        result = JSONExtractor._extract_by_braces(response)
        if result:
            return result, "brace_matching"
        
        # Strategy 4: Remove common preambles and try again
        result = JSONExtractor._remove_preamble_and_parse(response)
        if result:
            return result, "preamble_removal"
        
        # Strategy 5: Try to fix common JSON issues
        result = JSONExtractor._fix_and_parse(response)
        if result:
            return result, "auto_fix"
        
        # All strategies failed
        if log_failures:
            logger.error("All JSON extraction strategies failed")
            logger.debug(f"Response (first 1000 chars): {response[:1000]}")
            logger.debug(f"Response (last 500 chars): {response[-500:]}")
        
        return None, "all_failed"
    
    @staticmethod
    def _try_direct_parse(text: str) -> Optional[Dict[str, Any]]:
        """Try to parse text directly as JSON."""
        try:
            return json.loads(text)
        except:
            return None
    
    @staticmethod
    def _extract_from_markdown(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        # Try ```json
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`(.*?)`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        return None
    
    @staticmethod
    def _extract_by_braces(text: str) -> Optional[Dict[str, Any]]:
        """Find JSON by matching opening and closing braces."""
        # Find first {
        start = text.find('{')
        if start == -1:
            return None
        
        # Count braces to find matching close
        brace_count = 0
        in_string = False
        escape = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
                if brace_count == 0:
                    # Found matching brace
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except:
                        return None
        
        return None
    
    @staticmethod
    def _remove_preamble_and_parse(text: str) -> Optional[Dict[str, Any]]:
        """Remove common preambles and try parsing."""
        # Common preambles to remove
        preambles = [
            "Here is the JSON:",
            "Here's the JSON:",
            "Here is the semantic model:",
            "Here's the semantic model:",
            "The semantic model is:",
            "JSON output:",
            "Output:",
            "Result:",
        ]
        
        cleaned = text
        for preamble in preambles:
            if preamble.lower() in cleaned.lower():
                idx = cleaned.lower().find(preamble.lower())
                cleaned = cleaned[idx + len(preamble):].strip()
        
        return JSONExtractor._try_direct_parse(cleaned)
    
    @staticmethod
    def _fix_and_parse(text: str) -> Optional[Dict[str, Any]]:
        """Try to fix common JSON issues."""
        # Extract potential JSON
        json_text = JSONExtractor._extract_by_braces(text)
        if not json_text:
            # Try to find anything that looks like JSON
            start = text.find('{')
            if start == -1:
                return None
            json_text = text[start:]
        
        if isinstance(json_text, dict):
            return json_text
        
        # Try common fixes
        fixes = [
            lambda s: s.replace('\n', ' '),  # Remove newlines
            lambda s: s.replace('\t', ' '),  # Remove tabs
            lambda s: re.sub(r',\s*}', '}', s),  # Remove trailing commas
            lambda s: re.sub(r',\s*]', ']', s),  # Remove trailing commas in arrays
        ]
        
        current = json_text if isinstance(json_text, str) else str(json_text)
        
        for fix in fixes:
            try:
                fixed = fix(current)
                return json.loads(fixed)
            except:
                continue
        
        return None