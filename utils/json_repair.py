"""
JSON Repair Utility for handling malformed LLM responses.
Provides robust parsing with multiple fallback strategies.
"""

import json
import re
from typing import Dict, Any, Optional, List


class JSONRepairError(Exception):
    """Custom exception for JSON repair failures."""
    pass


def strip_markdown_code_blocks(text: str) -> str:
    """
    Remove markdown code block delimiters from text.

    Args:
        text: Text potentially containing markdown code blocks

    Returns:
        Cleaned text without code block markers
    """
    # Remove ```json and ``` markers
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return text.strip()


def fix_common_json_errors(text: str) -> str:
    """
    Fix common JSON syntax errors.

    Args:
        text: Potentially malformed JSON string

    Returns:
        JSON string with common errors fixed
    """
    # Fix trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Fix single quotes to double quotes (but be careful with apostrophes)
    # This is a simple approach - more sophisticated would use state machine
    text = text.replace("'", '"')

    # Fix unquoted keys (simple pattern - may not catch all cases)
    text = re.sub(r'(\w+):', r'"\1":', text)

    # Fix multiple consecutive commas
    text = re.sub(r',\s*,', ',', text)

    # Fix missing commas between key-value pairs (very basic)
    text = re.sub(r'"\s*"', '","', text)

    return text


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain additional content.

    Args:
        text: Text potentially containing JSON

    Returns:
        Extracted JSON string or None
    """
    # Try to find JSON object using regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # Return the longest match (likely the most complete)
        return max(matches, key=len)

    return None


def parse_json_with_repair(text: str, max_attempts: int = 9) -> Dict[str, Any]:
    """
    Parse JSON with multiple repair strategies.

    Args:
        text: JSON string or text containing JSON
        max_attempts: Maximum repair attempts to make

    Returns:
        Parsed JSON dictionary

    Raises:
        JSONRepairError: If all repair strategies fail
    """
    if not text or not isinstance(text, str):
        raise JSONRepairError("Input is empty or not a string")

    original_text = text
    strategies = []

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        strategies.append(f"Direct parse failed: {e}")

    # Strategy 2: Strip whitespace and try again
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        strategies.append(f"Stripped whitespace parse failed: {e}")

    # Strategy 3: Remove markdown code blocks
    try:
        cleaned = strip_markdown_code_blocks(text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        strategies.append(f"Markdown removal parse failed: {e}")

    # Strategy 4: Extract JSON from text
    try:
        extracted = extract_json_from_text(text)
        if extracted:
            return json.loads(extracted)
    except (json.JSONDecodeError, TypeError) as e:
        strategies.append(f"JSON extraction parse failed: {e}")

    # Strategy 5: Fix common errors
    try:
        fixed = fix_common_json_errors(text)
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        strategies.append(f"Common error fix parse failed: {e}")

    # Strategy 6: Combine markdown removal + error fixes
    try:
        cleaned = strip_markdown_code_blocks(text)
        fixed = fix_common_json_errors(cleaned)
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        strategies.append(f"Combined cleanup parse failed: {e}")

    # Strategy 7: Extract + fix errors
    try:
        extracted = extract_json_from_text(text)
        if extracted:
            fixed = fix_common_json_errors(extracted)
            return json.loads(fixed)
    except (json.JSONDecodeError, TypeError) as e:
        strategies.append(f"Extract + fix parse failed: {e}")

    # Strategy 8: Try to parse just the content between first { and last }
    try:
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace:last_brace + 1]
            return json.loads(candidate)
    except json.JSONDecodeError as e:
        strategies.append(f"Brace extraction parse failed: {e}")

    # Strategy 9: Last resort - aggressive cleanup
    try:
        # Remove all content before first {
        first_brace = text.find('{')
        if first_brace > 0:
            text = text[first_brace:]

        # Remove all content after last }
        last_brace = text.rfind('}')
        if last_brace != -1:
            text = text[:last_brace + 1]

        # Apply all fixes
        text = strip_markdown_code_blocks(text)
        text = fix_common_json_errors(text)

        return json.loads(text)
    except json.JSONDecodeError as e:
        strategies.append(f"Aggressive cleanup parse failed: {e}")

    # All strategies failed
    error_msg = "\n".join([f"  - {s}" for s in strategies])
    raise JSONRepairError(
        f"Failed to parse JSON after {len(strategies)} attempts:\n{error_msg}\n\nOriginal text:\n{original_text[:500]}"
    )


def validate_decision_structure(data: Dict[str, Any]) -> bool:
    """
    Validate that a dictionary has the expected structure for a decision.

    Args:
        data: Dictionary to validate

    Returns:
        True if valid decision structure
    """
    required_keys = ["continue_task", "action_type"]
    optional_keys = ["target_element", "input_text", "reasoning", "friction_noted"]

    # Check required keys exist
    if not all(key in data for key in required_keys):
        return False

    # Validate types
    if not isinstance(data["continue_task"], bool):
        return False

    valid_action_types = ["click", "scroll", "type", "wait", "abandon"]
    if data["action_type"] not in valid_action_types:
        return False

    return True


def validate_analysis_structure(data: Dict[str, Any]) -> bool:
    """
    Validate that a dictionary has the expected structure for page analysis.

    Args:
        data: Dictionary to validate

    Returns:
        True if valid analysis structure
    """
    required_keys = ["visible_elements", "relevant_to_task"]

    # Check required keys exist
    if not all(key in data for key in required_keys):
        return False

    # Validate types
    if not isinstance(data["visible_elements"], list):
        return False

    if not isinstance(data["relevant_to_task"], list):
        return False

    return True


def repair_and_validate_decision(text: str) -> Dict[str, Any]:
    """
    Parse and validate a decision response from LLM.

    Args:
        text: JSON text from LLM response

    Returns:
        Valid decision dictionary

    Raises:
        JSONRepairError: If parsing or validation fails
    """
    data = parse_json_with_repair(text)

    if not validate_decision_structure(data):
        raise JSONRepairError(f"Invalid decision structure: {data}")

    return data


def repair_and_validate_analysis(text: str) -> Dict[str, Any]:
    """
    Parse and validate an analysis response from LLM.

    Args:
        text: JSON text from LLM response

    Returns:
        Valid analysis dictionary

    Raises:
        JSONRepairError: If parsing or validation fails
    """
    data = parse_json_with_repair(text)

    if not validate_analysis_structure(data):
        raise JSONRepairError(f"Invalid analysis structure: {data}")

    return data


def safe_parse_llm_response(
    text: str,
    expected_type: str = "decision",
    fallback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely parse LLM response with fallback on failure.

    Args:
        text: Response text from LLM
        expected_type: Type of response ("decision" or "analysis")
        fallback: Fallback dictionary to return on failure

    Returns:
        Parsed and validated dictionary, or fallback
    """
    if fallback is None:
        fallback = {
            "continue_task": False,
            "action_type": "wait",
            "target_element": None,
            "input_text": None,
            "reasoning": "Failed to parse LLM response",
            "friction_noted": "JSON parsing error",
            "error": True
        }

    try:
        if expected_type == "decision":
            return repair_and_validate_decision(text)
        elif expected_type == "analysis":
            return repair_and_validate_analysis(text)
        else:
            return parse_json_with_repair(text)

    except JSONRepairError as e:
        print(f"JSON repair failed: {e}")
        return fallback
    except Exception as e:
        print(f"Unexpected error during JSON parsing: {e}")
        return fallback
