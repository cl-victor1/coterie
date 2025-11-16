"""
Google Gemini client for vision analysis of screenshots.
Uses gemini-2.5-pro for analyzing webpage screenshots.
Enhanced with coordinate-based element detection and friction point analysis.
"""

import os
import json
import base64
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
from utils.json_repair import safe_parse_llm_response, parse_json_with_repair

load_dotenv()


class GeminiClient:
    """Client for interacting with Google Gemini API for vision analysis."""

    def __init__(self):
        """Initialize Gemini client with API key from environment."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        print("Gemini client initialized successfully")

    def analyze_screenshot(self, screenshot_bytes: bytes, prompt: str, device_type: str = "desktop") -> Dict[str, Any]:
        """
        Analyze a screenshot using Gemini's vision capabilities.

        Args:
            screenshot_bytes: Raw PNG bytes of the screenshot
            prompt: Analysis prompt with persona context
            device_type: "desktop" or "mobile" to determine appropriate image size

        Returns:
            Structured analysis of the screenshot
        """
        try:
            if not screenshot_bytes:
                raise ValueError("Screenshot bytes are empty")

            with Image.open(io.BytesIO(screenshot_bytes)) as img:
                # Set max size based on device type to avoid unnecessary scaling
                if device_type == "mobile":
                    # Mobile: preserve original size, avoid upscaling small screenshots
                    max_size = (800, 1600)  # Sufficient for iPhone and similar devices
                else:
                    # Desktop: limit oversized screenshots
                    max_size = (1920, 1080)
                
                # Only downscale if image exceeds limit, never upscale
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

            image = Image.open(io.BytesIO(img_byte_arr))

            # Generate response
            response = self.model.generate_content([prompt, image])

            # Parse the response
            response_text = response.text

            # Try to parse as JSON
            try:
                # Clean up the response if it contains markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                result = json.loads(response_text.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, structure the response ourselves
                result = {
                    "visible_elements": ["Unable to parse structured response"],
                    "relevant_to_task": ["See raw analysis"],
                    "friction_points": [],
                    "recommended_action": "Review raw analysis",
                    "raw_analysis": response_text
                }

            return result

        except Exception as e:
            print(f"Error analyzing screenshot with Gemini: {str(e)}")
            return {
                "error": str(e),
                "visible_elements": [],
                "relevant_to_task": [],
                "friction_points": ["Error analyzing page"],
                "recommended_action": "retry"
            }

    def analyze_with_retry(self, screenshot_bytes: bytes, prompt: str, device_type: str = "desktop", max_retries: int = 3) -> Dict[str, Any]:
        """
        Analyze screenshot with retry logic for robustness.

        Args:
            screenshot_bytes: Raw PNG bytes
            prompt: Analysis prompt
            device_type: "desktop" or "mobile" to determine appropriate image size
            max_retries: Maximum number of retry attempts

        Returns:
            Analysis result or error information
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                result = self.analyze_screenshot(screenshot_bytes, prompt, device_type)
                if "error" not in result:
                    return result
                last_error = result["error"]
            except Exception as e:
                last_error = str(e)
                print(f"Attempt {attempt + 1} failed: {e}")

        return {
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "visible_elements": [],
            "relevant_to_task": [],
            "friction_points": ["Unable to analyze page"],
            "recommended_action": "manual_review"
        }

    def analyze_page_elements(self,
                              screenshot_bytes: bytes,
                              persona_focus: List[str],
                              include_coordinates: bool = True,
                              device_type: str = "desktop") -> Dict[str, Any]:
        """
        Analyze page elements with detailed detection and optional coordinates.
        Gemini returns coordinates as [y, x] pairs normalized to a 0–1000 range
        from the top-left origin.

        Args:
            screenshot_bytes: Raw PNG bytes of the screenshot
            persona_focus: List of elements this persona would focus on
            include_coordinates: Whether to include approximate element coordinates
            device_type: "desktop" or "mobile" to determine appropriate image size

        Returns:
            Detailed element analysis with clickable elements and coordinates
        """
        focus_str = ", ".join(persona_focus) if persona_focus else "all interactive elements"
        coordinate_instruction = (
            "5. Normalized coordinates for each clickable element as [y, x] "
            "(0-1000 range, top-left origin)\n"
            if include_coordinates else ""
        )
        coordinate_field = (
            '            "coordinates": [y_value_0_to_1000, x_value_0_to_1000],\n'
            if include_coordinates else ""
        )

        prompt = f"""
Analyze this webpage screenshot and identify interactive elements.

FOCUS ON: {focus_str}

Provide detailed analysis including:
1. All visible clickable elements (buttons, links, etc.)
2. Text input fields
3. Navigation elements
4. Visual hierarchy (what stands out most)
{coordinate_instruction}

Format your response as JSON:
{{
    "clickable_elements": [
        {{
            "text": "button/link text",
            "type": "button|link|icon",
            "description": "what it does",
{coordinate_field}            "prominence": "high|medium|low"
        }}
    ],
    "input_fields": [
        {{
            "type": "search|email|text",
            "placeholder": "placeholder text if visible",
            "label": "associated label if visible"
        }}
    ],
    "visual_hierarchy": ["most prominent element first", "less prominent..."],
    "page_layout": "description of overall layout"
}}
"""

        try:
            result = self.analyze_screenshot(screenshot_bytes, prompt, device_type)

            # Use JSON repair to handle malformed responses
            if isinstance(result, dict) and "raw_analysis" in result:
                result = safe_parse_llm_response(result["raw_analysis"], expected_type="analysis")

            return result

        except Exception as e:
            print(f"Error analyzing page elements: {e}")
            return {
                "clickable_elements": [],
                "input_fields": [],
                "visual_hierarchy": [],
                "page_layout": "Unable to analyze",
                "error": str(e)
            }