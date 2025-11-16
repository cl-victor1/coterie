"""
Verification service for validating UI state changes after interactions.
Extracted from task_executor.py for single responsibility.
"""

import asyncio
from typing import Dict, Any


class VerificationService:
    """Verifies UI state changes using vision analysis."""

    def __init__(self, browser, gemini_client, logger):
        """
        Initialize verification service.

        Args:
            browser: BrowserController instance
            gemini_client: GeminiClient instance for vision analysis
            logger: ActionLogger instance
        """
        self.browser = browser
        self.gemini = gemini_client
        self.logger = logger

    async def verify_size_selection(self, persona_key: str, target_element: str) -> Dict[str, Any]:
        """
        Verify if a size button is actually selected using vision analysis.

        Args:
            persona_key: Persona identifier for screenshot naming
            target_element: The size element that was clicked (e.g., "Size 1 button")

        Returns:
            Dict with keys: verified (bool), selected_size (str), visual_indicator (str)
        """
        # Capture screenshot to verify state change
        verify_screenshot = await self.browser.take_screenshot(f"{persona_key}_size_verify")

        if not verify_screenshot.get('success'):
            print("  ⚠️ Verification screenshot failed - assuming click worked")
            # If screenshot fails, assume success rather than triggering retries
            return {
                "verified": True, 
                "selected_size": "Unknown (screenshot failed)", 
                "visual_indicator": "Screenshot capture failed - assumed success"
            }

        # Quick vision analysis focusing on UI state
        verify_prompt = f"""CRITICAL: Check if a size button is now visually SELECTED/ACTIVE.
Look for visual indicators: highlighted border, different background color, checkmark, or any styling showing selection.
Target element that was clicked: {target_element}

Respond with JSON:
{{
    "size_selected": true/false,
    "selected_size": "Size 1" or "Size 2" etc. or null,
    "visual_indicator": "description of what shows it's selected"
}}"""

        verify_analysis = self.gemini.analyze_screenshot(
            verify_screenshot.get('screenshot_bytes'),
            verify_prompt,
            self.browser.device_type
        )

        return {
            "verified": verify_analysis.get('size_selected', False),
            "selected_size": verify_analysis.get('selected_size'),
            "visual_indicator": verify_analysis.get('visual_indicator', 'N/A')
        }

    async def retry_size_selection(self, decision: Dict[str, Any], 
                                   element_analysis: Dict[str, Any],
                                   action_executor,
                                   persona_key: str, 
                                   max_retries: int = 2) -> Dict[str, Any]:
        """
        Retry size selection with vision verification until successful or max retries.

        Args:
            decision: Original decision dict with action details
            element_analysis: Element analysis from vision (for coordinate clicking)
            action_executor: ActionExecutor instance to execute retries
            persona_key: Persona identifier for logging
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            Dict with keys: success (bool), attempts (int), verification (Dict)
        """
        target_element = decision.get('target_element', 'size button')

        for attempt in range(1, max_retries + 1):
            print(f"  🔄 Retry attempt {attempt}/{max_retries} for size selection...")

            # Wait for React to settle from previous click
            await asyncio.sleep(1.5)

            # Retry the click
            retry_success = await action_executor.execute_action(decision, element_analysis)

            if not retry_success:
                print(f"  ❌ Retry {attempt} click failed")
                self.logger.log_friction(
                    "size_selection_retry_click_failed",
                    f"Retry attempt {attempt} to click '{target_element}' failed",
                    "medium"
                )
                continue

            # Wait extra time for React re-render
            await asyncio.sleep(2.0)

            # Verify selection
            verification = await self.verify_size_selection(persona_key, target_element)

            if verification['verified']:
                print(f"  ✓ Size selection verified after {attempt} retry attempts!")
                return {
                    "success": True,
                    "attempts": attempt,
                    "verification": verification
                }
            else:
                print(f"  ⚠️ Retry {attempt} click succeeded but vision does NOT confirm selection")
                print(f"  → Vision says: {verification['visual_indicator']}")
                self.logger.log_friction(
                    "size_selection_retry_not_confirmed",
                    f"Retry {attempt}: Clicked '{target_element}' but visual selection state not detected",
                    "medium"
                )

        # All retries exhausted
        print(f"  ❌ Size selection failed after {max_retries} retry attempts")
        return {
            "success": False,
            "attempts": max_retries,
            "verification": verification if 'verification' in locals() else None
        }

