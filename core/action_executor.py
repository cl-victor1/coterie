"""
Action execution service for browser interactions.
Extracted from task_executor.py for single responsibility.
"""

import asyncio
import random
from typing import Dict, Any, Optional


class ActionExecutor:
    """Executes browser actions with intelligent retry and coordinate-based clicking."""

    def __init__(self, browser, element_matcher, logger):
        """
        Initialize action executor.

        Args:
            browser: BrowserController instance
            element_matcher: ElementMatcher instance
            logger: ActionLogger instance
        """
        self.browser = browser
        self.element_matcher = element_matcher
        self.logger = logger

    async def execute_action(self, decision: Dict[str, Any], 
                           element_analysis: Optional[Dict[str, Any]] = None,
                           recent_actions: Optional[list] = None) -> bool:
        """
        Execute the action decided by the AI (async).

        Args:
            decision: Decision dictionary from AI
            element_analysis: Optional element analysis with coordinates
            recent_actions: Optional list to track action coordinates

        Returns:
            Success status
        """
        action_type = decision.get("action_type", "wait")
        target = decision.get("target_element")
        input_text = decision.get("input_text")
        element_analysis = element_analysis or {}

        try:
            if action_type == "click":
                # Try coordinate-based click first
                coord_success = await self.execute_coordinate_click(
                    decision, element_analysis, recent_actions
                )
                if coord_success:
                    return True
                
                # If coordinate click failed, try text-based click with auto-retry
                max_retries = 2  # Total attempts: 1 initial + 2 retries = 3
                success = False
                
                for attempt in range(max_retries + 1):
                    if attempt > 0:
                        # This is a retry - add delay and log
                        print(f"  🔄 Auto-retry attempt {attempt}/{max_retries} for click on '{target}'")
                        await asyncio.sleep(random.uniform(0.5, 1.0))
                    
                    success = await self.browser.click_element(text=target) if target else False
                    
                    if success:
                        # Click succeeded
                        if attempt > 0:
                            print(f"  ✓ Click succeeded on retry {attempt}")
                        break
                    elif attempt < max_retries:
                        # Failed but will retry - don't log yet
                        continue
                
                # Only log the final result (success or final failure)
                self.logger.log_action("click", {
                    "target_element": target,
                    "success": success,
                    "reasoning": decision.get("reasoning"),
                    "auto_retries": attempt if not success else 0
                })
                return success

            elif action_type == "scroll":
                return await self.execute_scroll(decision)

            elif action_type == "type":
                return await self.execute_type(target, input_text, decision)

            elif action_type == "wait":
                return await self.execute_wait(decision)

            else:
                print(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            print(f"Action execution error: {str(e)}")
            self.logger.log_friction("execution_error", str(e), "high")
            return False

    async def execute_coordinate_click(self, decision: Dict[str, Any],
                                     element_analysis: Optional[Dict[str, Any]] = None,
                                     recent_actions: Optional[list] = None) -> bool:
        """
        Attempt a coordinate-based click using Gemini element analysis data (async).
        Uses intelligent keyword matching to find the right element.

        Args:
            decision: Decision dictionary from AI
            element_analysis: Element analysis with clickable coordinates
            recent_actions: Optional list to track coordinates

        Returns:
            Whether the coordinate click succeeded
        """
        if not element_analysis:
            return False

        target = decision.get("target_element")
        if not target:
            return False

        clickable_elements = element_analysis.get("clickable_elements") or []
        if not clickable_elements:
            return False

        viewport = self.element_matcher.get_viewport_size()
        reasoning = decision.get("reasoning")
        
        # Find best matching element
        best_match = self.element_matcher.find_best_match(target, clickable_elements, viewport)
        
        # If we found a good match, click it
        if best_match and best_match.get("pixel_coordinates"):
            pixel_coords = best_match["pixel_coordinates"]
            success = await self.browser.click_at_coordinates(pixel_coords["x"], pixel_coords["y"])

            # Update recent_actions with clicked coordinates
            if recent_actions is not None and len(recent_actions) > 0:
                recent_actions[-1]['coordinates'] = (pixel_coords["x"], pixel_coords["y"])

            if self.logger:
                self.logger.log_action("click_coordinates", {
                    "target_element": target,
                    "matched_text": best_match['label'],
                    "match_score": best_match['score'],
                    "raw_coordinates": best_match["coordinates"],
                    "converted_coordinates": (pixel_coords["x"], pixel_coords["y"]),
                    "viewport": viewport,
                    "success": success,
                    "reasoning": reasoning
                })

            return success
        
        return False

    async def execute_scroll(self, decision: Dict[str, Any]) -> bool:
        """
        Execute scroll action.

        Args:
            decision: Decision dictionary

        Returns:
            Success status
        """
        target = decision.get("target_element")
        
        # Extract scroll direction from decision (check both target and reasoning)
        target_str = str(target).lower() if target else ""
        reasoning_str = str(decision.get("reasoning", "")).lower()

        # Only scroll up if explicitly mentioned AND not at page top
        if "up" in target_str or "up" in reasoning_str:
            scroll_pos = await self.browser.get_scroll_position()
            if scroll_pos.get("atTop", True):
                print("⚠️ Blocking scroll up - already at top of page")
                direction = "down"  # Override to natural behavior
            else:
                direction = "up"
        else:
            # Default to down (natural human behavior)
            direction = "down"

        success = await self.browser.scroll_page(direction)
        self.logger.log_action("scroll", {
            "direction": direction,
            "reasoning": decision.get("reasoning")
        })
        return success

    async def execute_type(self, target: str, input_text: str, 
                          decision: Dict[str, Any]) -> bool:
        """
        Execute type action.

        Args:
            target: Target element selector
            input_text: Text to type
            decision: Decision dictionary

        Returns:
            Success status
        """
        success = False
        if target and input_text:
            # Try different selectors
            selectors = [
                f'input[placeholder*="{target}"]',
                f'input[name*="{target}"]',
                f'input[id*="{target}"]',
                'input[type="search"]',
                'input[type="text"]'
            ]
            for selector in selectors:
                if await self.browser.type_text(selector, input_text):
                    success = True
                    break

        self.logger.log_action("type", {
            "target_element": target,
            "input_text": input_text,
            "success": success,
            "reasoning": decision.get("reasoning")
        })
        return success

    async def execute_wait(self, decision: Dict[str, Any]) -> bool:
        """
        Execute wait action.

        Args:
            decision: Decision dictionary

        Returns:
            Success status
        """
        await asyncio.sleep(3)
        self.logger.log_action("wait", {
            "duration": 3,
            "reasoning": decision.get("reasoning")
        })
        return True

