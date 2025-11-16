"""
Task completion detection service.
Extracted from task_executor.py for single responsibility.
"""

from typing import Dict, Any, List


class TaskCompletionChecker:
    """Tracks task state and detects when the diaper bundle purchase is complete."""

    def __init__(self):
        """Initialize task completion state."""
        self.size_one_selected = False
        self.bundle_selected = False
        self.size_selection_failed = False

    def reset(self):
        """Reset all task state flags."""
        self.size_one_selected = False
        self.bundle_selected = False
        self.size_selection_failed = False

    def update_selection_state(self, decision: Dict[str, Any], 
                               size_selection_verification: Dict[str, Any] = None):
        """
        Update task state based on action and verification results.

        Args:
            decision: Decision dictionary with action details
            size_selection_verification: Optional verification result from vision
        """
        if decision.get('action_type') == 'click':
            target_text = str(decision.get('target_element', '')).lower()
            reasoning_text = str(decision.get('reasoning', '')).lower()
            combined = f"{target_text} {reasoning_text}"

            # Detect Size 1 selection - ONLY if vision verification confirmed it
            if any(phrase in combined for phrase in ['size 1', 'size one', '8-12 lbs']):
                # Check if we have vision verification for this size selection
                if size_selection_verification and size_selection_verification.get('verified'):
                    self.size_one_selected = True
                    print("  ✓ Size 1 selection detected (vision-confirmed)")
                elif size_selection_verification is None:
                    # No verification ran (not a size selector click on product page)
                    # Use text-based detection as fallback
                    self.size_one_selected = True
                    print("  ✓ Size 1 selection detected (text-based, no verification available)")
                else:
                    # Vision verification ran but failed - do NOT set as selected
                    print("  ⚠️ Size 1 click detected but NOT marked as selected (vision verification failed)")

            # Detect bundle/cheapest/auto-renew selection
            if any(phrase in combined for phrase in ['bundle', 'auto-renew', 'auto renew', 'cheapest', 'trial', 'starter', 'sample']):
                self.bundle_selected = True
                print("  ✓ Bundle/cheapest option detected")

            # Check for successful add-to-cart navigation
            post_click_url = decision.get('post_click_url', '')
            if any(phrase in target_text for phrase in ['add to cart', 'add-to-cart', 'addtocart']) and 'cart' in post_click_url:
                print(f"  ✓ Add to cart successful - detected cart page navigation")
                self.size_one_selected = True
                self.bundle_selected = True

    def check_task_completion(self, page_context: Dict[str, Any], 
                             decision: Dict[str, Any],
                             recent_actions: List[Dict[str, Any]]) -> bool:
        """
        Check if the task has been completed using multiple detection strategies.

        Priority order:
        1. Immediate add-to-cart detection (state-based)
        2. Upsell modal detection (confirms successful add-to-cart)
        3. Cart URL detection (fallback)

        Args:
            page_context: Current page information
            decision: Latest decision from AI
            recent_actions: Recent action history

        Returns:
            Whether task is completed
        """
        text = page_context.get("visible_text", "").lower()
        url = page_context.get("url", "").lower()
        action_type = str(decision.get("action_type", "")).lower()
        target_text = str(decision.get("target_element", "")).lower()
        reasoning_text = str(decision.get("reasoning", "")).lower()
        combined_text = f"{target_text} {reasoning_text}"

        # Extract common signals
        add_to_cart_intent = action_type == "click" and any(
            phrase in combined_text for phrase in ["add to cart", "add-to-cart", "add to bag", "add to shopping bag"]
        )

        size_one_confirmed = any(
            phrase in combined_text for phrase in ["size 1", "size one"]
        ) or any(phrase in text for phrase in ["size 1", "size1", "size one"])

        bundle_confirmed = any(
            phrase in combined_text for phrase in ["bundle", "trial", "starter", "sample"]
        ) or any(phrase in text for phrase in ["diaper bundle", "trial bundle", "starter bundle", "bundle"])

        # STRATEGY 1 (HIGHEST PRIORITY): Immediate add-to-cart detection with state tracking
        if add_to_cart_intent and self.size_one_selected and self.bundle_selected:
            print("  🎯 Task completion detected: Add to cart clicked for Size 1 bundle")
            return True

        # STRATEGY 2: Upsell modal detection
        upsell_modal_detected = any(phrase in text for phrase in [
            "complete your bundle",
            "add the wipe",
            "free shipping when you add",
            "would you like to add"
        ])

        if upsell_modal_detected and self._last_action_was_add_to_cart(recent_actions):
            print("  🎯 Task completion detected: Upsell modal appeared after add-to-cart")
            return True

        # STRATEGY 3 (FALLBACK): Direct cart URL confirmation
        if "cart" in url and size_one_confirmed and bundle_confirmed:
            print("  🎯 Task completion detected: Cart URL with Size 1 bundle")
            return True

        # Legacy check (kept for backwards compatibility)
        cheapest_signals = any(
            phrase in combined_text for phrase in ["cheapest", "cheaper", "lowest", "trial", "starter", "sample"]
        ) or any(phrase in text for phrase in ["cheapest bundle", "trial bundle", "starter bundle", "sample bundle"])

        if add_to_cart_intent and size_one_confirmed and bundle_confirmed and cheapest_signals:
            print("  🎯 Task completion detected: All signals present in current decision")
            return True

        return False

    def _last_action_was_add_to_cart(self, recent_actions: List[Dict[str, Any]]) -> bool:
        """
        Check if the most recent action was clicking "Add to cart".

        Args:
            recent_actions: Recent action history

        Returns:
            True if last action was add-to-cart click
        """
        if not recent_actions:
            return False

        last_action = recent_actions[-1]
        action_type = last_action.get('action_type', '')
        target_element = str(last_action.get('target_element', '')).lower()
        reasoning = str(last_action.get('reasoning', '')).lower()

        return (action_type == 'click' and
                any(phrase in target_element or phrase in reasoning
                    for phrase in ['add to cart', 'add-to-cart', 'add to bag']))

