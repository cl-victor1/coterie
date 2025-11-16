"""
Loop detection and retry logic for preventing stuck behavior.
Extracted from task_executor.py for single responsibility.
"""

from typing import Dict, Any, Optional, List


class LoopDetector:
    """Detects action loops and provides warnings to prevent stuck behavior."""

    @staticmethod
    def detect_action_loop(recent_actions: List[Dict[str, Any]], 
                          recent_urls: List[str],
                          page_state_history: List[str],
                          persona_playbook: Dict[str, Any]) -> Optional[str]:
        """
        Detect if persona is stuck in a repetitive action loop.
        Enhanced to catch coordinate and target element repetition.

        Args:
            recent_actions: List of recent action dictionaries
            recent_urls: List of recent URLs
            page_state_history: List of page state hashes
            persona_playbook: Persona-specific behavioral rules

        Returns:
            Warning message if loop detected, None otherwise
        """
        if len(recent_actions) < 2:
            return None

        # Get persona-specific loop thresholds
        max_scrolls = persona_playbook.get('max_scrolls_per_page', 3)

        # NEW: Check for same coordinates clicked 3+ times consecutively
        if len(recent_actions) >= 3:
            last_three = recent_actions[-3:]
            coords = [a.get('coordinates') for a in last_three]
            
            # Check if all three have the same coordinates
            if all(c is not None for c in coords) and len(set(map(tuple, coords))) == 1:
                # Check if it's the same element too
                target = last_three[-1].get('target_element', '')
                return (f"🚨 REPETITION DETECTED: You clicked coordinates {coords[0]} "
                       f"('{target}') THREE times in a row. The page didn't change meaningfully. "
                       f"You MUST try something DIFFERENT:\n"
                       f"- On mobile: SCROLL DOWN to see other options\n"
                       f"- Try a different button or link\n"
                       f"- Or ABANDON if truly stuck")

        # NEW: Check for same target clicked 3+ times without page change
        if len(recent_actions) >= 3:
            last_three = recent_actions[-3:]
            # Check if all three are clicks on the same target without page change
            targets = [a.get('target_element') for a in last_three]
            action_types = [a.get('action_type') for a in last_three]
            page_changes = [a.get('page_changed', True) for a in last_three]
            
            # Only trigger if clicked same element 3 times with no page changes
            if (len(set(targets)) == 1 and  # Same target
                all(at == 'click' for at in action_types) and  # All clicks
                not any(page_changes)):  # No page changes
                target = targets[0] if targets else 'the same element'
                return (f"🚨 STUCK DETECTED: You clicked '{target}' three times but the page didn't change. "
                       f"This element is not working. Try:\n"
                       f"- SCROLL to find other options\n"
                       f"- Click a DIFFERENT element\n"
                       f"- ABANDON if no other path exists")

        # Check for repetitive scrolling
        recent_5 = recent_actions[-5:]
        scroll_count = sum(1 for a in recent_5 if a.get('action_type') == 'scroll')
        same_direction_scrolls = []
        for a in recent_5:
            if a.get('action_type') == 'scroll':
                direction = a.get('target_element', '').lower()
                same_direction_scrolls.append(direction)

        # Count consecutive scrolls in same direction
        if len(same_direction_scrolls) >= 3:
            if same_direction_scrolls[-3:] == [same_direction_scrolls[-1]] * 3:
                return f"You've scrolled {same_direction_scrolls[-1]} 3+ times in a row. The content isn't changing. Try CLICKING an element instead or ABANDON if stuck."

        # Check for excessive scrolling beyond persona limit
        if scroll_count >= max_scrolls:
            return f"You've scrolled {scroll_count} times recently. Your max is {max_scrolls}. STOP scrolling. Click a specific element or abandon."

        # Check for same action type repeated
        recent_action_types = [a.get('action_type') for a in recent_5]
        if len(recent_action_types) >= 3:
            # For click actions, trigger on 3 repetitions
            last_action = recent_action_types[-1]
            if last_action == 'click' and recent_action_types[-3:] == [last_action] * 3:
                # Check if these are actually different, productive clicks
                last_3_actions = recent_5[-3:]
                different_targets = set(a.get('target_element', '') for a in last_3_actions)
                any_page_changed = any(a.get('page_changed', False) for a in last_3_actions)

                # Only trigger if clicking same elements OR no progress made at all
                if len(different_targets) <= 1 or not any_page_changed:
                    return f"You've clicked 3 times in a row with no progress. Try SCROLLING or a DIFFERENT approach."
            # For other actions, keep 4 repetitions threshold
            elif len(recent_action_types) >= 4 and recent_action_types[-4:] == [last_action] * 4:
                return f"You've repeated the same action ({last_action}) 4 times. Try a DIFFERENT action type."

        # Check if page state hasn't changed (stuck)
        if len(page_state_history) >= 3:
            last_3_states = page_state_history[-3:]
            if last_3_states[0] == last_3_states[-1]:
                return "The page hasn't changed in your last 3 actions. You're stuck. Try clicking a DIFFERENT element or ABANDON."

        return None

    @staticmethod
    def detect_retry_attempt(current_decision: Dict[str, Any], 
                            action_history: List[Dict[str, Any]]) -> bool:
        """
        Detect if the current decision is a retry of a recently failed critical action.

        Args:
            current_decision: The decision just made by the AI
            action_history: Recent action history

        Returns:
            True if this is a retry attempt of a critical action
        """
        if not action_history or len(action_history) < 2:
            return False

        current_target = str(current_decision.get('target_element', '')).lower()
        current_action_type = str(current_decision.get('action_type', '')).lower()

        # Only check click actions for retries
        if current_action_type != 'click':
            return False

        # Check if this action is task-critical
        if not LoopDetector.is_task_critical_target(current_target):
            return False

        # Look at the last few actions (excluding the current one which was just added)
        recent_actions = action_history[-6:-1]  # Last 5 actions before current

        for past_action in reversed(recent_actions):
            past_target = str(past_action.get('target_element', '')).lower()
            past_type = str(past_action.get('action_type', '')).lower()

            # Check if we're targeting the same element
            if past_type == 'click' and (current_target in past_target or past_target in current_target):
                # Check if that past action failed
                if not past_action.get('action_result') or not past_action.get('page_changed'):
                    # This is a retry!
                    return True

        return False

    @staticmethod
    def is_task_critical_target(target_element: str) -> bool:
        """
        Check if a target element is critical to task completion.

        Args:
            target_element: The element being targeted

        Returns:
            True if element is task-critical
        """
        target_lower = str(target_element).lower()

        # Task-critical keywords
        critical_keywords = [
            'size 1', 'size one', 'size_1', '8-12 lbs', '8-12lbs',
            'bundle', 'auto-renew', 'auto renew', 'trial', 'starter', 'sample', 'cheapest',
            'add to cart', 'add-to-cart', 'add to bag'
        ]

        return any(keyword in target_lower for keyword in critical_keywords)

    @staticmethod
    def count_retry_attempts(target_element: str, recent_actions: List[Dict[str, Any]]) -> int:
        """
        Count how many times we've attempted this specific target.

        Args:
            target_element: The target element to count
            recent_actions: Recent action history

        Returns:
            Number of attempts for this target
        """
        if not recent_actions:
            return 1

        target_lower = str(target_element).lower()
        count = 0

        for action in reversed(recent_actions[-10:]):
            action_target = str(action.get('target_element', '')).lower()
            if target_lower in action_target or action_target in target_lower:
                count += 1

        return count

    @staticmethod
    def add_loop_warning_to_prompt(base_prompt: str, loop_warning: str) -> str:
        """
        Add loop warning to decision prompt.

        Args:
            base_prompt: Original prompt
            loop_warning: Warning message about detected loop

        Returns:
            Enhanced prompt with warning
        """
        warning_section = f"\n\n🚨 LOOP DETECTED 🚨\n{loop_warning}\n\nYou MUST try a DIFFERENT strategy now:\n- If you've been scrolling: CLICK a specific element instead\n- If you've been clicking same element: Try a DIFFERENT element\n- If truly stuck: Set continue_task=false and ABANDON\n\n"

        # Insert warning before "DECISION TIME"
        if "DECISION TIME" in base_prompt:
            return base_prompt.replace("DECISION TIME", warning_section + "DECISION TIME")
        else:
            return base_prompt + warning_section

