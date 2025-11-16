"""
Action history formatting and analysis utilities.
Extracted from persona_manager.py for single responsibility.
"""

from typing import Dict, Any, List


class ActionHistoryFormatter:
    """Formats and analyzes action history for decision-making context."""

    @staticmethod
    def format_detailed_action_history(actions: List[Dict[str, Any]], 
                                      max_actions: int = 10) -> str:
        """
        Format action history with coordinates and results for decision-making context.

        Args:
            actions: List of action dictionaries with detailed tracking
            max_actions: Maximum number of recent actions to include

        Returns:
            Formatted history string showing actions, results, and page changes
        """
        if not actions:
            return "No previous actions yet. This is your first decision."

        recent = actions[-max_actions:]
        lines = ["Your Recent Actions (most recent at bottom):"]
        for i, action in enumerate(recent, 1):
            action_type = action.get('action_type', 'unknown').upper()
            target = action.get('target_element', 'N/A')
            coords = action.get('coordinates')
            result = action.get('action_result')
            page_changed = action.get('page_changed')

            # Format result indicator
            if result is None:
                result_str = "⏳ PENDING"
            elif result:
                result_str = "✓ SUCCESS"
            else:
                result_str = "✗ FAILED"

            # Format page change indicator
            if page_changed is None or page_changed:
                change_str = "→ Page changed"
            else:
                change_str = "→ NO CHANGE (stuck)"

            # Format coordinates if available
            coord_str = f" @ coordinates {coords}" if coords else ""

            lines.append(f"{i}. {action_type} '{target}'{coord_str} | {result_str} {change_str}")

        return "\n".join(lines)

    @staticmethod
    def get_failed_actions(actions: List[Dict[str, Any]], lookback: int = 5) -> List[Dict[str, Any]]:
        """
        Extract recently failed or ineffective actions.

        Args:
            actions: List of action dictionaries
            lookback: Number of recent actions to examine

        Returns:
            List of failed/stuck actions
        """
        recent = actions[-lookback:] if actions else []
        failed = []
        for action in recent:
            # Consider action failed if it didn't succeed OR didn't change the page
            if not action.get('action_result') or not action.get('page_changed'):
                failed.append(action)
        return failed

    @staticmethod
    def is_task_critical_action(action: Dict[str, Any]) -> bool:
        """
        Determine if an action is critical to task completion and should allow retries.

        Args:
            action: Action dictionary with target_element

        Returns:
            True if action targets a task-critical element
        """
        target = str(action.get('target_element', '')).lower()
        action_type = str(action.get('action_type', '')).lower()

        # Task-critical keywords for the diaper bundle task
        critical_keywords = [
            # Size 1 selectors
            'size 1', 'size one', 'size_1', '8-12 lbs', '8-12lbs', '8 to 12',
            # Bundle/cheapest options
            'bundle', 'auto-renew', 'auto renew', 'trial', 'starter', 'sample', 'cheapest', 'cheaper',
            # Add to cart actions
            'add to cart', 'add-to-cart', 'add to bag', 'add to shopping'
        ]

        # Check if target contains any critical keywords
        is_critical = any(keyword in target for keyword in critical_keywords)

        # Click actions on critical elements are prioritized for retry
        if action_type == 'click' and is_critical:
            return True

        return False

    @staticmethod
    def count_consecutive_failures(actions: List[Dict[str, Any]], 
                                   target_element: str) -> int:
        """
        Count how many times the same target element failed consecutively.

        Args:
            actions: Full action history
            target_element: Target element to count failures for

        Returns:
            Number of consecutive failures for this target
        """
        if not actions:
            return 0

        count = 0
        target_lower = str(target_element).lower()

        # Walk backwards through recent actions
        for action in reversed(actions[-10:]):
            action_target = str(action.get('target_element', '')).lower()

            # If we find a match for this target
            if action_target == target_lower or target_lower in action_target or action_target in target_lower:
                # Check if it failed
                if not action.get('action_result') or not action.get('page_changed'):
                    count += 1
                else:
                    # Success found, stop counting
                    break
            # If we hit a different target, stop counting consecutive failures
            elif action.get('action_type') == 'click':
                break

        return count

    @staticmethod
    def build_forbidden_actions_section(failed_actions: List[Dict[str, Any]], 
                                       action_history: List[Dict[str, Any]] = None) -> str:
        """
        Build explicit list of actions that should NOT be repeated.
        Task-critical actions get retry attempts before being blocked.

        Args:
            failed_actions: List of recently failed actions
            action_history: Full action history for counting consecutive failures

        Returns:
            Formatted forbidden actions warning
        """
        if not failed_actions:
            return "None - all recent actions were effective."

        action_history = action_history or []
        lines = []
        MAX_RETRIES = 3  # Allow up to 3 attempts for critical actions

        for action in failed_actions:
            action_type = action.get('action_type', 'unknown')
            target = action.get('target_element', 'unknown')
            coords = action.get('coordinates')

            # Check if this is a task-critical action
            is_critical = ActionHistoryFormatter.is_task_critical_action(action)

            # Count consecutive failures for this specific target
            failure_count = ActionHistoryFormatter.count_consecutive_failures(
                action_history, target
            )

            if is_critical and failure_count < MAX_RETRIES:
                # Allow retry for critical actions that haven't exceeded max attempts
                retry_msg = f"⚠️ RETRY ALLOWED ({failure_count}/{MAX_RETRIES}): '{target}' is task-critical - you may retry with a fresh screenshot"
                lines.append(f"- {retry_msg}")
            else:
                # Block non-critical actions or critical actions that exceeded retries
                if is_critical:
                    block_msg = f"❌ BLOCKED after {failure_count} attempts: '{target}' - try scrolling or different approach"
                else:
                    if coords:
                        block_msg = f"❌ Do NOT {action_type} coordinates {coords} again ('{target}' - didn't work)"
                    else:
                        block_msg = f"❌ Do NOT {action_type} '{target}' again (didn't work or no page change)"
                lines.append(f"- {block_msg}")

        return "\n".join(lines)

