"""
Action logger for tracking all persona interactions and decisions.
Records detailed logs of each action with timestamps and reasoning.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import os


class ActionLogger:
    """Logs and tracks all actions taken during testing."""

    def __init__(self, persona_name: str):
        """
        Initialize action logger for a specific persona.

        Args:
            persona_name: Name of the persona being simulated
        """
        self.persona_name = persona_name
        self.start_time = datetime.now()
        self.actions: List[Dict[str, Any]] = []
        self.friction_points: List[Dict[str, Any]] = []
        self.screenshots: List[Dict[str, Any]] = []
        self.auto_dismissed_popups: int = 0  # Track auto-dismissed popups

    def log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """
        Log an action taken by the persona.

        Args:
            action_type: Type of action (click, scroll, type, etc.)
            details: Dictionary with action details
        """
        action = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            "action": action_type,
            "details": details,
            "sequence_number": len(self.actions) + 1
        }

        # Add reasoning if provided
        if "reasoning" in details:
            action["reasoning"] = details["reasoning"]

        # Add target element if provided
        if "target_element" in details:
            action["target"] = details["target_element"]

        # Add URL context
        if "url" in details:
            action["url"] = details["url"]

        self.actions.append(action)
        self._log_to_console(action)

    def log_popup_event(self, popup_info: Dict[str, Any], event_type: str = "appeared") -> None:
        """
        Log a popup/modal event.

        Args:
            popup_info: Popup detection information from browser controller
            event_type: Type of event (appeared, dismissed, etc.)
        """
        popup_text = popup_info.get("text", "")[:100]
        popup_id = popup_info.get("id", "")
        popup_classes = popup_info.get("classes", "")

        description = f"Popup {event_type}"
        if popup_id:
            description += f" (#{popup_id})"
        elif popup_classes:
            first_class = popup_classes.split()[0] if popup_classes else ""
            description += f" (.{first_class})"
        if popup_text:
            description += f": {popup_text}"

        self.log_action("popup_event", {
            "event_type": event_type,
            "popup_info": popup_info,
            "description": description
        })

    def log_auto_dismissed_popup(self, dismissal_result: Dict[str, Any]) -> None:
        """
        Log an automatically dismissed popup.

        Args:
            dismissal_result: Result from auto_dismiss_popup() method
        """
        self.auto_dismissed_popups += 1

        popup_info = dismissal_result.get('popup_info', {})
        popup_text = popup_info.get("text", "")[:100] if popup_info else ""

        description = f"Auto-dismissed popup"
        if popup_text:
            description += f": {popup_text}"

        self.log_action("auto_dismiss_popup", {
            "success": dismissal_result.get('dismissed', False),
            "reason": dismissal_result.get('reason', ''),
            "popup_info": popup_info,
            "description": description,
            "total_auto_dismissed": self.auto_dismissed_popups
        })

    def log_friction(self, friction_type: str, description: str, severity: str = "medium") -> None:
        """
        Log a friction point encountered.

        Args:
            friction_type: Type of friction (navigation, content, trust, etc.)
            description: Description of the friction point
            severity: Severity level (low, medium, high)
        """
        friction = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            "type": friction_type,
            "description": description,
            "severity": severity,
            "action_context": self.actions[-1] if self.actions else None
        }

        self.friction_points.append(friction)
        print(f"⚠️ Friction logged [{severity}]: {description}")

    def log_screenshot(self, reference: str = "", context: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a screenshot taken during testing.

        Args:
            reference: Identifier for the screenshot (path or in-memory label)
            context: Context or reason for screenshot
            metadata: Optional metadata payload (timestamps, errors, etc.)
        """
        screenshot_log = {
            "timestamp": datetime.now().isoformat(),
            "reference": reference,
            "context": context,
            "action_number": len(self.actions),
            "metadata": metadata or {}
        }
        self.screenshots.append(screenshot_log)

    def log_decision(self, decision: Dict[str, Any]) -> None:
        """
        Log a decision made by the AI.

        Args:
            decision: Decision dictionary from LLM
        """
        self.log_action("decision", {
            "continue_task": decision.get("continue_task"),
            "action_type": decision.get("action_type"),
            "reasoning": decision.get("reasoning"),
            "friction_noted": decision.get("friction_noted")
        })

        # Log friction if noted
        if decision.get("friction_noted"):
            self.log_friction("user_experience", decision["friction_noted"])

    def log_page_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Log page analysis results.

        Args:
            analysis: Analysis results from vision AI
        """
        self.log_action("page_analysis", {
            "visible_elements": len(analysis.get("visible_elements", [])),
            "relevant_elements": analysis.get("relevant_to_task", []),
            "recommended_action": analysis.get("recommended_action"),
            "friction_points": analysis.get("friction_points", [])
        })

        # Log any friction points found
        for friction in analysis.get("friction_points", []):
            self.log_friction("page_analysis", friction, "low")

    def get_recent_actions(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent actions.

        Args:
            count: Number of recent actions to return

        Returns:
            List of recent actions
        """
        return self.actions[-count:] if self.actions else []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the logged session.

        Returns:
            Summary dictionary
        """
        total_time = (datetime.now() - self.start_time).total_seconds()

        # Count action types
        action_types = {}
        for action in self.actions:
            action_type = action["action"]
            action_types[action_type] = action_types.get(action_type, 0) + 1

        # Determine if task was completed
        task_completed = any(
            "cart" in str(action.get("details", {})).lower() and
            "add" in str(action.get("details", {})).lower()
            for action in self.actions
        )

        return {
            "persona": self.persona_name,
            "start_time": self.start_time.isoformat(),
            "total_time_seconds": total_time,
            "total_actions": len(self.actions),
            "action_breakdown": action_types,
            "friction_points_count": len(self.friction_points),
            "screenshots_taken": len(self.screenshots),
            "auto_dismissed_popups": self.auto_dismissed_popups,
            "task_completed": task_completed
        }

    def export_log(self) -> Dict[str, Any]:
        """
        Export the complete log for saving.

        Returns:
            Complete log dictionary
        """
        return {
            "persona": self.persona_name,
            "session": {
                "start_time": self.start_time.isoformat(),
                "total_duration_seconds": (datetime.now() - self.start_time).total_seconds()
            },
            "actions": self.actions,
            "friction_points": self.friction_points,
            "screenshots": self.screenshots,
            "summary": self.get_summary()
        }

    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """
        Save log to a JSON file.

        Args:
            filepath: Optional custom filepath

        Returns:
            Path to saved file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"output/logs/{self.persona_name}_{timestamp}.json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.export_log(), f, indent=2)

        print(f"Log saved to: {filepath}")
        return filepath

    def _log_to_console(self, action: Dict[str, Any]) -> None:
        """
        Print action to console for real-time monitoring.

        Args:
            action: Action dictionary
        """
        timestamp = action["timestamp"].split("T")[1].split(".")[0]  # HH:MM:SS
        action_type = action["action"]
        details = action.get("details", {})

        # Format based on action type
        if action_type == "click":
            target = details.get("target_element", "unknown")
            print(f"[{timestamp}] 🖱️ CLICK: {target}")
        elif action_type == "click_coordinates":
            target = details.get("target_element", "unknown")
            matched_text = details.get("matched_text", "")
            coords = details.get("converted_coordinates")

            # Show both target and matched text if they differ
            if matched_text and matched_text != target:
                display_text = f"{target} → matched: '{matched_text}'"
            else:
                display_text = target

            # Include coordinates for debugging
            if coords:
                print(f"[{timestamp}] 🖱️ CLICK: {display_text} @ ({coords[0]}, {coords[1]})")
            else:
                print(f"[{timestamp}] 🖱️ CLICK: {display_text}")
        elif action_type == "scroll":
            direction = details.get("direction", "down")
            print(f"[{timestamp}] 📜 SCROLL: {direction}")
        elif action_type == "type":
            text = details.get("input_text", "")
            print(f"[{timestamp}] ⌨️ TYPE: '{text}'")
        elif action_type == "navigate":
            url = details.get("url", "")
            print(f"[{timestamp}] 🌐 NAVIGATE: {url}")
        elif action_type == "decision":
            continue_task = details.get("continue_task")
            print(f"[{timestamp}] 🤔 DECISION: Continue={continue_task}")
        elif action_type == "popup_event":
            description = details.get("description", "Popup event")
            print(f"[{timestamp}] 🪟 POPUP: {description}")
        else:
            print(f"[{timestamp}] ▶️ {action_type.upper()}")

        # Print reasoning if available
        if "reasoning" in action:
            reasoning = action["reasoning"][:100] + "..." if len(action["reasoning"]) > 100 else action["reasoning"]
            print(f"    💭 {reasoning}")