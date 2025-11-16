"""
Context Manager for AI Persona Testing System.
Tracks state, history, and behavioral patterns throughout persona journey.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class PersonaState:
    """Tracks the current mental and emotional state of a persona."""
    emotional_state: str = "neutral"
    frustration_level: float = 0.0  # 0.0 to 1.0
    patience_remaining: float = 1.0  # 1.0 to 0.0
    trust_level: float = 0.5  # 0.0 to 1.0
    confusion_count: int = 0
    success_count: int = 0
    last_page_url: str = ""
    time_on_current_page: float = 0.0


class ContextManager:
    """
    Manages context and state for a persona throughout their testing journey.
    Tracks actions, emotional state, and provides behavioral guidance.
    """

    def __init__(self, persona_name: str, persona_dict: Dict[str, Any]):
        """
        Initialize context manager for a persona.

        Args:
            persona_name: Name of the persona
            persona_dict: Full persona definition dictionary
        """
        self.persona_name = persona_name
        self.persona_dict = persona_dict
        self.state = PersonaState()

        # Action tracking
        self.action_history: List[Dict[str, Any]] = []
        self.page_history: List[Dict[str, Any]] = []
        self.friction_accumulation: List[Dict[str, Any]] = []

        # Timing
        self.start_time = datetime.now()
        self.last_action_time = datetime.now()

        # Behavioral flags
        self.has_explored = False
        self.exploration_threshold = 3  # Minimum actions before purchase for some personas
        self.max_patience_threshold = 0.2  # Abandonment threshold

    def track_action(self, action: Dict[str, Any], success: bool) -> None:
        """
        Track an action and update state accordingly.

        Args:
            action: Action dictionary with type, target, reasoning
            success: Whether action was successful
        """
        now = datetime.now()
        time_since_last = (now - self.last_action_time).total_seconds()

        action_record = {
            "timestamp": now.isoformat(),
            "action_type": action.get("action_type"),
            "target": action.get("target_element"),
            "reasoning": action.get("reasoning"),
            "success": success,
            "time_since_last": time_since_last
        }

        self.action_history.append(action_record)
        self.last_action_time = now

        # Update state based on action result
        if success:
            self.state.success_count += 1
            self.state.frustration_level = max(0.0, self.state.frustration_level - 0.1)
            self.state.trust_level = min(1.0, self.state.trust_level + 0.05)
        else:
            self.state.confusion_count += 1
            self.state.frustration_level = min(1.0, self.state.frustration_level + 0.15)
            self.state.patience_remaining = max(0.0, self.state.patience_remaining - 0.1)
            self.state.trust_level = max(0.0, self.state.trust_level - 0.1)

        # Track exploration
        if action.get("action_type") in ["scroll", "click"]:
            self.has_explored = len(self.action_history) >= self.exploration_threshold

    def track_page_change(self, url: str, page_context: Dict[str, Any]) -> None:
        """
        Track navigation to a new page.

        Args:
            url: New page URL
            page_context: Page analysis context
        """
        if url != self.state.last_page_url:
            page_record = {
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "context": page_context,
                "time_spent": self.state.time_on_current_page
            }
            self.page_history.append(page_record)

            self.state.last_page_url = url
            self.state.time_on_current_page = 0.0

    def update_emotional_state(self, friction_points: List[str]) -> None:
        """
        Update emotional state based on encountered friction.

        Args:
            friction_points: List of friction point descriptions
        """
        if not friction_points:
            # No friction - slightly improve state
            self.state.emotional_state = "calm" if self.state.frustration_level < 0.3 else self.state.emotional_state
            return

        # Accumulate friction
        for friction in friction_points:
            self.friction_accumulation.append({
                "timestamp": datetime.now().isoformat(),
                "description": friction,
                "frustration_before": self.state.frustration_level
            })

        # Increase frustration
        friction_increase = len(friction_points) * 0.1
        self.state.frustration_level = min(1.0, self.state.frustration_level + friction_increase)
        self.state.patience_remaining = max(0.0, self.state.patience_remaining - friction_increase)

        # Update emotional state based on frustration level
        if self.state.frustration_level > 0.7:
            self.state.emotional_state = "frustrated"
        elif self.state.frustration_level > 0.4:
            self.state.emotional_state = "annoyed"
        elif self.state.frustration_level > 0.2:
            self.state.emotional_state = "slightly_irritated"
        else:
            self.state.emotional_state = "calm"

    def should_abandon(self) -> Tuple[bool, str]:
        """
        Determine if persona should abandon task based on accumulated state.

        Returns:
            Tuple of (should_abandon, reason)
        """
        # Check patience threshold
        if self.state.patience_remaining <= self.max_patience_threshold:
            return True, "Patience exhausted - too many difficulties encountered"

        # Check frustration threshold
        if self.state.frustration_level >= 0.9:
            return True, "Frustration level critical - abandoning task"

        # Check confusion threshold (persona-specific)
        if self.state.confusion_count >= 5:
            return True, "Too confused to continue - unclear how to proceed"

        # Check trust level
        if self.state.trust_level <= 0.1:
            return True, "Lost trust in website - abandoning"

        return False, ""

    def should_explore_first(self) -> bool:
        """
        Determine if persona should explore before making purchase.
        Relevant for personas like Maya Rodriguez.

        Returns:
            Whether persona should explore more before buying
        """
        return len(self.action_history) < self.exploration_threshold

    def get_behavioral_instruction(self) -> str:
        """
        Get behavioral instruction based on current state and persona type.

        Returns:
            Instruction string for decision-making prompt
        """
        persona_lower = self.persona_name.lower()

        instructions = []

        # Maya Rodriguez - eco-conscious
        if "maya" in persona_lower:
            if self.should_explore_first():
                instructions.append("IMPORTANT: You MUST explore sustainability info before buying. Look for eco-friendly claims, ingredients, certifications.")
            else:
                instructions.append("You've explored enough. Now focus on finding the right product.")

        # Sarah Kim - efficiency focused
        elif "sarah" in persona_lower:
            instructions.append("PRIORITY: Subscription options and bulk purchases. You value efficiency above all.")

        # Lauren Peterson - sleep-deprived
        elif "lauren" in persona_lower:
            if self.state.patience_remaining < 0.5:
                instructions.append("WARNING: Your patience is wearing thin. You need this to work quickly or you'll give up.")
            instructions.append("You're exhausted. Any confusion or extra steps will cause you to abandon.")

        # Jasmine Lee - influencer follower
        elif "jasmine" in persona_lower:
            instructions.append("FOCUS: Visual elements, reviews, social proof. You want to see what others think.")

        # Priya Desai - convenience first
        elif "priya" in persona_lower:
            instructions.append("EFFICIENCY: Take the most direct path. No browsing, no exploration. Get to checkout fast.")

        # State-based instructions
        if self.state.frustration_level > 0.5:
            instructions.append(f"You're feeling {self.state.emotional_state}. Your tolerance for more issues is low.")

        if self.state.confusion_count >= 3:
            instructions.append("You're getting confused. One more unclear step will make you abandon.")

        return "\n".join(instructions) if instructions else "Proceed naturally based on your personality."

    def get_persona_context(self, include_history: bool = True) -> Dict[str, Any]:
        """
        Get full context for decision-making.

        Args:
            include_history: Whether to include action history

        Returns:
            Context dictionary
        """
        context = {
            "current_state": {
                "emotional_state": self.state.emotional_state,
                "frustration_level": self.state.frustration_level,
                "patience_remaining": self.state.patience_remaining,
                "trust_level": self.state.trust_level,
                "confusion_count": self.state.confusion_count
            },
            "action_summary": self._summarize_actions(),
            "time_elapsed": (datetime.now() - self.start_time).total_seconds()
        }

        if include_history:
            context["full_history"] = self.action_history[-10:]  # Last 10 actions

        return context

    def format_action_history_for_prompt(self, max_actions: int = 5) -> str:
        """
        Format action history for inclusion in prompts.

        Args:
            max_actions: Maximum number of recent actions to include

        Returns:
            Formatted history string
        """
        if not self.action_history:
            return "No actions taken yet. This is your first step."

        recent_actions = self.action_history[-max_actions:]

        lines = ["Recent Actions:"]
        for i, action in enumerate(recent_actions, 1):
            success_marker = "✓" if action["success"] else "✗"
            lines.append(f"{i}. {success_marker} {action['action_type']}: {action.get('reasoning', 'N/A')}")

        return "\n".join(lines)

    def _summarize_actions(self) -> str:
        """
        Summarize action history into concise description.

        Returns:
            Summary string
        """
        if not self.action_history:
            return "Journey just beginning"

        total_actions = len(self.action_history)
        successful_actions = sum(1 for a in self.action_history if a["success"])
        failed_actions = total_actions - successful_actions

        action_types = {}
        for action in self.action_history:
            action_type = action["action_type"]
            action_types[action_type] = action_types.get(action_type, 0) + 1

        summary_parts = [
            f"Taken {total_actions} actions",
            f"({successful_actions} successful, {failed_actions} failed)"
        ]

        if action_types:
            type_summary = ", ".join([f"{count} {atype}" for atype, count in action_types.items()])
            summary_parts.append(f"Actions: {type_summary}")

        return " | ".join(summary_parts)
