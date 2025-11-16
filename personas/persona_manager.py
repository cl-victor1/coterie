"""
Persona manager for generating persona-specific prompts and managing behavior.
Refactored for single responsibility - delegates to specialized services.
"""

import sys
import os
from typing import Dict, Any, Optional, List


from .persona_definitions import Persona, PERSONAS
from core.context_manager import ContextManager
from .prompt_builder import PromptBuilder
from .behavior_rules import BehaviorRules
from .action_history_formatter import ActionHistoryFormatter
from .friction_descriptor import FrictionDescriptor


class PersonaManager:
    """Manages persona selection and coordinates persona-specific behavior."""

    def __init__(self):
        self.personas = PERSONAS
        self.current_persona: Optional[Persona] = None
        self.context_manager: Optional[ContextManager] = None

    def set_persona(self, persona_key: str) -> Persona:
        """
        Set the current persona for testing and initialize context manager.

        Args:
            persona_key: Key identifying the persona

        Returns:
            The selected Persona object
        """
        if persona_key not in self.personas:
            raise ValueError(f"Unknown persona: {persona_key}")
        self.current_persona = self.personas[persona_key]

        # Initialize context manager for this persona
        self.context_manager = ContextManager(
            self.current_persona.name,
            self.current_persona.to_dict()
        )

        return self.current_persona

    def get_vision_prompt(self, screenshot_context: str) -> str:
        """
        Generate vision analysis prompt for Gemini based on current persona.

        Args:
            screenshot_context: Context about the current screenshot

        Returns:
            Vision prompt string
        """
        # Use enhanced version if context manager available
        if self.context_manager:
            return self.get_enhanced_vision_prompt(screenshot_context, include_history=True)

        # Original implementation for backward compatibility
        if not self.current_persona:
            raise ValueError("No persona set")

        return PromptBuilder.build_vision_prompt(self.current_persona, screenshot_context)

    def get_enhanced_vision_prompt(self, screenshot_context: str,
                                  include_history: bool = True) -> str:
        """
        Generate enhanced vision analysis prompt with context and history.

        Args:
            screenshot_context: Context about the current screenshot
            include_history: Whether to include action history

        Returns:
            Enhanced vision prompt string
        """
        if not self.current_persona or not self.context_manager:
            raise ValueError("No persona set")

        # Get context from context manager
        context = self.context_manager.get_persona_context(include_history)
        action_history = self.context_manager.format_action_history_for_prompt()

        return PromptBuilder.build_enhanced_vision_prompt(
            self.current_persona,
            context,
            action_history,
            screenshot_context
        )

    def get_contextualized_decision_prompt(self, vision_analysis: Dict[str, Any],
                                          current_url: str,
                                          action_history: List[Dict[str, Any]] = None) -> str:
        """
        Generate enhanced decision prompt with full context, behavioral rules, and anti-repetition warnings.

        Args:
            vision_analysis: Vision analysis results
            current_url: Current page URL
            action_history: List of recent action dictionaries

        Returns:
            Decision prompt string
        """
        if not self.current_persona or not self.context_manager:
            raise ValueError("No persona set")

        # Get full context
        context = self.context_manager.get_persona_context(include_history=True)
        behavioral_instruction = self.context_manager.get_behavioral_instruction()
        playbook = self.get_persona_action_playbook()
        page_guidance = self.get_page_specific_guidance(current_url, vision_analysis)

        # Check for abandonment
        should_abandon, abandon_reason = self.context_manager.should_abandon()

        # Format detailed action history
        detailed_history = ActionHistoryFormatter.format_detailed_action_history(
            action_history or [], max_actions=10
        )

        # Identify failed actions and build forbidden list
        failed_actions = ActionHistoryFormatter.get_failed_actions(
            action_history or [], lookback=5
        )
        forbidden_section = ActionHistoryFormatter.build_forbidden_actions_section(
            failed_actions, action_history or []
        )

        return PromptBuilder.build_decision_prompt(
            self.current_persona,
            context,
            vision_analysis,
            current_url,
            detailed_history,
            forbidden_section,
            behavioral_instruction,
            page_guidance,
            should_abandon,
            abandon_reason,
            playbook
        )

    def get_initial_navigation_prompt(self) -> Dict[str, Any]:
        """
        Generate prompt for initial navigation - always starts at Coterie homepage.

        Returns:
            Dictionary with navigation info
        """
        if not self.current_persona:
            raise ValueError("No persona set")

        return PromptBuilder.get_initial_navigation_prompt(self.current_persona)

    def apply_persona_behavior_rules(self, decision: Dict[str, Any],
                                    page_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply persona-specific behavioral rules to override decisions if needed.

        Args:
            decision: Initial decision from LLM
            page_analysis: Current page analysis

        Returns:
            Modified decision with persona-specific rules applied
        """
        if not self.current_persona or not self.context_manager:
            return decision

        return BehaviorRules.apply_persona_behavior_rules(
            self.current_persona,
            decision,
            page_analysis,
            self.context_manager
        )

    def should_force_exploration(self) -> bool:
        """
        Determine if persona should explore before direct action.

        Returns:
            True if persona should explore first
        """
        if not self.context_manager:
            return False

        return self.context_manager.should_explore_first()

    def get_persona_specific_focus(self) -> List[str]:
        """
        Get list of elements this persona would focus on.

        Returns:
            List of focus keywords
        """
        if not self.current_persona:
            return []

        return BehaviorRules.get_persona_specific_focus(self.current_persona.name)

    def update_context(self, action: Dict[str, Any], result: bool, 
                      page_analysis: Dict[str, Any]) -> None:
        """
        Update context manager with action results.

        Args:
            action: Action dictionary
            result: Action success status
            page_analysis: Page analysis results
        """
        if not self.context_manager:
            return

        # Track the action
        self.context_manager.track_action(action, result)

        # Track page change
        if page_analysis:
            url = page_analysis.get("url", "")
            self.context_manager.track_page_change(url, page_analysis)

        # Update emotional state based on friction
        friction_points = page_analysis.get("friction_points", [])
        if friction_points:
            self.context_manager.update_emotional_state(friction_points)

    def get_persona_action_playbook(self) -> Dict[str, Any]:
        """
        Get persona-specific action preferences and behavioral rules.

        Returns:
            Dictionary with preferred actions, prohibited actions, and behavioral rules
        """
        if not self.current_persona:
            return {}

        return BehaviorRules.get_persona_action_playbook(self.current_persona.name)

    def get_page_specific_guidance(self, current_url: str, 
                                   page_analysis: Dict[str, Any]) -> str:
        """
        Get page-specific behavioral guidance for current persona.

        Args:
            current_url: Current page URL
            page_analysis: Vision analysis of current page

        Returns:
            Specific guidance string for this persona on this page type
        """
        if not self.current_persona:
            return ""

        return BehaviorRules.get_page_specific_guidance(
            self.current_persona.name,
            current_url,
            page_analysis,
            self.context_manager
        )

    def get_modal_friction_description(self, modal_info: Dict[str, Any], 
                                      attempt_count: int = 1) -> Dict[str, str]:
        """
        Generate persona-specific friction description for modal interruptions.

        Args:
            modal_info: Dictionary with modal details
            attempt_count: Number of attempts to close the modal

        Returns:
            Dictionary with friction description and severity level
        """
        if not self.current_persona:
            return {
                "description": "Unexpected modal interruption",
                "severity": "medium"
            }

        return FrictionDescriptor.get_modal_friction_description(
            self.current_persona,
            modal_info,
            attempt_count
        )
