"""
Prompt building utilities for LLM interactions.
Extracted from persona_manager.py for single responsibility.
"""

from typing import Dict, Any


class PromptBuilder:
    """Builds context-aware prompts for vision analysis and decision making."""

    @staticmethod
    def build_vision_prompt(persona, screenshot_context: str) -> str:
        """
        Generate basic vision analysis prompt for Gemini based on current persona.

        Args:
            persona: Current Persona object
            screenshot_context: Context about the current screenshot

        Returns:
            Vision prompt string
        """
        return f"""You are analyzing a website screenshot for usability testing from the perspective of {persona.name}.

Persona Profile:
- {persona.profile}
- Visual Preference: {persona.visual_preference}
- Current Emotional State: {persona.emotional_state}
- Device: {persona.device}

Task: The user wants to purchase Size 1, cheapest diaper bundle from Coterie.

Analyze this screenshot and provide:
1. What elements are visible on the page (buttons, links, text, images)
2. Which elements are relevant to finding or purchasing diapers
3. Any potential friction points for this specific persona
4. The most logical next action based on the persona's goals

Format your response as JSON with keys:
- "visible_elements": list of important elements
- "relevant_to_task": elements related to diaper purchase
- "friction_points": issues this persona will encounter
- "recommended_action": next logical step

Additional context: {screenshot_context}"""

    @staticmethod
    def build_enhanced_vision_prompt(persona, context: Dict[str, Any], 
                                    action_history: str, screenshot_context: str) -> str:
        """
        Generate enhanced vision analysis prompt with context and history.

        Args:
            persona: Current Persona object
            context: Context dictionary from context manager
            action_history: Formatted action history string
            screenshot_context: Context about the current screenshot

        Returns:
            Enhanced vision prompt string
        """
        return f"""You are analyzing a website screenshot for {persona.name}.

PERSONA PROFILE:
- {persona.profile}
- Visual Preference: {persona.visual_preference}
- Navigation Style: {persona.navigation_style}
- Current Emotional State: {context['current_state']['emotional_state']}
- Frustration Level: {context['current_state']['frustration_level']:.1%}
- Device: {persona.device}

JOURNEY CONTEXT:
{action_history}

TASK: Find and purchase Size 1, cheapest diaper bundle from Coterie.

Analyze this screenshot considering:
1. What would {persona.name} notice FIRST based on their visual preferences?
2. Which elements align with their values: {', '.join(persona.values)}?
3. What would frustrate someone who fears: {persona.purchase_fears}?
4. Given their {persona.navigation_style} style, what's the logical next step?
5. Are there elements that would trigger their "{persona.emotional_state}" state?

🔍 CRITICAL: UI STATE DETECTION
For ALL interactive elements (buttons, size selectors, toggles, checkboxes, radio buttons), you MUST:
- Identify the CURRENT STATE: selected/unselected, active/inactive, enabled/disabled, checked/unchecked
- Describe VISUAL INDICATORS of state: borders, background colors, checkmarks, highlights, opacity changes, or any styling suggesting an element is active/selected
- On {persona.device} devices, selection states may be SUBTLE - look carefully for:
  * Border color/thickness changes (e.g., blue vs gray border)
  * Background color shifts (e.g., filled vs outline buttons)
  * Small checkmarks, dots, or icons indicating selection
  * Text color or weight changes
  * Slight opacity or shadow changes

EXAMPLE UI States to Report:
- "Size 1 button: SELECTED (visible blue border, darker background)"
- "Size 2 button: UNSELECTED (gray outline, lighter background)"
- "Add to Cart button: DISABLED (grayed out text)"
- "Add to Cart - Size 1 button: ENABLED (green background, bold text)"
- "Auto-Renew toggle: ON (switch positioned right, blue color)"

Format your response as JSON:
{{
    "visible_elements": ["elements {persona.name} would notice"],
    "relevant_to_task": ["elements for Size 1 diaper purchase"],
    "ui_states": {{"element_name": "STATE - visual indicator description"}},
    "friction_points": ["specific issues for THIS persona"],
    "recommended_action": "next action matching their style",
    "emotional_triggers": ["elements affecting their state"],
    "trust_signals": ["elements building/breaking trust"]
}}

Current page: {screenshot_context}"""

    @staticmethod
    def build_decision_prompt(persona, context: Dict[str, Any], 
                            vision_analysis: Dict[str, Any],
                            current_url: str, detailed_history: str,
                            forbidden_section: str, behavioral_instruction: str,
                            page_guidance: str, should_abandon: bool,
                            abandon_reason: str, playbook: Dict[str, Any]) -> str:
        """
        Generate enhanced decision prompt with full context and behavioral rules.

        Args:
            persona: Current Persona object
            context: Context dictionary
            vision_analysis: Vision analysis results
            current_url: Current page URL
            detailed_history: Formatted action history
            forbidden_section: Forbidden actions warning
            behavioral_instruction: Behavioral rules
            page_guidance: Page-specific guidance
            should_abandon: Whether persona should abandon
            abandon_reason: Reason for abandonment
            playbook: Persona action playbook

        Returns:
            Decision prompt string
        """
        constraints = "\n".join([f"  - {rule}" for rule in playbook.get('behavioral_rules', [])])
        prohibited = ", ".join(playbook.get('prohibited_actions', []))
        preferred = ", ".join(playbook.get('preferred_actions', []))

        prompt = f"""You ARE {persona.name}, {persona.title}.

YOUR IDENTITY:
- Age: {persona.age}
- Profile: {persona.profile}
- Values: {', '.join(persona.values)}
- Motivators: {', '.join(persona.motivators)}
- Personality: {persona.personality}
- Behavior: {persona.behavior}

YOUR MENTAL STATE:
- Emotional: {context['current_state']['emotional_state']}
- Frustration: {context['current_state']['frustration_level']:.0%}
- Patience: {context['current_state']['patience_remaining']:.0%}
- Trust: {context['current_state']['trust_level']:.0%}
- Confusion count: {context['current_state']['confusion_count']}

YOUR COMPLETE ACTION HISTORY:
{detailed_history}

🚫 CRITICAL - DO NOT REPEAT THESE FAILED ACTIONS:
{forbidden_section}

CURRENT SITUATION:
- URL: {current_url}
- Page analysis: {vision_analysis}
- Time spent: {context['time_elapsed']:.0f} seconds

⚠️ ANTI-REPETITION RULES (MUST FOLLOW):
1. NEVER click the same coordinates twice - if you just clicked something, try DIFFERENT element
2. NEVER click the same element twice if the page didn't change - it's not working
3. If an action didn't work (marked ✗ FAILED or NO CHANGE), try a COMPLETELY different approach
4. On mobile, if you can't find what you need, SCROLL DOWN to reveal more content
5. Review your action history above - avoid repeating what already failed
6. If stuck after 2-3 failed attempts, consider SCROLLING or ABANDONING

🔍 UI STATE VERIFICATION RULES (CRITICAL):
1. NEVER assume a button is selected just because you clicked it in a previous action
2. ALWAYS check the vision analysis 'ui_states' field to verify the CURRENT state of interactive elements
3. ONLY click "Add to Cart - Size 1" if vision analysis confirms Size 1 is in SELECTED state
4. If you clicked Size 1 but vision still shows it as UNSELECTED, the click failed - try clicking again
5. Before making decisions about enabled/disabled buttons, verify their state from 'ui_states'
6. Trust the vision analysis state detection over your action history

YOUR BEHAVIORAL RULES (YOU MUST FOLLOW):
{constraints}

ACTIONS YOU PREFER: {preferred}
ACTIONS YOU NEVER DO: {prohibited}

{behavioral_instruction}

{page_guidance if page_guidance else ""}

{"⚠️ CRITICAL: " + abandon_reason if should_abandon else ""}

DECISION TIME - What do you do next?

Think EXACTLY as {persona.name} would based on:
- Cognitive Style: {persona.cognitive_style}
- Navigation Style: {persona.navigation_style}
- Response Style: {persona.response_style}
- Purchase Fears: {persona.purchase_fears}
- Wow Factors: {persona.wow_factors}

IMPORTANT CONSTRAINTS:
- Use your {playbook.get('decision_style', 'natural')} decision style
- Max scrolls on this page: {playbook.get('max_scrolls_per_page', 3)}
- Your patience threshold: {playbook.get('patience_threshold', 0.5):.0%}

Respond in YOUR voice ({persona.response_style}) as JSON:
{{
    "continue_task": {str(not should_abandon).lower()},
    "action_type": "click|scroll|type|wait|abandon",
    "target_element": "specific element or null",
    "input_text": "text to type or null",
    "reasoning": "your reasoning in your voice style",
    "friction_noted": "specific UX issue or null",
    "internal_thought": "what you're really thinking"
}}"""

        return prompt

    @staticmethod
    def get_initial_navigation_prompt(persona) -> Dict[str, Any]:
        """
        Generate prompt for initial navigation - always starts at Coterie homepage.

        Args:
            persona: Current Persona object

        Returns:
            Dictionary with navigation info
        """
        return {
            "start_url": "https://www.coterie.com",
            "needs_search": False,
            "search_query": None,
            "entry_context": persona.entry_point
        }

