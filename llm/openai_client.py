"""
OpenAI client for decision-making logic using GPT-5.1.
Handles persona-based reasoning and action generation.
Enhanced with context-aware analysis and state management.
"""

import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from utils.json_repair import safe_parse_llm_response, parse_json_with_repair

load_dotenv()


class OpenAIClient:
    """Client for interacting with OpenAI GPT-5.1 for logical reasoning."""

    def __init__(self):
        """Initialize OpenAI client with API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5.1"
        print("OpenAI client initialized successfully")

    def make_decision(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Make a decision based on persona context and current state.

        Args:
            prompt: Decision-making prompt with full context
            temperature: Creativity level (0.0-1.0)

        Returns:
            Structured decision including action and reasoning
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are simulating user behavior for usability testing. "
                            "Always respond with valid JSON that matches the requested format. "
                            "Think and act exactly as the described persona would."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                response_format={"type": "json_object"}  # Force JSON response
            )

            # Parse the response
            response_text = response.choices[0].message.content

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "continue_task": True,
                    "action_type": "wait",
                    "target_element": None,
                    "input_text": None,
                    "reasoning": "Error parsing decision",
                    "friction_noted": None,
                    "raw_response": response_text
                }

            return result

        except Exception as e:
            print(f"Error making decision with GPT-5.1: {str(e)}")
            return {
                "continue_task": False,
                "action_type": "abandon",
                "target_element": None,
                "input_text": None,
                "reasoning": f"Error: {str(e)}",
                "friction_noted": "System error prevented continuation"
            }
    

    def evaluate_completion(self, persona_context: str, action_history: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate whether the persona successfully completed the task.

        Args:
            persona_context: Full persona context
            action_history: List of all actions taken

        Returns:
            Evaluation of task completion and experience
        """
        history_str = json.dumps(action_history, indent=2)

        prompt = f"""
Based on the following persona and their journey, evaluate their experience:

{persona_context}

Action History:
{history_str}

Evaluate:
1. Was the task completed (Size 1, cheapest diaper bundle added to cart)?
2. What friction points were encountered?
3. Why did they complete or abandon the task?
4. Overall satisfaction with the experience?

Respond in JSON format:
{{
    "task_completed": true/false,
    "completion_reason": "why they completed or abandoned",
    "major_friction_points": ["list of significant issues"],
    "positive_elements": ["what worked well"],
    "persona_satisfaction": 1-10,
    "recommendations": ["UX improvements for this persona type"]
}}
"""

        return self.make_decision(prompt, temperature=0.3)  # Lower temperature for evaluation

    def _format_action_history(self, actions: List[Dict[str, Any]]) -> str:
        """
        Format action history into readable text.

        Args:
            actions: List of action dictionaries

        Returns:
            Formatted history string
        """
        if not actions:
            return "No actions yet"

        lines = []
        for i, action in enumerate(actions, 1):
            action_type = action.get("action_type", "unknown")
            reasoning = action.get("reasoning", "N/A")
            success = action.get("success", False)
            marker = "✓" if success else "✗"

            lines.append(f"{i}. {marker} {action_type}: {reasoning[:100]}")

        return "\n".join(lines)