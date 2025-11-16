"""
Persona-specific friction description generator.
Extracted from persona_manager.py for single responsibility.
"""

from typing import Dict, Any


class FrictionDescriptor:
    """Generates persona-specific friction descriptions for UX issues."""

    @staticmethod
    def get_modal_friction_description(persona, modal_info: Dict[str, Any], 
                                      attempt_count: int = 1) -> Dict[str, str]:
        """
        Generate persona-specific friction description for modal interruptions.

        Args:
            persona: Current Persona object
            modal_info: Dictionary with modal details (type, success, etc.)
            attempt_count: Number of attempts to close the modal

        Returns:
            Dictionary with friction description and severity level
        """
        if not persona:
            return {
                "description": "Unexpected modal interruption",
                "severity": "medium"
            }

        persona_lower = persona.name.lower()
        modal_name = modal_info.get("modal", "unknown")
        success = modal_info.get("success", False)

        # Get persona characteristics
        patience = getattr(persona, 'patience_threshold', 0.5)
        emotional_state = getattr(persona, 'emotional_state', 'neutral')

        # Base severity on attempt count and persona patience
        if success:
            severity = "low"
        elif attempt_count == 1:
            severity = "medium" if patience < 0.3 else "low"
        else:  # Multiple failed attempts
            severity = "high"

        # Generate persona-specific descriptions
        if "sarah" in persona_lower:
            # Sarah Kim: Subscription-Savvy, Time-Constrained, Control-Seeking
            if success:
                description = "Pop-up interrupted my planned flow. I need predictable, linear paths to subscription options—these interruptions waste precious time."
            elif attempt_count == 1:
                description = "This pop-up is blocking me from seeing subscription details and Size 1 pricing. As a busy new parent, I don't have time for extra steps that don't add value."
            else:
                description = f"Tried {attempt_count} times to close this pop-up. This is unacceptable friction when I just want to quickly set up a diaper auto-renew. I'm losing trust in this site's efficiency."

        elif "maya" in persona_lower:
            # Maya Rodriguez: Eco-Conscious, Research-Driven, Skeptical
            if success:
                description = "Pop-up interrupted my research flow. I need uninterrupted access to sustainability info, certifications, and ingredient transparency."
            elif attempt_count == 1:
                description = "This modal is blocking my view of eco-credentials and product details. I can't verify this brand's environmental claims if their site keeps interrupting me."
            else:
                description = f"After {attempt_count} attempts, still blocked by this pop-up. This lack of transparency and user respect raises red flags about the brand's values."

        elif "lauren" in persona_lower:
            # Lauren Peterson: Sleep-Deprived, Premium-Seeking, Impatient
            if success:
                description = "Already exhausted, and now I have to deal with pop-ups. I just want premium diapers that work—keep the shopping simple."
            elif attempt_count == 1:
                description = "I'm too tired for this. Close button should work instantly. Every extra second dealing with pop-ups feels like torture when you're this sleep-deprived."
            else:
                description = f"Seriously? {attempt_count} tries and it's still here? I don't have the energy for this. Abandoning and ordering from somewhere that respects my time."

        elif "jasmine" in persona_lower:
            # Jasmine Lee: Visual-First, Social Proof Driven, Aesthetic-Focused
            if success:
                description = "Pop-up broke the visual flow. I want to enjoy browsing beautiful product images and Instagram content, not fight overlays."
            elif attempt_count == 1:
                description = "This pop-up is covering up the gorgeous product photography and layout. It's ruining my browsing experience and blocking me from seeing social proof."
            else:
                description = f"This stubborn pop-up ({attempt_count} attempts!) is completely ruining the aesthetic experience. Not the polished, Instagram-worthy vibe I expected."

        elif "priya" in persona_lower:
            # Priya Desai: Convenience-First, Speed-Obsessed, Urban Professional
            if success:
                description = "Pop-up wasted 2 seconds. I'm here for fastest checkout possible—every obstacle increases chance I'll abandon for Amazon Prime."
            elif attempt_count == 1:
                description = "Pop-up blocking my speed run to checkout. This better close immediately or I'm switching to Amazon 1-hour delivery."
            else:
                description = f"{attempt_count} attempts?! Unacceptable. Amazon would never. Abandoning for faster option—time is money."

        else:
            # Default generic description
            if success:
                description = "Modal interruption requiring dismissal."
            elif attempt_count == 1:
                description = f"Modal blocking navigation. Close button not responsive on first attempt."
            else:
                description = f"Persistent modal could not be dismissed after {attempt_count} attempts. Significant UX friction."

        return {
            "description": description,
            "severity": severity
        }

