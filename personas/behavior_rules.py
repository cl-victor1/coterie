"""
Behavioral rules and constraints for different personas.
Extracted from persona_manager.py for single responsibility.
"""

from typing import Dict, Any, List


class BehaviorRules:
    """Defines persona-specific behavioral rules, constraints, and page guidance."""

    @staticmethod
    def get_persona_action_playbook(persona_name: str) -> Dict[str, Any]:
        """
        Get persona-specific action preferences and behavioral rules.

        Args:
            persona_name: Name of the persona

        Returns:
            Dictionary with preferred actions, prohibited actions, and behavioral rules
        """
        persona_lower = persona_name.lower()

        # Sarah Kim - Subscription-Savvy, Efficiency-focused
        if "sarah" in persona_lower:
            return {
                "preferred_actions": [
                    "click_auto_renew_option",
                    "click_shop_the_diaper",
                    "click_the_diaper_nav",
                    "select_size_1_from_grid",
                    "click_choose_a_size_button"
                ],
                "prohibited_actions": [
                    "endless_scrolling",
                    "read_long_content",
                    "explore_about_pages",
                    "browse_image_galleries"
                ],
                "max_scrolls_per_page": 3,
                "max_actions_before_decision": 8,
                "behavioral_rules": [
                    "Click 'Shop The Diaper' or 'The Diaper' in nav immediately",
                    "Look for Auto-Renew (10% off = $95/month vs $105.50)",
                    "Select Size 1 from size grid, click 'Choose a Size'",
                    "Linear path - no exploration or backtracking",
                    "Abandon if subscription/delivery terms unclear after 3 attempts"
                ],
                "decision_style": "analytical_efficient",
                "patience_threshold": 0.3
            }

        # Maya Rodriguez - Eco-Conscious, Research-oriented
        elif "maya" in persona_lower:
            return {
                "preferred_actions": [
                    "scroll_to_find_about_impact",
                    "click_safety_reports",
                    "read_ingredient_list",
                    "verify_oeko_tex_certification",
                    "check_25_percent_plant_based",
                    "read_detailed_content"
                ],
                "prohibited_actions": [
                    "quick_purchase_without_research",
                    "skip_ingredient_verification",
                    "ignore_certifications"
                ],
                "max_scrolls_per_page": 12,
                "max_actions_before_decision": 20,
                "behavioral_rules": [
                    "Scroll to footer/nav to find 'About', 'Impact', or 'Safety Reports'",
                    "Verify OEKO-TEX certification and 25% plant-based materials claim",
                    "Read ingredient list (confirm no fragrance/latex/parabens)",
                    "Research before clicking Size 1 - don't rush to purchase",
                    "Abandon if no real eco credentials or greenwashing detected"
                ],
                "decision_style": "thoughtful_researcher",
                "patience_threshold": 0.6
            }

        # Lauren Peterson - Sleep-Deprived, Impatient
        elif "lauren" in persona_lower:
            return {
                "preferred_actions": [
                    "quick_check_4_8_star_rating",
                    "click_shop_the_diaper_immediately",
                    "click_size_1_from_grid",
                    "click_choose_a_size",
                    "abandon_if_confused"
                ],
                "prohibited_actions": [
                    "detailed_research",
                    "long_reading",
                    "complex_navigation",
                    "explore_multiple_pages"
                ],
                "max_scrolls_per_page": 3,
                "max_actions_before_decision": 6,
                "behavioral_rules": [
                    "Quick glance at 4.8/5 rating (8,939 reviews) for reassurance",
                    "Click 'Shop The Diaper' or size selection immediately",
                    "Make fast decision - Size 1 → 'Choose a Size' → done",
                    "Abandon QUICKLY if navigation unclear or confusing",
                    "No patience for slow sites, complex flows, or hidden info"
                ],
                "decision_style": "emotional_quick",
                "patience_threshold": 0.2
            }

        # Jasmine Lee - Influencer-Following, Visual
        elif "jasmine" in persona_lower:
            return {
                "preferred_actions": [
                    "scroll_to_see_instagram_gallery",
                    "browse_product_image_gallery",
                    "check_4_8_star_reviews",
                    "look_for_best_of_babylist_badge",
                    "view_lifestyle_photos",
                    "explore_award_badges"
                ],
                "prohibited_actions": [
                    "read_technical_details",
                    "analyze_pricing_deeply",
                    "skip_visual_content",
                    "rush_through_images"
                ],
                "max_scrolls_per_page": 10,
                "max_actions_before_decision": 12,
                "behavioral_rules": [
                    "Scroll homepage to see Instagram gallery and award badges",
                    "Browse product image gallery and lifestyle photos on product page",
                    "Check 4.8/5 star reviews (8,939 reviews) for social proof",
                    "Look for 'Best of Babylist', 'The Bump' awards",
                    "Make decision based on aesthetics and community validation"
                ],
                "decision_style": "visual_social",
                "patience_threshold": 0.4
            }

        # Priya Desai - Convenience-First, Speed-focused
        elif "priya" in persona_lower:
            return {
                "preferred_actions": [
                    "click_shop_the_diaper_now",
                    "click_quick_buy",
                    "select_size_1_immediately",
                    "click_choose_a_size_button",
                    "check_delivery_schedule"
                ],
                "prohibited_actions": [
                    "any_exploration",
                    "long_scrolling",
                    "reading_lengthy_content",
                    "browsing_galleries",
                    "checking_reviews"
                ],
                "max_scrolls_per_page": 3,
                "max_actions_before_decision": 6,
                "behavioral_rules": [
                    "Click 'Shop The Diaper' or 'Quick Buy' instantly",
                    "Select Size 1 from grid → Wait for button to update to 'Add to Cart - Size 1' → Click it → Done",
                    "After clicking Size 1, button text changes (this is normal, not a failure)",
                    "Check delivery schedule (3-5 weeks) only if shown",
                    "Zero exploration, zero browsing, zero reading",
                    "Abandon if process takes more than 5 clicks total or truly stuck"
                ],
                "decision_style": "goal_driven_fast",
                "patience_threshold": 0.25
            }

        return {}

    @staticmethod
    def get_page_specific_guidance(persona_name: str, current_url: str, 
                                   page_analysis: Dict[str, Any],
                                   context_manager) -> str:
        """
        Get page-specific behavioral guidance for current persona.

        Args:
            persona_name: Name of the persona
            current_url: Current page URL
            page_analysis: Vision analysis of current page
            context_manager: Context manager for action history

        Returns:
            Specific guidance string for this persona on this page type
        """
        persona_lower = persona_name.lower()
        url_lower = current_url.lower()

        # Detect page type
        is_homepage = 'coterie.com' in url_lower and url_lower.count('/') <= 3
        is_product_page = any(term in url_lower for term in ['product', 'diaper', '/p/', 'bundle'])
        is_about_page = 'about' in url_lower or 'our-story' in url_lower or 'mission' in url_lower
        is_cart = 'cart' in url_lower or 'checkout' in url_lower

        guidance = []

        # Homepage guidance
        if is_homepage:
            if "sarah" in persona_lower:
                guidance.append("HOMEPAGE STRATEGY: Look for 'Auto-Renew' mention (10% off) or 'Quick Buy' buttons. Click 'Shop The Diaper' or 'The Diaper' nav menu. Goal: Find Size 1 with auto-renew/subscription.")
            elif "maya" in persona_lower:
                guidance.append("HOMEPAGE STRATEGY: Scroll to find 'About', 'Safety Reports', 'Impact', or eco/sustainability info in footer/nav. You MUST verify eco claims before product purchase. Don't rush.")
            elif "lauren" in persona_lower:
                guidance.append("HOMEPAGE STRATEGY: Quick look for comfort/quality claims. If 'Shop The Diaper' or 'Quick Buy' visible, click it. Max 1-2 scrolls only. Abandon if navigation unclear.")
            elif "jasmine" in persona_lower:
                guidance.append("HOMEPAGE STRATEGY: Scroll to see Instagram gallery, award badges (The Bump, Babylist), customer reviews. Enjoy aesthetics and social proof before clicking products.")
            elif "priya" in persona_lower:
                guidance.append("HOMEPAGE STRATEGY: Click 'Shop The Diaper' or 'Quick Buy' button NOW. Direct to Size 1 selection. Zero exploration. Fast path only.")

        # Product page guidance
        elif is_product_page:
            if "sarah" in persona_lower:
                guidance.append("PRODUCT PAGE STRATEGY: Check 'Auto-Renew' benefits (10% off = $95/month vs $105.50). IMPORTANT: Check the vision analysis 'ui_states' to see if Size 1 is already selected. If NOT selected, click Size 1 from grid. After clicking, the button will update to 'Add to cart - Size 1' (normal React behavior). Only proceed to click 'Add to cart' if vision confirms Size 1 is SELECTED. Plan defaults to Auto-Renew (cheapest). Fast, linear path.")
            elif "maya" in persona_lower:
                guidance.append("PRODUCT PAGE STRATEGY: First scroll to verify '25% plant-based', OEKO-TEX certification, ingredient list (no fragrance/latex/parabens). After eco verification, check vision analysis 'ui_states' to confirm Size 1 selection status. If NOT selected, click Size 1 from grid. The button will update from 'Choose a Size' to 'Add to cart - Size 1' - this is normal React behavior, NOT a failure. Wait for vision to confirm Size 1 is SELECTED before clicking Add to Cart. Plan defaults to Auto-Renew.")
            elif "lauren" in persona_lower:
                guidance.append("PRODUCT PAGE STRATEGY: You're on the diaper page! Flow: 1) Check vision analysis 'ui_states' - if Size 1 NOT selected, click it from grid, 2) Plan defaults to Auto-Renew (cheaper at $95 vs $105.50) - keep default, 3) Only after vision confirms Size 1 is SELECTED, click 'Add to cart - Size 1' button. IMPORTANT: The button text CHANGES after size selection - this is NORMAL React behavior, NOT broken. Don't overthink, just verify state and click through.")
            elif "jasmine" in persona_lower:
                guidance.append("PRODUCT PAGE STRATEGY: Browse product gallery and lifestyle photos first. Check 4.8/5 reviews for social proof. Look for award badges. Once satisfied with aesthetics, check vision 'ui_states' for Size 1 status. If NOT selected, click Size 1 from grid. Button will change to 'Add to cart - Size 1' - this is the React app updating, totally normal. Only click Add to Cart after vision confirms Size 1 is SELECTED. Plan defaults to Auto-Renew.")
            elif "priya" in persona_lower:
                guidance.append("PRODUCT PAGE STRATEGY: Check vision 'ui_states' for Size 1 status. If NOT selected, click Size 1 from grid IMMEDIATELY. CRITICAL: After clicking Size 1, the page's React app updates the button text from 'Choose a Size' to 'Add to cart - Size 1'. Wait 1-2 seconds for this update. This is NOT a failure - it's how modern web apps work. Only click 'Add to cart - Size 1' after vision confirms Size 1 is SELECTED. Plan defaults to Auto-Renew (cheapest). Zero reading. Direct path only.")

        # About/Sustainability page guidance
        elif is_about_page:
            if "maya" in persona_lower:
                guidance.append("ABOUT PAGE: Perfect! Read carefully for eco claims, certifications, and transparency. This is exactly what you need.")
            elif "sarah" in persona_lower or "priya" in persona_lower:
                guidance.append("ABOUT PAGE WARNING: You don't care about this. Navigate back to products/pricing immediately.")

        # Cart page guidance
        elif is_cart:
            guidance.append("CART PAGE: Task nearly complete! Verify Size 1 diaper bundle is in cart. Check price. Consider checkout if all looks good.")

        # Add scroll count warnings
        if context_manager:
            recent_scrolls = sum(1 for action in context_manager.action_history[-5:]
                               if action.get('action_type') == 'scroll')
            playbook = BehaviorRules.get_persona_action_playbook(persona_name)
            max_scrolls = playbook.get('max_scrolls_per_page', 3)

            if recent_scrolls >= max_scrolls:
                guidance.append(f"⚠️ SCROLL WARNING: You've scrolled {recent_scrolls} times on this page. Your max is {max_scrolls}. STOP scrolling and CLICK something or ABANDON.")

        return "\n".join(guidance) if guidance else ""

    @staticmethod
    def apply_persona_behavior_rules(persona, decision: Dict[str, Any],
                                    page_analysis: Dict[str, Any],
                                    context_manager) -> Dict[str, Any]:
        """
        Apply persona-specific behavioral rules to override decisions if needed.

        Args:
            persona: Current Persona object
            decision: Initial decision from LLM
            page_analysis: Current page analysis
            context_manager: Context manager instance

        Returns:
            Modified decision with persona-specific rules applied
        """
        if not persona or not context_manager:
            return decision

        persona_lower = persona.name.lower()

        # Maya Rodriguez - Must explore before buying
        if "maya" in persona_lower:
            if context_manager.should_explore_first():
                # Check if trying to buy too early
                if decision.get("action_type") == "click" and "diaper" in str(decision.get("target_element", "")).lower():
                    # Override to explore instead
                    for element in page_analysis.get("visible_elements", []):
                        element_lower = str(element).lower()
                        if any(term in element_lower for term in ["sustain", "eco", "ingredient", "about"]):
                            decision["action_type"] = "click"
                            decision["target_element"] = element
                            decision["reasoning"] = "I need to understand their sustainability practices before buying anything."
                            decision["internal_thought"] = "Can't just buy without knowing the environmental impact"
                            break
                    else:
                        # If no sustainability info visible, scroll to find it
                        decision["action_type"] = "scroll"
                        decision["target_element"] = "down"
                        decision["reasoning"] = "Looking for information about ingredients and environmental practices."

        # Sarah Kim - Efficiency first
        elif "sarah" in persona_lower:
            # Look for subscription or bulk options
            for element in page_analysis.get("visible_elements", []):
                element_lower = str(element).lower()
                if "Shop the Diaper" in element_lower:
                    if decision.get("action_type") != "click":
                        decision["action_type"] = "click"
                        decision["target_element"] = element
                        decision["reasoning"] = "Shop the Diaper link found - this saves time with auto-delivery."
                        break

        # Lauren Peterson - Quick abandonment
        elif "lauren" in persona_lower:
            if context_manager.state.patience_remaining < 0.3:
                if decision.get("action_type") not in ["click", "abandon"]:
                    # Force decisive action or abandonment
                    decision["continue_task"] = False
                    decision["action_type"] = "abandon"
                    decision["reasoning"] = "Too exhausted to figure this out. Will try again when I have more energy."

        # Jasmine Lee - Visual focus
        elif "jasmine" in persona_lower:
            # Prioritize visually appealing elements
            for element in page_analysis.get("visible_elements", []):
                element_lower = str(element).lower()
                if any(term in element_lower for term in ["gallery", "photo", "review", "influencer", "testimonial"]):
                    if context_manager.state.success_count < 2:  # Early in journey
                        decision["action_type"] = "click"
                        decision["target_element"] = element
                        decision["reasoning"] = f"Ooh, this looks interesting! Want to see {element}"
                        break

        # Priya Desai - Direct path
        elif "priya" in persona_lower:
            # Skip any exploration
            if decision.get("action_type") == "scroll" and len(context_manager.action_history) < 3:
                # Override scroll with direct action if possible
                for element in page_analysis.get("relevant_to_task", []):
                    if "diaper" in str(element).lower() or "shop" in str(element).lower():
                        decision["action_type"] = "click"
                        decision["target_element"] = element
                        decision["reasoning"] = "Going directly to diapers. No time for browsing."
                        break

        return decision

    @staticmethod
    def get_persona_specific_focus(persona_name: str) -> List[str]:
        """
        Get list of elements this persona would focus on.

        Args:
            persona_name: Name of the persona

        Returns:
            List of focus keywords
        """
        persona_lower = persona_name.lower()

        if "maya" in persona_lower:
            return ["sustainability", "eco", "natural", "ingredients", "certifications"]
        elif "sarah" in persona_lower:
            return ["subscription", "delivery", "auto-ship", "bulk", "save"]
        elif "lauren" in persona_lower:
            return ["overnight", "sleep", "absorbent", "leak", "comfortable"]
        elif "jasmine" in persona_lower:
            return ["review", "photo", "gallery", "influencer", "testimonial", "cute"]
        elif "priya" in persona_lower:
            return ["cart", "checkout", "buy", "delivery", "fast", "express"]

        return []

