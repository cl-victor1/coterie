"""
Main task executor that orchestrates the persona testing loop.
Coordinates browser, LLMs, and logging to simulate user behavior.
Refactored for single responsibility - delegates to specialized services.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from browser.browser_controller import BrowserController
from llm.gemini_client import GeminiClient
from llm.openai_client import OpenAIClient
from personas.persona_manager import PersonaManager
from core.action_logger import ActionLogger
from core.element_matcher import ElementMatcher
from core.action_executor import ActionExecutor
from core.loop_detector import LoopDetector
from core.verification_service import VerificationService
from core.task_completion_checker import TaskCompletionChecker


class TaskExecutor:
    """Executes usability testing tasks with AI personas using three-phase action loop."""

    def __init__(self, headless: bool = False, persistent_context: bool = False,
                 context_name: str = "default", human_delays: bool = True,
                 auto_dismiss_popups: bool = True):
        """
        Initialize enhanced task executor.

        Args:
            headless: Whether to run browser in headless mode
            persistent_context: Whether to use persistent browser context (saves cookies/storage)
            context_name: Name for persistent context (allows per-persona contexts)
            human_delays: Whether to add random delays to simulate human behavior
            auto_dismiss_popups: Whether to automatically dismiss popups (default: True)
        """
        self.browser = BrowserController(
            headless=headless,
            persistent_context=persistent_context,
            context_name=context_name,
            human_delays=human_delays
        )
        self.gemini = GeminiClient()
        self.openai = OpenAIClient()
        self.persona_manager = PersonaManager()
        self.logger: Optional[ActionLogger] = None
        
        # Initialize service classes
        self.element_matcher: Optional[ElementMatcher] = None
        self.action_executor: Optional[ActionExecutor] = None
        self.verification_service: Optional[VerificationService] = None
        self.task_completion_checker = TaskCompletionChecker()
        
        # State tracking
        self.max_actions = 20
        self.action_count = 0
        self.last_screenshot: Optional[str] = None
        self.recent_actions = []
        self.page_state_history = []
        self.recent_urls = []
        self.auto_dismiss_popups = auto_dismiss_popups

    async def execute_persona_test(self, persona_key: str) -> Dict[str, Any]:
        """
        Execute a complete test session for a persona (async).

        Args:
            persona_key: Key identifying the persona

        Returns:
            Test results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting test for persona: {persona_key}")
        print(f"{'='*60}\n")

        # Set up persona and logger
        persona = self.persona_manager.set_persona(persona_key)
        self.logger = ActionLogger(persona.name)
        self.action_count = 0
        
        # Initialize services with dependencies
        self.element_matcher = ElementMatcher(self.browser)
        self.action_executor = ActionExecutor(self.browser, self.element_matcher, self.logger)
        self.verification_service = VerificationService(self.browser, self.gemini, self.logger)
        
        # Reset task completion state
        self.task_completion_checker.reset()

        try:
            # Start browser with appropriate device
            await self.browser.start_browser(device_type=persona.device)

            # Login with credentials if available
            email = "victor.long.cheng@gmail.com"
            password = "vgi5*@n3LLjxgjA"
            if email and password:
                try:
                    login_success = await self.browser.login_to_coterie(email, password)
                    if login_success:
                        self.logger.log_action("login", {
                            "email": email,
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                            "note": "Pre-test authentication completed"
                        })
                    else:
                        self.logger.log_friction("login_failure",
                            "Failed to authenticate with provided credentials", "medium")
                except Exception as e:
                    self.logger.log_friction("login_error", str(e), "medium")
            else:
                print("\nℹ️ No login credentials provided - continuing as guest user\n")

            # Get initial navigation
            nav_info = self.persona_manager.get_initial_navigation_prompt()

            # Navigate to Coterie homepage
            await self.browser.navigate_to(nav_info["start_url"])
            self.logger.log_action("navigate", {
                "url": nav_info["start_url"],
                "entry_point": persona.entry_point,
                "entry_context": nav_info.get("entry_context", persona.entry_point)
            })

            # Main testing loop
            task_completed = False
            abandoned = False

            while self.action_count < self.max_actions and not task_completed and not abandoned:
                self.action_count += 1
                print(f"\n--- Action {self.action_count}/{self.max_actions} ---")

                # Capture current state
                snapshot = await self.browser.take_screenshot(f"{persona_key}_action{self.action_count}")
                self._log_screenshot_event(snapshot, f"Before action {self.action_count}")
                page_context = await self.browser.get_page_content()

                # Analyze current page with Gemini
                vision_prompt = self.persona_manager.get_vision_prompt(
                    f"Current URL: {page_context['url']}\nPage Title: {page_context['title']}"
                )
                if snapshot.get("bytes"):
                    vision_analysis = self.gemini.analyze_with_retry(
                        snapshot["bytes"], vision_prompt, self.browser.device_type
                    )
                else:
                    vision_analysis = {
                        "error": snapshot.get("error", "screenshot_unavailable"),
                        "visible_elements": [],
                        "relevant_to_task": [],
                        "friction_points": ["screenshot_not_available"],
                        "recommended_action": "retry"
                    }
                self.logger.log_page_analysis(vision_analysis)

                # Gather element-level analysis for coordinate-based clicks
                element_analysis: Dict[str, Any] = {}
                if snapshot.get("bytes"):
                    persona_focus = self.persona_manager.get_persona_specific_focus()
                    element_analysis = self.gemini.analyze_page_elements(
                        snapshot["bytes"],
                        persona_focus,
                        include_coordinates=True,
                        device_type=self.browser.device_type
                    )

                # Detect action loops
                loop_warning = LoopDetector.detect_action_loop(
                    self.recent_actions,
                    self.recent_urls,
                    self.page_state_history,
                    self.persona_manager.get_persona_action_playbook()
                )

                # Make decision with GPT
                decision_prompt = self.persona_manager.get_contextualized_decision_prompt(
                    vision_analysis,
                    page_context["url"],
                    self.recent_actions
                )

                # Add loop warning if detected
                if loop_warning:
                    print(f"⚠️ Loop detected: {loop_warning}")
                    decision_prompt = LoopDetector.add_loop_warning_to_prompt(
                        decision_prompt, loop_warning
                    )
                    self.logger.log_friction("action_loop", loop_warning, "high")

                # Add size selection failure warning if detected
                if self.task_completion_checker.size_selection_failed:
                    size_fail_warning = """
⚠️ CRITICAL: Size selector appears UNRESPONSIVE. Previous attempts to click the size button failed to visually select it (tried 3 times).
The size button may be broken, blocked by an element, or require a different interaction method.
SUGGESTION: Try alternative strategies:
- Scroll to ensure the size selector is fully visible
- Look for alternative size selection UI (dropdown, radio buttons, etc.)
- Try selecting a different size first, then Size 1
- Navigate away and return to refresh the page state
- Consider abandoning if the UI is truly broken
"""
                    print(f"⚠️ Size selection failed - adding context to decision")
                    decision_prompt = f"{decision_prompt}\n\n{size_fail_warning}"

                decision = self.openai.make_decision(decision_prompt)
                self.logger.log_decision(decision)

                # Track action for loop detection
                action_record = {
                    'action_type': decision.get('action_type'),
                    'target_element': decision.get('target_element'),
                    'reasoning': decision.get('reasoning'),
                    'coordinates': None,
                    'page_url': page_context['url'],
                    'action_result': None,
                    'page_changed': False
                }
                self.recent_actions.append(action_record)
                if len(self.recent_actions) > 15:
                    self.recent_actions.pop(0)

                # RETRY DETECTION
                is_retry = LoopDetector.detect_retry_attempt(decision, self.recent_actions)
                if is_retry:
                    print(f"  🔄 Retry detected for task-critical action: '{decision.get('target_element')}'")
                    print(f"  → Capturing fresh screenshot and re-analyzing for better coordinates...")

                    await asyncio.sleep(0.5)
                    retry_snapshot = await self.browser.take_screenshot(
                        f"{persona_key}_retry_action{self.action_count}"
                    )

                    if retry_snapshot.get("bytes"):
                        persona_focus = self.persona_manager.get_persona_specific_focus()
                        element_analysis = self.gemini.analyze_page_elements(
                            retry_snapshot["bytes"],
                            persona_focus,
                            include_coordinates=True,
                            device_type=self.browser.device_type
                        )
                        print(f"  ✓ Fresh element analysis completed - using updated coordinates")

                        self.logger.log_action("retry_detected", {
                            "target_element": decision.get('target_element'),
                            "retry_count": LoopDetector.count_retry_attempts(
                                decision.get('target_element'), self.recent_actions
                            ),
                            "fresh_screenshot": True,
                            "reasoning": "Task-critical action failed - retrying with fresh screenshot analysis"
                        })
                    else:
                        print(f"  ⚠️ Retry screenshot capture failed - proceeding with original analysis")

                # Execute the decided action
                if not decision.get("continue_task", True):
                    abandoned = True
                    print(f"❌ Persona decided to abandon task: {decision.get('reasoning')}")
                    break

                # PRE-ACTION POPUP CHECK
                if self.auto_dismiss_popups and decision.get('action_type') == 'click':
                    pre_dismissal_result = await self.browser.auto_dismiss_popup()
                    if pre_dismissal_result.get('dismissed'):
                        print("  ✓ Pre-action popup dismissed (cleared blocking overlay)")
                        self.logger.log_auto_dismissed_popup(pre_dismissal_result)
                        await asyncio.sleep(0.3)

                action_success = await self.action_executor.execute_action(
                    decision, element_analysis, self.recent_actions
                )

                # Special handling for Coterie product page
                current_url = page_context.get('url', '').lower()
                is_product_page = '/products/the-diaper' in current_url or '/products/' in current_url

                size_selection_verification = None

                if action_success and is_product_page and decision.get('action_type') == 'click':
                    print("  → Product page: Waiting extra time for React button state update...")
                    await asyncio.sleep(2.0)

                    # VERIFICATION: Check if this was a size selector click
                    target_element = decision.get('target_element', '').lower()
                    
                    is_add_to_cart = any(term in target_element for term in 
                                        ['add to cart', 'add-to-cart', 'addtocart', 'add to bag'])
                    
                    has_size_keyword = any(term in target_element for term in [
                        'size 1', 'size 2', 'size 3', 'size 4', 'size 5', 'size 6', 'size 7',
                        '8-12 lbs', '10-16 lbs', '12-18 lbs', '14-24 lbs', '16-28 lbs',
                        "'1'", "'2'", "'3'", "'4'", "'5'", "'6'", "'7'",
                        'pick your size', 'size grid'
                    ])
                    
                    is_size_selector_click = has_size_keyword and not is_add_to_cart

                    if is_size_selector_click:
                        print("  🔍 Size selector clicked - verifying selection state...")

                        verification = await self.verification_service.verify_size_selection(
                            persona_key, decision.get('target_element')
                        )

                        if not verification['verified']:
                            print(f"  ⚠️ WARNING: Size selector was clicked but vision does NOT confirm selection!")
                            print(f"  → Vision says: {verification['visual_indicator']}")
                            self.logger.log_friction(
                                "size_selection_not_confirmed",
                                f"Clicked '{decision.get('target_element')}' but visual selection state not detected after 2 seconds",
                                "medium"
                            )

                            # Attempt automatic retries
                            retry_result = await self.verification_service.retry_size_selection(
                                decision, element_analysis, self.action_executor,
                                persona_key, max_retries=2
                            )

                            if retry_result['success']:
                                size_selection_verification = retry_result['verification']
                                print(f"  ✓ Size selection confirmed after retries: {size_selection_verification['selected_size']}")
                                print(f"  → Visual indicator: {size_selection_verification['visual_indicator']}")
                            else:
                                size_selection_verification = retry_result.get('verification', verification)
                                self.task_completion_checker.size_selection_failed = True
                                print(f"  ❌ Size selection verification failed after all retry attempts")
                                self.logger.log_friction(
                                    "size_selection_failed_all_retries",
                                    f"Clicked size selector 3 times but visual selection never confirmed. Button may be unresponsive.",
                                    "high"
                                )
                        else:
                            size_selection_verification = verification
                            print(f"  ✓ Size selection confirmed: {verification['selected_size']}")
                            print(f"  → Visual indicator: {verification['visual_indicator']}")

                # Auto-dismiss popups if enabled
                if self.auto_dismiss_popups:
                    dismissal_result = await self.browser.auto_dismiss_popup()
                    if dismissal_result.get('dismissed'):
                        self.logger.log_auto_dismissed_popup(dismissal_result)

                # Check if a popup was detected (legacy tracking)
                if hasattr(self.browser, '_last_detected_popup') and self.browser._last_detected_popup:
                    popup_info = self.browser._last_detected_popup
                    self.logger.log_popup_event(popup_info, "appeared")
                    self.browser._last_detected_popup = None

                # Post-click URL verification
                if action_success and decision.get('action_type') == 'click':
                    target_element = decision.get('target_element', '').lower()
                    
                    post_click_url = (await self.browser.get_page_content()).get('url', '')
                    if any(phrase in target_element for phrase in ['add to cart', 'add-to-cart', 'addtocart']) and 'cart' in post_click_url:
                        print(f"  ✓ Add to cart successful - detected cart page navigation")
                        self.task_completion_checker.size_one_selected = True
                        self.task_completion_checker.bundle_selected = True

                    if any(phrase in target_element for phrase in ['shop', 'buy', 'product', 'cta']):
                        if post_click_url == current_url:
                            print(f"  ⚠️ Navigation expected but URL unchanged: {post_click_url}")
                            self.logger.log_friction("failed_navigation",
                                                    f"Clicked '{target_element}' but page didn't navigate",
                                                    "medium")

                # Track page state for loop detection
                page_state_hash = await self.browser.get_page_state_hash()
                page_changed = (page_state_hash != self.page_state_history[-1]
                               if len(self.page_state_history) >= 1 else True)

                # ROBUSTNESS FIX: Wait for UI updates
                if (action_success and not page_changed and
                    decision.get('action_type') == 'click' and
                    self.action_count > 1):

                    print("  ⚠️ Click succeeded but no DOM change detected - waiting for UI update...")
                    await asyncio.sleep(1.5)

                    new_page_state_hash = await self.browser.get_page_state_hash()
                    page_changed = (new_page_state_hash != page_state_hash)

                    if page_changed:
                        print("  ✓ DOM change detected after retry!")
                        page_state_hash = new_page_state_hash
                    else:
                        print("  → Still no DOM change (may be expected for some clicks)")

                self.page_state_history.append(page_state_hash)
                if len(self.page_state_history) > 5:
                    self.page_state_history.pop(0)

                # Update recent action record with results
                if self.recent_actions:
                    self.recent_actions[-1]['action_result'] = action_success
                    self.recent_actions[-1]['page_changed'] = page_changed

                # Get current page state for URL tracking
                post_action_context = await self.browser.get_page_content()

                # Track URL for auth trap detection
                current_url = post_action_context.get("url", "")
                self.recent_urls.append(current_url)
                if len(self.recent_urls) > 3:
                    self.recent_urls.pop(0)

                # Check for auth flow trap and escape if detected
                if await self._detect_and_escape_auth_trap():
                    post_action_context = await self.browser.get_page_content()

                # Update persona context
                if self.persona_manager.context_manager:
                    self.persona_manager.update_context(decision, action_success, vision_analysis)

                # Track task completion state
                self.task_completion_checker.update_selection_state(decision, size_selection_verification)

                # Check if task is completed
                if self.task_completion_checker.check_task_completion(
                    post_action_context, decision, self.recent_actions
                ):
                    task_completed = True
                    print(f"✅ Task completed! Diaper bundle added to cart.")

                # Brief pause between actions
                await asyncio.sleep(2)

            # Final screenshot
            final_snapshot = await self.browser.take_screenshot(f"{persona_key}_final")
            self._log_screenshot_event(final_snapshot, "Final state")

            # Generate final evaluation
            evaluation = self.openai.evaluate_completion(
                str(persona.to_dict()),
                self.logger.actions
            )

            # Compile results
            results = {
                "persona": persona.to_dict(),
                "test_results": {
                    "task_completed": task_completed,
                    "abandoned": abandoned,
                    "total_actions": self.action_count,
                    "total_time": self.logger.get_summary()["total_time_seconds"],
                    "friction_points": self.logger.friction_points,
                    "evaluation": evaluation
                },
                "action_log": self.logger.export_log()
            }

            print(f"\n{'='*60}")
            print(f"Test completed for {persona.name}")
            print(f"Task completed: {task_completed}")
            print(f"Actions taken: {self.action_count}")
            print(f"Friction points: {len(self.logger.friction_points)}")
            print(f"{'='*60}\n")

            return results

        except Exception as e:
            print(f"❌ Error during test execution: {str(e)}")
            return {
                "persona": persona.to_dict() if persona else {},
                "error": str(e),
                "action_log": self.logger.export_log() if self.logger else {}
            }

        finally:
            await self.browser.close_browser()

    def _log_screenshot_event(self, screenshot_payload: Dict[str, Any], context: str = "") -> None:
        """
        Log screenshot event from browser controller.

        Args:
            screenshot_payload: Dictionary with screenshot data or error
            context: Context description for the screenshot
        """
        if screenshot_payload.get("path"):
            self.logger.log_screenshot(screenshot_payload["path"], context)
        elif screenshot_payload.get("error"):
            self.logger.log_friction("screenshot_error", screenshot_payload["error"], "low")

    async def _detect_and_escape_auth_trap(self) -> bool:
        """
        Detect if persona is stuck in /auth flow and force escape (async).

        Returns:
            True if auth trap was detected and escape was attempted
        """
        if len(self.recent_urls) < 2:
            return False

        # Check if stuck on /auth page for 2+ consecutive actions
        auth_count = sum(1 for url in self.recent_urls if '/auth' in url.lower())

        if auth_count >= 2:
            print(f"⚠️ AUTH TRAP DETECTED: Stuck on /auth page for {auth_count} actions")
            self.logger.log_friction(
                "auth_flow_trap",
                f"Forced into authentication flow (/auth) for {auth_count} consecutive actions, blocking product exploration",
                "critical"
            )

            # Try to escape by navigating to product page
            try:
                print("→ Escaping auth trap by navigating to product page...")
                await self.browser.navigate_to("https://www.coterie.com/products/the-diaper")
                self.recent_urls.clear()
                await asyncio.sleep(2)
                return True
            except Exception as e:
                print(f"Auth trap escape failed: {str(e)}")
                # Try homepage as fallback
                try:
                    print("→ Fallback: Navigating to homepage...")
                    await self.browser.navigate_to("https://www.coterie.com")
                    self.recent_urls.clear()
                    await asyncio.sleep(2)
                    return True
                except Exception as e2:
                    print(f"Homepage fallback also failed: {str(e2)}")
                    return False

        return False
