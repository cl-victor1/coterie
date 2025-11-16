"""
Browser controller using Playwright for web automation.
Handles navigation, interactions, and screenshot capture.
"""

import asyncio
import re
import os
import random
from typing import Optional, Dict, Any, List
from playwright.async_api import async_playwright, Browser, Page, BrowserContext, Playwright
from datetime import datetime


class BrowserController:
    """Controls browser automation using Playwright."""

    def __init__(self, headless: bool = False, persistent_context: bool = False, context_name: str = "default", human_delays: bool = True):
        """
        Initialize browser controller.

        Args:
            headless: Whether to run browser in headless mode (False for visible browser)
            persistent_context: Whether to use persistent browser context (saves cookies/storage)
            context_name: Name for persistent context (allows per-persona contexts)
            human_delays: Whether to add random delays to simulate human behavior (default: True)
        """
        self.headless = headless
        self.persistent_context = persistent_context
        self.context_name = context_name
        self.human_delays = human_delays
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._login_in_progress = False  # Flag to track login process

    async def _human_delay(self, min_ms: int = 100, max_ms: int = 500) -> None:
        """
        Add a random delay to simulate human behavior (async).

        Args:
            min_ms: Minimum delay in milliseconds
            max_ms: Maximum delay in milliseconds
        """
        if self.human_delays:
            delay = random.uniform(min_ms / 1000, max_ms / 1000)
            await asyncio.sleep(delay)

    async def start_browser(self, device_type: str = "desktop") -> Page:
        """
        Start browser with appropriate settings (async).

        Args:
            device_type: Type of device to emulate ("desktop" or "mobile")

        Returns:
            Page object for interaction
        """
        # Store device type for later use in scroll methods
        self.device_type = device_type

        # Start Playwright
        self.playwright = await async_playwright().start()

        # Use persistent context if enabled
        if self.persistent_context:
            context_dir = os.path.join(os.getcwd(), ".browser_contexts", self.context_name)
            os.makedirs(context_dir, exist_ok=True)

            if device_type == "mobile":
                device = self.playwright.devices["iPhone 13"]
                # Filter device config to only include valid parameters for launch_persistent_context
                valid_device_params = {
                    k: v for k, v in device.items()
                    if k in ['viewport', 'user_agent', 'device_scale_factor', 'is_mobile', 'has_touch', 'locale', 'timezone_id', 'geolocation', 'permissions']
                }
                self.context = await self.playwright.chromium.launch_persistent_context(
                    context_dir,
                    headless=self.headless,
                    args=['--disable-blink-features=AutomationControlled'],
                    **valid_device_params,
                    ignore_https_errors=True,
                    accept_downloads=True
                )
            else:
                self.context = await self.playwright.chromium.launch_persistent_context(
                    context_dir,
                    headless=self.headless,
                    args=['--disable-blink-features=AutomationControlled'],
                    viewport={'width': 1920, 'height': 1080},
                    ignore_https_errors=True,
                    accept_downloads=True,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )

            print(f"✓ Using persistent browser context: {context_dir}")
            # Note: persistent context doesn't have a separate browser object
            self.browser = None
        else:
            # Configure browser based on device type (non-persistent)
            if device_type == "mobile":
                # Emulate iPhone 13
                device = self.playwright.devices["iPhone 13"]
                self.browser = await self.playwright.chromium.launch(
                    headless=self.headless,
                    args=['--disable-blink-features=AutomationControlled']
                )
                self.context = await self.browser.new_context(
                    **device,
                    ignore_https_errors=True,
                    accept_downloads=True
                )
            else:
                # Desktop configuration
                self.browser = await self.playwright.chromium.launch(
                    headless=self.headless,
                    args=['--disable-blink-features=AutomationControlled']
                )
                self.context = await self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    ignore_https_errors=True,
                    accept_downloads=True,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )

        # Create page (persistent context may already have pages)
        if self.persistent_context and len(self.context.pages) > 0:
            self.page = self.context.pages[0]
            print("✓ Reusing existing page from persistent context")
        else:
            self.page = await self.context.new_page()

        # Set default timeout
        self.page.set_default_timeout(30000)  # 30 seconds

       

        mode = 'persistent' if self.persistent_context else ('headless' if self.headless else 'visible')
        print(f"Browser started in {mode} mode with {device_type} configuration")

        return self.page

    async def login_to_coterie(self, email: str, password: str) -> bool:
        """
        Login to Coterie with provided credentials.

        Args:
            email: User email address
            password: User password

        Returns:
            True if login successful, False otherwise
        """
        if not self.page:
            return False

        try:
            # Step 1: Disable auth guards during login
            self._login_in_progress = True

            # Step 2: Navigate to auth page
            auth_url = "https://www.coterie.com/auth"

            # Attempt navigation
            await self.page.goto(auth_url, wait_until="domcontentloaded")
            await asyncio.sleep(2)  # Let page stabilize

            # Check URL after navigation
            current_url_after = self.page.url

            # Check for auto-login via cached session (redirect to account page)
            if "/account/" in current_url_after.lower():
                self._login_in_progress = False
                return True

            # Step 4: Find and fill email field
            email_selectors = [
                'input[type="email"]',
                'input[name="email"]',
                'input[id*="email"]',
                'input[placeholder*="email" i]',
                'input[placeholder*="Email" i]',
                'input[autocomplete="email"]',
                'input[autocomplete="username"]'
            ]

            email_filled = False
            for selector in email_selectors:
                try:
                    count = await self.page.locator(selector).count()
                    if count > 0:
                        await self.page.fill(selector, email)
                        email_filled = True
                        break
                except Exception:
                    continue

            if not email_filled:
                self._login_in_progress = False
                return False

            await self._human_delay(300, 700)

            # Step 5: Click "Next" button to proceed to password step
            next_button_clicked = False
            next_keywords = ["next", "continue", "proceed"]

            # Try role-based button selection first
            for keyword in next_keywords:
                try:
                    locator = self.page.get_by_role("button", name=keyword, exact=False)
                    count = await locator.count()
                    if count > 0:
                        await locator.first.click()
                        next_button_clicked = True
                        break
                except Exception:
                    continue

            # Fallback to selector-based search
            if not next_button_clicked:
                next_selectors = [
                    'button:has-text("Next")',
                    'button:has-text("Continue")',
                    'button[type="submit"]',
                    'input[type="submit"]'
                ]
                for selector in next_selectors:
                    try:
                        count = await self.page.locator(selector).count()
                        if count > 0:
                            await self.page.click(selector)
                            next_button_clicked = True
                            break
                    except Exception:
                        continue

            if not next_button_clicked:
                self._login_in_progress = False
                return False

            # Step 6: Wait for page transition to password step
            # Wait for either navigation or password field to appear
            try:
                # Wait up to 10 seconds for password field to appear
                await self.page.wait_for_selector('input[type="password"]', timeout=10000)
            except Exception:
                # Fallback: wait and check URL/page state
                await asyncio.sleep(3)

            # Wait for password field to be ready
            try:
                await self.page.wait_for_selector('input[type="password"]', timeout=8000)
            except Exception:
                # Wait a bit more and try alternative approaches
                await asyncio.sleep(2)

            # Step 7: Find and fill password field (now it should exist)
            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[id*="password"]',
                'input[placeholder*="password" i]',
                'input[placeholder*="Password" i]',
                'input[autocomplete="current-password"]',
                'input[autocomplete="password"]'
            ]

            password_filled = False
            for selector in password_selectors:
                try:
                    count = await self.page.locator(selector).count()
                    if count > 0:
                        await self.page.fill(selector, password)
                        password_filled = True
                        break
                except Exception:
                    continue

            if not password_filled:
                self._login_in_progress = False
                return False

            await self._human_delay(300, 700)

            # Step 8: Find and click Login/Submit button (second submit)
            # Keywords for login button (second step - avoid "next"/"continue" used in step 1)
            login_keywords = ["log in", "login", "sign in", "signin", "submit", "enter"]
            login_clicked = False

            # Try buttons with role="button" first
            for keyword in login_keywords:
                try:
                    locator = self.page.get_by_role("button", name=keyword, exact=False)
                    count = await locator.count()
                    if count > 0:
                        await locator.first.click()
                        login_clicked = True
                        break
                except Exception:
                    continue

            # Try generic button selectors if above didn't work
            if not login_clicked:
                login_selectors = [
                    'button:has-text("Log In")',
                    'button:has-text("Login")',
                    'button:has-text("Sign In")',
                    'button:has-text("Signin")',
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button[form]'  # Button with form attribute
                ]
                for selector in login_selectors:
                    try:
                        count = await self.page.locator(selector).count()
                        if count > 0:
                            await self.page.click(selector)
                            login_clicked = True
                            break
                    except Exception:
                        continue

            if not login_clicked:
                self._login_in_progress = False
                return False

            # Step 9: Wait for navigation/login to complete
            await asyncio.sleep(5)  # Give time for authentication to process

            # Step 10: Verify login success
            current_url = self.page.url

            # Check for successful auto-login (redirect to account page)
            if "/account/" in current_url.lower():
                self._login_in_progress = False
                return True

            # Check if we're still on auth page (login failed)
            if "/auth" in current_url.lower():
                # Check for error messages
                page_text = await self.page.inner_text("body")
                if any(err in page_text.lower() for err in ["invalid", "incorrect", "error", "failed"]):
                    self._login_in_progress = False
                    return False

            # Step 11: Navigate to homepage
            homepage = "https://www.coterie.com/"
            await self.page.goto(homepage, wait_until="domcontentloaded")
            await asyncio.sleep(2)

            # Step 12: Mark login as complete
            self._login_in_progress = False

            # Verify we're on homepage or account page
            final_url = self.page.url
            if "coterie.com" in final_url and ("/auth" not in final_url or "/account/" in final_url):
                return True
            else:
                return False

        except Exception:
            self._login_in_progress = False
            return False

    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to a specific URL (async).

        Args:
            url: Target URL

        Returns:
            Success status
        """
        try:
            await self._human_delay(300, 800)  # Random delay before navigation
            await self.page.goto(url, wait_until="domcontentloaded")

            # Wait for page to stabilize and popup blocker to take effect
            await asyncio.sleep(2)

            current_url = self.page.url if self.page else ""
            print(f"[Navigate] Landed on: {current_url or 'unknown'}")

           

            await self._human_delay(500, 1200)  # Delay after navigation (reading time)
            return True
        except Exception as e:
            print(f"Navigation error while loading {url}: {str(e)}")
            return False

    async def take_screenshot(self, name_prefix: str = "screenshot") -> Dict[str, Any]:
        """
        Capture screenshot of current page without writing to disk (async).

        Args:
            name_prefix: Identifier for tracing the screenshot purpose

        Returns:
            Dictionary containing raw PNG bytes and metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload: Dict[str, Any] = {
            "bytes": b"",
            "timestamp": timestamp,
            "label": name_prefix
        }

        try:
            payload["bytes"] = await self.page.screenshot(full_page=False, type="png")
            
        except Exception as e:
            payload["error"] = str(e)
            print(f"Screenshot error: {str(e)}")

        return payload

    async def _click_locator_device_aware(self, locator, description: str = "") -> bool:
        """
        Click a locator in a device-aware manner (async).
        Uses touch tap for mobile devices, mouse click for desktop.

        Args:
            locator: Playwright locator object
            description: Description of the element for logging

        Returns:
            Success status
        """
        try:
            # Wait for element to be visible (especially important for mobile SPAs)
            if hasattr(self, 'device_type') and self.device_type == "mobile":
                await locator.first.wait_for(state='visible', timeout=5000)

            # Device-aware clicking
            if hasattr(self, 'device_type') and self.device_type == "mobile":
                # Mobile: Get bounding box and tap center
                box = await locator.first.bounding_box()
                if box:
                    center_x = box['x'] + box['width'] / 2
                    center_y = box['y'] + box['height'] / 2
                    await self.page.touchscreen.tap(center_x, center_y)
                    print(f"Tapped element: {description} [mobile touch]")
                    await self._human_delay(300, 800)
                    return True
                else:
                    print(f"⚠️ Could not get bounding box for mobile tap: {description}")
                    # Fallback to regular click
                    await locator.first.click()
                    print(f"Clicked element (fallback): {description} [mobile click]")
                    await self._human_delay(300, 800)
                    return True
            else:
                # Desktop: Regular click
                await locator.first.click()
                print(f"Clicked element: {description} [desktop mouse]")
                await self._human_delay(300, 800)
                return True

        except Exception as e:
            print(f"Device-aware click error for {description}: {str(e)}")
            return False

    async def click_element(self, selector: str = None, text: str = None) -> bool:
        """
        Click an element on the page (async).

        Args:
            selector: CSS selector or element description
            text: Text content to find element

        Returns:
            Success status
        """
        try:
            await self._human_delay(200, 600)  # Delay before click (thinking time)

            if text:
                candidates = self._extract_click_targets(text)
                if text not in candidates:
                    candidates.append(text)

                for candidate in candidates:
                    candidate = candidate.strip()
                    if not candidate:
                        continue

                    try:
                        locator = self.page.get_by_role("button", name=candidate, exact=False)
                        if await locator.count():
                            if await self._click_locator_device_aware(locator, f"button: {candidate}"):
                                return True
                    except Exception:
                        pass

                    try:
                        locator = self.page.get_by_role("link", name=candidate, exact=False)
                        if await locator.count():
                            if await self._click_locator_device_aware(locator, f"link: {candidate}"):
                                return True
                    except Exception:
                        pass

                    try:
                        locator = self.page.get_by_text(candidate)
                        if await locator.count():
                            if await self._click_locator_device_aware(locator, f"text: {candidate}"):
                                return True
                    except Exception:
                        pass

                print(f"No clickable element found for text target: {text}")
                return False

            if selector:
                # Convert selector to locator and use device-aware click
                locator = self.page.locator(selector)
                if await locator.count():
                    if await self._click_locator_device_aware(locator, f"selector: {selector}"):
                        return True
                else:
                    print(f"No element found for selector: {selector}")
                    return False

            return False
        except Exception as e:
            print(f"Click error: {str(e)}")
            return False

    async def _get_element_at_coordinates(self, x: int, y: int) -> Dict[str, Any]:
        """
        Get detailed information about the element at given coordinates (async).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Dictionary with element details (tag, id, classes, text, etc.)
        """
        result = {"found": False}

        try:
            if not self.page:
                return result

            element_info_script = """
                (coords) => {
                    const el = document.elementFromPoint(coords.x, coords.y);
                    if (!el) return { found: false };

                    return {
                        found: true,
                        tag: el.tagName,
                        id: el.id || "",
                        classes: el.className || "",
                        text: el.textContent?.trim().substring(0, 100) || "",
                        href: el.href || "",
                        type: el.type || "",
                        role: el.getAttribute('role') || ""
                    };
                }
            """

            result = await self.page.evaluate(element_info_script, {"x": x, "y": y})
            return result

        except Exception:
            return {"found": False}

    async def _check_element_clickable(self, x: int, y: int) -> Dict[str, Any]:
        """
        Verify if an element at given coordinates is clickable (async).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Dictionary with clickability status and details
        """
        result = {
            "clickable": False,
            "reason": "",
            "in_viewport": False,
            "visible": False,
            "covered": False
        }

        try:
            if not self.page:
                result["reason"] = "No page available"
                return result

            # Check if coordinates are within viewport bounds
            viewport = self.page.viewport_size
            if viewport:
                width, height = viewport.get("width", 1920), viewport.get("height", 1080)
                if x < 0 or y < 0 or x >= width or y >= height:
                    result["reason"] = f"Coordinates ({x}, {y}) outside viewport ({width}x{height})"
                    return result
                result["in_viewport"] = True

            # Use JavaScript to check if element is visible and not covered by overlay
            check_script = """
                (coords) => {
                    const element = document.elementFromPoint(coords.x, coords.y);
                    if (!element) {
                        return { visible: false, covered: false, tagName: null };
                    }

                    const rect = element.getBoundingClientRect();
                    const style = window.getComputedStyle(element);

                    const visible = (
                        rect.width > 0 &&
                        rect.height > 0 &&
                        style.visibility !== 'hidden' &&
                        style.display !== 'none' &&
                        parseFloat(style.opacity) > 0
                    );

                    // Check if another element is covering this one
                    const topElement = document.elementFromPoint(coords.x, coords.y);
                    const covered = topElement !== element && !element.contains(topElement);

                    return {
                        visible: visible,
                        covered: covered,
                        tagName: element.tagName,
                        className: element.className,
                        id: element.id
                    };
                }
            """

            check_result = await self.page.evaluate(check_script, {"x": x, "y": y})

            result["visible"] = check_result.get("visible", False)
            result["covered"] = check_result.get("covered", False)

            if not result["visible"]:
                result["reason"] = "Element at coordinates is not visible"
                return result

            if result["covered"]:
                result["reason"] = f"Element covered by overlay (found {check_result.get('tagName', 'unknown')})"
                return result

            result["clickable"] = True
            result["reason"] = "Element is clickable"
            return result

        except Exception as e:
            result["reason"] = f"Clickability check failed: {str(e)}"
            return result

    async def click_at_coordinates(self, x: int, y: int) -> bool:
        """
        Click at specific coordinates on the page (async).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Success status
        """
        try:
            if not self.page:
                print("No page available for clicking")
                return False

            # Pre-click human delay (thinking time before clicking)
            await self._human_delay(200, 600)

            # CRITICAL: Check for popups immediately before clicking
            # This catches popups that appear after page load but before user interaction
            popup_check = await self.detect_popup_or_modal()
            if popup_check:
                print(f"⚠️ Popup detected before click - attempting auto-dismissal")
                dismissal = await self.auto_dismiss_popup()
                if dismissal.get('dismissed'):
                    print("  ✓ Popup cleared before click")
                    await asyncio.sleep(0.2)  # Brief wait for DOM to settle

            # Verify coordinates are within viewport and element is clickable
            clickability = await self._check_element_clickable(x, y)
            if not clickability["clickable"]:
                print(f"⚠️ Cannot click at ({x}, {y}): {clickability['reason']}")
                # Still attempt the click (coordinates may be valid despite check failure)
                # but log the warning for debugging

            # Click/Tap at the specified coordinates (device-aware)
            if hasattr(self, 'device_type') and self.device_type == "mobile":
                # Use touch events for mobile devices
                await self.page.touchscreen.tap(x, y)
                print(f"Tapped at coordinates: ({x}, {y}) [mobile touch]")
            else:
                # Use mouse events for desktop devices
                await self.page.mouse.click(x, y)
                print(f"Clicked at coordinates: ({x}, {y}) [desktop mouse]")

            # Log what element was actually clicked (for debugging popup intercepts)
            actual_element = await self._get_element_at_coordinates(x, y)
            if actual_element.get("found"):
                tag = actual_element.get("tag", "unknown")
                classes = actual_element.get("classes", "")
                element_id = actual_element.get("id", "")
                text = actual_element.get("text", "")[:50]  # First 50 chars

                element_desc = f"{tag}"
                if element_id:
                    element_desc += f"#{element_id}"
                if classes:
                    element_desc += f".{classes.split()[0]}" if classes else ""
                if text:
                    element_desc += f" ('{text}...')"

                print(f"  → Clicked element: {element_desc}")

            # Wait for network activity to settle (critical for dynamic content)
            # Increased timeout from 3s to 8s for complex SPAs and slow API responses
            network_settled = False
            try:
                await self.page.wait_for_load_state('networkidle', timeout=8000)
                print("  → Network settled after click")
                network_settled = True
            except Exception:
                # Timeout is acceptable - page does not always trigger network activity
                print("  → Network idle timeout (acceptable)")

            # Smart wait for DOM updates based on network activity
            # If network settled quickly, short wait; if timeout, longer wait for JS/React
            if network_settled:
                await asyncio.sleep(0.5)  # Quick DOM update check
            else:
                await asyncio.sleep(2.0)  # Longer wait for complex JS rendering

            # Check for popup/modal appearance after click
            popup_info = await self.detect_popup_or_modal()
            if popup_info:
                popup_text = popup_info.get("text", "")[:80]
                popup_id = popup_info.get("id", "")
                popup_classes = popup_info.get("classes", "")

                popup_desc = f"Popup/Modal detected"
                if popup_id:
                    popup_desc += f" #{popup_id}"
                elif popup_classes:
                    popup_desc += f" .{popup_classes.split()[0]}" if popup_classes else ""
                if popup_text:
                    popup_desc += f": '{popup_text}...'"

                print(f"  ⚠️ {popup_desc}")

                # Store popup info for logging (can be accessed by caller)
                if not hasattr(self, '_last_detected_popup'):
                    self._last_detected_popup = None
                self._last_detected_popup = popup_info

            # Post-click human delay (reading/processing time after action)
            await self._human_delay(300, 800)

            return True
        except Exception as e:
            print(f"Coordinate click error: {str(e)}")
            return False


    def _extract_click_targets(self, raw_text: str) -> List[str]:
        """
        Extract plausible target strings from descriptive action text.
        Enhanced to generate multiple candidate variations for better matching.
        """
        if not raw_text:
            return []

        candidates: List[str] = []
        
        # Extract quoted text first (highest priority)
        quoted_matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw_text)
        for match in quoted_matches:
            candidate = match[0] or match[1]
            if candidate:
                candidates.append(candidate.strip())

        # Extract text after colon
        if ":" in raw_text:
            after_colon = raw_text.split(":", 1)[1].strip()
            if after_colon and after_colon not in candidates:
                candidates.append(after_colon)

        # If no candidates yet, use the raw text
        if not candidates:
            candidates.append(raw_text.strip())
        
        # Now generate variations from all candidates
        expanded_candidates = []
        for candidate in candidates:
            # Add original
            expanded_candidates.append(candidate)
            
            # Generate keyword-based variations
            # Extract keywords: split by underscore, hyphen, space
            keywords = re.split(r'[_\-\s]+', candidate.lower())
            keywords = [kw.strip() for kw in keywords if kw.strip()]
            
            # Extract numbers separately
            numbers = re.findall(r'\d+', candidate)
            
            # Generate variations:
            # 1. Just the number (if exists) - e.g., "size_1_button" -> "1"
            for num in numbers:
                if num not in expanded_candidates:
                    expanded_candidates.append(num)
            
            # 2. Keywords with spaces - e.g., "size_1_button" -> "size 1"
            if len(keywords) > 1:
                space_joined = " ".join(keywords)
                if space_joined not in expanded_candidates:
                    expanded_candidates.append(space_joined)
            
            # 3. First two keywords - e.g., "size_1_button" -> "size 1"
            if len(keywords) >= 2:
                first_two = " ".join(keywords[:2])
                if first_two not in expanded_candidates:
                    expanded_candidates.append(first_two)
            
            # 4. Last two keywords - e.g., "size_1_button" -> "1 button"
            if len(keywords) >= 2:
                last_two = " ".join(keywords[-2:])
                if last_two not in expanded_candidates:
                    expanded_candidates.append(last_two)
            
            # 5. Number + first keyword - e.g., "size_1_button" -> "1 size"
            if numbers and keywords:
                for num in numbers:
                    num_first = f"{num} {keywords[0]}"
                    first_num = f"{keywords[0]} {num}"
                    if num_first not in expanded_candidates:
                        expanded_candidates.append(num_first)
                    if first_num not in expanded_candidates:
                        expanded_candidates.append(first_num)
            
            # 6. Individual significant keywords (length >= 3)
            for kw in keywords:
                if len(kw) >= 3 and kw not in expanded_candidates:
                    expanded_candidates.append(kw)
        
        # Filter and deduplicate
        final_candidates = []
        for c in expanded_candidates:
            c = c.strip()
            # Keep non-empty strings that aren't too generic
            if c and c not in final_candidates:
                # Skip very generic single-letter words (but keep numbers)
                if len(c) >= 1 and (c.isdigit() or len(c) >= 2):
                    final_candidates.append(c)
        
        return final_candidates

    async def type_text(self, selector: str, text: str) -> bool:
        """
        Type text into an input field with human-like delays (async).

        Args:
            selector: Input field selector
            text: Text to type

        Returns:
            Success status
        """
        try:
            await self._human_delay(200, 500)  # Delay before typing

            # Use type instead of fill for more realistic typing
            if self.human_delays:
                await self.page.type(selector, text, delay=random.randint(50, 150))
            else:
                await self.page.fill(selector, text)

            print(f"Typed text into: {selector}")
            await self._human_delay(200, 400)  # Brief delay after typing
            return True
        except Exception as e:
            print(f"Type error: {str(e)}")
            return False

    async def scroll_page(self, direction: str = "down", amount: int = 800) -> bool:
        """
        Scroll the page using multiple strategies for better compatibility (async).

        Args:
            direction: "up" or "down"
            amount: Pixels to scroll (default 800px for better content revelation)

        Returns:
            Success status
        """
        try:
            # Get initial scroll position (both window and potential containers)
            before_state = await self._get_comprehensive_scroll_state()

            scroll_amount = amount if direction == "down" else -amount

            # Strategy 0: For mobile devices, try touch swipe gesture first
            if hasattr(self, 'device_type') and self.device_type == "mobile":
                viewport = self.page.viewport_size
                if viewport:
                    center_x = viewport.get("width", 390) // 2  # iPhone 13 width
                    # Start and end points for swipe
                    start_y = viewport.get("height", 844) // 2
                    end_y = start_y - scroll_amount  # Negative for scroll down, positive for up

                    # Perform touch swipe gesture
                    try:
                        # Simulate touch swipe for scrolling
                        await self.page.evaluate("""
                            async ({startY, endY, centerX}) => {
                                const touchStart = new TouchEvent('touchstart', {
                                    touches: [new Touch({
                                        identifier: 0,
                                        target: document.body,
                                        clientX: centerX,
                                        clientY: startY
                                    })]
                                });

                                const touchMove = new TouchEvent('touchmove', {
                                    touches: [new Touch({
                                        identifier: 0,
                                        target: document.body,
                                        clientX: centerX,
                                        clientY: endY
                                    })]
                                });

                                const touchEnd = new TouchEvent('touchend', {
                                    changedTouches: [new Touch({
                                        identifier: 0,
                                        target: document.body,
                                        clientX: centerX,
                                        clientY: endY
                                    })]
                                });

                                document.body.dispatchEvent(touchStart);
                                await new Promise(r => setTimeout(r, 50));
                                document.body.dispatchEvent(touchMove);
                                await new Promise(r => setTimeout(r, 50));
                                document.body.dispatchEvent(touchEnd);
                            }
                        """, {"startY": start_y, "endY": end_y, "centerX": center_x})

                        await asyncio.sleep(0.5)

                        after_touch = await self._get_comprehensive_scroll_state()
                        if self._has_scrolled(before_state, after_touch):
                            print(f"Scrolled {direction} by {amount}px (touch swipe)")
                            await asyncio.sleep(0.5)
                            return True
                    except Exception as e:
                        print(f"Touch swipe failed: {str(e)}, trying other methods...")

            # Strategy 1: Try mouse wheel scrolling (most reliable for modern SPAs)
            # Mouse wheel at center of viewport for better compatibility
            viewport = self.page.viewport_size
            if viewport:
                center_x = viewport.get("width", 1920) // 2
                center_y = viewport.get("height", 1080) // 2

                # Move mouse to center and scroll
                await self.page.mouse.move(center_x, center_y)
                await self.page.mouse.wheel(0, scroll_amount)
                await asyncio.sleep(0.5)  # Brief pause for scroll animation

                # Check if Strategy 1 worked
                after_wheel = await self._get_comprehensive_scroll_state()
                if self._has_scrolled(before_state, after_wheel):
                    print(f"Scrolled {direction} by {amount}px (mouse wheel)")
                    await asyncio.sleep(0.5)  # Additional time for content loading
                    return True

            # Strategy 2: Try scrolling specific containers
            scrollable_container = await self._find_scrollable_container()
            if scrollable_container:
                # Scroll the container directly
                scroll_script = f"""
                    (element) => {{
                        element.scrollTop += {scroll_amount};
                        return element.scrollTop;
                    }}
                """
                await self.page.evaluate(scroll_script, scrollable_container)
                await asyncio.sleep(0.5)

                after_container = await self._get_comprehensive_scroll_state()
                if self._has_scrolled(before_state, after_container):
                    print(f"Scrolled {direction} by {amount}px (container)")
                    await asyncio.sleep(0.5)
                    return True

            # Strategy 3: Fallback to window scrolling (original method)
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            else:
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")

            await asyncio.sleep(0.5)

            # Final check
            after_window = await self._get_comprehensive_scroll_state()
            if self._has_scrolled(before_state, after_window):
                print(f"Scrolled {direction} by {amount}px (window)")
                await asyncio.sleep(0.5)
                return True

            # If still no scroll, try smooth scroll as last resort
            smooth_scroll_script = f"""
                () => {{
                    // Try scrolling the document element
                    document.documentElement.scrollBy({{
                        top: {scroll_amount},
                        behavior: 'smooth'
                    }});

                    // Also try body element
                    document.body.scrollBy({{
                        top: {scroll_amount},
                        behavior: 'smooth'
                    }});
                }}
            """
            await self.page.evaluate(smooth_scroll_script)
            await asyncio.sleep(1)  # Wait for smooth scroll animation

            after_smooth = await self._get_comprehensive_scroll_state()
            if self._has_scrolled(before_state, after_smooth):
                print(f"Scrolled {direction} by {amount}px (smooth scroll)")
                return True

            print(f"Scroll {direction} had no effect (at boundary or no scrollable content)")
            return False

        except Exception as e:
            print(f"Scroll error: {str(e)}")
            return False

    async def _find_scrollable_container(self):
        """
        Find the main scrollable container on the page.

        Returns:
            Element handle of scrollable container or None
        """
        try:
            # JavaScript to find the most likely scrollable container
            find_container_script = """
                () => {
                    // Common patterns for scrollable containers in SPAs
                    const candidates = [];

                    // Check for elements with overflow set to scroll/auto
                    const allElements = document.querySelectorAll('*');
                    for (let el of allElements) {
                        const style = window.getComputedStyle(el);
                        const isScrollable = (
                            (style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                            el.scrollHeight > el.clientHeight
                        );

                        if (isScrollable) {
                            // Calculate how much of viewport this element covers
                            const rect = el.getBoundingClientRect();
                            const viewportCoverage = (rect.width * rect.height) / (window.innerWidth * window.innerHeight);

                            candidates.push({
                                element: el,
                                coverage: viewportCoverage,
                                scrollHeight: el.scrollHeight,
                                clientHeight: el.clientHeight,
                                className: el.className,
                                id: el.id,
                                tagName: el.tagName
                            });
                        }
                    }

                    // Sort by viewport coverage (largest first)
                    candidates.sort((a, b) => b.coverage - a.coverage);

                    // Return the element that covers most of the viewport (likely main container)
                    // Exclude html and body as we handle those separately
                    for (let candidate of candidates) {
                        if (candidate.element.tagName.toLowerCase() !== 'html' &&
                            candidate.element.tagName.toLowerCase() !== 'body' &&
                            candidate.coverage > 0.3) {  // Covers at least 30% of viewport
                            return candidate.element;
                        }
                    }

                    // Check for common container patterns
                    const commonSelectors = [
                        'main', '[role="main"]', '.main-content', '#main-content',
                        '.content', '#content', '.container', '.app-content',
                        '[class*="scroll"]', '[id*="scroll"]', '.page-content'
                    ];

                    for (let selector of commonSelectors) {
                        const el = document.querySelector(selector);
                        if (el && el.scrollHeight > el.clientHeight) {
                            return el;
                        }
                    }

                    return null;
                }
            """

            container_handle = await self.page.evaluate_handle(find_container_script)

            # Check if we found a valid element
            is_element = await self.page.evaluate("el => el && el.nodeType === 1", container_handle)

            if is_element:
                # Log what we found for debugging
                container_info = await self.page.evaluate("""
                    el => ({
                        tagName: el.tagName,
                        id: el.id,
                        className: el.className,
                        scrollHeight: el.scrollHeight,
                        clientHeight: el.clientHeight
                    })
                """, container_handle)
                print(f"Found scrollable container: {container_info['tagName']} " +
                      f"(id='{container_info['id']}', class='{container_info['className']}')")
                return container_handle
            else:
                return None

        except Exception as e:
            print(f"Error finding scrollable container: {str(e)}")
            return None

    async def _get_comprehensive_scroll_state(self) -> Dict[str, Any]:
        """
        Get comprehensive scroll state including window and container positions.

        Returns:
            Dictionary with scroll positions for various elements
        """
        try:
            state = await self.page.evaluate("""
                () => {
                    // Get window scroll
                    const windowScroll = {
                        x: window.pageXOffset,
                        y: window.pageYOffset
                    };

                    // Get body scroll
                    const bodyScroll = {
                        x: document.body.scrollLeft,
                        y: document.body.scrollTop
                    };

                    // Get document element scroll
                    const docScroll = {
                        x: document.documentElement.scrollLeft,
                        y: document.documentElement.scrollTop
                    };

                    // Find any scrolled containers
                    const scrolledContainers = [];
                    const allElements = document.querySelectorAll('*');
                    for (let el of allElements) {
                        if (el.scrollTop > 0 || el.scrollLeft > 0) {
                            scrolledContainers.push({
                                tagName: el.tagName,
                                id: el.id,
                                className: el.className,
                                scrollTop: el.scrollTop,
                                scrollLeft: el.scrollLeft,
                                scrollHeight: el.scrollHeight,
                                clientHeight: el.clientHeight
                            });
                        }
                    }

                    // Get visible content hash (to detect content changes)
                    const visibleText = document.body.innerText.substring(0, 500);

                    return {
                        window: windowScroll,
                        body: bodyScroll,
                        documentElement: docScroll,
                        containers: scrolledContainers,
                        contentHash: visibleText
                    };
                }
            """)
            return state
        except Exception as e:
            print(f"Error getting comprehensive scroll state: {str(e)}")
            return {
                "window": {"x": 0, "y": 0},
                "body": {"x": 0, "y": 0},
                "documentElement": {"x": 0, "y": 0},
                "containers": [],
                "contentHash": ""
            }

    def _has_scrolled(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> bool:
        """
        Check if scrolling actually occurred by comparing states.

        Args:
            before_state: Scroll state before attempting to scroll
            after_state: Scroll state after attempting to scroll

        Returns:
            True if any scrolling occurred
        """
        try:
            # Check window scroll
            if (abs(after_state["window"]["y"] - before_state["window"]["y"]) > 10 or
                abs(after_state["window"]["x"] - before_state["window"]["x"]) > 10):
                return True

            # Check body scroll
            if (abs(after_state["body"]["y"] - before_state["body"]["y"]) > 10 or
                abs(after_state["body"]["x"] - before_state["body"]["x"]) > 10):
                return True

            # Check document element scroll
            if (abs(after_state["documentElement"]["y"] - before_state["documentElement"]["y"]) > 10 or
                abs(after_state["documentElement"]["x"] - before_state["documentElement"]["x"]) > 10):
                return True

            # Check if any container scrolled
            after_containers = {f"{c['tagName']}#{c['id']}.{c['className']}": c['scrollTop']
                               for c in after_state.get("containers", [])}
            before_containers = {f"{c['tagName']}#{c['id']}.{c['className']}": c['scrollTop']
                                for c in before_state.get("containers", [])}

            for key, after_scroll in after_containers.items():
                before_scroll = before_containers.get(key, 0)
                if abs(after_scroll - before_scroll) > 10:
                    return True

            # Check if visible content changed significantly (for infinite scroll)
            if before_state.get("contentHash", "") != after_state.get("contentHash", ""):
                # Content changed, likely due to scrolling
                return True

            return False

        except Exception as e:
            print(f"Error checking scroll: {str(e)}")
            # Conservative: assume scroll didn't work
            return False

    async def get_scroll_position(self) -> Dict[str, int]:
        """
        Get current scroll position and page dimensions (async).

        Returns:
            Dictionary with scroll position and page info
        """
        try:
            result = await self.page.evaluate("""
                () => ({
                    x: window.pageXOffset,
                    y: window.pageYOffset,
                    scrollHeight: document.documentElement.scrollHeight,
                    clientHeight: document.documentElement.clientHeight,
                    atTop: window.pageYOffset === 0,
                    atBottom: window.pageYOffset + window.innerHeight >= document.documentElement.scrollHeight - 10
                })
            """)
            return result
        except Exception as e:
            print(f"Error getting scroll position: {str(e)}")
            return {'x': 0, 'y': 0, 'atTop': False, 'atBottom': False}

    async def get_page_state_hash(self) -> str:
        """
        Get a hash of current page state to detect changes (async).
        Enhanced to detect React/SPA state changes (button states, selections, form values).

        Returns:
            String hash of visible content and interactive element states
        """
        try:
            # JavaScript to extract comprehensive page state including React updates
            state_info = await self.page.evaluate("""
                () => {
                    // Get visible text
                    const visibleText = document.body.innerText.substring(0, 1000);

                    // Get button states and text (catches React button updates)
                    const buttons = Array.from(document.querySelectorAll('button')).map(btn => ({
                        text: btn.textContent.trim(),
                        disabled: btn.disabled,
                        classes: btn.className
                    }));

                    // Get selected elements (checkboxes, radio buttons, selected options)
                    const selections = Array.from(document.querySelectorAll('[aria-selected="true"], [aria-checked="true"], .selected, input:checked, [class*="selected"]')).map(el => el.textContent.trim() + '|' + el.className);

                    // Get form input values
                    const inputs = Array.from(document.querySelectorAll('input, select, textarea')).map(input => input.value);

                    return {
                        url: window.location.href,
                        visibleText: visibleText,
                        buttons: buttons.slice(0, 10),  // First 10 buttons
                        selections: selections,
                        inputs: inputs.filter(v => v)  // Non-empty values
                    };
                }
            """)

            # Create hash from comprehensive state
            state_str = f"{state_info['url']}:{state_info['visibleText']}:{str(state_info['buttons'])}:{str(state_info['selections'])}:{str(state_info['inputs'])}"
            return str(hash(state_str))
        except Exception as e:
            print(f"Error getting page state hash: {str(e)}")
            # Fallback to simple version
            try:
                visible_text = await self.page.inner_text("body")
                url = self.page.url
                state = f"{url}:{visible_text[:1000]}"
                return str(hash(state))
            except:
                return ""

    async def get_page_content(self) -> Dict[str, Any]:
        """
        Extract current page content and metadata (async).

        Returns:
            Dictionary with page information
        """
        try:
            return {
                "url": self.page.url,
                "title": await self.page.title(),
                "html": await self.page.content(),
                "visible_text": await self.page.inner_text("body"),
            }
        except Exception as e:
            print(f"Content extraction error: {str(e)}")
            return {
                "url": self.page.url if self.page else "",
                "title": "",
                "html": "",
                "visible_text": "",
                "error": str(e)
            }

    async def find_elements(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Find elements on page matching search terms (async).

        Args:
            search_terms: List of terms to search for

        Returns:
            List of found elements with details
        """
        found_elements = []

        for term in search_terms:
            try:
                # Search for buttons
                buttons = await self.page.get_by_role("button").filter(has_text=term).all()
                for btn in buttons:
                    found_elements.append({
                        "type": "button",
                        "text": await btn.inner_text(),
                        "selector": f"button:has-text('{term}')"
                    })

                # Search for links
                links = await self.page.get_by_role("link").filter(has_text=term).all()
                for link in links:
                    found_elements.append({
                        "type": "link",
                        "text": await link.inner_text(),
                        "selector": f"a:has-text('{term}')",
                        "href": await link.get_attribute("href")
                    })

                # Search for any element with text
                elements = await self.page.get_by_text(term).all()
                for elem in elements:
                    text = await elem.inner_text()
                    found_elements.append({
                        "type": "text",
                        "text": text[:100],  # Truncate long text
                        "selector": f":has-text('{term}')"
                    })

            except Exception as e:
                print(f"Error finding elements for '{term}': {str(e)}")

        return found_elements

    async def wait_for_element(self, selector: str, timeout: int = 10000) -> bool:
        """
        Wait for an element to appear (async).

        Args:
            selector: Element selector
            timeout: Maximum wait time in milliseconds

        Returns:
            Whether element appeared
        """
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except:
            return False

    async def detect_popup_or_modal(self) -> Optional[Dict[str, Any]]:
        """
        Detect if a popup or modal is currently visible on the page (async).

        Returns:
            Dictionary with popup details if found, None otherwise
        """
        if not self.page:
            return None

        try:
            popup_detection_script = """
                () => {
                    // Look for common popup/modal patterns
                    const elements = Array.from(document.querySelectorAll('*'));

                    // Filter for potential popups/modals
                    const candidates = elements.filter(el => {
                        const style = window.getComputedStyle(el);
                        const zIndex = parseInt(style.zIndex) || 0;
                        const position = style.position;
                        const display = style.display;
                        const visibility = style.visibility;

                        // Check for high z-index overlays
                        if (zIndex < 1000) return false;
                        if (position !== 'fixed' && position !== 'absolute') return false;
                        if (display === 'none' || visibility === 'hidden') return false;

                        // Check size (must cover significant portion of screen)
                        const rect = el.getBoundingClientRect();
                        const area = rect.width * rect.height;
                        const viewportArea = window.innerWidth * window.innerHeight;

                        // Either a large overlay (>25% of viewport) or a centered modal
                        const isLargeOverlay = area > (viewportArea * 0.25);
                        const isCentered = (
                            rect.left > window.innerWidth * 0.1 &&
                            rect.right < window.innerWidth * 0.9 &&
                            rect.top > window.innerHeight * 0.1
                        );

                        return isLargeOverlay || isCentered;
                    });

                    if (candidates.length === 0) return null;

                    // Get the topmost candidate (highest z-index)
                    const popup = candidates.reduce((max, el) => {
                        const maxZ = parseInt(window.getComputedStyle(max).zIndex) || 0;
                        const elZ = parseInt(window.getComputedStyle(el).zIndex) || 0;
                        return elZ > maxZ ? el : max;
                    });

                    // Extract popup information
                    const rect = popup.getBoundingClientRect();
                    const text = popup.textContent?.trim().substring(0, 200) || "";

                    // Look for close button within popup
                    const closeButton = popup.querySelector('[aria-label*="close" i], [aria-label*="dismiss" i], .close, .modal-close, button[aria-label*="Close"]');
                    let closeButtonInfo = null;

                    if (closeButton) {
                        const btnRect = closeButton.getBoundingClientRect();
                        closeButtonInfo = {
                            x: Math.round(btnRect.left + btnRect.width / 2),
                            y: Math.round(btnRect.top + btnRect.height / 2),
                            text: closeButton.textContent?.trim() || "×",
                            ariaLabel: closeButton.getAttribute('aria-label') || ""
                        };
                    }

                    return {
                        found: true,
                        tag: popup.tagName,
                        id: popup.id || "",
                        classes: popup.className || "",
                        text: text,
                        zIndex: parseInt(window.getComputedStyle(popup).zIndex) || 0,
                        position: {
                            x: Math.round(rect.left),
                            y: Math.round(rect.top),
                            width: Math.round(rect.width),
                            height: Math.round(rect.height)
                        },
                        closeButton: closeButtonInfo
                    };
                }
            """

            result = await self.page.evaluate(popup_detection_script)

            if result and result.get("found"):
                return result

            return None

        except Exception as e:
            print(f"Popup detection error: {str(e)}")
            return None

    

    async def auto_dismiss_popup(self) -> Dict[str, Any]:
        """
        Automatically detect and dismiss any visible popup/modal (async).

        Returns:
            Dictionary with dismissal status and details:
            {
                'dismissed': bool,
                'popup_found': bool,
                'popup_info': dict or None,
                'reason': str
            }
        """
        result = {
            'dismissed': False,
            'popup_found': False,
            'popup_info': None,
            'reason': ''
        }

        try:
            # Detect if popup is present
            popup_info = await self.detect_popup_or_modal()

            if not popup_info:
                result['reason'] = 'No popup detected'
                return result

            result['popup_found'] = True
            result['popup_info'] = popup_info

            # Get close button coordinates
            close_btn = popup_info.get('closeButton')

            if not close_btn:
                result['reason'] = 'Popup found but no close button detected'
                return result

            # Click the close button using coordinates
            x = close_btn.get('x')
            y = close_btn.get('y')

            if x is None or y is None:
                result['reason'] = 'Close button found but coordinates invalid'
                return result

            print(f"🔘 Auto-dismissing popup: '{popup_info.get('text', '')[:50]}...'")

            # Click close button (device-aware)
            if hasattr(self, 'device_type') and self.device_type == "mobile":
                await self.page.touchscreen.tap(x, y)
                print(f"  → Tapped close button at ({x}, {y}) [mobile]")
            else:
                await self.page.mouse.click(x, y)
                print(f"  → Clicked close button at ({x}, {y}) [desktop]")

            # Wait for popup to close
            await asyncio.sleep(0.5)

            # Verify popup is gone
            popup_still_present = await self.detect_popup_or_modal()

            if not popup_still_present:
                result['dismissed'] = True
                result['reason'] = 'Popup successfully dismissed'
                print("  ✓ Popup dismissed successfully")
            else:
                result['reason'] = 'Click attempted but popup still visible'
                print("  ⚠️ Popup still visible after dismiss attempt")

            return result

        except Exception as e:
            result['reason'] = f'Error during auto-dismissal: {str(e)}'
            print(f"Auto-dismiss error: {str(e)}")
            return result

    async def close_browser(self):
        """Close browser and cleanup resources (async)."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            print("Browser closed")
        except Exception as e:
            print(f"Error closing browser: {str(e)}")
