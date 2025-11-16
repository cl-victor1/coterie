"""
Element matching and coordinate conversion utilities.
Extracted from task_executor.py for single responsibility.
"""

import re
from typing import Dict, Any, Optional, List


class ElementMatcher:
    """Handles element matching, keyword extraction, and coordinate conversion."""

    def __init__(self, browser):
        """
        Initialize element matcher.

        Args:
            browser: BrowserController instance for viewport information
        """
        self.browser = browser

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for intelligent matching.
        
        Args:
            text: Input text (e.g., "size_1_button", "Size 1")
            
        Returns:
            List of keywords extracted from text
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Split by common separators: underscore, hyphen, space
        keywords = re.split(r'[_\-\s]+', text)
        
        # Also extract numbers separately
        numbers = re.findall(r'\d+', text)
        
        # Combine and deduplicate
        all_keywords = keywords + numbers
        
        # Filter out empty strings and very short words (except numbers)
        filtered = []
        for kw in all_keywords:
            kw = kw.strip()
            if kw and (kw.isdigit() or len(kw) >= 2):
                if kw not in filtered:
                    filtered.append(kw)
        
        return filtered
    
    def calculate_match_score(self, target_keywords: List[str], element_keywords: List[str]) -> float:
        """
        Calculate match score between target and element keywords.
        
        Args:
            target_keywords: Keywords from target element description
            element_keywords: Keywords from actual element text
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not target_keywords or not element_keywords:
            return 0.0
        
        # Count matching keywords
        matches = 0
        number_match = False
        
        for target_kw in target_keywords:
            for element_kw in element_keywords:
                # Exact match
                if target_kw == element_kw:
                    matches += 1
                    if target_kw.isdigit():
                        number_match = True
                    break
                # Partial match (one contains the other)
                elif target_kw in element_kw or element_kw in target_kw:
                    matches += 0.5
                    break
        
        # Calculate base score
        score = matches / max(len(target_keywords), len(element_keywords))
        
        # Boost score if numbers match (very important for size buttons)
        if number_match:
            score = min(1.0, score + 0.3)
        
        return score

    def find_best_match(self, target: str, clickable_elements: List[Dict[str, Any]], 
                       viewport: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        """
        Find the best matching element from a list of clickable elements.

        Args:
            target: Target element description
            clickable_elements: List of elements with text and coordinates
            viewport: Viewport size for coordinate conversion

        Returns:
            Dictionary with matched element info or None
        """
        if not target or not clickable_elements:
            return None

        viewport = viewport or self.get_viewport_size()
        
        # Extract keywords from target
        target_keywords = self.extract_keywords(str(target))
        
        # Find best matching element
        best_match = None
        best_score = 0.0
        
        for element in clickable_elements:
            label = str(element.get("text", "")).strip()
            raw_coordinates = element.get("coordinates")
            if not label or not raw_coordinates:
                continue

            # Extract keywords from element label
            element_keywords = self.extract_keywords(label)
            
            # Calculate match score
            score = self.calculate_match_score(target_keywords, element_keywords)
            
          
            # Keep track of best match
            if score > best_score and score >= 0.4:  # Threshold for considering a match
                best_score = score
                best_match = {
                    "label": label,
                    "coordinates": raw_coordinates,
                    "element": element,
                    "score": score
                }
        
        # Convert coordinates if we found a match
        if best_match:
            pixel_coords = self.convert_coordinates_to_pixels(
                best_match["coordinates"], 
                viewport
            )
            if pixel_coords:
                best_match["pixel_coordinates"] = pixel_coords
                return best_match
        
        # No good match found
        print(f"  ⚠️ No good coordinate match found for '{target}' (best score: {best_score:.2f})")
        return None

    def convert_coordinates_to_pixels(self, coordinates: Any, 
                                     viewport: Optional[Dict[str, int]] = None) -> Optional[Dict[str, int]]:
        """
        Convert Gemini [y, x] coordinates (0-1000 scale) into viewport pixel positions.

        Args:
            coordinates: [y, x] coordinates in 0-1000 scale
            viewport: Viewport dimensions

        Returns:
            Dictionary with x, y pixel coordinates or None
        """
        if not coordinates or len(coordinates) != 2:
            return None

        try:
            y_raw = float(coordinates[0])
            x_raw = float(coordinates[1])
        except (ValueError, TypeError):
            return None

        viewport = viewport or self.get_viewport_size()
        width = max(1, int(viewport.get("width", 1920)))
        height = max(1, int(viewport.get("height", 1080)))

        def clamp_ratio(value: float) -> float:
            return max(0.0, min(1.0, value))

        x_ratio = clamp_ratio(x_raw / 1000.0)
        y_ratio = clamp_ratio(y_raw / 1000.0)

        x_px = min(width - 1, max(0, int(round(x_ratio * width))))
        y_px = min(height - 1, max(0, int(round(y_ratio * height))))

        return {"x": x_px, "y": y_px}

    def get_viewport_size(self) -> Dict[str, int]:
        """
        Retrieve the current browser viewport size with sensible defaults.

        Returns:
            Dictionary with width and height
        """
        default_viewport = {"width": 1920, "height": 1080}

        try:
            if self.browser and self.browser.page:
                viewport = self.browser.page.viewport_size
                if viewport and viewport.get("width") and viewport.get("height"):
                    return {
                        "width": int(viewport["width"]),
                        "height": int(viewport["height"])
                    }
        except Exception as exc:
            print(f"Viewport lookup failed: {exc}")

        return default_viewport.copy()

