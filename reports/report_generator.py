"""
Report generator for creating comprehensive test reports.
Generates timestamped JSON files with all test results.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


class ReportGenerator:
    """Generates and manages test reports."""

    def __init__(self):
        """Initialize report generator."""
        self.output_dir = "output"
        self.report_data = {
            "test_metadata": {},
            "personas": []
        }
        self.current_report_path: Optional[str] = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_report(self) -> str:
        """
        Initialize a new report with timestamp.

        Returns:
            Path to the report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_report_path = os.path.join(self.output_dir, f"test_{timestamp}.json")

        self.report_data = {
            "test_metadata": {
                "start_time": datetime.now().isoformat(),
                "timestamp": timestamp,
                "target_website": "https://www.coterie.com/",
                "task": "Add Size 1, cheapest diaper bundle to cart",
                "personas_tested": [],
                "status": "in_progress"
            },
            "personas": []
        }

        # Save initial report
        self._save_report()
        print(f"📝 Report initialized: {self.current_report_path}")
        return self.current_report_path

    def add_persona_result(self, persona_result: Dict[str, Any]) -> None:
        """
        Add a persona's test results to the report.

        Args:
            persona_result: Complete test results for one persona
        """
        if not self.current_report_path:
            self.initialize_report()

        # Extract key information for summary
        persona_name = persona_result.get("persona", {}).get("name", "Unknown")
        task_completed = persona_result.get("test_results", {}).get("task_completed", False)
        abandoned = persona_result.get("test_results", {}).get("abandoned", False)

        # Create persona report entry
        persona_entry = {
            "persona_name": persona_name,
            "test_timestamp": datetime.now().isoformat(),
            "entry_point": persona_result.get("persona", {}).get("context", {}).get("entry_point"),
            "device": persona_result.get("persona", {}).get("context", {}).get("device"),
            "task_completed": task_completed,
            "abandoned": abandoned,
            "journey": self._extract_journey(persona_result),
            "friction_points": self._extract_friction_points(persona_result),
            "metrics": self._extract_metrics(persona_result),
            "evaluation": persona_result.get("test_results", {}).get("evaluation", {}),
            "full_log": persona_result.get("action_log", {})
        }

        # Add to report
        self.report_data["personas"].append(persona_entry)
        self.report_data["test_metadata"]["personas_tested"].append(persona_name)

        # Save updated report
        self._save_report()
        print(f"✅ Added results for {persona_name} to report")

    def _extract_journey(self, persona_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract simplified journey from action log.

        Args:
            persona_result: Complete persona test results

        Returns:
            Simplified journey list
        """
        actions = persona_result.get("action_log", {}).get("actions", [])
        journey = []

        for action in actions:
            # Only include significant actions
            if action.get("action") in ["navigate", "click", "type", "decision"]:
                journey_step = {
                    "step": len(journey) + 1,
                    "action": action.get("action"),
                    "target": action.get("target") or action.get("details", {}).get("target_element"),
                    "reasoning": action.get("reasoning") or action.get("details", {}).get("reasoning", ""),
                    "timestamp": action.get("timestamp")
                }
                journey.append(journey_step)

        return journey

    def _extract_friction_points(self, persona_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and categorize friction points.

        Args:
            persona_result: Complete persona test results

        Returns:
            Categorized friction points
        """
        friction_points = persona_result.get("action_log", {}).get("friction_points", [])

        # Categorize by type
        categorized = {}
        for friction in friction_points:
            friction_type = friction.get("type", "other")
            if friction_type not in categorized:
                categorized[friction_type] = []
            categorized[friction_type].append({
                "description": friction.get("description"),
                "severity": friction.get("severity"),
                "timestamp": friction.get("timestamp")
            })

        return categorized

    def _extract_metrics(self, persona_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from test results.

        Args:
            persona_result: Complete persona test results

        Returns:
            Metrics dictionary
        """
        test_results = persona_result.get("test_results", {})
        summary = persona_result.get("action_log", {}).get("summary", {})

        return {
            "total_actions": test_results.get("total_actions", 0),
            "total_time_seconds": test_results.get("total_time", 0),
            "friction_count": len(test_results.get("friction_points", [])),
            "action_breakdown": summary.get("action_breakdown", {}),
            "screenshots_taken": summary.get("screenshots_taken", 0),
            "conversion_likelihood": persona_result.get("persona", {}).get("metrics", {}).get("conversion_likelihood", 0),
            "expected_satisfaction": persona_result.get("persona", {}).get("metrics", {}).get("satisfaction_score", 0),
            "actual_satisfaction": test_results.get("evaluation", {}).get("persona_satisfaction", 0)
        }

    def finalize_report(self) -> Dict[str, Any]:
        """
        Finalize the report with summary statistics.

        Returns:
            Complete report data
        """
        if not self.current_report_path:
            return {}

        # Calculate summary statistics
        total_personas = len(self.report_data["personas"])
        completed_count = sum(1 for p in self.report_data["personas"] if p["task_completed"])
        abandoned_count = sum(1 for p in self.report_data["personas"] if p["abandoned"])

        # Aggregate friction points
        all_friction = []
        for persona in self.report_data["personas"]:
            for category, points in persona.get("friction_points", {}).items():
                for point in points:
                    all_friction.append({
                        "persona": persona["persona_name"],
                        "category": category,
                        "description": point["description"],
                        "severity": point["severity"]
                    })

        # Update metadata
        self.report_data["test_metadata"].update({
            "end_time": datetime.now().isoformat(),
            "status": "completed",
            "summary": {
                "total_personas_tested": total_personas,
                "tasks_completed": completed_count,
                "tasks_abandoned": abandoned_count,
                "completion_rate": completed_count / total_personas if total_personas > 0 else 0,
                "total_friction_points": len(all_friction),
                "high_severity_friction": len([f for f in all_friction if f["severity"] == "high"])
            }
        })

        # Add aggregated insights
        self.report_data["insights"] = self._generate_insights()

        # Save final report
        self._save_report()
        print(f"\n📊 Report finalized: {self.current_report_path}")
        self._print_summary()

        return self.report_data

    def _generate_insights(self) -> Dict[str, Any]:
        """
        Generate insights from test results.

        Returns:
            Insights dictionary
        """
        personas = self.report_data["personas"]

        # Find common friction points
        friction_descriptions = []
        for persona in personas:
            for category, points in persona.get("friction_points", {}).items():
                for point in points:
                    friction_descriptions.append(point["description"])

        # Count occurrences
        friction_counts = {}
        for desc in friction_descriptions:
            friction_counts[desc] = friction_counts.get(desc, 0) + 1

        # Sort by frequency
        common_friction = sorted(friction_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Persona-specific insights
        persona_insights = []
        for persona in personas:
            persona_insights.append({
                "name": persona["persona_name"],
                "completed": persona["task_completed"],
                "key_friction": list(persona.get("friction_points", {}).keys()),
                "journey_length": len(persona.get("journey", []))
            })

        return {
            "most_common_friction_points": common_friction,
            "persona_performance": persona_insights,
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """
        Generate UX recommendations based on test results.

        Returns:
            List of recommendations
        """
        recommendations = []
        personas = self.report_data["personas"]

        # Check completion rates
        completion_rate = self.report_data["test_metadata"]["summary"]["completion_rate"]
        if completion_rate < 0.7:
            recommendations.append("Low completion rate suggests significant UX barriers")

        # Check for mobile issues
        mobile_personas = [p for p in personas if p.get("device") == "mobile"]
        mobile_issues = sum(len(p.get("friction_points", {})) for p in mobile_personas)
        if mobile_issues > len(mobile_personas) * 3:
            recommendations.append("Mobile experience needs optimization - high friction on mobile devices")

        # Check for specific friction patterns
        all_friction = []
        for persona in personas:
            for points in persona.get("friction_points", {}).values():
                all_friction.extend([p["description"] for p in points])

        if any("price" in f.lower() or "cost" in f.lower() for f in all_friction):
            recommendations.append("Pricing transparency should be improved")

        if any("navigation" in f.lower() for f in all_friction):
            recommendations.append("Navigation flow needs simplification")

        if any("trust" in f.lower() or "certification" in f.lower() for f in all_friction):
            recommendations.append("Add more trust signals and certifications visibility")

        return recommendations if recommendations else ["No critical issues identified"]

    def _save_report(self) -> None:
        """Save current report to file."""
        if self.current_report_path:
            with open(self.current_report_path, 'w') as f:
                json.dump(self.report_data, f, indent=2)

    def _print_summary(self) -> None:
        """Print report summary to console."""
        summary = self.report_data["test_metadata"]["summary"]
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Personas Tested: {summary['total_personas_tested']}")
        print(f"Tasks Completed: {summary['tasks_completed']}")
        print(f"Tasks Abandoned: {summary['tasks_abandoned']}")
        print(f"Completion Rate: {summary['completion_rate']:.1%}")
        print(f"Total Friction Points: {summary['total_friction_points']}")
        print(f"High Severity Issues: {summary['high_severity_friction']}")
        print("="*60)