#!/usr/bin/env python
"""
Main entry point for AI Persona Usability Testing System.
Orchestrates testing of Coterie website with 5 distinct personas.
Async implementation to support async BrowserController and TaskExecutor.
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from core.task_executor import TaskExecutor
from reports.report_generator import ReportGenerator
from personas.persona_definitions import PERSONAS


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("       AI PERSONA USABILITY TESTING SYSTEM")
    print("       Target: Coterie.com - Size 1 Diaper Bundle")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


async def run_persona_test(persona_key: str, headless: bool = False,
                     persistent_context: bool = True, human_delays: bool = True,
                     auto_dismiss_popups: bool = True) -> dict:
    """
    Run test for a single persona (async).

    Args:
        persona_key: Key identifying the persona
        headless: Whether to run browser in headless mode
        persistent_context: Whether to use persistent browser context
        human_delays: Whether to add random delays to simulate human behavior
        auto_dismiss_popups: Whether to automatically dismiss popups

    Returns:
        Test results dictionary
    """
    # Use persona_key as context name for per-persona contexts
    executor = TaskExecutor(
        headless=headless,
        persistent_context=persistent_context,
        context_name=persona_key if persistent_context else "default",
        human_delays=human_delays,
        auto_dismiss_popups=auto_dismiss_popups
    )
    return await executor.execute_persona_test(persona_key)


async def main():
    """Main execution function (async)."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI Persona Usability Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all personas with visible browser
  python main.py

  # Test single persona in headless mode
  python main.py --single sarah_kim --headless

  # Use persistent context to reduce popups (like a returning visitor)
  python main.py --persistent-context

  # Speed test (no human delays)
  python main.py --no-human-delays --headless
        """
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        choices=list(PERSONAS.keys()) + ["all"],
        default=["all"],
        help="Personas to test (default: all)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible browser)"
    )
    parser.add_argument(
        "--single",
        type=str,
        choices=list(PERSONAS.keys()),
        help="Test only a single specific persona"
    )
    parser.add_argument(
        "--persistent-context",
        action="store_true",
        help="Use persistent browser context (saves cookies/storage, reduces popups)"
    )
    parser.add_argument(
        "--no-human-delays",
        action="store_true",
        help="Disable random delays that simulate human behavior (faster but less realistic)"
    )
    parser.add_argument(
        "--no-auto-dismiss-popups",
        action="store_true",
        help="Disable automatic popup dismissal (personas will manually close popups)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Verify API keys are present
    required_env_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {missing_vars}")
        print("Please ensure .env file contains all required API keys.")
        sys.exit(1)

    # Print banner
    print_banner()

    # Determine which personas to test
    if args.single:
        personas_to_test = [args.single]
    elif "all" in args.personas:
        personas_to_test = list(PERSONAS.keys())
    else:
        personas_to_test = args.personas

    print(f"📋 Personas to test: {', '.join(personas_to_test)}")
    print(f"🖥️  Browser mode: {'Headless' if args.headless else 'Visible'}")
    print(f"💾 Persistent context: {'Enabled (cookies saved)' if not args.persistent_context else 'Disabled (fresh browser)'}")
    print(f"🤖 Human delays: {'Disabled (fast)' if args.no_human_delays else 'Enabled (realistic)'}")
    print(f"🔘 Auto-dismiss popups: {'Disabled (manual)' if args.no_auto_dismiss_popups else 'Enabled (automatic)'}")
    print("\n" + "="*70 + "\n")

    # Initialize report generator
    report_generator = ReportGenerator()
    report_path = report_generator.initialize_report()

    # Test each persona
    successful_tests = 0
    failed_tests = 0

    for i, persona_key in enumerate(personas_to_test, 1):
        print(f"\n{'='*70}")
        print(f"TESTING PERSONA {i}/{len(personas_to_test)}: {PERSONAS[persona_key].name}")
        print(f"{'='*70}\n")

        try:
            # Run test for this persona
            result = await run_persona_test(
                persona_key,
                headless=args.headless,
                persistent_context=args.persistent_context,
                human_delays=not args.no_human_delays,
                auto_dismiss_popups=not args.no_auto_dismiss_popups
            )

            # Add to report
            report_generator.add_persona_result(result)

            # Track success
            if "error" not in result:
                successful_tests += 1
                task_completed = result.get("test_results", {}).get("task_completed", False)
                status = "✅ COMPLETED" if task_completed else "⚠️ ABANDONED"
                print(f"\n{status} - {PERSONAS[persona_key].name}")
            else:
                failed_tests += 1
                print(f"\n❌ FAILED - {PERSONAS[persona_key].name}: {result.get('error')}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Testing interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error testing {persona_key}: {str(e)}")
            failed_tests += 1
            continue

        # Brief pause between personas
        if i < len(personas_to_test):
            print("\n⏸️  Pausing before next persona...\n")
            await asyncio.sleep(5)

    # Finalize report
    print("\n" + "="*70)
    print("FINALIZING REPORT")
    print("="*70)

    final_report = report_generator.finalize_report()

    # Print final summary
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print(f"✅ Successful tests: {successful_tests}")
    print(f"❌ Failed tests: {failed_tests}")
    print(f"📊 Report saved to: {report_path}")
    print(f"⏱️  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # Print recommendations if available
    if "insights" in final_report:
        recommendations = final_report["insights"].get("recommendations", [])
        if recommendations:
            print("\n📝 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
            print()

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)