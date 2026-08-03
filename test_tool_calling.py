import argparse

from src.inference import Reply, ToolCall, ToolCallRouter
from src.tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS

TEST_CASES = [
    # Easy: single required argument, unambiguous.
    "What time is it in Tokyo right now?",
    # Easy-medium: multiple required arguments, all clearly stated.
    "Convert 100 USD to EUR for me.",
    # Medium: required + optional argument, only some mentioned.
    "Find me some Italian restaurants in Bandung.",
    # Medium: required-only, but no optional args mentioned at all.
    "Search for restaurants in Jakarta.",
    # Hard: multiple required arguments, one of them a list.
    "Schedule a meeting called 'Sprint Planning' on 2026-08-10 at 09:00 for 45 minutes with Alice and Bob.",
    # No-tool: should NOT trigger any function call.
    "What's the difference between a list and a tuple in Python?",
    # No-tool: conversational, no tool applies.
    "Thanks, that's all I needed for now!",
]


def run_tests(checkpoint: str) -> None:
    """Run each test case through the router and print + execute the result.

    Args:
        checkpoint: Local path or HF Hub repo id of the fine-tuned model.
    """
    router = ToolCallRouter(checkpoint)

    for i, user_text in enumerate(TEST_CASES, start=1):
        result = router.route(user_text, TOOL_SCHEMAS)

        print(f"\n{'=' * 70}")
        print(f"Test {i}: {user_text}")
        print(f"{'=' * 70}")

        if isinstance(result, ToolCall):
            print(f"-> Tool call: {result.name}({result.arguments})")
            implementation = TOOL_IMPLEMENTATIONS.get(result.name)
            if implementation is None:
                print(f"   [WARNING] Model called unknown tool '{result.name}'")
                continue
            try:
                mock_result = implementation(**result.arguments)
                print(f"   Mock execution result: {mock_result}")
            except TypeError as e:
                print(f"   [WARNING] Argument mismatch: {e}")
        elif isinstance(result, Reply):
            print(f"-> Text reply: {result.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tool-calling routing on real examples.")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    run_tests(args.checkpoint)
