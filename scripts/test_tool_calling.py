import argparse

from src.inference import Reply, ToolCall, ToolCallRouter
from src.tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS, execute_tool

TEST_CASES = [
    "What's the weather like in Jakarta?",
    "What is (25 + 15) * 3?",
    "Tell me about the Eiffel Tower.",
    "Send an email to bob@example.com with subject 'Hello' and body 'How are you?'",
    "What's the difference between a list and a tuple in Python?",
    "Thanks, that's all I needed for now!",
]


def run_tests(checkpoint: str) -> None:
    """Run each test case through the trained model and print + execute the result.

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
            if result.name not in TOOL_IMPLEMENTATIONS:
                print(f"   [WARNING] Model called unknown tool '{result.name}'")
                continue
            try:
                print(f"   Execution result: {execute_tool(result.name, result.arguments)}")
            except TypeError as e:
                print(f"   [WARNING] Argument mismatch: {e}")
        elif isinstance(result, Reply):
            print(f"-> Text reply: {result.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check the freshly trained model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    run_tests(args.checkpoint)
