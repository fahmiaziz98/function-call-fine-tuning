import argparse
import json

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from tools import TOOL_SCHEMAS

MAX_TOKENS = 256
CONTEXT_SIZE = 1024


def build_system_prompt(tool_schemas: list[dict]) -> str:
    """Build the system message listing available tools.

    Args:
        tool_schemas: List of tool schema dicts.

    Returns:
        Formatted system message string, matching the training format.
    """
    tools_json = json.dumps(tool_schemas, indent=4)
    return f"You are a helpful assistant with access to the following functions. Use them if required -\n{tools_json}"


def load_model(repo_id: str, filename: str) -> Llama:
    """Download (if needed) and load a GGUF model from the HF Hub.

    Args:
        repo_id: HF Hub repo id containing the GGUF file.
        filename: Exact GGUF filename within that repo.

    Returns:
        A loaded Llama instance ready for chat completion.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Llama(model_path=model_path, n_ctx=CONTEXT_SIZE, verbose=False)


def chat_loop(model: Llama, system_prompt: str) -> None:
    """Run an interactive terminal chat loop.

    Args:
        model: Loaded Llama instance.
        system_prompt: System message to prepend to every conversation.
    """
    print("Chat started. Type 'exit' to quit.\n")
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        response = model.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        assistant_text = response["choices"][0]["message"]["content"].strip()

        print(f"Assistant: {assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the GGUF tool-calling model.")
    parser.add_argument("--repo_id", type=str, required=True, help="HF Hub repo id, e.g. your-username/qwen2.5-1.5b-tool-calling-gguf")
    parser.add_argument("--filename", type=str, required=True, help="GGUF filename in that repo, e.g. model.q4_k_m.gguf")
    args = parser.parse_args()

    model = load_model(args.repo_id, args.filename)
    system_prompt = build_system_prompt(TOOL_SCHEMAS)
    chat_loop(model, system_prompt)
