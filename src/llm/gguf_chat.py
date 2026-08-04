from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


@dataclass(frozen=True)
class ChatConfig:
    """Configuration for GGUF inference."""

    context_size: int = 4096
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int = 42
    n_batch: int = 512
    max_history_pairs: int = 5


@dataclass
class ToolCall:
    """Represents a parsed tool call."""

    name: str
    arguments: dict[str, Any]


def build_system_prompt(tool_schemas: list[dict[str, Any]]) -> str:
    """Build the system prompt listing available tools.

    Args:
        tool_schemas: List of tool schema dicts.

    Returns:
        The formatted system prompt string.
    """
    return (
        "You are a helpful AI assistant.\n"
        "You have access to the following functions.\n"
        "If a function is required, respond ONLY with a valid JSON object "
        "using the following format:\n\n"
        "{\n"
        '  "name": "<tool_name>",\n'
        '  "arguments": {\n'
        '    "arg": "value"\n'
        "  }\n"
        "}\n\n"
        "Do not include markdown.\n"
        "Do not explain the tool call.\n"
        "If no tool is needed, answer normally.\n\n"
        f"{json.dumps(tool_schemas, indent=2)}"
    )


def load_model(repo_id: str, filename: str, config: ChatConfig) -> Llama:
    """Download (if needed) and load a GGUF model.

    Args:
        repo_id: HF Hub repo id containing the GGUF file.
        filename: Exact GGUF filename within that repo.
        config: Chat configuration (context size, batch size, seed).

    Returns:
        A loaded Llama instance.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Llama(
        model_path=model_path,
        n_ctx=config.context_size,
        n_batch=config.n_batch,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=0,  # deployment target (e.g. Streamlit Cloud) is CPU-only
        seed=config.seed,
        verbose=False,
    )


def trim_history(messages: list[dict[str, str]], max_pairs: int) -> list[dict[str, str]]:
    """Keep only the latest conversation pairs plus the system message.

    Args:
        messages: Full message history, with the system message first.
        max_pairs: Maximum number of user/assistant pairs to retain.

    Returns:
        The trimmed message list.
    """
    if not messages:
        return messages

    system, history = messages[0], messages[1:]
    max_messages = max_pairs * 2

    if len(history) > max_messages:
        history = history[-max_messages:]

    return [system, *history]


def stream_chat(model: Llama, messages: list[dict[str, str]], config: ChatConfig):
    """Yield response tokens as they're generated.

    Args:
        model: Loaded Llama instance.
        messages: Conversation history to generate a reply for.
        config: Chat configuration.

    Yields:
        Generated text tokens.
    """
    stream = model.create_chat_completion(
        messages=messages, temperature=config.temperature, max_tokens=config.max_tokens, stream=True
    )
    for chunk in stream:
        token = chunk["choices"][0]["delta"].get("content")
        if token:
            yield token


def generate_response(model: Llama, messages: list[dict[str, str]], config: ChatConfig) -> str:
    """Generate a complete (non-streamed) response.

    Args:
        model: Loaded Llama instance.
        messages: Conversation history to generate a reply for.
        config: Chat configuration.

    Returns:
        The generated response text.
    """
    response = model.create_chat_completion(
        messages=messages, temperature=config.temperature, max_tokens=config.max_tokens, stream=False
    )
    return (response["choices"][0]["message"].get("content") or "").strip()


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse a JSON tool call from generated text.

    Args:
        text: Raw generated text, expected to contain
            {"name": ..., "arguments": {...}} if a tool call was made.

    Returns:
        A ToolCall if parsing succeeds, otherwise None (treat as plain text).
    """
    text = text.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or "name" not in payload or "arguments" not in payload:
        return None
    if not isinstance(payload["arguments"], dict):
        return None

    return ToolCall(name=payload["name"], arguments=payload["arguments"])
