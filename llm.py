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
    """
    Build the system prompt.

    Args:
        tool_schemas (list[dict[str, Any]]): A list of tool schemas.

    Returns:
        str: The built system prompt.
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


def load_model(
    repo_id: str,
    filename: str,
    config: ChatConfig,
) -> Llama:
    """
    Download and load a GGUF model.

    Args:
        repo_id (str): The repository ID of the model on the Hugging Face Hub.
        filename (str): The filename of the model to download.
        config (ChatConfig): The configuration for the chat model.

    Returns:
        Llama: The loaded chat model.
    """

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    return Llama(
        model_path=model_path,
        n_ctx=config.context_size,
        n_batch=config.n_batch,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=-1,
        seed=config.seed,
        verbose=False,
    )


def trim_history(
    messages: list[dict[str, str]],
    max_pairs: int,
) -> list[dict[str, str]]:
    """
    Keep only the latest conversation pairs.

    Args:
        messages (list[dict[str, str]]): The list of messages to trim.
        max_pairs (int): The maximum number of conversation pairs to keep.

    Returns:
        list[dict[str, str]]: The trimmed list of messages.
    """

    if not messages:
        return messages

    system = messages[0]
    history = messages[1:]

    max_messages = max_pairs * 2

    if len(history) > max_messages:
        history = history[-max_messages:]

    return [system, *history]


def stream_chat(
    model: Llama,
    messages: list[dict[str, str]],
    config: ChatConfig,
):
    """
    Yield response tokens.

    Args:
        model (Llama): The chat model to use.
        messages (list[dict[str, str]]): The list of messages to send to the model.
        config (ChatConfig): The configuration for the chat model.

    Yields:
        str: The response tokens.
    """

    stream = model.create_chat_completion(
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )

    for chunk in stream:

        delta = chunk["choices"][0]["delta"]

        token = delta.get("content")

        if token:
            yield token


def generate_response(
    model: Llama,
    messages: list[dict[str, str]],
    config: ChatConfig,
) -> str:
    """
    Generate a complete response without streaming.

    Args:
        model (Llama): The chat model to use.
        messages (list[dict[str, str]]): The list of messages to send to the model.
        config (ChatConfig): The configuration for the chat model.

    Returns:
        str: The generated response.
    """

    response = model.create_chat_completion(
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=False,
    )

    message = response["choices"][0]["message"]

    return (message.get("content") or "").strip()


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse a JSON tool call returned by the model.

    Expected format:

    {
        "name": "...",
        "arguments": {...}
    }
    """

    text = text.strip()

    if not text:
        return None

    try:
        payload = json.loads(text)

    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    if "name" not in payload:
        return None

    if "arguments" not in payload:
        return None

    if not isinstance(payload["arguments"], dict):
        return None

    return ToolCall(
        name=payload["name"],
        arguments=payload["arguments"],
    )
