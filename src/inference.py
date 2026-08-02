import json
from dataclasses import dataclass

import torch
from unsloth import FastLanguageModel

MAX_NEW_TOKENS = 256
MAX_SEQ_LENGTH = 1024


@dataclass
class ToolCall:
    """A parsed tool call decision.

    Attributes:
        name: The name of the tool to call.
        arguments: Dict of arguments to call it with.
    """

    name: str
    arguments: dict


@dataclass
class Reply:
    """A plain-text response, used when no tool call was made.

    Attributes:
        text: The model's plain-text reply.
    """

    text: str


class ToolCallRouter:
    """Routes a user request to either a tool call or a plain-text reply."""

    def __init__(self, checkpoint: str, device: str | None = None):
        """Load the fine-tuned model and tokenizer.

        Args:
            checkpoint: Local path or HF Hub repo id of the fine-tuned adapter.
            device: Torch device. Defaults to "cuda" if available.

        Raises:
            OSError: If the checkpoint cannot be loaded.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True
        )
        FastLanguageModel.for_inference(self.model)

    def _build_system_text(self, tool_schemas: list[dict]) -> str:
        """Build the system message listing available tools.

        Args:
            tool_schemas: List of tool schema dicts (name/description/parameters).

        Returns:
            Formatted system message string.
        """
        tools_json = json.dumps(tool_schemas, indent=4)
        return f"You are a helpful assistant with access to the following functions. Use them if required -\n{tools_json}"

    def _generate(self, system_text: str, user_text: str) -> str:
        """Run generation for one system+user turn.

        Args:
            system_text: System message including tool schemas.
            user_text: The user's request.

        Returns:
            Decoded assistant response text (generated tokens only).
        """
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def route(self, user_text: str, tool_schemas: list[dict]) -> ToolCall | Reply:
        """Decide whether to call a tool or reply in plain text.

        Args:
            user_text: The user's request.
            tool_schemas: List of available tool schema dicts.

        Returns:
            A ToolCall if the model emitted a valid, parseable tool call;
            otherwise a Reply containing the raw text (this includes cases
            where the model attempted a tool call but produced invalid
            JSON — the raw text is preserved rather than silently dropped).
        """
        system_text = self._build_system_text(tool_schemas)
        raw_output = self._generate(system_text, user_text)

        if raw_output.startswith("{") and '"name"' in raw_output:
            try:
                parsed = json.loads(raw_output)
                return ToolCall(name=parsed["name"], arguments=parsed.get("arguments", {}))
            except (json.JSONDecodeError, KeyError):
                pass  # Fall through: treat as plain text if parsing fails.

        return Reply(text=raw_output)
