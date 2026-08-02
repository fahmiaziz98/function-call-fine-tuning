from dataclasses import dataclass
from enum import Enum


class ResponseType(str, Enum):
    """Whether an example's assistant response is a tool call or plain text.

    Attributes:
        TOOL_CALL: The assistant response is a structured function call.
        NO_TOOL: The assistant response is a plain-text reply, used to
            teach the model when NOT to call a tool.
    """

    TOOL_CALL = "tool_call"
    NO_TOOL = "no_tool"


@dataclass
class ToolCallExample:
    """A single unified training example.

    Attributes:
        response_type: Whether this example expects a tool call or text.
        system_text: System message listing available tool schemas.
        user_text: The user's request.
        assistant_text: Expected assistant output — either a JSON tool
            call string or a plain-text reply.
    """

    response_type: ResponseType
    system_text: str
    user_text: str
    assistant_text: str

    def to_dict(self) -> dict:
        """Convert the example into a JSON-serializable dictionary."""
        return {
            "response_type": self.response_type.value,
            "system_text": self.system_text,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
        }

    def full_text(self) -> str:
        """Concatenate all fields into one string, used for deduplication.

        Returns:
            A single string representing the full example content.
        """
        return f"{self.system_text}\n{self.user_text}\n{self.assistant_text}"
