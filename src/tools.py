from __future__ import annotations

from typing import Any, Callable

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name."}},
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression, e.g. (10+5)*2",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_wikipedia",
        "description": "Search information about a topic.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The topic to search."}},
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject."},
                "body": {"type": "string", "description": "Email body."},
            },
            "required": ["recipient", "subject", "body"],
        },
    },
]


def get_weather(city: str) -> dict[str, Any]:
    """Return a deterministic mock weather response.

    Args:
        city: The city name.

    Returns:
        Mock weather data for the given city.
    """
    return {"city": city, "temperature": 31, "condition": "Sunny", "humidity": 78, "unit": "C"}


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression, e.g. "(10+5)*2".

    Returns:
        The evaluated result, or an error message if evaluation fails.

    Note:
        Uses eval() for demonstration purposes only — never expose this
        directly to untrusted input in production.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


def search_wikipedia(query: str) -> dict[str, Any]:
    """Return a deterministic mock encyclopedia result.

    Args:
        query: The topic to search.

    Returns:
        Mock article data for the given query.
    """
    return {
        "title": query,
        "summary": f"{query} is a mock Wikipedia article returned for demonstration purposes.",
        "url": f"https://example.com/wiki/{query.replace(' ', '_')}",
    }


def send_email(recipient: str, subject: str, body: str) -> dict[str, Any]:
    """Return a mock email delivery result.

    Args:
        recipient: Recipient email address.
        subject: Email subject.
        body: Email body.

    Returns:
        Mock delivery confirmation with a truncated body preview.
    """
    return {
        "status": "sent",
        "recipient": recipient,
        "subject": subject,
        "message_id": "mock-email-001",
        "preview": body[:80],
    }


TOOL_IMPLEMENTATIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_wikipedia": search_wikipedia,
    "send_email": send_email,
}


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool by name.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments to call it with.

    Returns:
        The tool's result dict.

    Raises:
        ValueError: If `tool_name` is not registered.
    """
    tool = TOOL_IMPLEMENTATIONS.get(tool_name)
    if tool is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return tool(**arguments)
