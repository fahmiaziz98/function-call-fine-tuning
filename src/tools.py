TOOL_SCHEMAS = [
    # 1. Easy: single required string argument.
    {
        "name": "get_current_time",
        "description": "Get the current time in a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city to get the time for"}
            },
            "required": ["city"],
        },
    },
    # 2. Easy-medium: two required arguments, both simple types.
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "The amount to convert"},
                "from_currency": {"type": "string", "description": "The source currency code, e.g. USD"},
                "to_currency": {"type": "string", "description": "The target currency code, e.g. EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    # 3. Medium: required + optional argument, needs the model to omit
    # the optional field when not mentioned by the user.
    {
        "name": "search_restaurants",
        "description": "Search for restaurants in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city or area to search in"},
                "cuisine": {"type": "string", "description": "Optional cuisine type, e.g. Italian, Japanese"},
                "max_price": {"type": "integer", "description": "Optional maximum price level, 1-4"},
            },
            "required": ["location"],
        },
    },
    # 4. Hard: multiple required arguments including an array/enum-like
    # field, requiring the model to correctly structure a list argument.
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting with one or more attendees",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The meeting title"},
                "date": {"type": "string", "description": "Meeting date, format YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "Start time, format HH:MM"},
                "duration_minutes": {"type": "integer", "description": "Meeting duration in minutes"},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee names or emails",
                },
            },
            "required": ["title", "date", "start_time", "duration_minutes", "attendees"],
        },
    },
]


def mock_get_current_time(city: str) -> dict:
    """Return a deterministic mock time for a city."""
    return {"city": city, "time": "14:32", "timezone": "mock/UTC+0"}


def mock_convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Return a deterministic mock currency conversion."""
    return {
        "converted_amount": round(amount * 0.92, 2),
        "from_currency": from_currency,
        "to_currency": to_currency,
    }


def mock_search_restaurants(location: str, cuisine: str = None, max_price: int = None) -> dict:
    """Return a deterministic mock restaurant search result."""
    return {
        "location": location,
        "cuisine": cuisine,
        "max_price": max_price,
        "results": ["Mock Bistro", "Mock Kitchen", "Mock Diner"],
    }


def mock_schedule_meeting(
    title: str, date: str, start_time: str, duration_minutes: int, attendees: list[str]
) -> dict:
    """Return a deterministic mock meeting confirmation."""
    return {
        "status": "scheduled",
        "title": title,
        "date": date,
        "start_time": start_time,
        "duration_minutes": duration_minutes,
        "attendees": attendees,
    }


TOOL_IMPLEMENTATIONS = {
    "get_current_time": mock_get_current_time,
    "convert_currency": mock_convert_currency,
    "search_restaurants": mock_search_restaurants,
    "schedule_meeting": mock_schedule_meeting,
}
