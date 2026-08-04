import json

import streamlit as st

from src.llm.gguf_chat import (
    ChatConfig,
    build_system_prompt,
    load_model,
    parse_tool_call,
    stream_chat,
    trim_history,
)
from src.tools import TOOL_SCHEMAS, execute_tool


st.set_page_config(
    page_title="Tool Calling Demo",
    page_icon="🤖",
    layout="centered",
)

CONFIG = ChatConfig()


@st.cache_resource
def get_model(repo_id: str, filename: str):
    """Load GGUF model only once."""
    return load_model(
        repo_id=repo_id,
        filename=filename,
        config=CONFIG,
    )


with st.sidebar:

    st.title("⚙️ Settings")

    repo_id = st.text_input(
        "HF Repository",
        value="fahmiaziz/qwen2.5-1.5b-tool-calling-gguf",
    )

    filename = st.text_input(
        "GGUF Filename",
        value="model-q4_k_m.gguf",
    )

    st.divider()

    st.caption(f"Context : {CONFIG.context_size}")
    st.caption(f"History : {CONFIG.max_history_pairs} pairs")
    st.caption(f"Temperature : {CONFIG.temperature}")

    st.divider()

    if st.button(
        "🗑 Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    TOOL_SCHEMAS,
                ),
            }
        ]

        st.rerun()


st.title("🤖 Function Calling Demo")
st.caption(
    "Powered by GGUF + llama.cpp"
)

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                TOOL_SCHEMAS,
            ),
        }
    ]


model = get_model(
    repo_id,
    filename,
)


for message in st.session_state.messages:
    if message["role"] == "system":
        continue

    if message["role"] == "tool":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask anything...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.session_state.messages = trim_history(
        st.session_state.messages,
        CONFIG.max_history_pairs,
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):

        placeholder = st.empty()

        first_response = ""

        for token in stream_chat(
            model,
            st.session_state.messages,
            CONFIG,
        ):

            first_response += token

            placeholder.markdown(first_response + "▌")

        placeholder.markdown(first_response)

    tool_call = parse_tool_call(first_response)

    if tool_call is None:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": first_response,
            }
        )

        st.stop()

    with st.chat_message("assistant"):

        st.success("🟢 Tool Call")

        st.code(
            json.dumps(
                {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
                indent=2,
            ),
            language="json",
        )

    tool_result = execute_tool(
        tool_call.name,
        tool_call.arguments,
    )

    with st.chat_message("assistant"):

        st.info("📦 Tool Result")

        st.json(tool_result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": first_response,
        }
    )

    st.session_state.messages.append(
        {
            "role": "tool",
            "content": json.dumps(tool_result),
        }
    )

    st.session_state.messages = trim_history(
        st.session_state.messages,
        CONFIG.max_history_pairs,
    )


    with st.chat_message("assistant"):

        placeholder = st.empty()

        final_response = ""

        for token in stream_chat(
            model,
            st.session_state.messages,
            CONFIG,
        ):

            final_response += token

            placeholder.markdown(final_response + "▌")

        placeholder.markdown(final_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_response,
        }
    )

    st.session_state.messages = trim_history(
        st.session_state.messages,
        CONFIG.max_history_pairs,
    )
