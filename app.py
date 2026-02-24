import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import pydantic
import streamlit as st
from anyio import create_memory_object_stream, create_task_group
from dotenv import load_dotenv
from google import genai
from httpx_sse import SSEError
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.message import SessionMessage

# Page settings
st.set_page_config(page_title="MCP Chat", page_icon="🤖", layout="wide")

# Load environment variables and initialize client
GEMINI_MODEL = "gemini-flash-lite-latest"


load_dotenv()


# Chat management functions
def create_new_chat():
    """Create a new chat session."""
    st.session_state.chat = {
        "messages": [],
        "server": st.session_state.get("selected_server", "None"),
    }
    if "selected_server" in st.session_state:
        st.session_state.chat["messages"].append(
            {
                "role": "assistant",
                "content": f"Connected to server: {st.session_state.get('selected_server', 'None')}",
            }
        )


# Load config file
@st.cache_data
def load_mcp_config() -> Dict[str, Any]:
    """Load MCP configuration file."""
    config_path = "mcp.json"
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "mcpServers" not in config:
            raise ValueError("Config file does not contain 'mcpServers' key.")

        return config
    except FileNotFoundError:
        st.error(f"Config file not found: {config_path}")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Configuration file error: {e}")
        st.stop()


def validate_server_config(config: Dict[str, Any]) -> bool:
    """Validate server configuration."""
    if "url" in config:
        return True

    required_fields = ["command", "args"]
    for field in required_fields:
        if field not in config:
            st.error(f"Missing required field in server configuration: {field}")
            return False
    return True


def create_server_parameters(server_config: Dict[str, Any]) -> Any:
    """Create server parameters."""
    if not validate_server_config(server_config):
        raise ValueError("Invalid server configuration")

    if "url" in server_config:
        return server_config

    env = os.environ.copy()
    if "env" in server_config:
        env.update(server_config["env"])

    return StdioServerParameters(
        command=server_config["command"],
        args=server_config["args"],
        env=env,
    )


async def safe_inspect_server(session: ClientSession):
    """Safely inspect the server and store in session state."""
    inspection: Dict[str, Any] = {
        "prompts": None,
        "prompts_error": None,
        "resources": None,
        "resources_error": None,
        "tools": None,
        "tools_error": None,
    }

    # Prompts inspection
    try:
        prompts = await asyncio.wait_for(session.list_prompts(), timeout=5.0)
        if prompts and prompts.prompts:
            inspection["prompts"] = [
                f"- **{p.name}**: {p.description or 'No description'}"
                for p in prompts.prompts
            ]
    except asyncio.TimeoutError:
        inspection["prompts_error"] = "Prompt list request timed out"
    except Exception as e:
        inspection["prompts_error"] = f"Unable to fetch prompt list: {e}"

    # Resources inspection
    try:
        resources = await asyncio.wait_for(session.list_resources(), timeout=5.0)
        if resources and resources.resources:
            inspection["resources"] = [
                f"- `{r.uri}`: {r.name or 'No name'}" for r in resources.resources
            ]
    except asyncio.TimeoutError:
        inspection["resources_error"] = "Resource list request timed out"
    except Exception as e:
        inspection["resources_error"] = f"Unable to fetch resource list: {e}"

    # Tools inspection
    try:
        tools = await asyncio.wait_for(session.list_tools(), timeout=5.0)
        if tools and tools.tools:
            inspection["tools"] = [
                f"- **{t.name}**: {t.description or 'No description'}"
                for t in tools.tools
            ]
    except asyncio.TimeoutError:
        inspection["tools_error"] = "Tool list request timed out"
    except Exception as e:
        inspection["tools_error"] = f"Unable to fetch tool list: {e}"

    st.session_state.server_inspection = inspection


@asynccontextmanager
async def http_client(url: str, headers: Dict[str, str]):
    """HTTP client transport for stateless MCP servers."""
    read_send, read_receive = create_memory_object_stream(100)
    write_send, write_receive = create_memory_object_stream(100)

    async def handle_outgoing():
        async with httpx.AsyncClient() as client:
            async with write_receive:
                async for message in write_receive:
                    # Serialize message
                    msg = message.message if hasattr(message, "message") else message
                    json_body = pydantic.TypeAdapter(types.JSONRPCMessage).dump_python(
                        msg, mode="json", by_alias=True
                    )
                    if "params" in json_body and json_body["params"] is None:
                        del json_body["params"]

                    try:
                        # Ensure we accept both JSON and SSE, as some servers require it
                        req_headers = headers.copy()
                        req_headers["Accept"] = "application/json, text/event-stream"

                        response = await client.post(
                            url, json=json_body, headers=req_headers, timeout=30.0
                        )

                        if response.status_code == 202:
                            continue

                        # Handle error responses that might contain JSON-RPC errors
                        if response.is_error:
                            try:
                                response_data = response.json()
                                # If we can parse it as a JSONRPCMessage, treat it as a valid protocol message
                                parsed_msg = pydantic.TypeAdapter(
                                    types.JSONRPCMessage
                                ).validate_python(response_data)

                                await read_send.send(
                                    SessionMessage(message=parsed_msg, metadata=None)
                                )
                                continue
                            except Exception:
                                pass

                        response.raise_for_status()

                        # Check for SSE response
                        content_type = response.headers.get("content-type", "")
                        if "text/event-stream" in content_type:
                            async for line in response.aiter_lines():
                                line = line.strip()
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    try:
                                        response_data = json.loads(data_str)
                                        parsed_msg = pydantic.TypeAdapter(
                                            types.JSONRPCMessage
                                        ).validate_python(response_data)
                                        await read_send.send(
                                            SessionMessage(
                                                message=parsed_msg, metadata=None
                                            )
                                        )
                                    except Exception:
                                        pass
                        else:
                            # Parse standard JSON response
                            response_data = response.json()
                            parsed_msg = pydantic.TypeAdapter(
                                types.JSONRPCMessage
                            ).validate_python(response_data)

                            await read_send.send(
                                SessionMessage(message=parsed_msg, metadata=None)
                            )
                    except Exception as e:
                        # Log error and potentially stop
                        st.error(f"HTTP Transport error: {e}")
                        raise e

    async with create_task_group() as tg:
        tg.start_soon(handle_outgoing)
        yield read_receive, write_send
        tg.cancel_scope.cancel()


@asynccontextmanager
async def get_mcp_session(server_params: Any):
    """Context manager for safely managing MCP session."""
    if not server_params:
        yield None
        return

    session_started = False
    try:
        if isinstance(server_params, StdioServerParameters):
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=30.0)
                    session_started = True
                    yield session

        elif isinstance(server_params, dict) and "url" in server_params:
            try:
                # Try SSE first
                async with sse_client(
                    url=server_params["url"], headers=server_params.get("headers", {})
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=30.0)
                        session_started = True
                        yield session
            except Exception as e:
                # Fallback to HTTP if SSE fails with 405 Method Not Allowed or SSEError
                is_405 = False
                if (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code == 405
                ):
                    is_405 = True
                elif isinstance(e, SSEError):
                    is_405 = True
                elif hasattr(e, "exceptions"):
                    for sub_exc in e.exceptions:
                        if (
                            isinstance(sub_exc, httpx.HTTPStatusError)
                            and sub_exc.response.status_code == 405
                        ):
                            is_405 = True
                            break
                        if isinstance(sub_exc, SSEError):
                            is_405 = True
                            break

                if is_405:
                    async with http_client(
                        url=server_params["url"],
                        headers=server_params.get("headers", {}),
                    ) as (read, write):
                        async with ClientSession(read, write) as session:
                            await asyncio.wait_for(session.initialize(), timeout=30.0)
                            session_started = True
                            yield session
                else:
                    raise e
        else:
            raise ValueError("Invalid server parameters")

    except asyncio.TimeoutError:
        if session_started:
            raise
        st.error("MCP server connection timed out")
        yield None
    except Exception as e:
        if session_started:
            raise
        if hasattr(e, "exceptions"):
            for sub_exc in e.exceptions:
                st.error(f"Failed to connect to MCP server: {sub_exc}")
        else:
            st.error(f"Failed to connect to MCP server: {e}")
        yield None


async def send_message_with_mcp(prompt: str, server_params: Any):
    """Send message with MCP server using Gemini chat."""
    client = genai.Client()
    try:
        async with get_mcp_session(server_params) as session:
            # Get conversation history for context
            messages = st.session_state.chat["messages"]

            # Prepare full conversation context including the current prompt
            contents = []
            for msg in messages:
                if msg["role"] == "user":
                    contents.append(
                        {"role": "user", "parts": [{"text": msg["content"]}]}
                    )
                elif msg["role"] == "assistant":
                    contents.append(
                        {"role": "model", "parts": [{"text": msg["content"]}]}
                    )

            # Add current user prompt
            contents.append({"role": "user", "parts": [{"text": prompt}]})

            # Create configuration with tools if available
            config = genai.types.GenerateContentConfig()
            if session is not None:
                config.tools = [session]
                config.system_instruction = (
                    "You are a helpful AI assistant connected to an MCP server. "
                    "You have access to a set of tools provided by the server. "
                    "Use these tools whenever necessary to fulfill the user's request. "
                    "Always answer in the language used by the user."
                )

            with st.spinner("Generating response..."):
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=config,
                    ),
                    timeout=30.0,
                )

            # Handle tool calls
            while response.function_calls:
                # Display tool calls
                for call in response.function_calls:
                    with st.chat_message("assistant"):
                        st.markdown(f"🛠️ Calling tool: `{call.name}`")
                        with st.expander("Arguments"):
                            st.json(call.args)

                # Add model's function call message to history
                model_parts = response.candidates[0].content.parts
                contents.append({"role": "model", "parts": model_parts})

                # Execute tools and collect responses
                tool_responses = []
                for call in response.function_calls:
                    try:
                        result = await session.call_tool(call.name, arguments=call.args)

                        # Extract text content from result
                        tool_output_text = ""
                        if result.content:
                            for content_item in result.content:
                                if content_item.type == "text":
                                    tool_output_text += content_item.text

                        # Display tool result
                        with st.chat_message("assistant"):
                            with st.expander(f"Result: {call.name}"):
                                st.markdown(tool_output_text)

                        tool_responses.append(
                            {
                                "function_response": {
                                    "name": call.name,
                                    "response": {"result": tool_output_text},
                                }
                            }
                        )
                    except Exception as e:
                        st.error(f"Error executing tool {call.name}: {e}")
                        tool_responses.append(
                            {
                                "function_response": {
                                    "name": call.name,
                                    "response": {"error": str(e)},
                                }
                            }
                        )

                # Add tool responses to history
                contents.append({"role": "tool", "parts": tool_responses})

                # Generate next response
                with st.spinner("Generating response..."):
                    response = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=contents,
                            config=config,
                        ),
                        timeout=30.0,
                    )

            if response and response.text:
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.chat["messages"].append(
                    {"role": "assistant", "content": response.text}
                )
            else:
                st.warning("Received empty response.")

    except asyncio.TimeoutError:
        st.error("Response generation timed out.")
    except Exception as e:
        if hasattr(e, "exceptions"):
            for sub_exc in e.exceptions:
                st.error(f"Error occurred while sending message: {sub_exc}")
        else:
            st.error(f"Error occurred while sending message: {e}")


async def initialize_session_safely(server_params: Any):
    """Safely initialize session."""
    st.session_state.server_inspection = None
    if not server_params:
        return

    try:
        async with get_mcp_session(server_params) as session:
            if session:
                await safe_inspect_server(session)
    except Exception as e:
        st.error(f"Error during session initialization: {e}")


def main():
    """Main application function."""
    # MCP Server configuration in sidebar
    with st.sidebar:
        st.header("MCP Chat")

        if "chat" not in st.session_state:
            create_new_chat()
            st.rerun()

        # New chat button
        if st.button("New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        # Load MCP configuration
        mcp_config = load_mcp_config()

        # Server selection
        server_names = ["None"] + list(mcp_config["mcpServers"].keys())
        selected_server = st.selectbox("Select MCP Server", server_names)

        # Get server configuration
        server_config = (
            {}
            if selected_server == "None"
            else mcp_config["mcpServers"][selected_server]
        )
        server_params = None

        if server_config:
            try:
                server_params = create_server_parameters(server_config)
            except ValueError as e:
                st.error(f"Server configuration error: {e}")
                server_params = None

        # Display server configuration
        with st.expander("Server Configuration"):
            if server_config:
                st.json(server_config)
            else:
                st.info("No server selected.")

        # Reinitialize session on server change
        if ("selected_server" not in st.session_state) or (
            st.session_state.selected_server != selected_server
        ):
            with st.spinner("Connecting to server..."):
                asyncio.run(initialize_session_safely(server_params))
            st.session_state.selected_server = selected_server
            # Add assistant message to chat
            st.session_state.chat["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Connected to server: {st.session_state.get('selected_server', 'None')}",
                }
            )

        if st.session_state.get("server_inspection"):
            with st.expander("🔎 MCP Server Inspection"):
                insp = st.session_state.server_inspection

                st.subheader("📑 Prompts")
                if insp.get("prompts_error"):
                    st.warning(insp["prompts_error"])
                elif insp.get("prompts"):
                    st.markdown("\n".join(insp["prompts"]))
                else:
                    st.info("No available prompts.")

                st.subheader("📂 Resources")
                if insp.get("resources_error"):
                    st.warning(insp["resources_error"])
                elif insp.get("resources"):
                    st.markdown("\n".join(insp["resources"]))
                else:
                    st.info("No available resources.")

                st.subheader("🛠️ Tools")
                if insp.get("tools_error"):
                    st.warning(insp["tools_error"])
                elif insp.get("tools"):
                    st.markdown("\n".join(insp["tools"]))
                else:
                    st.info("No available tools.")

    # Display chat history
    messages = st.session_state.chat["messages"]
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Enter your message..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to current chat
        st.session_state.chat["messages"].append({"role": "user", "content": prompt})

        # Generate response
        asyncio.run(send_message_with_mcp(prompt, server_params))


if __name__ == "__main__":
    main()
