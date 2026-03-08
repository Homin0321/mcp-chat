import asyncio
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

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

from utils import fix_markdown_symbol_issue

# Page settings
st.set_page_config(page_title="MCP Chat", page_icon="🤖", layout="wide")

# Load environment variables and initialize client
MODEL_OPTIONS = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

load_dotenv()


def extract_grounding_info(candidate):
    """Extract grounding and URL context information from response candidate."""
    info = {
        "queries": [],
        "chunks": [],
        "rendered_content": None,
        "url_contexts": [],
    }

    has_data = False

    if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
        gm = candidate.grounding_metadata
        info["queries"] = getattr(gm, "web_search_queries", [])

        if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    title = getattr(chunk.web, "title", "Unknown Source")
                    uri = getattr(chunk.web, "uri", "")
                    if uri:
                        info["chunks"].append({"title": title, "uri": uri})

        if hasattr(gm, "search_entry_point") and gm.search_entry_point:
            info["rendered_content"] = getattr(
                gm.search_entry_point, "rendered_content", None
            )

        if info["queries"] or info["chunks"] or info["rendered_content"]:
            has_data = True

    if hasattr(candidate, "url_context_metadata") and candidate.url_context_metadata:
        ucm = candidate.url_context_metadata
        if hasattr(ucm, "url_metadata") and ucm.url_metadata:
            for um in ucm.url_metadata:
                url = getattr(um, "retrieved_url", "")
                status = getattr(um, "url_retrieval_status", "")
                status_str = str(status).split(".")[-1] if status else "UNKNOWN_STATUS"
                if url:
                    info["url_contexts"].append({"url": url, "status": status_str})
                    has_data = True

    return info if has_data else None


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
                "role": "system",
                "content": f"Connected to server: {st.session_state.get('selected_server', 'None')}",
            }
        )


@st.dialog("Markdown Source", width="large")
def show_markdown():
    if "chat" in st.session_state and st.session_state.chat.get("messages"):
        messages = st.session_state.chat["messages"]
        if len(messages) > 0:
            st.code(messages[-1]["content"], language="markdown")
            return

    st.info("No messages to display.")


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


CONNECTION_TIMEOUT = 30.0


def _needs_http_fallback(e: Exception) -> bool:
    """SSE에서 HTTP 통신으로 Fallback 해야 하는 에러인지 판별합니다."""
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 405:
        return True
    if isinstance(e, SSEError):
        return True

    if hasattr(e, "exceptions"):
        for sub_exc in getattr(e, "exceptions", []):
            if (
                isinstance(sub_exc, httpx.HTTPStatusError)
                and sub_exc.response.status_code == 405
            ):
                return True
            if isinstance(sub_exc, SSEError):
                return True

    return False


@asynccontextmanager
async def get_mcp_session(
    server_params: Union[StdioServerParameters, dict, None],
) -> AsyncGenerator[Union[ClientSession, None], None]:
    """Context manager for safely managing MCP session."""
    if not server_params:
        yield None
        return

    async with AsyncExitStack() as stack:
        try:
            if isinstance(server_params, StdioServerParameters):
                read, write = await stack.enter_async_context(
                    stdio_client(server_params)
                )

            elif isinstance(server_params, dict) and "url" in server_params:
                try:
                    read, write = await stack.enter_async_context(
                        sse_client(
                            url=server_params["url"],
                            headers=server_params.get("headers", {}),
                        )
                    )
                except Exception as e:
                    if _needs_http_fallback(e):
                        read, write = await stack.enter_async_context(
                            http_client(
                                url=server_params["url"],
                                headers=server_params.get("headers", {}),
                            )
                        )
                    else:
                        raise e
            else:
                raise ValueError("Invalid server parameters")

            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=CONNECTION_TIMEOUT)

        except asyncio.TimeoutError:
            st.error("MCP server connection timed out")
            yield None
            return
        except Exception as e:
            if hasattr(e, "exceptions"):
                for sub_exc in getattr(e, "exceptions", []):
                    st.error(f"Failed to connect to MCP server: {sub_exc}")
            else:
                st.error(f"Failed to connect to MCP server: {e}")
            yield None
            return

        yield session


async def send_message_with_mcp(prompt: str, server_params: Any):
    """Send message with MCP server using Gemini chat."""
    model_name = st.session_state.get("selected_model", MODEL_OPTIONS[0])
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
                elif msg["role"] == "model_tool_call":
                    contents.append({"role": "model", "parts": msg["parts"]})
                elif msg["role"] == "tool_response":
                    contents.append({"role": "user", "parts": msg["parts"]})

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
            else:
                tools = []
                if st.session_state.get("use_google_search", False):
                    tools.append(
                        genai.types.Tool(google_search=genai.types.GoogleSearch())
                    )
                if st.session_state.get("use_url_context", False):
                    tools.append(genai.types.Tool(url_context=genai.types.UrlContext()))

                if tools:
                    config.tools = tools

            with st.spinner("Generating response..."):
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    ),
                    timeout=120.0,
                )

            # Handle AFC history
            if getattr(response, "automatic_function_calling_history", None):
                for content in response.automatic_function_calling_history[
                    len(contents) :
                ]:
                    # Model tool call
                    if content.role == "model" and content.parts:
                        function_calls = [
                            p.function_call for p in content.parts if p.function_call
                        ]
                        if function_calls:
                            for call in function_calls:
                                with st.chat_message("assistant"):
                                    st.markdown(f"🛠️ Calling tool: `{call.name}`")
                                    with st.expander("Arguments"):
                                        st.json(call.args)

                            # Add model tool call to history
                            st.session_state.chat["messages"].append(
                                {
                                    "role": "model_tool_call",
                                    "parts": [
                                        {
                                            "function_call": {
                                                "name": c.name,
                                                "args": c.args,
                                            }
                                        }
                                        for c in function_calls
                                    ],
                                    "function_calls": function_calls,
                                }
                            )

                    # Tool response
                    elif content.role in ["user", "tool"] and content.parts:
                        function_responses = [
                            p.function_response
                            for p in content.parts
                            if p.function_response
                        ]
                        if function_responses:
                            tool_responses = []
                            for resp in function_responses:
                                tool_responses.append(
                                    {
                                        "function_response": {
                                            "name": resp.name,
                                            "response": resp.response,
                                        }
                                    }
                                )
                                with st.chat_message("assistant"):
                                    with st.expander(f"Result: {resp.name}"):
                                        if "result" in resp.response:
                                            res = resp.response["result"]
                                            if (
                                                isinstance(res, dict)
                                                and "content" in res
                                            ):
                                                texts = [
                                                    c.get("text", "")
                                                    for c in res["content"]
                                                    if isinstance(c, dict)
                                                    and c.get("type") == "text"
                                                ]
                                                st.markdown("".join(texts))
                                            elif hasattr(res, "content"):
                                                texts = [
                                                    getattr(c, "text", "")
                                                    for c in res.content
                                                    if getattr(c, "type", "") == "text"
                                                ]
                                                st.markdown("".join(texts))
                                            else:
                                                st.markdown(str(res))
                                        elif "error" in resp.response:
                                            st.error(resp.response["error"])
                                        else:
                                            st.markdown(str(resp.response))

                            st.session_state.chat["messages"].append(
                                {"role": "tool_response", "parts": tool_responses}
                            )

            if response and response.text:
                fixed_text = fix_markdown_symbol_issue(response.text)

                grounding_info = None
                if hasattr(response, "candidates") and response.candidates:
                    grounding_info = extract_grounding_info(response.candidates[0])

                with st.chat_message("assistant"):
                    st.markdown(fixed_text)

                    if grounding_info:
                        with st.expander("🔍 Sources & Context"):
                            if grounding_info.get("queries"):
                                st.markdown("**Search Queries:**")
                                for q in grounding_info["queries"]:
                                    st.markdown(f"- `{q}`")

                            if grounding_info.get("chunks"):
                                st.markdown("**Sources:**")
                                for i, chunk in enumerate(grounding_info["chunks"], 1):
                                    st.markdown(
                                        f"{i}. [{chunk['title']}]({chunk['uri']})"
                                    )

                            if grounding_info.get("rendered_content"):
                                st.components.v1.html(
                                    grounding_info["rendered_content"],
                                    height=150,
                                    scrolling=True,
                                )

                            if grounding_info.get("url_contexts"):
                                st.markdown("**URL Contexts:**")
                                for uc in grounding_info["url_contexts"]:
                                    status_emoji = (
                                        "✅" if "SUCCESS" in uc["status"] else "❌"
                                    )
                                    st.markdown(
                                        f"- {status_emoji} [{uc['url']}]({uc['url']}) ({uc['status']})"
                                    )

                st.session_state.chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": fixed_text,
                        "grounding_info": grounding_info,
                    }
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

        st.selectbox("Select Model", MODEL_OPTIONS, index=0, key="selected_model")

        if "chat" not in st.session_state:
            create_new_chat()
            st.rerun()

        # New chat button
        if st.button("New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        # Show Markdown button
        if st.button("Show Markdown", use_container_width=True):
            show_markdown()

        # Load MCP configuration
        mcp_config = load_mcp_config()

        # Server selection
        server_names = ["None"] + list(mcp_config["mcpServers"].keys())
        server_index = 0
        if st.session_state.get("selected_server") in server_names:
            server_index = server_names.index(st.session_state.selected_server)
        selected_server = st.selectbox(
            "Select MCP Server", server_names, index=server_index
        )

        # Get server configuration
        server_config = (
            {}
            if selected_server == "None"
            else mcp_config["mcpServers"][selected_server]
        )
        server_params = None

        if selected_server == "None":
            st.checkbox("Google Search Grounding", key="use_google_search")
            st.checkbox("URL Context", key="use_url_context")

        if server_config:
            try:
                server_params = create_server_parameters(server_config)
            except ValueError as e:
                st.error(f"Server configuration error: {e}")
                server_params = None

        # Display server configuration
        if selected_server != "None":
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
            # Add system message to chat
            st.session_state.chat["messages"].append(
                {
                    "role": "system",
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
        if message["role"] == "model_tool_call":
            for call in message["function_calls"]:
                with st.chat_message("assistant"):
                    st.markdown(f"🛠️ Calling tool: `{call.name}`")
                    with st.expander("Arguments"):
                        st.json(call.args)
        elif message["role"] == "tool_response":
            for resp in message["parts"]:
                func_resp = resp["function_response"]
                with st.chat_message("assistant"):
                    with st.expander(f"Result: {func_resp['name']}"):
                        if "result" in func_resp["response"]:
                            res = func_resp["response"]["result"]
                            if isinstance(res, dict) and "content" in res:
                                texts = [
                                    c.get("text", "")
                                    for c in res["content"]
                                    if isinstance(c, dict) and c.get("type") == "text"
                                ]
                                st.markdown("".join(texts))
                            elif hasattr(res, "content"):
                                texts = [
                                    getattr(c, "text", "")
                                    for c in res.content
                                    if getattr(c, "type", "") == "text"
                                ]
                                st.markdown("".join(texts))
                            else:
                                st.markdown(str(res))
                        elif "error" in func_resp["response"]:
                            st.error(func_resp["response"]["error"])
                        else:
                            st.markdown(str(func_resp["response"]))
        else:
            avatar = "⚙️" if message["role"] == "system" else None
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                if message.get("grounding_info"):
                    gi = message["grounding_info"]
                    with st.expander("🔍 Sources & Context"):
                        if gi.get("queries"):
                            st.markdown("**Search Queries:**")
                            for q in gi["queries"]:
                                st.markdown(f"- `{q}`")

                        if gi.get("chunks"):
                            st.markdown("**Sources:**")
                            for i, chunk in enumerate(gi["chunks"], 1):
                                st.markdown(f"{i}. [{chunk['title']}]({chunk['uri']})")

                        if gi.get("rendered_content"):
                            st.components.v1.html(
                                gi["rendered_content"], height=150, scrolling=True
                            )

                        if gi.get("url_contexts"):
                            st.markdown("**URL Contexts:**")
                            for uc in gi["url_contexts"]:
                                status_emoji = (
                                    "✅" if "SUCCESS" in uc["status"] else "❌"
                                )
                                st.markdown(
                                    f"- {status_emoji} [{uc['url']}]({uc['url']}) ({uc['status']})"
                                )

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
