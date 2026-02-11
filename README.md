# MCP Chat

MCP (Model Context Protocol) is an open-source standard for connecting AI applications to external systems. This is a Streamlit-based web application that serves as a client for MCP servers. It allows you to interact with a Gemini model that can leverage tools from a selected MCP server.

## Features

- **Web-based UI**: A simple and intuitive chat interface built with Streamlit.
- **Multi-server Support**: Connect to different MCP servers by selecting them from a dropdown menu. Supports both stdio-based local servers and HTTP/SSE-based remote servers.
- **Server Inspection**: View the available prompts, resources, and tools of the selected MCP server.
- **Chat with Tools**: Interact with the Gemini model, which can use the connected MCP server's capabilities as tools to answer your prompts.
- **Configuration Display**: The configuration of the selected server is displayed in the sidebar.

## How to Run

1.  **Set up your environment**:
    Create a `.env` file by copying the example file and editing it with your GEMINI_API_KEY value.
    ```bash
    cp .env.example .env
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure your MCP servers**:
    Make sure you have a `mcp.json` file in the same directory. See the **Configuration** section below for details.
    ```bash
    cp mcp.json.example mcp.json
    ```

4.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Configuration

The `mcp.json` file is used to configure the MCP servers that the application can connect to.

The file should contain a single JSON object with a key `mcpServers`. The value of this key is another JSON object where each key is the server name.

### Stdio-based Servers (Local)

For servers that run locally via command line:

```json
{
  "mcpServers": {
    "my-local-server": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "MY_VAR": "value"
      }
    }
  }
}
```

-   `command`: The command to execute to start the server.
-   `args`: A list of arguments to pass to the command.
-   `env`: (Optional) A dictionary of environment variables to set for the server process.

### HTTP/SSE-based Servers (Remote)

For remote servers or servers running via HTTP/SSE:

```json
{
  "mcpServers": {
    "my-remote-server": {
      "url": "http://localhost:8000/sse",
      "headers": {
        "Authorization": "Bearer token"
      }
    }
  }
}
```

-   `url`: The URL endpoint for the MCP server.
-   `headers`: (Optional) A dictionary of HTTP headers to include in requests to the server.
