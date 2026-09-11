# Client setup

Install the `mcp` extra, then use the **absolute path** to that environment's `agent-memory-mcp` executable. Set one absolute store path for every client working on the same project. This avoids dependence on the client's launch directory or shell activation.

## Claude Code

Run inside the target project, replacing the paths:

```bash
claude mcp add --env AGENT_MEMORY_AGENT=claude-code AGENT_MEMORY_PATH=/absolute/project/.agent_memory/store.json AGENT_MEMORY_EMBEDDER=hashing --transport stdio agent-memory -- /absolute/venv/bin/agent-memory-mcp
```

This uses local scope, private to you in that project. Check the connection with `/mcp`. The command structure follows the [official Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

Optional hooks:

```bash
agent-memory install-hooks
agent-memory install-hooks --with-prompt-recall
agent-memory install-hooks --uninstall
```

These merge the project's Claude settings and preserve other hooks. Startup includes a handoff no older than 14 days and a worklog no older than 42 days. Per-prompt recall adds a separate Python process and potentially a model load per message. Session-end notes describe Git changes observed since session start; overlapping sessions can observe the same work.

The prompt handler consumes the documented `prompt` field, retaining the older `user_input` fallback. See [hook input documentation](https://code.claude.com/docs/en/hooks#userpromptsubmit-input).

## Codex CLI

Add to the applicable `config.toml`, with your absolute paths:

```toml
[mcp_servers.agent-memory]
command = "/absolute/venv/bin/agent-memory-mcp"

[mcp_servers.agent-memory.env]
AGENT_MEMORY_AGENT = "codex"
AGENT_MEMORY_PATH = "/absolute/project/.agent_memory/store.json"
AGENT_MEMORY_EMBEDDER = "hashing"
```

Follow the [official Codex MCP documentation](https://developers.openai.com/codex/mcp) for config scope and connection controls. This server does not require an OpenAI API key; your coding client retains its own authentication and permissions.

## Cursor

In the project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agent-memory": {
      "type": "stdio",
      "command": "/absolute/venv/bin/agent-memory-mcp",
      "env": {
        "AGENT_MEMORY_AGENT": "cursor",
        "AGENT_MEMORY_PATH": "${workspaceFolder}/.agent_memory/store.json",
        "AGENT_MEMORY_EMBEDDER": "hashing"
      }
    }
  }
}
```

See [Cursor's official MCP documentation](https://cursor.com/docs/mcp) for configuration and variable expansion. On Windows, use an executable such as `C:\\project\\.venv\\Scripts\\agent-memory-mcp.exe` in JSON.

## Verify the connection

Ask the client to write a harmless project fact, list it, read its revision, correct it with `expected_revision`, and recall it. Run `agent-memory --path /absolute/project/.agent_memory/store.json doctor` if a client sees an empty store. An empty answer under a tiny budget can be expected: whole memories are omitted if their rendered text does not fit.

| Integration | Verification in this candidate |
| --- | --- |
| MCP tool API | Automated tests for both SDK major versions |
| MCP stdio transport | Two-process scripted handoff demo |
| Claude hook payloads | Real subprocess JSON input/output and temporary Git repositories |
| Claude Code, Codex and Cursor applications | Configuration checked against official documentation; live application sessions still need verification |

The server sends workflow instructions at initialization. Whether an agent follows them depends on the client and model; automatic use across every client is not guaranteed.
