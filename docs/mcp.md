# Using embd as an MCP Tool

embd ships a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that
lets Claude CLI and Claude Desktop query your document index as a first-class tool — no
copy-pasting, no switching windows.

The MCP server is a **local client-side process**. It runs on your machine, speaks the
stdio MCP protocol to Claude, and makes authenticated HTTPS calls to your embd retrieval
server. Nothing changes on the server side.

---

## Prerequisites

- Python 3.11+
- An embd retrieval server running and accessible over HTTPS
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

**With uv (recommended):** no separate install step — uv resolves the environment on first
run via the config (see below).

**With pip:**
```bash
pip install "embd[mcp]"
```

---

## Environment variables

The MCP server reads its configuration from environment variables. These are set inside
your Claude config file (see below) and never touch the repo.

| Variable | Required | Description |
|----------|----------|-------------|
| `EMBD_BASE_URL` | Yes | Base URL of your embd server (e.g. `https://embd.example.com`) |
| `EMBD_API_KEY`  | Yes | Bearer token — must match the key configured on the server |

---

## Claude CLI setup

Add the `embd` server to your Claude CLI settings. The config file lives at
`~/.claude/settings.json` (global) or `.claude/settings.json` (project-level).

**With uv (recommended):** uv resolves and caches the environment automatically on first
run — no separate install step.

```json
{
  "mcpServers": {
    "embd": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/embd", "--extra", "mcp", "embd-mcp"],
      "env": {
        "EMBD_BASE_URL": "<YOUR_SERVER_URL>",
        "EMBD_API_KEY": "<YOUR_API_KEY>"
      }
    }
  }
}
```

**With pip (after `pip install "embd[mcp]"`):**

```json
{
  "mcpServers": {
    "embd": {
      "command": "embd-mcp",
      "env": {
        "EMBD_BASE_URL": "<YOUR_SERVER_URL>",
        "EMBD_API_KEY": "<YOUR_API_KEY>"
      }
    }
  }
}
```

Replace `/path/to/embd`, `<YOUR_SERVER_URL>`, and `<YOUR_API_KEY>` with your actual values.

After saving, start a new Claude CLI session. Run `/mcp` to confirm the `embd` server
appears in the list with status **connected**.

---

## Claude Desktop setup

Add the same block to Claude Desktop's config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "embd": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/embd", "--extra", "mcp", "embd-mcp"],
      "env": {
        "EMBD_BASE_URL": "<YOUR_SERVER_URL>",
        "EMBD_API_KEY": "<YOUR_API_KEY>"
      }
    }
  }
}
```

Restart Claude Desktop after saving. The `embd` tool should appear in the tool picker.

---

## Available tool

### `search_documents`

Search the document index for passages relevant to a query.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query`     | string | required | Natural-language question or search phrase |
| `source_id` | string | — | Restrict to one document (e.g. `report.pdf`, `papers/report.pdf`) |
| `top_k`     | int    | 5 | Number of passages to return (1–20) |

Results are returned in the format:

```
[report.pdf p.12] (score 0.87)
The relevant passage text…

---

[papers/overview.pdf p.4] (score 0.81)
Another relevant passage…
```

---

## Testing

**Verify the server starts:**

```bash
EMBD_BASE_URL=https://embd.example.com EMBD_API_KEY=your-key embd-mcp
```

You should see MCP initialization output on stderr. Press Ctrl-C to stop.

**Verify the health endpoint is reachable:**

```bash
curl -s https://embd.example.com/health
# Expected: {"status":"ok"}
```

**In Claude CLI:**

```
/mcp
```

Shows connected MCP servers. Then try asking Claude a question about your documents —
it will automatically invoke `search_documents` when relevant.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `EMBD_BASE_URL is not set` | Env var missing from Claude config | Add `EMBD_BASE_URL` to the `env` block in your Claude config |
| `Authentication failed` (401) | Wrong API key | Check `EMBD_API_KEY` matches the key on the server |
| `Access denied` (403) | Your IP not in server allowlist | Add your IP to `deploy/custom-ips.conf` on the server, reload nginx |
| `Could not reach embd server` | Server down or URL wrong | Check `EMBD_BASE_URL` and run `curl <url>/health` |
| `embd-mcp: command not found` | Package not installed or not on PATH | Switch to the `uv run --directory ...` form, or run `pip install "embd[mcp]"` |
| MCP server shows as disconnected | Process crash at startup | Run `embd-mcp` manually in a terminal to see the error |

---

## Alternative: CLI command

If you are running from a directory that contains a `config.toml`, you can also start the
MCP server via the `embd` CLI:

```bash
EMBD_BASE_URL=... EMBD_API_KEY=... embd mcp-serve
```

This is equivalent to `embd-mcp` but goes through the CLI entry point. The `--config`
option is accepted but ignored for this subcommand — configuration always comes from
environment variables.
