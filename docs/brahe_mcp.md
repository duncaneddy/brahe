# Brahe MCP

[Brahe MCP](https://github.com/duncaneddy/brahe-mcp) is a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes Brahe's astrodynamics capabilities to language models and AI-assisted development tools. Once connected, a model can call Brahe functions directly — time conversions, coordinate and frame transformations, orbit propagation, access computation, and TLE/ephemeris lookups — without the model having to write or run Brahe code manually. This can significantly improve the accuracy of astrodynamics-related responses and enable new use cases like interactive analysis and natural language querying of satellite data.

The server is published to PyPI as [`brahe-mcp`](https://pypi.org/project/brahe-mcp/). It runs locally over stdio and works with any MCP-compatible client, including Claude Desktop, Claude Code, Gemini CLI, and the OpenAI Codex CLI.

## Installation

Install the MCP server with `uv` (recommended) or `pip`:

```bash
uv tool install brahe-mcp
```

```bash
pip install brahe-mcp
```

Both methods expose a `brahe-mcp` command on the `PATH`. MCP clients launch that command to start the server over stdio.

## Client Configuration

The server entry is the same across clients — register a server named `brahe` that runs the `brahe-mcp` command. Only the config file location and syntax differ by client.

### Claude Desktop

Edit the Claude Desktop config file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add a `brahe` entry under `mcpServers`:

```json
{
  "mcpServers": {
    "brahe": {
      "command": "brahe-mcp"
    }
  }
}
```

Restart Claude Desktop for the change to take effect.

### Claude Code

Claude Code reads MCP servers from `.claude/settings.json` (project-scoped) or from the user-level settings file. Use the same JSON block as Claude Desktop:

```json
{
  "mcpServers": {
    "brahe": {
      "command": "brahe-mcp"
    }
  }
}
```

Alternatively, add the server via the CLI without editing JSON directly:

```bash
claude mcp add brahe brahe-mcp
```

### OpenAI Codex CLI

[Codex CLI](https://developers.openai.com/codex/mcp) stores MCP configuration in TOML at `~/.codex/config.toml` (or project-scoped `.codex/config.toml`):

```toml
[mcp_servers.brahe]
command = "brahe-mcp"
args = []
```

Or add it via the Codex CLI:

```bash
codex mcp add brahe -- brahe-mcp
```

### ChatGPT Desktop and Web

ChatGPT Desktop and the ChatGPT web client do not support local stdio MCP servers — they require remote HTTPS endpoints. Brahe MCP is currently a local-only server, so it cannot be connected to ChatGPT at this time. Use Claude Desktop, Claude Code, Codex CLI, or another MCP client with stdio support.

## SpaceTrack Credentials

The SpaceTrack-backed tools (TLE lookups, conjunction data) require a [Space-Track.org](https://www.space-track.org) account. Provide credentials through the `env` block in the server config:

```json
{
  "mcpServers": {
    "brahe": {
      "command": "brahe-mcp",
      "env": {
        "SPACETRACK_USER": "your@email.com",
        "SPACETRACK_PASS": "your-password"
      }
    }
  }
}
```

For Codex, pass them through the CLI:

```bash
codex mcp add brahe \
  --env SPACETRACK_USER=your@email.com \
  --env SPACETRACK_PASS=your-password \
  -- brahe-mcp
```

Claude Code inherits the parent shell environment, so credentials already exported in `~/.zshrc` or `~/.bashrc` are available without an `env` block. Claude Desktop does **not** expand shell variables like `${SPACETRACK_USER}` — literal values must be written in the config file.

Without these credentials, the CelesTrak-backed tools continue to function; only the SpaceTrack tools will return an authentication error.

## Local Development

To run the server from a local clone of the repository (useful for development or testing unreleased features):

```bash
git clone https://github.com/duncaneddy/brahe-mcp.git
cd brahe-mcp
uv sync --group dev
```

Point your MCP client at the local clone with `uv run`:

```json
{
  "mcpServers": {
    "brahe": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/brahe-mcp", "brahe-mcp"]
    }
  }
}
```

Replace `/absolute/path/to/brahe-mcp` with the full path to your clone.

## Further Reading

- [Brahe MCP on GitHub](https://github.com/duncaneddy/brahe-mcp) — source, issue tracker, and contribution guide
- [Brahe MCP on PyPI](https://pypi.org/project/brahe-mcp/) — release index
- [Model Context Protocol specification](https://modelcontextprotocol.io) — protocol background and client list
