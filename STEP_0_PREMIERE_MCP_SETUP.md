# Step 0 — Universal Setup: Any IDE ↔ Adobe Premiere Pro MCP Bridge

Run **Step 0** once on any machine or before switching IDEs (Antigravity, Cursor, VS Code / GitHub Copilot, Windsurf, Claude Code, Claude Desktop, OpenAI Codex CLI, or any stdio-compatible MCP client) so your AI assistant can communicate reliably with Adobe Premiere Pro.

---

## Quick Start (One-Command Setup & Diagnostics)

> Close **Adobe Premiere Pro** before installing or updating the CEP connector.

**Windows (PowerShell):**
```powershell
# Full install + CEP plugin install + doctor check + print IDE configs
powershell -ExecutionPolicy Bypass -File .\setup_premiere_bridge.ps1 -Install

# Diagnostics only (no writes): verifies Node, CEP, doctor, and live heartbeat
powershell -ExecutionPolicy Bypass -File .\setup_premiere_bridge.ps1 -CheckOnly

# Auto-write workspace configs (.mcp.json, .cursor/mcp.json, .vscode/mcp.json)
powershell -ExecutionPolicy Bypass -File .\setup_premiere_bridge.ps1 -WriteWorkspaceConfig
```

**macOS / Linux (Bash):**
```bash
chmod +x ./setup_premiere_bridge.sh
./setup_premiere_bridge.sh --install
# Or read-only diagnostics:
./setup_premiere_bridge.sh --check-only
```

---

## Step 1 — Install the MCP Server & Premiere CEP Bridge

1. **Prerequisite:** Install **Node.js 20.19+** (`node -v`).
2. **Install the MCP server globally** (package `premiere-pro-mcp`, currently tested on `v1.16.1`):
   ```bash
   npm install -g premiere-pro-mcp
   ```
3. **Install the Premiere Pro CEP Extension + CSXS Debug Mode:**
   ```bash
   premiere-pro-mcp --install-cep
   premiere-pro-mcp --diagnose-cep
   premiere-pro-mcp --doctor
   ```
   *(Built-in helper to print IDE configs directly: `premiere-pro-mcp --print-client-config <claude|cursor|vscode|codex>`)*

---

## Step 2 — Connect Any IDE / MCP Client

If your IDE supports workspace configs, this repository already includes [`.mcp.json`](.mcp.json). Otherwise, use the exact absolute `node` + `dist/index.js` paths printed by `setup_premiere_bridge.ps1` (recommended over bare command names when GUI apps strip `PATH`).

### A. Universal `mcpServers` JSON
Works in:
- **Google Antigravity:** Global MCP config (`%USERPROFILE%\.gemini\antigravity\mcp_config.json` on Windows / `~/.gemini/antigravity/mcp_config.json` on macOS) or workspace `.mcp.json`
- **Cursor:** `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global)
- **Windsurf / Cascade:** `~/.codeium/windsurf/mcp_config.json`
- **Claude Code:** `.mcp.json` (project root) or `claude mcp add premiere-pro -- premiere-pro-mcp`
- **Claude Desktop:** `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "premiere-pro": {
      "command": "premiere-pro-mcp",
      "env": {
        "PREMIERE_MCP_TELEMETRY": "0"
      }
    }
  }
}
```
*(If the IDE cannot launch `premiere-pro-mcp` from `PATH`, set `"command"` to your `node` executable and `"args": ["<global-npm-root>/premiere-pro-mcp/dist/index.js"]` — `setup_premiere_bridge.ps1` prints the exact escaped path for your machine.)*

### B. VS Code / GitHub Copilot (`.vscode/mcp.json`)
VS Code expects `"servers"` with `"type": "stdio"`:
```json
{
  "servers": {
    "premiere-pro": {
      "type": "stdio",
      "command": "premiere-pro-mcp",
      "env": {
        "PREMIERE_MCP_TELEMETRY": "0"
      }
    }
  }
}
```

### C. OpenCode CLI / IDE (`opencode.json` or `~/.config/opencode/opencode.json`)
OpenCode uses a dedicated `"mcp": { "servers": { ... } }` schema with `"type": "local"`:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "premiere-pro": {
        "type": "local",
        "command": [
          "premiere-pro-mcp"
        ],
        "environment": {
          "PREMIERE_MCP_TELEMETRY": "0"
        }
      }
    }
  }
}
```
*(This repo already includes a pre-configured [`opencode.json`](opencode.json) in the workspace root. If you prefer using the OpenCode CLI directly, run: `opencode mcp add premiere-pro -- premiere-pro-mcp` or `opencode mcp add premiere-pro --global -- premiere-pro-mcp`)*

### D. OpenAI Codex CLI (`~/.codex/config.toml`)
```toml
[mcp_servers.premiere-pro]
command = "premiere-pro-mcp"
```

### E. Legacy Handshake Fallback (Any Client Failing `server/discover`)
If an older IDE or MCP client fails modern `2026-07-28` capability negotiation, add `"PREMIERE_MCP_PROTOCOL_MODE": "legacy"` under `"env"` (or pass `-LegacyProtocol` to `setup_premiere_bridge.ps1`) and restart the IDE.

---

## Step 3 — Activate & Verify Inside Adobe Premiere Pro

1. Launch **Adobe Premiere Pro** and open a project with an active sequence.
2. Open **Window > Extensions > MCP for Adobe Premiere Pro** (if listed under legacy/short name: **MCP Bridge**).
3. Confirm the panel shows a green **Running** status and uses the default temp bridge directory:
   - **Windows:** `%TEMP%\premiere-mcp-bridge`
   - **macOS:** `$TMPDIR/premiere-mcp-bridge`
   - Do **not** override `PREMIERE_TEMP_DIR` unless both your IDE MCP `"env"` and the Premiere panel point to the exact same absolute directory.
4. Reload MCP servers in your IDE and run a **read-only health check**:
   > `Call ping and verify_premiere_connection. Make no changes.`

---

## Troubleshooting Quick Reference

| Symptom | Fix |
| :--- | :--- |
| `The CEP connector heartbeat is stale` | Premiere is closed, no panel is open, or Premiere blocked the extension. Open **Window > Extensions > MCP for Adobe Premiere Pro** inside Premiere and retry `ping`. |
| Extension missing from `Window > Extensions` | Fully quit Premiere and run `premiere-pro-mcp --install-cep` (ensures CSXS `PlayerDebugMode = 1`), then inspect `premiere-pro-mcp --diagnose-cep`. |
| IDE cannot spawn `premiere-pro-mcp` | Run `.\setup_premiere_bridge.ps1` and paste the printed absolute `"command"` (`node.exe`) + `"args"` (`dist/index.js`) snippet instead of `"premiere-pro-mcp"`. |
| Safe auto-repair plan | Run `premiere-pro-mcp --doctor --plan-fixes` (or `--apply-fixes --confirm-premiere-closed` with Premiere quit). |

