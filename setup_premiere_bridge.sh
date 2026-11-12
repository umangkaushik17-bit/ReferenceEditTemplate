#!/usr/bin/env bash
# Step 0 setup & diagnostic script to bridge Adobe Premiere Pro MCP to any IDE (macOS / Linux).
set -euo pipefail

IDE="all"
INSTALL=0
CHECK_ONLY=0
WRITE_WORKSPACE_CONFIG=0
LEGACY_PROTOCOL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ide) IDE="${2:-all}"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --write-workspace-config) WRITE_WORKSPACE_CONFIG=1; shift ;;
    --legacy) LEGACY_PROTOCOL=1; shift ;;
    -h|--help)
      echo "Usage: ./setup_premiere_bridge.sh [--install] [--check-only] [--write-workspace-config] [--legacy] [--ide all|antigravity|cursor|vscode|windsurf|claude-code|claude-desktop|codex]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=== Step 0: Runtime & Premiere MCP Server ==="
if ! command -v node >/dev/null 2>&1; then
  echo "[FAIL] Node.js not found. Install Node.js 20.19+ from https://nodejs.org/"
  exit 1
fi
NODE_EXE="$(command -v node)"
echo "[OK] Node.js $(node -v) ($NODE_EXE)"

if [[ "$INSTALL" -eq 1 && "$CHECK_ONLY" -eq 0 ]]; then
  npm install -g premiere-pro-mcp
fi

if ! command -v premiere-pro-mcp >/dev/null 2>&1; then
  echo "[FAIL] premiere-pro-mcp not found. Re-run with: ./setup_premiere_bridge.sh --install"
  exit 1
fi

NPM_ROOT="$(npm root -g)"
ENTRY_JS="${NPM_ROOT}/premiere-pro-mcp/dist/index.js"
echo "[OK] premiere-pro-mcp v$(premiere-pro-mcp --version) (${ENTRY_JS})"

if [[ "$INSTALL" -eq 1 && "$CHECK_ONLY" -eq 0 ]]; then
  echo "[INFO] Installing Premiere Pro CEP Bridge (--install-cep)..."
  premiere-pro-mcp --install-cep
fi

echo "=== CEP Bridge & Doctor Checks ==="
premiere-pro-mcp --diagnose-cep || true
premiere-pro-mcp --doctor || true

echo "=== Live Bridge Heartbeat ==="
BRIDGE_DIR="${PREMIERE_TEMP_DIR:-${TMPDIR:-/tmp}/premiere-mcp-bridge}"
HB_FILE="${BRIDGE_DIR%/}/bridge-heartbeat.json"
echo "Bridge Temp Dir: $BRIDGE_DIR"
if [[ -f "$HB_FILE" ]]; then
  echo "[OK] Heartbeat file present: $HB_FILE"
else
  echo "[WAIT] Heartbeat not found. Open Premiere Pro -> Window > Extensions > MCP for Adobe Premiere Pro."
fi

ENV_JSON="\"PREMIERE_MCP_TELEMETRY\": \"0\""
if [[ "$LEGACY_PROTOCOL" -eq 1 ]]; then
  ENV_JSON="${ENV_JSON}, \"PREMIERE_MCP_PROTOCOL_MODE\": \"legacy\""
fi

MCP_SERVERS_JSON=$(cat <<EOF
{
  "mcpServers": {
    "premiere-pro": {
      "command": "${NODE_EXE}",
      "args": ["${ENTRY_JS}"],
      "env": { ${ENV_JSON} }
    }
  }
}
EOF
)

VSCODE_JSON=$(cat <<EOF
{
  "servers": {
    "premiere-pro": {
      "type": "stdio",
      "command": "${NODE_EXE}",
      "args": ["${ENTRY_JS}"],
      "env": { ${ENV_JSON} }
    }
  }
}
EOF
)

echo "=== IDE MCP Configurations (${IDE}) ==="
if [[ "$IDE" =~ ^(all|antigravity|cursor|windsurf|claude-code|claude-desktop)$ ]]; then
  echo "--- Universal mcpServers JSON ---"
  echo "$MCP_SERVERS_JSON"
fi
if [[ "$IDE" =~ ^(all|vscode)$ ]]; then
  echo "--- VS Code / Copilot (.vscode/mcp.json) ---"
  echo "$VSCODE_JSON"
fi
if [[ "$IDE" =~ ^(all|codex)$ ]]; then
  echo "--- OpenAI Codex (~/.codex/config.toml) ---"
  printf '[mcp_servers.premiere-pro]\ncommand = "%s"\nargs = ["%s"]\n' "$NODE_EXE" "$ENTRY_JS"
fi

if [[ "$WRITE_WORKSPACE_CONFIG" -eq 1 && "$CHECK_ONLY" -eq 0 ]]; then
  if [[ "$IDE" =~ ^(all|claude-code|antigravity)$ ]]; then
    printf '%s\n' "$MCP_SERVERS_JSON" > .mcp.json
    echo "[OK] Wrote .mcp.json"
  fi
  if [[ "$IDE" =~ ^(all|cursor)$ ]]; then
    mkdir -p .cursor
    printf '%s\n' "$MCP_SERVERS_JSON" > .cursor/mcp.json
    echo "[OK] Wrote .cursor/mcp.json"
  fi
  if [[ "$IDE" =~ ^(all|vscode)$ ]]; then
    mkdir -p .vscode
    printf '%s\n' "$VSCODE_JSON" > .vscode/mcp.json
    echo "[OK] Wrote .vscode/mcp.json"
  fi
fi

echo ""
echo "Next inside Premiere Pro: open a project/sequence, check Window > Extensions > MCP for Adobe Premiere Pro, reload your IDE MCP servers, and ask your assistant to run ping + verify_premiere_connection."

