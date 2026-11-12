<#
.SYNOPSIS
    Step 0 setup and diagnostic script to bridge Adobe Premiere Pro MCP to any IDE.
.DESCRIPTION
    Checks Node.js (>= 20.19), installs/verifies premiere-pro-mcp and the Adobe CEP
    bridge plugin, runs diagnostics (--diagnose-cep, --doctor), checks the live
    Premiere heartbeat file, and prints or writes ready-to-use MCP configs for
    Antigravity, Cursor, VS Code (Copilot), Windsurf, Claude Code, Claude Desktop,
    Codex, or any generic stdio MCP client.
#>
[CmdletBinding()]
param(
    [ValidateSet("all", "antigravity", "cursor", "vscode", "windsurf", "claude-code", "claude-desktop", "codex")]
    [string]$Ide = "all",
    [switch]$Install,
    [switch]$CheckOnly,
    [switch]$WriteWorkspaceConfig,
    [switch]$LegacyProtocol
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
}

function Test-NodeVersion {
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeCmd) {
        Write-Host "[FAIL] Node.js not found. Install Node.js 20.19+ from https://nodejs.org/" -ForegroundColor Red
        return $null
    }
    $rawVer = (& node -v).Trim().TrimStart("v")
    $parts = $rawVer.Split(".")
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    $ok = ($major -gt 20) -or ($major -eq 20 -and $minor -ge 19)
    $status = if ($ok) { "[OK]" } else { "[WARN]" }
    $color = if ($ok) { "Green" } else { "Yellow" }
    Write-Host "$status Node.js v$rawVer ($($nodeCmd.Source)) — minimum required: 20.19+" -ForegroundColor $color
    return $nodeCmd.Source
}

function Get-NpmExecutable {
    $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($npmCmd) { return $npmCmd.Source }
    $npmAny = Get-Command npm -ErrorAction SilentlyContinue
    if ($npmAny) { return $npmAny.Source }
    return $null
}

function Resolve-McpEntrypoint {
    $mcpCmd = Get-Command premiere-pro-mcp -ErrorAction SilentlyContinue
    if ($mcpCmd -and $mcpCmd.Source) {
        $shimDir = Split-Path $mcpCmd.Source -Parent
        if ($shimDir) {
            $localEntry = Join-Path $shimDir "node_modules\premiere-pro-mcp\dist\index.js"
            if (Test-Path $localEntry) {
                return $localEntry
            }
        }
    }
    $npmExe = Get-NpmExecutable
    if ($npmExe) {
        $npmRoot = (& $npmExe root -g | Out-String).Trim()
        if (-not [string]::IsNullOrWhiteSpace($npmRoot)) {
            $entry = Join-Path $npmRoot "premiere-pro-mcp\dist\index.js"
            if (Test-Path $entry) {
                return $entry
            }
        }
    }
    return $null
}

function Ensure-PremiereMcpInstalled {
    $entry = Resolve-McpEntrypoint
    if ($entry -and -not $Install) {
        $ver = (& premiere-pro-mcp --version | Out-String).Trim()
        Write-Host "[OK] premiere-pro-mcp v$ver installed at $entry" -ForegroundColor Green
        return $entry
    }
    if ($CheckOnly) {
        Write-Host "[FAIL] premiere-pro-mcp is not installed globally. Run: npm install -g premiere-pro-mcp" -ForegroundColor Red
        return $null
    }
    $npmExe = Get-NpmExecutable
    if (-not $npmExe) {
        throw "npm not found on PATH. Install Node.js 20.19+ from https://nodejs.org/"
    }
    Write-Host "[INFO] Installing/updating premiere-pro-mcp globally via npm..." -ForegroundColor Yellow
    & $npmExe install -g premiere-pro-mcp
    $entry = Resolve-McpEntrypoint
    if (-not $entry) {
        throw "premiere-pro-mcp installation completed, but dist/index.js was not found under global npm root."
    }
    $ver = (& premiere-pro-mcp --version | Out-String).Trim()
    Write-Host "[OK] premiere-pro-mcp v$ver ready ($entry)" -ForegroundColor Green
    return $entry
}

function Install-And-CheckCep {
    if ($Install -and -not $CheckOnly) {
        $premiereRunning = Get-Process -Name "Adobe Premiere Pro*" -ErrorAction SilentlyContinue
        if ($premiereRunning) {
            Write-Host "[WARN] Adobe Premiere Pro is currently running. Close Premiere before installing/updating the CEP extension if any file lock occurs." -ForegroundColor Yellow
        }
        Write-Host "[INFO] Installing Premiere Pro CEP Bridge (--install-cep)..." -ForegroundColor Yellow
        & premiere-pro-mcp --install-cep
    }

    Write-Host "`n[INFO] Running premiere-pro-mcp --diagnose-cep..." -ForegroundColor Gray
    & premiere-pro-mcp --diagnose-cep

    Write-Host "`n[INFO] Running premiere-pro-mcp --doctor..." -ForegroundColor Gray
    & premiere-pro-mcp --doctor
}

function Test-BridgeHeartbeat {
    $tempBase = if (-not [string]::IsNullOrWhiteSpace($env:TEMP)) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
    $bridgeDir = if (-not [string]::IsNullOrWhiteSpace($env:PREMIERE_TEMP_DIR)) { $env:PREMIERE_TEMP_DIR } else { Join-Path $tempBase "premiere-mcp-bridge" }
    $hbPath = Join-Path $bridgeDir "bridge-heartbeat.json"
    Write-Host "Bridge Temp Dir : $bridgeDir"
    if (-not (Test-Path $hbPath)) {
        Write-Host "[WAIT] Heartbeat not found ($hbPath). Open Premiere Pro -> Window > Extensions > MCP for Adobe Premiere Pro." -ForegroundColor Yellow
        return
    }
    $item = Get-Item $hbPath
    $ageSec = [math]::Round(((Get-Date) - $item.LastWriteTime).TotalSeconds, 1)
    if ($ageSec -le 20) {
        Write-Host "[OK] Bridge heartbeat is LIVE (updated ${ageSec}s ago)." -ForegroundColor Green
    } else {
        Write-Host "[WARN] Bridge heartbeat is STALE (${ageSec}s old). Open/refresh Window > Extensions > MCP for Adobe Premiere Pro in Premiere." -ForegroundColor Yellow
    }
}

function Test-PipelineDependencies {
    foreach ($tool in @("ffmpeg", "ffprobe", "python", "tesseract")) {
        $cmd = Get-Command $tool -ErrorAction SilentlyContinue
        if ($cmd) {
            Write-Host "[OK] $tool -> $($cmd.Source)" -ForegroundColor Green
        } elseif ($tool -eq "tesseract" -and (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe")) {
            Write-Host "[OK] tesseract -> C:\Program Files\Tesseract-OCR\tesseract.exe" -ForegroundColor Green
        } else {
            Write-Host "[WARN] $tool not found on PATH (needed for perception/render stages)." -ForegroundColor Yellow
        }
    }
}

function Build-ServerObject {
    param([string]$NodeExe, [string]$EntryJs)
    $envMap = [ordered]@{ "PREMIERE_MCP_TELEMETRY" = "0" }
    if ($LegacyProtocol) {
        $envMap["PREMIERE_MCP_PROTOCOL_MODE"] = "legacy"
    }
    if ($NodeExe -and $EntryJs) {
        return [ordered]@{
            command = $NodeExe
            args    = @($EntryJs)
            env     = $envMap
        }
    }
    return [ordered]@{
        command = "premiere-pro-mcp"
        env     = $envMap
    }
}

function Show-ClientConfigs {
    param([string]$NodeExe, [string]$EntryJs)
    $serverObj = Build-ServerObject -NodeExe $NodeExe -EntryJs $EntryJs

    $mcpServersJson = ([ordered]@{
        mcpServers = [ordered]@{
            "premiere-pro" = $serverObj
        }
    } | ConvertTo-Json -Depth 6)

    $vscodeObj = [ordered]@{
        type    = "stdio"
        command = $serverObj.command
        env     = $serverObj.env
    }
    if ($serverObj.Contains("args")) {
        $vscodeObj["args"] = $serverObj.args
    }
    $vscodeJson = ([ordered]@{
        servers = [ordered]@{
            "premiere-pro" = $vscodeObj
        }
    } | ConvertTo-Json -Depth 6)

    if ($Ide -in @("all", "antigravity", "cursor", "windsurf", "claude-code", "claude-desktop")) {
        Write-Host "`n--- Universal mcpServers JSON (Antigravity / Cursor / Windsurf / Claude) ---" -ForegroundColor White
        Write-Host "Paths:" -ForegroundColor DarkGray
        Write-Host "  Antigravity   : %USERPROFILE%\.gemini\antigravity\mcp_config.json (or workspace .mcp.json)"
        Write-Host "  Cursor        : .cursor\mcp.json (project) or %USERPROFILE%\.cursor\mcp.json"
        Write-Host "  Windsurf      : %USERPROFILE%\.codeium\windsurf\mcp_config.json"
        Write-Host "  Claude Code   : .mcp.json (project) or 'claude mcp add'"
        Write-Host "  Claude Desktop: %APPDATA%\Claude\claude_desktop_config.json"
        Write-Host $mcpServersJson
    }

    if ($Ide -in @("all", "vscode")) {
        Write-Host "`n--- VS Code / GitHub Copilot (.vscode/mcp.json) ---" -ForegroundColor White
        Write-Host $vscodeJson
    }

    if ($Ide -in @("all", "codex")) {
        Write-Host "`n--- OpenAI Codex CLI (%USERPROFILE%\.codex\config.toml) ---" -ForegroundColor White
        if ($EntryJs) {
            $nEsc = $NodeExe.Replace("\", "\\")
            $eEsc = $EntryJs.Replace("\", "\\")
            Write-Host "[mcp_servers.premiere-pro]"
            Write-Host "command = `"$nEsc`""
            Write-Host "args = [`"$eEsc`"]"
        } else {
            Write-Host "[mcp_servers.premiere-pro]"
            Write-Host "command = `"premiere-pro-mcp`""
        }
    }

    if ($WriteWorkspaceConfig -and -not $CheckOnly) {
        $root = (Get-Location).Path
        if ($Ide -in @("all", "claude-code", "antigravity")) {
            Set-Content -Path (Join-Path $root ".mcp.json") -Value $mcpServersJson -Encoding UTF8
            Write-Host "[OK] Wrote $(Join-Path $root '.mcp.json')" -ForegroundColor Green
        }
        if ($Ide -in @("all", "cursor")) {
            $cursorDir = Join-Path $root ".cursor"
            New-Item -ItemType Directory -Force -Path $cursorDir | Out-Null
            Set-Content -Path (Join-Path $cursorDir "mcp.json") -Value $mcpServersJson -Encoding UTF8
            Write-Host "[OK] Wrote $(Join-Path $cursorDir 'mcp.json')" -ForegroundColor Green
        }
        if ($Ide -in @("all", "vscode")) {
            $vsDir = Join-Path $root ".vscode"
            New-Item -ItemType Directory -Force -Path $vsDir | Out-Null
            Set-Content -Path (Join-Path $vsDir "mcp.json") -Value $vscodeJson -Encoding UTF8
            Write-Host "[OK] Wrote $(Join-Path $vsDir 'mcp.json')" -ForegroundColor Green
        }
    }
}

Write-Section "Step 0: Runtime & Premiere MCP Server"
$nodeExe = Test-NodeVersion
$entryJs = Ensure-PremiereMcpInstalled

if ($entryJs) {
    Write-Section "CEP Bridge & Doctor Checks"
    Install-And-CheckCep
}

Write-Section "Live Bridge Heartbeat"
Test-BridgeHeartbeat

Write-Section "Media & Perception Dependencies"
Test-PipelineDependencies

Write-Section "IDE MCP Configurations"
Show-ClientConfigs -NodeExe $nodeExe -EntryJs $entryJs

Write-Section "Next Step Inside Your IDE"
Write-Host "1. Open Adobe Premiere Pro and open a project + active sequence."
Write-Host "2. Open Window > Extensions > MCP for Adobe Premiere Pro (verify green 'Running')."
Write-Host "3. Restart/reload MCP servers in your IDE and run a read-only check:"
Write-Host "   `"Call ping and verify_premiere_connection. Make no changes.`"" -ForegroundColor Green
