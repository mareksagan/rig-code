# Rig Code CLI

A local-first, autonomous AI coding agent built in Rust. Powered by [Ollama](https://ollama.com) and [rig](https://github.com/0xPlaygrounds/rig) — no API keys, no cloud, no telemetry.

![Rust](https://img.shields.io/badge/rust-2024-orange?logo=rust)
![Ollama](https://img.shields.io/badge/ollama-local-blue)
![License](https://img.shields.io/badge/license-MIT-green)

```
╔══════════════════════════════════════════════════════════╗
║        🚀 Rig Code CLI — Powered by Ollama + rig         ║
╚══════════════════════════════════════════════════════════╝
Model: qwen2.5:3b | Type 'exit' to exit

You: Add a healthcheck endpoint to the server
Rig: 🔧 shell({"command": "find src -name '*.rs' | head -20"})
    ✓ src/main.rs
      src/routes/mod.rs
      ...
    🔧 read_file({"path": "src/routes/mod.rs"})
    ✓ ...
    🔧 str_replace_file({"path": "src/routes/mod.rs", ...})
    ✓ Replaced 1 occurrence in src/routes/mod.rs

Done. Added a `/health` GET endpoint in `src/routes/mod.rs` that
returns `{"status":"ok"}` with HTTP 200.
```

## Features

- **🔒 Fully Local** — Runs any Ollama model on your machine. No data leaves your computer.
- **🔧 11 Built-in Tools** — Shell, file read/write, grep, glob, web search, URL fetching, todo list, user questions, and plan mode.
- **⚡ Parallel Execution** — When the model requests multiple independent tools, they run concurrently.
- **🎯 Native Tool Calling** — Uses rig's native tool schemas for models that support it (e.g. `qwen2.5`, `mistral`). Automatically falls back to text-based parsing for models that don't (e.g. `llama3`).
- **🛡️ Safe by Default** — Destructive commands (`rm`, `mv`, overwrite) require confirmation. Use `--auto-approve` only if you know what you're doing.
- **💬 Interactive & One-Shot** — Chat loop for exploration, or single `-p` prompts for automation.
- **🧠 Smart Context** — Conversation history is automatically trimmed to stay within the model's context window.

## Requirements

- [Rust](https://rustup.rs) (2024 edition)
- [Ollama](https://ollama.com/download) running locally

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/rig-code.git
cd rig-code

# 2. Build
cargo build --release

# 3. Pull a model (default is qwen2.5:3b)
ollama pull qwen2.5:3b

# 4. Run!
./target/release/rig-code
```

## Usage

### Interactive mode

```bash
./target/release/rig-code
```

Type prompts naturally. The agent will use tools to explore, edit, and answer. Type `exit` or `quit` to leave.

### One-shot mode

```bash
./target/release/rig-code -p "List all TODO comments in src/"
```

### Use a different model

```bash
# Models with native tool support (recommended)
./target/release/rig-code -m qwen2.5:7b
./target/release/rig-code -m mistral

# Models without native tool support (uses text fallback)
./target/release/rig-code -m llama3:latest
./target/release/rig-code -m phi3:medium
```

### Auto-approve destructive operations

```bash
./target/release/rig-code --auto-approve -p "Delete all .bak files"
```

> ⚠️ Only use `--auto-approve` in CI/automation or when you fully trust the model's output.

## Available Tools

| Tool | Description |
|------|-------------|
| `shell` | Execute bash commands with optional timeout. Confirms destructive ops. |
| `read_file` | Read text files with line numbers and pagination. |
| `write_file` | Write or append to files. Confirms overwrites. |
| `str_replace_file` | Precise string replacement. Warns on multiple matches. |
| `glob` | Find files matching a glob pattern. |
| `grep` | Search file contents with regex across directories. |
| `search_web` | Web search via DuckDuckGo. |
| `fetch_url` | Fetch and extract text content from URLs. |
| `todo_list` | Track task progress across multi-turn sessions. |
| `ask_user` | Ask clarifying questions with optional predefined answers. |
| `plan_mode` | Enter/exit planning mode where the agent writes a plan before executing. |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User CLI  │────▶│  RigAgent    │────▶│   Ollama    │
│  (clap)     │     │  (rig-core)  │     │  (local LLM)│
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                    ┌───────┴───────┐
                    ▼               ▼
              ┌─────────┐    ┌──────────┐
              │  Shell  │    │ File Ops │
              │  Grep   │    │ Web/Fetch│
              └─────────┘    └──────────┘
```

- **Agent loop**: Multi-turn reasoning with up to 25 turns. Extracts native `ToolCall` items when available, falls back to robust regex-based JSON extraction for text-only models.
- **Parallel execution**: Independent tool calls in the same turn are executed concurrently via `futures::join_all`.
- **Context management**: History is trimmed to the 28 most recent messages. Tool results over 6,000 chars and shell output over 8,000 chars are summarized to prevent context overflow.
- **Deduplication**: Identical tool calls (same name + arguments) within a session are skipped automatically.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_API_BASE_URL` | Ollama API URL (default: `http://localhost:11434`) |
| `RIG_CODE_AUTO_APPROVE` | Auto-approve destructive operations when set to any value |
| `RUST_LOG` | Logging level, e.g. `RUST_LOG=info` |

## Development

```bash
# Check
cargo check

# Run tests
cargo test

# Format
cargo fmt

# Lint
cargo clippy
```

## License

MIT
