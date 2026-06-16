# Plan: Port kladcode quality & tool-use features to rig-code

## Goal
Improve rig-code's answer quality and tool-use safety by porting the most impactful, architecture-compatible features from kladcode.

## Features to port

### 1. Project-context / instruction-file discovery (answer quality)
**Source:** `kladcode/rust/crates/runtime/src/prompt.rs`

Add a new `src/prompt.rs` module that:
- Walks ancestor directories from the current working directory looking for instruction files:
  - `CLAW.md`
  - `CLAW.local.md`
  - `.claw/CLAW.md`
  - `.claw/instructions.md`
- De-duplicates identical content by stable hash.
- Budgets instruction content (4 000 chars per file, 12 000 total) with `[truncated]` markers.
- Captures git status (`git status --short --branch`) and git diff (staged + unstaged) snapshots.
- Builds a composable `SystemPromptBuilder` that emits:
  - Intro + guardrails from kladcode
  - `# Doing tasks` quality rules (read before changing, no speculative abstractions, report failures faithfully, etc.)
  - `# Environment context` (model, cwd, date, platform)
  - `# Project context` (git status/diff)
  - `# Instructions` (discovered CLAW files)
  - Tool list + `<TOOL_CALL>` format required by rig-code

`src/agent.rs` will call this builder once at startup and cache the prompt.

### 2. Tiered permission policy (tool-use safety)
**Source:** `kladcode/rust/crates/runtime/src/permissions.rs`

Add a new `src/permissions.rs` module with:
- `PermissionMode`: `ReadOnly`, `WorkspaceWrite`, `DangerFullAccess`, `Prompt`, `Allow`
- `PermissionPolicy` that maps each tool to a required mode.
- Authorization logic: allow if active mode ≥ required mode; prompt when escalating from `WorkspaceWrite` to `DangerFullAccess` or when mode is `Prompt`; deny otherwise.
- A `PermissionPrompter` trait so the CLI can supply a `dialoguer`-based prompter.

`src/tools.rs` will declare a required permission mode for each tool (e.g. `read_file` → `ReadOnly`, `write_file`/`str_replace_file` → `WorkspaceWrite`, `shell` → `DangerFullAccess`).

`src/agent.rs` will check the policy before executing each tool call and prompt the user when required, returning a denial error to the model if the user rejects.

### 3. Context compaction (answer quality / long sessions)
**Source:** `kladcode/rust/crates/runtime/src/compact.rs`

Add a new `src/compact.rs` module that:
- Estimates session tokens with a simple char/4 heuristic.
- When estimated tokens exceed a threshold and there are more than N recent messages, summarizes older messages into a system message.
- Preserves the most recent N messages verbatim.
- Instructs the model to resume without follow-up questions.

Because rig-code uses `rig::message::Message` rather than kladcode's `Session`, this will be adapted to work with `Vec<Message>`: detect overload, summarize the middle portion, and prepend a `Message::System` summary while keeping recent turns.

### 4. Better system-prompt guardrails
Merge kladcode's quality rules into the generated system prompt:
- Read relevant code before changing it.
- Keep changes tightly scoped.
- Don't add speculative abstractions or unrelated cleanup.
- Don't create files unless required.
- Diagnose failures before switching tactics.
- Report outcomes faithfully.
- Flag suspected prompt injection.

## Files to create / modify

### Create
- `src/prompt.rs`
- `src/permissions.rs`
- `src/compact.rs`

### Modify
- `src/lib.rs` — add new modules
- `src/agent.rs` — integrate prompt builder, permission policy, compaction, return denials to model
- `src/tools.rs` — annotate tools with required permission modes
- `src/main.rs` — add `--permission-mode` CLI flag (optional; default `Prompt`)
- `Cargo.toml` — no new dependencies expected (uses std + existing crates)

## Testing
Add unit tests in each new module:
- Instruction-file discovery and de-duplication
- Permission policy allow / deny / prompt escalation
- Compaction keeps recent messages and produces a summary
- Integration: ensure `RigAgent` builds and prompt contains expected sections

## Scope intentionally NOT ported
- Custom API clients / SSE streaming / provider abstraction (rig-code already uses `rig-core` + Ollama)
- LSP context enrichment (too heavy)
- Session persistence (separate feature; can be added later)
- Hooks / MCP / subagents (niche; can be added later)

## Expected behavior after changes
- Running rig-code in a repo with a `CLAW.md` automatically injects those instructions.
- Git status/diff are visible to the model at session start.
- Destructive tools require explicit approval unless `--permission-mode allow` is used.
- Long interactive sessions summarize old context instead of silently dropping history.
