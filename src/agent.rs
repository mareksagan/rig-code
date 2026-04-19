use anyhow::Result;
use colored::Colorize;
use futures::future::join_all;
use rig::client::{CompletionClient, Nothing};
use rig::completion::{Completion, CompletionResponse};
use rig::message::{
    AssistantContent, Message, ToolCall as RigToolCall, ToolResult,
    ToolResultContent, UserContent,
};
use rig::providers::ollama;
use rig::tool::Tool;
use rig::OneOrMany;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::tools::*;

/// Maximum turns in the agent loop before giving up.
const MAX_TURNS: usize = 25;
/// Target max messages in history before trimming.
const MAX_HISTORY_MESSAGES: usize = 28;
/// Very long tool results are truncated to keep context healthy.
const MAX_TOOL_RESULT_LEN: usize = 6000;
/// Very long shell output is truncated.
const MAX_SHELL_OUTPUT_LEN: usize = 8000;

pub struct RigAgent {
    pub todo_state: TodoState,
    pub in_plan_mode: Arc<Mutex<bool>>,
    pub model: String,
    /// Detected at runtime: whether the model supports native tool calling.
    pub supports_native_tools: Arc<AtomicBool>,
}

impl RigAgent {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            todo_state: TodoState::default(),
            in_plan_mode: Arc::new(Mutex::new(false)),
            model: model.into(),
            supports_native_tools: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Build the rig Agent with all tools registered natively.
    fn build_rig_agent_with_tools(&self) -> rig::agent::Agent<ollama::CompletionModel> {
        let client = ollama::Client::new(Nothing).expect("Failed to create Ollama client");

        client
            .agent(&self.model)
            .preamble(&build_system_prompt())
            .max_tokens(4096)
            .temperature(0.2)
            .tool(Shell)
            .tool(ReadFile)
            .tool(WriteFile)
            .tool(StrReplaceFile)
            .tool(GlobTool)
            .tool(GrepTool)
            .tool(SearchWeb)
            .tool(FetchURL)
            .tool(TodoListTool {
                state: self.todo_state.clone(),
            })
            .tool(AskUser)
            .tool(PlanModeTool {
                in_plan_mode: self.in_plan_mode.clone(),
            })
            .build()
    }

    /// Build the rig Agent WITHOUT native tools (text-only fallback).
    fn build_rig_agent_text_only(&self) -> rig::agent::Agent<ollama::CompletionModel> {
        let client = ollama::Client::new(Nothing).expect("Failed to create Ollama client");

        client
            .agent(&self.model)
            .preamble(&build_system_prompt())
            .max_tokens(4096)
            .temperature(0.2)
            .build()
    }

    pub async fn run_interactive(&self) -> Result<()> {
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════╗"
                .bright_blue()
                .bold()
        );
        println!(
            "{}",
            "║        🚀 Rig Code CLI — Powered by Ollama + rig         ║"
                .bright_blue()
                .bold()
        );
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════╝"
                .bright_blue()
                .bold()
        );
        println!(
            "Model: {} | Type {} to exit\n",
            self.model.cyan(),
            "'exit'".dimmed()
        );

        let mut history: Vec<Message> = Vec::new();

        loop {
            let input = dialoguer::Input::<String>::new()
                .with_prompt("You")
                .allow_empty(false)
                .interact_text()?;

            if input.trim().eq_ignore_ascii_case("exit")
                || input.trim().eq_ignore_ascii_case("quit")
            {
                println!("{}", "Goodbye! 👋".green());
                break;
            }

            let spinner = indicatif::ProgressBar::new_spinner();
            spinner.set_style(
                indicatif::ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .unwrap(),
            );
            spinner.set_message("Thinking...");
            spinner.enable_steady_tick(std::time::Duration::from_millis(100));

            let result = self
                .run_agent_loop(&input, &mut history, true, &spinner)
                .await;

            spinner.finish_and_clear();

            match result {
                Ok(output) => {
                    println!("{} {}\n", "Rig:".bright_green().bold(), output);
                }
                Err(e) => {
                    eprintln!("{} {}", "Error:".red().bold(), e);
                }
            }
        }

        Ok(())
    }

    pub async fn run_once(&self, prompt: &str) -> Result<String> {
        let mut history = Vec::new();
        self.run_agent_loop(prompt, &mut history, false, &indicatif::ProgressBar::hidden())
            .await
    }

    async fn run_agent_loop(
        &self,
        input: &str,
        history: &mut Vec<Message>,
        show_intermediate: bool,
        spinner: &indicatif::ProgressBar,
    ) -> Result<String> {
        let mut current_prompt = Message::user(input);
        let mut executed_calls: HashSet<(String, String)> = HashSet::new();

        for _turn in 0..MAX_TURNS {
            // Build agent with or without native tools depending on what we've detected.
            let rig_agent = if self.supports_native_tools.load(Ordering::SeqCst) {
                self.build_rig_agent_with_tools()
            } else {
                self.build_rig_agent_text_only()
            };

            // Build and send the completion request.
            let request = rig_agent
                .completion(current_prompt.clone(), history.clone())
                .await?;

            let response: CompletionResponse<ollama::CompletionResponse> = match request.send().await {
                Ok(resp) => resp,
                Err(e) => {
                    let err_str = e.to_string().to_lowercase();
                    if err_str.contains("does not support tools") {
                        println!("{}", "⚠️ Model doesn't support native tools, falling back to text mode.".yellow());
                        self.supports_native_tools.store(false, Ordering::SeqCst);
                        // Retry this turn with text-only agent.
                        continue;
                    }
                    return Err(e.into());
                }
            };

            // Extract text and native tool calls from the model's response.
            let (text, native_calls) = extract_native_tool_calls(&response.choice);

            // In interactive mode, show reasoning before executing tools.
            if show_intermediate && !text.trim().is_empty() && !native_calls.is_empty() {
                spinner.finish_and_clear();
                println!("{}", text.trim());
                println!();
            }

            // Store the assistant's raw response (text + tool calls) in history.
            history.push(current_prompt.clone());
            history.push(Message::Assistant {
                id: None,
                content: response.choice.clone(),
            });

            // Trim history to avoid context overflow.
            trim_history(history);

            if native_calls.is_empty() {
                // Fallback: the model may have embedded tool calls as text.
                let fallback_calls = parse_tool_calls(&text);
                if fallback_calls.is_empty() {
                    return Ok(clean_final_answer(&text));
                }

                // Execute fallback tool calls and continue.
                let tool_results = self
                    .execute_fallback_tools(&fallback_calls, &mut executed_calls)
                    .await;
                current_prompt = build_tool_result_message(tool_results);
                continue;
            }

            // Execute native tool calls in parallel.
            let tool_results = self
                .execute_native_tools(&native_calls, &mut executed_calls)
                .await;

            // If all calls were duplicates, nudge the model to answer.
            if tool_results.is_empty() {
                current_prompt = Message::user(
                    "NOTE: All requested tools were already executed. Provide your final answer based on previous results.",
                );
                continue;
            }

            current_prompt = build_tool_result_message(tool_results);
        }

        anyhow::bail!("Exceeded maximum turns ({}) without final answer", MAX_TURNS)
    }

    /// Execute native rig ToolCalls in parallel. Returns (id, call_id, result) tuples.
    async fn execute_native_tools(
        &self,
        calls: &[RigToolCall],
        executed: &mut HashSet<(String, String)>,
    ) -> Vec<(String, Option<String>, String)> {
        let futures = calls.iter().filter_map(|call| {
            let args_str = call.function.arguments.to_string();
            let key = (call.function.name.clone(), args_str.clone());

            if executed.contains(&key) {
                println!(
                    "{} {} (skipped duplicate)",
                    "↻".yellow(),
                    call.function.name.cyan()
                );
                return None;
            }
            executed.insert(key);

            println!(
                "{} {}({})",
                "🔧".yellow(),
                call.function.name.cyan(),
                args_str.dimmed()
            );

            let name = call.function.name.clone();
            let args = call.function.arguments.clone();
            let id = call.id.clone();
            let call_id = call.call_id.clone();

            Some(async move {
                let result = self.execute_tool_by_name(&name, args).await;
                match result {
                    Ok(res) => {
                        let display = if res.len() > 500 {
                            format!("{}... (truncated)", &res[..500])
                        } else {
                            res.clone()
                        };
                        println!("{} {}", "✓".green(), display.dimmed());
                        (id, call_id, res)
                    }
                    Err(e) => {
                        println!("{} {}", "✗".red(), e.to_string().dimmed());
                        (id, call_id, format!("[ERROR] {}", e))
                    }
                }
            })
        });

        join_all(futures).await
    }

    /// Execute fallback-parsed tool calls.
    async fn execute_fallback_tools(
        &self,
        calls: &[ParsedToolCall],
        executed: &mut HashSet<(String, String)>,
    ) -> Vec<(String, Option<String>, String)> {
        let futures = calls.iter().filter_map(|call| {
            let args_str = call.arguments.to_string();
            let key = (call.name.clone(), args_str.clone());

            if executed.contains(&key) {
                println!("{} {} (skipped duplicate)", "↻".yellow(), call.name.cyan());
                return None;
            }
            executed.insert(key);

            println!(
                "{} {}({})",
                "🔧".yellow(),
                call.name.cyan(),
                args_str.dimmed()
            );

            let name = call.name.clone();
            let args = call.arguments.clone();

            Some(async move {
                let result = self.execute_tool_by_name(&name, args).await;
                match result {
                    Ok(res) => {
                        let display = if res.len() > 500 {
                            format!("{}... (truncated)", &res[..500])
                        } else {
                            res.clone()
                        };
                        println!("{} {}", "✓".green(), display.dimmed());
                        (name, None, res)
                    }
                    Err(e) => {
                        println!("{} {}", "✗".red(), e.to_string().dimmed());
                        (name, None, format!("[ERROR] {}", e))
                    }
                }
            })
        });

        join_all(futures).await
    }

    /// Dispatch a tool call by name and arguments.
    async fn execute_tool_by_name(
        &self,
        name: &str,
        args: Value,
    ) -> Result<String, anyhow::Error> {
        match name {
            "shell" => {
                let a: ShellArgs = serde_json::from_value(args)?;
                let r = Shell.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(truncate_string(r, MAX_SHELL_OUTPUT_LEN))
            }
            "read_file" => {
                let a: ReadFileArgs = serde_json::from_value(args)?;
                let r = ReadFile.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "write_file" => {
                let a: WriteFileArgs = serde_json::from_value(args)?;
                let r = WriteFile.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "str_replace_file" => {
                let a: StrReplaceFileArgs = serde_json::from_value(args)?;
                let r = StrReplaceFile
                    .call(a)
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "glob" => {
                let a: GlobArgs = serde_json::from_value(args)?;
                let r = GlobTool.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "grep" => {
                let a: GrepArgs = serde_json::from_value(args)?;
                let r = GrepTool.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "search_web" => {
                let a: SearchWebArgs = serde_json::from_value(args)?;
                let r = SearchWeb.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "fetch_url" => {
                let a: FetchURLArgs = serde_json::from_value(args)?;
                let r = FetchURL.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "todo_list" => {
                let a: TodoListArgs = serde_json::from_value(args)?;
                let tool = TodoListTool {
                    state: self.todo_state.clone(),
                };
                let r = tool.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "ask_user" => {
                let a: AskUserArgs = serde_json::from_value(args)?;
                let r = AskUser.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            "plan_mode" => {
                let a: PlanModeArgs = serde_json::from_value(args)?;
                let tool = PlanModeTool {
                    in_plan_mode: self.in_plan_mode.clone(),
                };
                let r = tool.call(a).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(r)
            }
            _ => anyhow::bail!("Unknown tool: {}", name),
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────

/// Extract text and native tool calls from the model's response choice.
fn extract_native_tool_calls(
    choice: &OneOrMany<AssistantContent>,
) -> (String, Vec<RigToolCall>) {
    let mut text_parts = Vec::new();
    let mut calls = Vec::new();

    for content in choice.iter() {
        match content {
            AssistantContent::Text(t) => text_parts.push(t.text.clone()),
            AssistantContent::ToolCall(tc) => calls.push(tc.clone()),
            _ => {}
        }
    }

    (text_parts.join("\n"), calls)
}

/// Build a user message containing multiple tool results.
fn build_tool_result_message(
    results: Vec<(String, Option<String>, String)>,
) -> Message {
    let contents: Vec<UserContent> = results
        .into_iter()
        .map(|(id, call_id, result)| {
            let truncated = truncate_string(result, MAX_TOOL_RESULT_LEN);
            UserContent::ToolResult(ToolResult {
                id,
                call_id,
                content: OneOrMany::one(ToolResultContent::text(truncated)),
            })
        })
        .collect();

    Message::User {
        content: OneOrMany::many(contents).expect("at least one tool result"),
    }
}

/// Trim history to keep context window healthy. Keeps the most recent messages,
/// dropping older ones from the middle.
fn trim_history(history: &mut Vec<Message>) {
    if history.len() <= MAX_HISTORY_MESSAGES {
        return;
    }
    let skip = history.len() - MAX_HISTORY_MESSAGES;
    history.drain(0..skip);
}

fn truncate_string(s: String, max: usize) -> String {
    if s.len() <= max {
        s
    } else {
        format!("{}\n\n[... truncated {} chars ...]", &s[..max], s.len() - max)
    }
}

// ──────────────────────────────────────────────────────────────
// Fallback text-based tool call parser (for robustness)
// ──────────────────────────────────────────────────────────────

#[derive(Debug)]
struct ParsedToolCall {
    name: String,
    arguments: Value,
}

fn parse_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let known_tools: HashSet<&str> = [
        "shell",
        "read_file",
        "write_file",
        "str_replace_file",
        "glob",
        "grep",
        "search_web",
        "fetch_url",
        "todo_list",
        "ask_user",
        "plan_mode",
    ]
    .iter()
    .cloned()
    .collect();

    // Strategy 1: Extract JSON objects and match by signature
    let json_re = regex::Regex::new(r"(?s:\{[^{}]*?\}(?:\s*\{[^{}]*?\})*?)").unwrap();

    for cap in json_re.captures_iter(text) {
        let json_block = &cap[0];
        if let Ok(val) = serde_json::from_str::<Value>(json_block.trim()) {
            if let Some(call) = try_match_tool_call(&val, &known_tools) {
                calls.push(call);
                continue;
            }
        }
        let objects: Vec<Value> = json_re
            .captures_iter(json_block)
            .filter_map(|c| serde_json::from_str::<Value>(&c[0]).ok())
            .collect();
        if objects.len() >= 2 {
            let name = objects[0]
                .get("name")
                .or_else(|| objects[0].get("tool"))
                .or_else(|| objects[0].get("command"))
                .and_then(|v| v.as_str());
            if let Some(name) = name {
                if known_tools.contains(name) {
                    let args = if objects[1].get("arguments").is_some() {
                        objects[1].get("arguments").cloned().unwrap_or(Value::Null)
                    } else {
                        objects[1].clone()
                    };
                    calls.push(ParsedToolCall {
                        name: name.to_string(),
                        arguments: args,
                    });
                }
            }
        }
    }

    if !calls.is_empty() {
        return calls;
    }

    // Strategy 2: standard <TOOL_CALL> format
    let re = regex::Regex::new(
        r"<?\s*TOOL_CALL\s*>\s*(\{.*?\})\s*<\s*/\s*TOOL_CALL\s*>",
    )
    .unwrap();
    for cap in re.captures_iter(text) {
        let json_str = &cap[1];
        if let Ok(val) = serde_json::from_str::<Value>(json_str) {
            if let Some(call) = try_match_tool_call(&val, &known_tools) {
                calls.push(call);
            }
        }
    }

    // Strategy 3: <tool_name>{args}</tool_name>
    if calls.is_empty() {
        for tool_name in &known_tools {
            let pattern = format!(
                r"<{}>\s*(\{{.*?\}})\s*</{}>",
                regex::escape(tool_name),
                regex::escape(tool_name)
            );
            if let Ok(re) = regex::Regex::new(&pattern) {
                for cap in re.captures_iter(text) {
                    let json_str = &cap[1];
                    if let Ok(args) = serde_json::from_str::<Value>(json_str) {
                        calls.push(ParsedToolCall {
                            name: tool_name.to_string(),
                            arguments: args,
                        });
                    }
                }
            }
        }
    }

    calls
}

fn try_match_tool_call(
    val: &Value,
    known_tools: &HashSet<&str>,
) -> Option<ParsedToolCall> {
    if let Some(name) = val.get("name").and_then(|v| v.as_str()) {
        if known_tools.contains(name) {
            let args = val.get("arguments").cloned().unwrap_or(Value::Null);
            return Some(ParsedToolCall {
                name: name.to_string(),
                arguments: args,
            });
        }
    }

    // Field-signature matching
    if val.get("command").is_some() && val.get("command").and_then(|v| v.as_str()).is_some() {
        return Some(ParsedToolCall {
            name: "shell".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("path").is_some() && val.get("content").is_some() {
        return Some(ParsedToolCall {
            name: "write_file".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("path").is_some() && val.get("old").is_some() && val.get("new").is_some() {
        return Some(ParsedToolCall {
            name: "str_replace_file".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("path").is_some() && val.get("line_offset").is_some() {
        return Some(ParsedToolCall {
            name: "read_file".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("path").is_some() && val.get("n_lines").is_some() {
        return Some(ParsedToolCall {
            name: "read_file".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("pattern").is_some() && val.get("path").is_none() && val.get("glob").is_none() {
        if val.get("limit").is_some() || val.get("query").is_some() {
            return Some(ParsedToolCall {
                name: "search_web".to_string(),
                arguments: val.clone(),
            });
        }
        return Some(ParsedToolCall {
            name: "glob".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("pattern").is_some() && (val.get("path").is_some() || val.get("glob").is_some()) {
        return Some(ParsedToolCall {
            name: "grep".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("url").is_some() {
        return Some(ParsedToolCall {
            name: "fetch_url".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("action").is_some() && val.get("todos").is_some() {
        return Some(ParsedToolCall {
            name: "todo_list".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("question").is_some() {
        return Some(ParsedToolCall {
            name: "ask_user".to_string(),
            arguments: val.clone(),
        });
    }
    if val.get("action").is_some() && val.get("plan").is_some() {
        return Some(ParsedToolCall {
            name: "plan_mode".to_string(),
            arguments: val.clone(),
        });
    }

    None
}

fn clean_final_answer(text: &str) -> String {
    let mut cleaned = text.trim().to_string();

    let re = regex::Regex::new(r"<TOOL_CALL>.*?</TOOL_CALL>").unwrap();
    cleaned = re.replace_all(&cleaned, "").to_string();

    let followups = [
        "what would you like to do next",
        "what else can i help you with",
        "how can i assist you further",
        "let me know if you need anything else",
        "is there anything else you'd like me to do",
        "let me know what's next",
        "anything else",
    ];
    let lowered = cleaned.to_lowercase();
    for phrase in &followups {
        if let Some(idx) = lowered.find(phrase) {
            cleaned = cleaned[..idx].trim().to_string();
        }
    }

    cleaned.trim().to_string()
}

// ──────────────────────────────────────────────────────────────
// System prompt
// ──────────────────────────────────────────────────────────────

fn build_system_prompt() -> String {
    format!(
        r#"You are Rig Code CLI, an expert software engineering AI agent running on the user's local machine.

CORE BEHAVIOR:
1. You have access to tools. USE them to gather information and make changes. Do not just describe — DO.
2. When you need to use a tool, the system will handle it automatically if you output the correct native tool call format.
3. You may call multiple tools in parallel when they are independent (e.g., read two different files at once).
4. NEVER repeat the same tool call. If you already received results, use them to answer.
5. After receiving tool results, give your final answer concisely. Do NOT ask follow-up questions.
6. Destructive operations may require user confirmation in interactive mode.
7. If a tool returns an error, report it truthfully. Do NOT claim success for failed operations.
8. When searching code, ignore `target/`, `node_modules/`, `.git/`, and build directories.
9. When using read_file to find text for str_replace_file, read enough lines (50-100) to locate the exact string.
10. Prefer `str_replace_file` over `write_file` for edits to avoid losing unrelated changes.

TOOLS:
shell: Execute bash commands. Args: {{"command": "string", "description": "string", "timeout_seconds": integer}}
read_file: Read a file. Args: {{"path": "string", "line_offset": integer, "n_lines": integer}}
write_file: Write/overwrite a file. Args: {{"path": "string", "content": "string", "append": boolean}}
str_replace_file: Precise text replacement. Args: {{"path": "string", "old": "string", "new": "string"}}
glob: Find files by pattern. Args: {{"pattern": "string"}}
grep: Search file contents with regex. Args: {{"pattern": "string", "path": "string", "glob": "string"}}
search_web: DuckDuckGo search. Args: {{"query": "string", "limit": integer}}
fetch_url: Fetch URL text. Args: {{"url": "string"}}
todo_list: Manage todos. Args: {{"action": "set|query|done", "todos": ["string"], "index": integer}}
ask_user: Ask the user. Args: {{"question": "string", "options": ["string"]}}
plan_mode: Enter/exit plan mode. Args: {{"action": "enter|exit", "plan": "string"}}

WORKING DIRECTORY: {}
"#,
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string())
    )
}
