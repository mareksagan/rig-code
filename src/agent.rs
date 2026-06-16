use anyhow::Result;
use async_trait::async_trait;
use colored::Colorize;
use dialoguer::Confirm;
use futures::future::join_all;
use is_terminal::IsTerminal;
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
use std::collections::{HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::compact::{compact_history, CompactionConfig};
use crate::hooks::{merge_hook_feedback, HookConfig, HookRunner};
use crate::lsp::{load_lsp_config, LspManager};
use crate::permissions::{
    PermissionMode, PermissionPolicy, PermissionPromptDecision, PermissionPrompter,
    PermissionRequest,
};
use crate::prompt;
use crate::session::{default_session_path, Session};
use crate::tools::{
    allowed_tools_for_subagent, required_permission_for_tool, AgentTool, AskUser,
    AskUserArgs, FetchURL, FetchURLArgs, GlobArgs, GlobTool, GrepArgs, GrepTool, PlanModeArgs,
    PlanModeTool, ReadFile, ReadFileArgs, SearchWeb, SearchWebArgs, Shell, ShellArgs,
    StrReplaceFile, StrReplaceFileArgs, SubagentSpawner, TodoListArgs, TodoListTool, TodoState,
    WriteFile, WriteFileArgs,
};
use std::path::PathBuf;

/// Interactive prompter used when the permission policy needs user approval.
struct DialoguerPrompter;

impl PermissionPrompter for DialoguerPrompter {
    fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision {
        println!(
            "{} Tool '{}' requires {} permission (current mode: {}).",
            "⚠️".yellow(),
            request.tool_name.cyan(),
            request.required_mode.as_str().yellow(),
            request.current_mode.as_str().dimmed()
        );
        println!("   Input: {}", request.input.dimmed());

        if std::io::stdin().is_terminal() {
            match Confirm::new()
                .with_prompt("Allow this tool call?")
                .default(false)
                .interact()
            {
                Ok(true) => PermissionPromptDecision::Allow,
                _ => PermissionPromptDecision::Deny {
                    reason: "User denied the tool call".to_string(),
                },
            }
        } else {
            PermissionPromptDecision::Deny {
                reason: "Tool requires interactive approval but stdin is not a terminal".to_string(),
            }
        }
    }
}

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
    pub permission_policy: PermissionPolicy,
    pub compaction_config: CompactionConfig,
    pub system_prompt: String,
    pub session_path: PathBuf,
    pub no_session: bool,
    pub hook_runner: HookRunner,
    pub lsp_manager: Option<LspManager>,
    pub allowed_tools: Option<HashSet<String>>,
}

impl RigAgent {
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_mode(model, PermissionMode::Prompt)
    }

    pub fn with_mode(model: impl Into<String>, mode: PermissionMode) -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let system_prompt = prompt::build_system_prompt(&cwd);
        let hook_runner = HookRunner::new(HookConfig::load(&cwd).unwrap_or_default());
        let lsp_manager = load_lsp_config(&cwd)
            .ok()
            .filter(|config| !config.servers.is_empty())
            .map(|config| LspManager::new(&cwd, config.servers));
        Self::with_options(model, mode, system_prompt, default_session_path(), false, hook_runner, lsp_manager, None)
    }

    pub fn with_mode_and_prompt(
        model: impl Into<String>,
        mode: PermissionMode,
        system_prompt: String,
    ) -> Self {
        Self::with_options(model, mode, system_prompt, default_session_path(), false, HookRunner::empty(), None, None)
    }

    pub fn with_options(
        model: impl Into<String>,
        mode: PermissionMode,
        system_prompt: String,
        session_path: PathBuf,
        no_session: bool,
        hook_runner: HookRunner,
        lsp_manager: Option<LspManager>,
        allowed_tools: Option<HashSet<String>>,
    ) -> Self {
        let policy = PermissionPolicy::new(mode)
            .with_tool_requirement("shell", required_permission_for_tool("shell"))
            .with_tool_requirement("read_file", required_permission_for_tool("read_file"))
            .with_tool_requirement("write_file", required_permission_for_tool("write_file"))
            .with_tool_requirement("str_replace_file", required_permission_for_tool("str_replace_file"))
            .with_tool_requirement("glob", required_permission_for_tool("glob"))
            .with_tool_requirement("grep", required_permission_for_tool("grep"))
            .with_tool_requirement("search_web", required_permission_for_tool("search_web"))
            .with_tool_requirement("fetch_url", required_permission_for_tool("fetch_url"))
            .with_tool_requirement("todo_list", required_permission_for_tool("todo_list"))
            .with_tool_requirement("ask_user", required_permission_for_tool("ask_user"))
            .with_tool_requirement("plan_mode", required_permission_for_tool("plan_mode"))
            .with_tool_requirement("agent", required_permission_for_tool("agent"));

        Self {
            todo_state: TodoState::default(),
            in_plan_mode: Arc::new(Mutex::new(false)),
            model: model.into(),
            supports_native_tools: Arc::new(AtomicBool::new(true)),
            permission_policy: policy,
            compaction_config: CompactionConfig::default(),
            system_prompt,
            session_path,
            no_session,
            hook_runner,
            lsp_manager,
            allowed_tools,
        }
    }

    fn is_tool_allowed(&self, name: &str) -> bool {
        self.allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.contains(name))
    }

    /// Refresh LSP context by opening source files and collecting diagnostics.
    pub async fn refresh_lsp_context(&self) {
        let Some(manager) = self.lsp_manager.as_ref() else {
            return;
        };

        let Ok(entries) = tokio::fs::read_dir(".").await else {
            return;
        };

        let mut opened = 0usize;
        let mut entries = entries;
        while let Ok(Some(entry)) = entries.next_entry().await {
            if opened >= 20 {
                break;
            }
            let path = entry.path();
            if path.is_file() {
                manager.open_document(&path).await;
                opened += 1;
            }
        }
    }

    fn system_prompt_with_lsp(&self) -> String {
        let Some(manager) = self.lsp_manager.as_ref() else {
            return self.system_prompt.clone();
        };
        let enrichment = manager.enrichment();
        if enrichment.is_empty() {
            self.system_prompt.clone()
        } else {
            format!("{}\n\n{}", self.system_prompt, enrichment.render_prompt_section())
        }
    }

    /// Load history from the session file if persistence is enabled.
    #[must_use]
    pub fn load_history(&self) -> Vec<Message> {
        if self.no_session {
            return Vec::new();
        }
        match Session::load_from_path(&self.session_path) {
            Ok(session) => session.messages,
            Err(_) => Vec::new(),
        }
    }

    /// Save the current history to the session file if persistence is enabled.
    pub fn save_history(&self, history: &[Message]) {
        if self.no_session {
            return;
        }
        let session = Session {
            version: 1,
            messages: history.to_vec(),
        };
        if let Some(parent) = self.session_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = session.save_to_path(&self.session_path);
    }

    /// Build the rig Agent with all tools registered natively.
    fn build_rig_agent_with_tools(&self) -> rig::agent::Agent<ollama::CompletionModel> {
        let client = ollama::Client::new(Nothing).expect("Failed to create Ollama client");

        let mut builder = client
            .agent(&self.model)
            .preamble(&self.system_prompt_with_lsp())
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
            });

        if self.is_tool_allowed("agent") {
            builder = builder.tool(AgentTool {
                spawner: std::sync::Arc::new(self.clone_for_subagent()),
            });
        }

        builder.build()
    }

    fn clone_for_subagent(&self) -> Self {
        Self {
            todo_state: TodoState::default(),
            in_plan_mode: Arc::new(Mutex::new(false)),
            model: self.model.clone(),
            supports_native_tools: Arc::new(AtomicBool::new(true)),
            permission_policy: PermissionPolicy::new(PermissionMode::DangerFullAccess)
                .with_tool_requirement("shell", required_permission_for_tool("shell"))
                .with_tool_requirement("read_file", required_permission_for_tool("read_file"))
                .with_tool_requirement("write_file", required_permission_for_tool("write_file"))
                .with_tool_requirement("str_replace_file", required_permission_for_tool("str_replace_file"))
                .with_tool_requirement("glob", required_permission_for_tool("glob"))
                .with_tool_requirement("grep", required_permission_for_tool("grep"))
                .with_tool_requirement("search_web", required_permission_for_tool("search_web"))
                .with_tool_requirement("fetch_url", required_permission_for_tool("fetch_url"))
                .with_tool_requirement("todo_list", required_permission_for_tool("todo_list"))
                .with_tool_requirement("ask_user", required_permission_for_tool("ask_user"))
                .with_tool_requirement("plan_mode", required_permission_for_tool("plan_mode")),
            compaction_config: self.compaction_config,
            system_prompt: self.system_prompt.clone(),
            session_path: self.session_path.clone(),
            no_session: true,
            hook_runner: HookRunner::empty(),
            lsp_manager: None,
            allowed_tools: Some(std::collections::HashSet::new()),
        }
    }

    /// Build the rig Agent WITHOUT native tools (text-only fallback).
    fn build_rig_agent_text_only(&self) -> rig::agent::Agent<ollama::CompletionModel> {
        let client = ollama::Client::new(Nothing).expect("Failed to create Ollama client");

        client
            .agent(&self.model)
            .preamble(&self.system_prompt_with_lsp())
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

        let mut history: Vec<Message> = self.load_history();

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
        let mut history = self.load_history();
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
        self.refresh_lsp_context().await;

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

            // Compact old history into a summary, then trim if still too long.
            compact_history(history, self.compaction_config);
            trim_history(history);
            self.save_history(history);

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

                // If all calls were duplicates, nudge the model to answer.
                if tool_results.is_empty() {
                    current_prompt = Message::user(
                        "NOTE: All requested tools were already executed. Provide your final answer based on previous results.",
                    );
                    continue;
                }

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
        if !self.is_tool_allowed(name) {
            return Err(anyhow::anyhow!(
                "tool '{}' is not enabled for this agent",
                name
            ));
        }

        let args_str = args.to_string();
        let mut prompter = DialoguerPrompter;
        match self.permission_policy.authorize(name, &args_str, Some(&mut prompter)) {
            crate::permissions::PermissionOutcome::Allow => {}
            crate::permissions::PermissionOutcome::Deny { reason } => {
                return Err(anyhow::anyhow!("[DENIED] {}", reason));
            }
        }

        let pre = self.hook_runner.run_pre_tool_use(name, &args_str);
        if pre.is_denied() {
            let feedback = merge_hook_feedback(
                pre.messages(),
                String::new(),
                true,
            );
            return Err(anyhow::anyhow!("[DENIED by hook] {}", feedback));
        }

        let (mut output, mut is_error) = match self.run_tool_inner(name, args).await {
            Ok(output) => (output, false),
            Err(error) => (error.to_string(), true),
        };

        output = merge_hook_feedback(pre.messages(), output, false);

        let post = self
            .hook_runner
            .run_post_tool_use(name, &args_str, &output, is_error);
        if post.is_denied() {
            is_error = true;
        }
        output = merge_hook_feedback(post.messages(), output, post.is_denied());

        if is_error {
            Err(anyhow::anyhow!("{}", output))
        } else {
            Ok(output)
        }
    }

    async fn run_tool_inner(&self, name: &str, args: Value) -> Result<String, anyhow::Error> {
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

#[async_trait]
impl SubagentSpawner for RigAgent {
    async fn spawn_subagent(
        &self,
        description: &str,
        prompt: &str,
        subagent_type: &str,
        name: Option<&str>,
    ) -> Result<String, crate::tools::ToolError> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let agent_id = format!("agent-{nanos}");
        let name = name
            .map(|n| n.to_string())
            .unwrap_or_else(|| description.to_string());

        let agents_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join(".rig-code")
            .join("agents");
        let _ = std::fs::create_dir_all(&agents_dir);

        let output_file = agents_dir.join(format!("{agent_id}.md"));
        let manifest_file = agents_dir.join(format!("{agent_id}.json"));

        let output_path = output_file.display().to_string();
        let manifest_path = manifest_file.display().to_string();

        let mut subagent_prompt = self.system_prompt.clone();
        subagent_prompt.push_str(&format!(
            "\n\nYou are a background sub-agent of type `{subagent_type}`. Work only on the delegated task, use only the tools available to you, do not ask the user questions, and finish with a concise result.\n\nTask: {description}"
        ));

        let allowed_tools = allowed_tools_for_subagent(subagent_type);
        let subagent = RigAgent::with_options(
            &self.model,
            PermissionMode::DangerFullAccess,
            subagent_prompt,
            agents_dir.join("session.json"),
            true,
            HookRunner::empty(),
            None,
            Some(allowed_tools),
        );

        let metadata = serde_json::json!({
            "agentId": &agent_id,
            "name": name,
            "description": description,
            "subagentType": subagent_type,
            "status": "running",
            "outputFile": &output_path,
            "manifestFile": &manifest_path,
        });

        let prompt_owned = prompt.to_string();
        let output_path_clone = output_path.clone();
        let manifest_path_clone = manifest_path.clone();
        let description_owned = description.to_string();
        let subagent_type_owned = subagent_type.to_string();
        let name_owned = name.clone();
        let agent_id_clone = agent_id.clone();

        std::thread::Builder::new()
            .name(format!("rig-agent-{agent_id}"))
            .spawn(move || {
                let result = (|| -> anyhow::Result<String> {
                    let rt = tokio::runtime::Runtime::new()?;
                    rt.block_on(subagent.run_once(&prompt_owned))
                })();

                let (status, error) = match result {
                    Ok(output) => {
                        let content = format!(
                            "# {description_owned}\n\n## Prompt\n\n{prompt_owned}\n\n## Result\n\n{output}"
                        );
                        let _ = std::fs::write(&output_path_clone, content);
                        ("completed", None)
                    }
                    Err(error) => {
                        let content = format!(
                            "# {description_owned}\n\n## Prompt\n\n{prompt_owned}\n\n## Error\n\n{error}"
                        );
                        let _ = std::fs::write(&output_path_clone, content);
                        ("failed", Some(error.to_string()))
                    }
                };

                let manifest = serde_json::json!({
                    "agentId": agent_id_clone,
                    "name": name_owned,
                    "description": description_owned,
                    "subagentType": subagent_type_owned,
                    "status": status,
                    "outputFile": output_path_clone,
                    "manifestFile": manifest_path_clone,
                    "error": error,
                });
                let _ = std::fs::write(&manifest_path_clone, manifest.to_string());
            })
            .map_err(|e| crate::tools::ToolError::Other(e.to_string()))?;

        Ok(metadata.to_string())
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

/// Normalize common model JSON key mistakes (trailing colons, extra spaces).
fn normalize_json_keys(val: Value) -> Value {
    match val {
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                let clean = k.trim_end_matches(':').trim().to_string();
                new_map.insert(clean, normalize_json_keys(v));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(normalize_json_keys).collect())
        }
        other => other,
    }
}

fn try_match_tool_call(
    val: &Value,
    known_tools: &HashSet<&str>,
) -> Option<ParsedToolCall> {
    let val = normalize_json_keys(val.clone());
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

