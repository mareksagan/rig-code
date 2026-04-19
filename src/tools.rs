use anyhow::Result;
use colored::Colorize;
use dialoguer::{Confirm, Input};
use is_terminal::IsTerminal;
use regex::Regex;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

// ────────────────────────────────
// Shared state
// ────────────────────────────────

#[derive(Clone, Default)]
pub struct TodoState {
    inner: Arc<Mutex<Vec<(String, bool)>>>,
}

impl TodoState {
    pub fn set(&self, todos: Vec<String>) {
        let mut lock = self.inner.lock().unwrap();
        *lock = todos.into_iter().map(|t| (t, false)).collect();
    }

    pub fn query(&self) -> Vec<(String, bool)> {
        self.inner.lock().unwrap().clone()
    }

    pub fn mark_done(&self, index: usize) -> Result<()> {
        let mut lock = self.inner.lock().unwrap();
        if index >= lock.len() {
            anyhow::bail!("Invalid todo index {}", index);
        }
        lock[index].1 = true;
        Ok(())
    }
}

// ────────────────────────────────
// Shell Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct Shell;

#[derive(Deserialize, Serialize, Debug)]
pub struct ShellArgs {
    pub command: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Command failed: {0}")]
    Command(String),
    #[error("Cancelled by user")]
    Cancelled,
    #[error("Timeout after {0}s")]
    Timeout(u64),
    #[error("{0}")]
    Other(String),
}

fn is_destructive(cmd: &str) -> bool {
    let destructive = [
        "rm ", "rmdir ", "mv ", "dd ", "mkfs", "format",
        "> ", ">> ", "|", "sed -i", "perl -pi",
    ];
    let lowered = cmd.to_lowercase();
    destructive.iter().any(|d| lowered.contains(d))
}

fn auto_approve() -> bool {
    std::env::var("RIG_CODE_AUTO_APPROVE").is_ok()
}

impl Tool for Shell {
    const NAME: &'static str = "shell";
    type Error = ToolError;
    type Args = ShellArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "shell".to_string(),
            description: "Execute a bash shell command. Use this to run commands, explore the filesystem, build/test code, etc.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The bash command to execute" },
                    "description": { "type": "string", "description": "Brief description of what this command does" },
                    "timeout_seconds": { "type": "integer", "description": "Optional timeout in seconds (default: 60)" }
                },
                "required": ["command"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let timeout = args.timeout_seconds.unwrap_or(60);
        let cmd_desc = if args.description.is_empty() {
            args.command.clone()
        } else {
            args.description.clone()
        };

        if is_destructive(&args.command) && !auto_approve() {
            println!("{}", format!("⚠️  Destructive command: {}", args.command).yellow().bold());
            if std::io::stdin().is_terminal() {
                let confirmed = Confirm::new()
                    .with_prompt("Execute this command?")
                    .default(false)
                    .interact()
                    .map_err(|e: dialoguer::Error| ToolError::Io(e.to_string()))?;
                if !confirmed {
                    return Err(ToolError::Cancelled);
                }
            } else {
                return Err(ToolError::Other(
                    "Destructive command requires terminal for confirmation. Run in interactive mode or set RIG_CODE_AUTO_APPROVE=1.".to_string()
                ));
            }
        }

        println!("{} {}", "▶".blue().bold(), cmd_desc.dimmed());

        let output = tokio::time::timeout(
            tokio::time::Duration::from_secs(timeout),
            Command::new("bash")
                .arg("-c")
                .arg(&args.command)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|_| ToolError::Timeout(timeout))?
        .map_err(|e| ToolError::Io(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut result = String::new();

        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr]\n");
            result.push_str(&stderr);
        }

        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            result.push_str(&format!("\n[exit code: {}]", code));
        }

        // Truncate very long output
        const MAX_LEN: usize = 8000;
        if result.len() > MAX_LEN {
            let truncated = result.chars().take(MAX_LEN).collect::<String>();
            result = format!("{}\n\n[... truncated {} chars ...]", truncated, result.len() - MAX_LEN);
        }

        if result.trim().is_empty() {
            result = "(no output)".to_string();
        }

        Ok(result)
    }
}

// ────────────────────────────────
// ReadFile Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct ReadFile;

#[derive(Deserialize, Serialize, Debug)]
pub struct ReadFileArgs {
    pub path: String,
    #[serde(default)]
    pub line_offset: Option<i32>,
    #[serde(default)]
    pub n_lines: Option<usize>,
}

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Error = ToolError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read the contents of a text file. Returns line-numbered content.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute or relative path to the file" },
                    "line_offset": { "type": "integer", "description": "Line to start from (1-indexed). Use negative to read from end." },
                    "n_lines": { "type": "integer", "description": "Max lines to read (default: 1000)" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);
        if !path.exists() {
            return Err(ToolError::Other(format!("File not found: {}", args.path)));
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::Io(e.to_string()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let n_lines = args.n_lines.unwrap_or(1000).min(1000);

        let (start, end) = match args.line_offset {
            Some(off) if off < 0 => {
                let abs = off.abs() as usize;
                let s = total.saturating_sub(abs);
                (s, (s + n_lines).min(total))
            }
            Some(off) => {
                // Handle both 0 and 1 as first line for robustness
                let s = if off == 0 { 0 } else { (off as usize).saturating_sub(1) };
                (s, (s + n_lines).min(total))
            }
            None => (0, n_lines.min(total)),
        };

        let mut result = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            result.push_str(&format!("{}| {}\n", start + i + 1, line));
        }

        if end < total {
            result.push_str(&format!("\n[... {} more lines ...]", total - end));
        }

        Ok(result)
    }
}

// ────────────────────────────────
// WriteFile Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct WriteFile;

#[derive(Deserialize, Serialize, Debug)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub append: bool,
}

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = ToolError;
    type Args = WriteFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Write or append content to a file. ALWAYS confirm with user before overwriting existing files.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path to write" },
                    "content": { "type": "string", "description": "Content to write" },
                    "append": { "type": "boolean", "description": "Append instead of overwrite" }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);

        if path.exists() && !args.append && !auto_approve() {
            println!("{}", format!("⚠️  File already exists: {}", args.path).yellow().bold());
            if std::io::stdin().is_terminal() {
                let confirmed = Confirm::new()
                    .with_prompt("Overwrite this file?")
                    .default(false)
                    .interact()
                    .map_err(|e: dialoguer::Error| ToolError::Io(e.to_string()))?;
                if !confirmed {
                    return Err(ToolError::Cancelled);
                }
            } else {
                return Err(ToolError::Other(
                    "File overwrite requires terminal for confirmation. Run in interactive mode or set RIG_CODE_AUTO_APPROVE=1.".to_string()
                ));
            }
        }

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolError::Io(e.to_string()))?;
        }

        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .append(args.append)
            .truncate(!args.append)
            .open(path)
            .await
            .map_err(|e| ToolError::Io(e.to_string()))?;

        file.write_all(args.content.as_bytes())
            .await
            .map_err(|e: std::io::Error| ToolError::Io(e.to_string()))?;

        let action = if args.append { "Appended to" } else { "Wrote" };
        Ok(format!("{} {} ({} bytes)", action, args.path, args.content.len()))
    }
}

// ────────────────────────────────
// StrReplaceFile Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct StrReplaceFile;

#[derive(Deserialize, Serialize, Debug)]
pub struct StrReplaceFileArgs {
    pub path: String,
    pub old: String,
    pub new: String,
}

impl Tool for StrReplaceFile {
    const NAME: &'static str = "str_replace_file";
    type Error = ToolError;
    type Args = StrReplaceFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "str_replace_file".to_string(),
            description: "Replace a specific string in a file with another string. Use for precise edits.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path" },
                    "old": { "type": "string", "description": "Exact text to replace" },
                    "new": { "type": "string", "description": "Replacement text" }
                },
                "required": ["path", "old", "new"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);
        if !path.exists() {
            return Err(ToolError::Other(format!("File not found: {}", args.path)));
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::Io(e.to_string()))?;

        if !content.contains(&args.old) {
            return Err(ToolError::Other(format!("String not found in file: {}", args.path)));
        }

        let occurrences = content.matches(&args.old).count();
        if occurrences > 1 {
            println!("{}", format!("⚠️  Found {} occurrences of the old string in {}", occurrences, args.path).yellow());
            if auto_approve() {
                // proceed without confirmation
            } else if std::io::stdin().is_terminal() {
                let confirmed = Confirm::new()
                    .with_prompt("Replace ALL occurrences?")
                    .default(false)
                    .interact()
                    .map_err(|e: dialoguer::Error| ToolError::Io(e.to_string()))?;
                if !confirmed {
                    return Err(ToolError::Cancelled);
                }
            } else {
                return Err(ToolError::Other(
                    "Multi-occurrence replace requires terminal for confirmation.".to_string()
                ));
            }
        } else {
            println!("{}", format!("📝 Editing {}", args.path).cyan());
        }

        let new_content = content.replace(&args.old, &args.new);
        tokio::fs::write(path, new_content)
            .await
            .map_err(|e| ToolError::Io(e.to_string()))?;

        Ok(format!("Replaced {} occurrence(s) in {}", occurrences, args.path))
    }
}

// ────────────────────────────────
// Glob Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct GlobTool;

#[derive(Deserialize, Serialize, Debug)]
pub struct GlobArgs {
    pub pattern: String,
}

impl Tool for GlobTool {
    const NAME: &'static str = "glob";
    type Error = ToolError;
    type Args = GlobArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_string(),
            description: "Find files matching a glob pattern (e.g., 'src/**/*.rs', '*.toml').".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern to match files" }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let entries: Vec<_> = glob::glob(&args.pattern)
            .map_err(|e| ToolError::Other(e.to_string()))?
            .filter_map(|e| e.ok())
            .filter(|p| p.is_file())
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        if entries.is_empty() {
            return Ok("No files found.".to_string());
        }

        Ok(entries.join("\n"))
    }
}

// ────────────────────────────────
// Grep Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct GrepTool;

#[derive(Deserialize, Serialize, Debug)]
pub struct GrepArgs {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub glob: Option<String>,
}

impl Tool for GrepTool {
    const NAME: &'static str = "grep";
    type Error = ToolError;
    type Args = GrepArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            description: "Search file contents using regex. Returns matching lines with file:line info.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regex pattern to search for" },
                    "path": { "type": "string", "description": "Directory or file to search in (default: current dir)" },
                    "glob": { "type": "string", "description": "Glob filter for files (e.g., '*.rs')" }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let regex = Regex::new(&args.pattern)
            .map_err(|e| ToolError::Other(format!("Invalid regex: {}", e)))?;

        let base_path = args.path.as_deref().unwrap_or(".");
        let glob_filter = args.glob.as_deref().unwrap_or("*");

        let mut results = Vec::new();
        let walker = walkdir::WalkDir::new(base_path)
            .max_depth(10)
            .follow_links(false);

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !glob::Pattern::new(glob_filter).map_err(|e| ToolError::Other(e.to_string()))?.matches(file_name) {
                continue;
            }

            let content = match tokio::fs::read_to_string(path).await {
                Ok(c) => c,
                Err(_) => continue, // skip binary files
            };

            for (i, line) in content.lines().enumerate() {
                if regex.is_match(line) {
                    let truncated = if line.len() > 200 {
                        format!("{}...", &line[..200])
                    } else {
                        line.to_string()
                    };
                    results.push(format!("{}:{}| {}", path.display(), i + 1, truncated));
                    if results.len() >= 100 {
                        break;
                    }
                }
            }
            if results.len() >= 100 {
                break;
            }
        }

        if results.is_empty() {
            Ok("No matches found.".to_string())
        } else {
            let mut out = results.join("\n");
            if results.len() >= 100 {
                out.push_str("\n\n[... truncated to 100 matches ...]");
            }
            Ok(out)
        }
    }
}

// ────────────────────────────────
// SearchWeb Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct SearchWeb;

#[derive(Deserialize, Serialize, Debug)]
pub struct SearchWebArgs {
    pub query: String,
    #[serde(default)]
    pub limit: Option<u8>,
}

impl Tool for SearchWeb {
    const NAME: &'static str = "search_web";
    type Error = ToolError;
    type Args = SearchWebArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "search_web".to_string(),
            description: "Search the web using DuckDuckGo. Returns search results with titles, URLs, and snippets.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "limit": { "type": "integer", "description": "Max results (default: 5, max: 10)" }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let limit = args.limit.unwrap_or(5).min(10);
        let query = urlencoding::encode(&args.query);
        let url = format!("https://html.duckduckgo.com/html/?q={}", query);

        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
            .build()
            .map_err(|e| ToolError::Io(e.to_string()))?;

        let resp = client.get(&url)
            .send()
            .await
            .map_err(|e| ToolError::Io(e.to_string()))?;

        let html = resp.text().await.map_err(|e| ToolError::Io(e.to_string()))?;

        // Simple HTML parsing
        let title_re = Regex::new(r#"<a[^>]*class="result__a"[^>]*>(.*?)</a>"#).unwrap();
        let snippet_re = Regex::new(r#"<a[^>]*class="result__snippet"[^>]*>(.*?)</a>"#).unwrap();
        let href_re = Regex::new(r#"<a[^>]*href="([^"]*)"[^>]*class="result__a""#).unwrap();

        let titles: Vec<_> = title_re.captures_iter(&html).map(|c| strip_html(&c[1])).collect();
        let snippets: Vec<_> = snippet_re.captures_iter(&html).map(|c| strip_html(&c[1])).collect();
        let hrefs: Vec<_> = href_re.captures_iter(&html).map(|c| c[1].to_string()).collect();

        let mut results = Vec::new();
        for i in 0..titles.len().min(limit as usize) {
            let title = titles.get(i).map(|s| s.as_str()).unwrap_or("No title");
            let snippet = snippets.get(i).map(|s| s.as_str()).unwrap_or("");
            let href = hrefs.get(i).map(|s| s.as_str()).unwrap_or("#");
            results.push(format!("{}. {}\n   URL: {}\n   {}\n", i + 1, title, href, snippet));
        }

        if results.is_empty() {
            Ok("No search results found.".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }
}

fn strip_html(html: &str) -> String {
    let re = Regex::new(r"<[^>]+>").unwrap();
    re.replace_all(html, "").replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
}

// ────────────────────────────────
// FetchURL Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct FetchURL;

#[derive(Deserialize, Serialize, Debug)]
pub struct FetchURLArgs {
    pub url: String,
}

impl Tool for FetchURL {
    const NAME: &'static str = "fetch_url";
    type Error = ToolError;
    type Args = FetchURLArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_url".to_string(),
            description: "Fetch and extract text content from a URL.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "URL to fetch" }
                },
                "required": ["url"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
            .build()
            .map_err(|e| ToolError::Io(e.to_string()))?;

        let resp = client.get(&args.url)
            .send()
            .await
            .map_err(|e| ToolError::Io(e.to_string()))?;

        let html = resp.text().await.map_err(|e| ToolError::Io(e.to_string()))?;

        // Very simple text extraction
        let body_re = Regex::new(r"<body[^>]*>(.*?)</body>").unwrap();
        let body = body_re.captures(&html).map(|c| c[1].to_string()).unwrap_or(html);

        let text = strip_html(&body);
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");

        const MAX_LEN: usize = 12000;
        if text.len() > MAX_LEN {
            Ok(format!("{}\n\n[... truncated {} chars ...]", &text[..MAX_LEN], text.len() - MAX_LEN))
        } else {
            Ok(text)
        }
    }
}

// ────────────────────────────────
// TodoList Tool
// ────────────────────────────────

#[derive(Clone)]
pub struct TodoListTool {
    pub state: TodoState,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TodoListArgs {
    pub action: String, // "set", "query", "done"
    #[serde(default)]
    pub todos: Option<Vec<String>>,
    #[serde(default)]
    pub index: Option<usize>,
}

impl Tool for TodoListTool {
    const NAME: &'static str = "todo_list";
    type Error = ToolError;
    type Args = TodoListArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "todo_list".to_string(),
            description: "Manage a todo list. Actions: 'set' (replace list), 'query' (get current), 'done' (mark index as done).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["set", "query", "done"], "description": "Action to perform" },
                    "todos": { "type": "array", "items": { "type": "string" }, "description": "List of todos for 'set' action" },
                    "index": { "type": "integer", "description": "0-based index for 'done' action" }
                },
                "required": ["action"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.action.as_str() {
            "set" => {
                if let Some(todos) = args.todos {
                    self.state.set(todos.clone());
                    Ok(format!("Set {} todo(s)", todos.len()))
                } else {
                    Err(ToolError::Other("'set' action requires 'todos'".to_string()))
                }
            }
            "query" => {
                let todos = self.state.query();
                if todos.is_empty() {
                    Ok("No todos.".to_string())
                } else {
                    let lines: Vec<String> = todos.iter().enumerate()
                        .map(|(i, (t, done))| format!("{} [{}] {}", i, if *done { "x" } else { " " }, t))
                        .collect();
                    Ok(lines.join("\n"))
                }
            }
            "done" => {
                if let Some(idx) = args.index {
                    self.state.mark_done(idx).map_err(|e| ToolError::Other(e.to_string()))?;
                    Ok(format!("Marked todo {} as done", idx))
                } else {
                    Err(ToolError::Other("'done' action requires 'index'".to_string()))
                }
            }
            _ => Err(ToolError::Other(format!("Unknown action: {}", args.action))),
        }
    }
}

// ────────────────────────────────
// AskUser Tool
// ────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct AskUser;

#[derive(Deserialize, Serialize, Debug)]
pub struct AskUserArgs {
    pub question: String,
    #[serde(default)]
    pub options: Option<Vec<String>>,
}

impl Tool for AskUser {
    const NAME: &'static str = "ask_user";
    type Error = ToolError;
    type Args = AskUserArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "ask_user".to_string(),
            description: "Ask the user a question and return their answer. Use when you need clarification or user preference.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "Question to ask the user" },
                    "options": { "type": "array", "items": { "type": "string" }, "description": "Optional predefined options (user can still type custom answer)" }
                },
                "required": ["question"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("{}", format!("🤔 {}", args.question).cyan().bold());

        if let Some(opts) = args.options {
            if !opts.is_empty() {
                let selection = Input::new()
                    .with_prompt("Your answer")
                    .default(opts[0].clone())
                    .interact_text()
                    .map_err(|e| ToolError::Io(e.to_string()))?;
                return Ok(selection);
            }
        }

        let answer = Input::new()
            .with_prompt("Your answer")
            .allow_empty(true)
            .interact_text()
            .map_err(|e| ToolError::Io(e.to_string()))?;

        Ok(answer)
    }
}

// ────────────────────────────────
// PlanMode Tool
// ────────────────────────────────

#[derive(Clone)]
pub struct PlanModeTool {
    pub in_plan_mode: Arc<Mutex<bool>>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct PlanModeArgs {
    pub action: String, // "enter", "exit"
    pub plan: Option<String>,
}

impl Tool for PlanModeTool {
    const NAME: &'static str = "plan_mode";
    type Error = ToolError;
    type Args = PlanModeArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "plan_mode".to_string(),
            description: "Enter or exit planning mode. In plan mode, you write a detailed plan before executing. The user must approve the plan.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["enter", "exit"], "description": "Enter or exit plan mode" },
                    "plan": { "type": "string", "description": "The plan text (required when entering)" }
                },
                "required": ["action"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.action.as_str() {
            "enter" => {
                let plan = args.plan.unwrap_or_default();
                println!("{}", "╔═══════════════════════════════════════╗".blue().bold());
                println!("{}", "║          📋 PLANNING MODE             ║".blue().bold());
                println!("{}", "╚═══════════════════════════════════════╝".blue().bold());
                println!("{}", plan);

                if std::io::stdin().is_terminal() {
                    let approved = Confirm::new()
                        .with_prompt("Approve this plan?")
                        .default(true)
                        .interact()
                        .map_err(|e: dialoguer::Error| ToolError::Io(e.to_string()))?;

                    if approved {
                        *self.in_plan_mode.lock().unwrap() = true;
                        Ok("Plan approved. Proceed with execution.".to_string())
                    } else {
                        Ok("Plan rejected. Please revise.".to_string())
                    }
                } else {
                    *self.in_plan_mode.lock().unwrap() = true;
                    Ok("Plan approved (non-interactive mode).".to_string())
                }
            }
            "exit" => {
                *self.in_plan_mode.lock().unwrap() = false;
                Ok("Exited planning mode.".to_string())
            }
            _ => Err(ToolError::Other(format!("Unknown action: {}", args.action))),
        }
    }
}
