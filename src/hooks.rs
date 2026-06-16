use serde_json::json;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookConfig {
    pub pre_tool_use: Vec<String>,
    pub post_tool_use: Vec<String>,
}

impl HookConfig {
    #[must_use]
    pub fn new(pre_tool_use: Vec<String>, post_tool_use: Vec<String>) -> Self {
        Self {
            pre_tool_use,
            post_tool_use,
        }
    }

    /// Load hook config from `.rig-code/hooks.json` if it exists.
    pub fn load(cwd: impl AsRef<Path>) -> std::io::Result<Self> {
        load_hook_config(cwd)
    }
}

impl Default for HookConfig {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookRunResult {
    denied: bool,
    messages: Vec<String>,
}

impl HookRunResult {
    #[must_use]
    pub fn allow(messages: Vec<String>) -> Self {
        Self {
            denied: false,
            messages,
        }
    }

    #[must_use]
    pub fn deny(messages: Vec<String>) -> Self {
        Self {
            denied: true,
            messages,
        }
    }

    #[must_use]
    pub fn is_denied(&self) -> bool {
        self.denied
    }

    #[must_use]
    pub fn messages(&self) -> &[String] {
        &self.messages
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HookRunner {
    config: HookConfig,
}

impl HookRunner {
    #[must_use]
    pub fn new(config: HookConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn empty() -> Self {
        Self::new(HookConfig::default())
    }

    pub fn run_pre_tool_use(&self, tool_name: &str, tool_input: &str) -> HookRunResult {
        self.run_hooks(
            HookEvent::PreToolUse,
            tool_name,
            tool_input,
            None,
            false,
            &self.config.pre_tool_use,
        )
    }

    pub fn run_post_tool_use(
        &self,
        tool_name: &str,
        tool_input: &str,
        tool_output: &str,
        is_error: bool,
    ) -> HookRunResult {
        self.run_hooks(
            HookEvent::PostToolUse,
            tool_name,
            tool_input,
            Some(tool_output),
            is_error,
            &self.config.post_tool_use,
        )
    }

    fn run_hooks(
        &self,
        event: HookEvent,
        tool_name: &str,
        tool_input: &str,
        tool_output: Option<&str>,
        is_error: bool,
        commands: &[String],
    ) -> HookRunResult {
        if commands.is_empty() {
            return HookRunResult::allow(Vec::new());
        }

        let mut all_messages = Vec::new();

        for command in commands {
            match run_command(command, event, tool_name, tool_input, tool_output, is_error) {
                CommandOutcome::Allow(stdout) => {
                    if !stdout.is_empty() {
                        all_messages.push(stdout);
                    }
                }
                CommandOutcome::Deny(stdout) => {
                    if !stdout.is_empty() {
                        all_messages.push(stdout);
                    } else {
                        all_messages.push(format!("Hook denied {tool_name}"));
                    }
                    return HookRunResult::deny(all_messages);
                }
                CommandOutcome::Warn(message) => {
                    all_messages.push(message);
                }
            }
        }

        HookRunResult::allow(all_messages)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HookEvent {
    PreToolUse,
    PostToolUse,
}

impl HookEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
        }
    }
}

enum CommandOutcome {
    Allow(String),
    Deny(String),
    Warn(String),
}

fn run_command(
    command: &str,
    event: HookEvent,
    tool_name: &str,
    tool_input: &str,
    tool_output: Option<&str>,
    is_error: bool,
) -> CommandOutcome {
    let payload = json!({
        "hook_event_name": event.as_str(),
        "tool_name": tool_name,
        "tool_input": parse_tool_input(tool_input),
        "tool_input_json": tool_input,
        "tool_output": tool_output,
        "tool_result_is_error": is_error,
    })
    .to_string();

    let child = std::process::Command::new("sh")
        .arg("-lc")
        .arg(command)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env("HOOK_EVENT", event.as_str())
        .env("HOOK_TOOL_NAME", tool_name)
        .env("HOOK_TOOL_INPUT", tool_input)
        .env(
            "HOOK_TOOL_IS_ERROR",
            if is_error { "1" } else { "0" },
        )
        .spawn();

    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            return CommandOutcome::Warn(format!(
                "Hook failed to spawn ({command}): {error}"
            ));
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = stdin.write_all(payload.as_bytes());
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(error) => {
            return CommandOutcome::Warn(format!(
                "Hook failed to run ({command}): {error}"
            ));
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    match output.status.code() {
        Some(0) => CommandOutcome::Allow(stdout),
        Some(2) => CommandOutcome::Deny(stdout),
        Some(code) => CommandOutcome::Warn(format!(
            "Hook exited {code}; allowing tool execution to continue. stdout: {stdout} stderr: {stderr}"
        )),
        None => CommandOutcome::Warn(format!(
            "Hook terminated by signal; allowing tool execution to continue. stdout: {stdout} stderr: {stderr}"
        )),
    }
}

fn parse_tool_input(tool_input: &str) -> serde_json::Value {
    serde_json::from_str(tool_input).unwrap_or_else(|_| json!({ "raw": tool_input }))
}

/// Merge hook feedback into tool output.
#[must_use]
pub fn merge_hook_feedback(messages: &[String], output: String, denied: bool) -> String {
    if messages.is_empty() {
        return output;
    }
    let mut sections = Vec::new();
    if !output.trim().is_empty() {
        sections.push(output);
    }
    let label = if denied {
        "Hook feedback (denied)"
    } else {
        "Hook feedback"
    };
    sections.push(format!("{label}:\n{}", messages.join("\n")));
    sections.join("\n\n")
}

/// Load hook config from `.rig-code/hooks.json` if it exists.
pub fn load_hook_config(cwd: impl AsRef<Path>) -> std::io::Result<HookConfig> {
    let path = cwd.as_ref().join(".rig-code").join("hooks.json");
    if !path.exists() {
        return Ok(HookConfig::default());
    }
    let contents = std::fs::read_to_string(&path)?;
    let value: serde_json::Value = serde_json::from_str(&contents)?;
    let hooks = value.get("hooks").and_then(|v| v.as_object());

    let pre = hooks
        .and_then(|obj| obj.get("PreToolUse"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let post = hooks
        .and_then(|obj| obj.get("PostToolUse"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(HookConfig::new(pre, post))
}

#[cfg(test)]
mod tests {
    use super::{merge_hook_feedback, HookConfig, HookRunner};

    #[test]
    fn allows_exit_code_zero_and_captures_stdout() {
        let runner = HookRunner::new(HookConfig::new(
            vec!["printf 'pre ok'".to_string()],
            Vec::new(),
        ));
        let result = runner.run_pre_tool_use("read_file", r#"{"path":"README.md"}"#);
        assert!(!result.is_denied());
        assert_eq!(result.messages(), &["pre ok".to_string()]);
    }

    #[test]
    fn denies_exit_code_two() {
        let runner = HookRunner::new(HookConfig::new(
            vec!["printf 'blocked'; exit 2".to_string()],
            Vec::new(),
        ));
        let result = runner.run_pre_tool_use("shell", r#"{"command":"pwd"}"#);
        assert!(result.is_denied());
        assert_eq!(result.messages(), &["blocked".to_string()]);
    }

    #[test]
    fn warns_for_other_non_zero_statuses() {
        let runner = HookRunner::new(HookConfig::new(
            vec!["printf 'warning'; exit 1".to_string()],
            Vec::new(),
        ));
        let result = runner.run_pre_tool_use("write_file", r#"{"file":"src/lib.rs"}"#);
        assert!(!result.is_denied());
        assert!(result
            .messages()
            .iter()
            .any(|m| m.contains("allowing tool execution to continue")));
    }

    #[test]
    fn merges_hook_feedback_into_output() {
        let merged = merge_hook_feedback(
            &["check passed".to_string()],
            "tool result".to_string(),
            false,
        );
        assert!(merged.contains("tool result"));
        assert!(merged.contains("Hook feedback:"));
        assert!(merged.contains("check passed"));
    }
}
