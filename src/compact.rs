use rig::message::{AssistantContent, Message, ToolCall, ToolResult, UserContent};

const COMPACT_CONTINUATION_PREAMBLE: &str =
    "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n";
const COMPACT_RECENT_MESSAGES_NOTE: &str = "Recent messages are preserved verbatim.";
const COMPACT_DIRECT_RESUME_INSTRUCTION: &str = "Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, and do not preface with continuation text.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionConfig {
    pub preserve_recent_messages: usize,
    pub max_estimated_tokens: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            preserve_recent_messages: 4,
            max_estimated_tokens: 10_000,
        }
    }
}

/// Estimate token count using a simple chars/4 heuristic.
#[must_use]
pub fn estimate_history_tokens(history: &[Message]) -> usize {
    history.iter().map(estimate_message_tokens).sum()
}

#[must_use]
pub fn should_compact(history: &[Message], config: CompactionConfig) -> bool {
    let compactable_start = compacted_summary_prefix_len(history);
    let compactable = &history[compactable_start..];

    compactable.len() > config.preserve_recent_messages
        && compactable
            .iter()
            .map(estimate_message_tokens)
            .sum::<usize>()
            >= config.max_estimated_tokens
}

/// If the history has grown too large, replace the oldest compactable messages
/// with a system summary while preserving the most recent messages verbatim.
pub fn compact_history(history: &mut Vec<Message>, config: CompactionConfig) {
    if !should_compact(history, config) {
        return;
    }

    let existing_summary = history
        .first()
        .and_then(extract_existing_compacted_summary);
    let compacted_prefix_len = usize::from(existing_summary.is_some());
    let keep_from = history
        .len()
        .saturating_sub(config.preserve_recent_messages);
    let removed = &history[compacted_prefix_len..keep_from];
    let preserved = history[keep_from..].to_vec();

    let summary = merge_compact_summaries(existing_summary.as_deref(), &summarize_messages(removed));
    let continuation = build_continuation_message(&summary, !preserved.is_empty());

    let mut compacted = vec![Message::System {
        content: continuation,
    }];
    compacted.extend(preserved);
    *history = compacted;
}

fn compacted_summary_prefix_len(history: &[Message]) -> usize {
    usize::from(
        history
            .first()
            .and_then(extract_existing_compacted_summary)
            .is_some(),
    )
}

fn extract_existing_compacted_summary(message: &Message) -> Option<String> {
    let Message::System { content } = message else {
        return None;
    };
    let summary = content.strip_prefix(COMPACT_CONTINUATION_PREAMBLE)?;
    let summary = summary
        .split_once(&format!("\n\n{COMPACT_RECENT_MESSAGES_NOTE}"))
        .map_or(summary, |(value, _)| value);
    let summary = summary
        .split_once(&format!("\n{COMPACT_DIRECT_RESUME_INSTRUCTION}"))
        .map_or(summary, |(value, _)| value);
    Some(summary.trim().to_string())
}

fn build_continuation_message(summary: &str, recent_messages_preserved: bool) -> String {
    let mut base = format!("{COMPACT_CONTINUATION_PREAMBLE}{summary}");

    if recent_messages_preserved {
        base.push_str("\n\n");
        base.push_str(COMPACT_RECENT_MESSAGES_NOTE);
    }

    base.push('\n');
    base.push_str(COMPACT_DIRECT_RESUME_INSTRUCTION);

    base
}

fn summarize_messages(messages: &[Message]) -> String {
    let user_messages = messages
        .iter()
        .filter(|message| matches!(message, Message::User { .. }))
        .count();
    let assistant_messages = messages
        .iter()
        .filter(|message| matches!(message, Message::Assistant { .. }))
        .count();

    let mut tool_names: Vec<String> = messages
        .iter()
        .flat_map(extract_tool_names)
        .collect();
    tool_names.sort();
    tool_names.dedup();

    let mut lines = vec![
        "<summary>".to_string(),
        "Conversation summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={}, assistant={}).",
            messages.len(),
            user_messages,
            assistant_messages
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }

    let recent_user_requests = collect_recent_user_requests(messages, 3);
    if !recent_user_requests.is_empty() {
        lines.push("- Recent user requests:".to_string());
        lines.extend(
            recent_user_requests
                .into_iter()
                .map(|request| format!("  - {request}")),
        );
    }

    let pending_work = infer_pending_work(messages);
    if !pending_work.is_empty() {
        lines.push("- Pending work:".to_string());
        lines.extend(pending_work.into_iter().map(|item| format!("  - {item}")));
    }

    let key_files = collect_key_files(messages);
    if !key_files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", key_files.join(", ")));
    }

    lines.push("- Key timeline:".to_string());
    for message in messages {
        lines.push(format!("  - {}", summarize_message(message)));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}

fn merge_compact_summaries(existing_summary: Option<&str>, new_summary: &str) -> String {
    let Some(existing_summary) = existing_summary else {
        return new_summary.to_string();
    };

    let previous_highlights = extract_summary_highlights(existing_summary);
    let new_highlights = extract_summary_highlights(new_summary);
    let new_timeline = extract_summary_timeline(new_summary);

    let mut lines = vec!["<summary>".to_string(), "Conversation summary:".to_string()];

    if !previous_highlights.is_empty() {
        lines.push("- Previously compacted context:".to_string());
        lines.extend(
            previous_highlights
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }

    if !new_highlights.is_empty() {
        lines.push("- Newly compacted context:".to_string());
        lines.extend(new_highlights.into_iter().map(|line| format!("  {line}")));
    }

    if !new_timeline.is_empty() {
        lines.push("- Key timeline:".to_string());
        lines.extend(new_timeline.into_iter().map(|line| format!("  {line}")));
    }

    lines.push("</summary>".to_string());
    lines.join("\n")
}

fn extract_summary_highlights(summary: &str) -> Vec<String> {
    let formatted = format_compact_summary(summary);
    let mut lines = Vec::new();
    let mut in_timeline = false;

    for line in formatted.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed == "Summary:" || trimmed == "Conversation summary:" {
            continue;
        }
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if in_timeline {
            continue;
        }
        lines.push(trimmed.to_string());
    }

    lines
}

fn extract_summary_timeline(summary: &str) -> Vec<String> {
    let formatted = format_compact_summary(summary);
    let mut lines = Vec::new();
    let mut in_timeline = false;

    for line in formatted.lines() {
        let trimmed = line.trim_end();
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if !in_timeline {
            continue;
        }
        if trimmed.is_empty() {
            break;
        }
        lines.push(trimmed.to_string());
    }

    lines
}

fn format_compact_summary(summary: &str) -> String {
    let without_analysis = strip_tag_block(summary, "analysis");
    if let Some(content) = extract_tag_block(&without_analysis, "summary") {
        without_analysis.replace(
            &format!("<summary>{content}</summary>"),
            &format!("Summary:\n{}", content.trim()),
        )
    } else {
        without_analysis
    }
}

fn extract_tag_block(content: &str, tag: &str) -> Option<String> {
    let start = format!("<{tag}>");
    let end = format!("</{tag}>");
    let start_index = content.find(&start)? + start.len();
    let end_index = content[start_index..].find(&end)? + start_index;
    Some(content[start_index..end_index].to_string())
}

fn strip_tag_block(content: &str, tag: &str) -> String {
    let start = format!("<{tag}>");
    let end = format!("</{tag}>");
    if let (Some(start_index), Some(end_index_rel)) = (content.find(&start), content.find(&end)) {
        let end_index = end_index_rel + end.len();
        let mut stripped = String::new();
        stripped.push_str(&content[..start_index]);
        stripped.push_str(&content[end_index..]);
        stripped
    } else {
        content.to_string()
    }
}

fn extract_tool_names(message: &Message) -> Vec<String> {
    match message {
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::ToolCall(ToolCall { function, .. }) => {
                    Some(function.name.clone())
                }
                _ => None,
            })
            .collect(),
        Message::User { content } => content
            .iter()
            .filter_map(|block| match block {
                UserContent::ToolResult(ToolResult { id, .. }) => Some(id.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn collect_recent_user_requests(messages: &[Message], limit: usize) -> Vec<String> {
    messages
        .iter()
        .rev()
        .filter_map(first_text_content)
        .filter(|text| !text.trim().is_empty())
        .take(limit)
        .map(|text| truncate_summary(text, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn infer_pending_work(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .rev()
        .filter_map(first_text_content)
        .filter(|text| {
            let lowered = text.to_ascii_lowercase();
            lowered.contains("todo")
                || lowered.contains("next")
                || lowered.contains("pending")
                || lowered.contains("follow up")
                || lowered.contains("remaining")
        })
        .take(3)
        .map(|text| truncate_summary(text, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn collect_key_files(messages: &[Message]) -> Vec<String> {
    let mut files: Vec<String> = messages
        .iter()
        .filter_map(first_text_content)
        .flat_map(extract_file_candidates)
        .collect();
    files.sort();
    files.dedup();
    files.into_iter().take(8).collect()
}

fn first_text_content(message: &Message) -> Option<&str> {
    match message {
        Message::System { content } => Some(content.as_str()),
        Message::User { content } => content.iter().find_map(|block| match block {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }),
        Message::Assistant { content, .. } => content.iter().find_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }),
    }
}

fn summarize_message(message: &Message) -> String {
    match message {
        Message::System { content } => format!("system: {}", truncate_summary(content, 160)),
        Message::User { content } => {
            let text = content
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => text.text.clone(),
                    UserContent::ToolResult(result) => {
                        format!("tool_result {}", result.id)
                    }
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join(" | ");
            format!("user: {}", truncate_summary(&text, 160))
        }
        Message::Assistant { content, .. } => {
            let text = content
                .iter()
                .map(|block| match block {
                    AssistantContent::Text(text) => text.text.clone(),
                    AssistantContent::ToolCall(call) => {
                        format!("tool_use {}", call.function.name)
                    }
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join(" | ");
            format!("assistant: {}", truncate_summary(&text, 160))
        }
    }
}

fn estimate_message_tokens(message: &Message) -> usize {
    first_text_content(message)
        .map(|text| text.len() / 4 + 1)
        .unwrap_or(1)
}

fn truncate_summary(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut truncated = content.chars().take(max_chars).collect::<String>();
    truncated.push('…');
    truncated
}

fn extract_file_candidates(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|c: char| {
                matches!(c, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'' | '`')
            });
            if candidate.contains('/') && has_interesting_extension(candidate) {
                Some(candidate.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn has_interesting_extension(candidate: &str) -> bool {
    std::path::Path::new(candidate)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            ["rs", "ts", "tsx", "js", "json", "md", "toml", "yaml", "yml"]
                .iter()
                .any(|expected| extension.eq_ignore_ascii_case(expected))
        })
}

#[cfg(test)]
mod tests {
    use super::{compact_history, estimate_history_tokens, should_compact, CompactionConfig};
    use rig::message::{Message, Text};
    use rig::OneOrMany;

    fn user_text(text: &str) -> Message {
        Message::User {
            content: OneOrMany::one(rig::message::UserContent::Text(Text {
                text: text.to_string(),
            })),
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(rig::message::AssistantContent::Text(Text {
                text: text.to_string(),
            })),
        }
    }

    #[test]
    fn leaves_small_history_unchanged() {
        let mut history = vec![user_text("hello")];
        compact_history(&mut history, CompactionConfig::default());
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn compacts_older_messages_into_system_summary() {
        let mut history = vec![
            user_text(&"one ".repeat(200)),
            assistant_text(&"two ".repeat(200)),
            user_text(&"three ".repeat(200)),
            assistant_text("recent"),
        ];

        compact_history(
            &mut history,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
            },
        );

        assert_eq!(history.len(), 3);
        assert!(matches!(history[0], Message::System { .. }));
        assert!(
            estimate_history_tokens(&history)
                < estimate_history_tokens(&[
                    user_text(&"one ".repeat(200)),
                    assistant_text(&"two ".repeat(200)),
                    user_text(&"three ".repeat(200)),
                    assistant_text("recent"),
                ])
        );
    }

    #[test]
    fn should_compact_returns_true_for_large_history() {
        let history = vec![
            user_text(&"x".repeat(40_000)),
            assistant_text(&"y".repeat(40_000)),
            user_text("recent"),
            assistant_text("recent reply"),
        ];

        assert!(should_compact(
            &history,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
            }
        ));
    }

    #[test]
    fn keeps_previous_compacted_summary_when_compacting_again() {
        let mut history = vec![
            user_text("Investigate src/compact.rs"),
            assistant_text("I will inspect the compact flow."),
            user_text("Also update src/agent.rs"),
            assistant_text("Next: preserve prior summary context."),
        ];

        compact_history(
            &mut history,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
            },
        );

        let mut follow_up = history;
        follow_up.push(user_text("Please add regression tests."));
        follow_up.push(assistant_text("Working on regression coverage."));

        compact_history(
            &mut follow_up,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
            },
        );

        if let Message::System { content } = &follow_up[0] {
            assert!(content.contains("Previously compacted context:"));
            assert!(content.contains("Newly compacted context:"));
        } else {
            panic!("expected first message to be system summary");
        }
    }
}
