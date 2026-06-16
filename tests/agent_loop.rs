use rig_code::agent::RigAgent;
use rig_code::compact::{compact_history, CompactionConfig};
use rig_code::permissions::PermissionMode;
use rig_code::prompt;
use rig_code::tools::required_permission_for_tool;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("rig-code-integration-{nanos}"))
}

#[test]
fn system_prompt_includes_claw_instructions() {
    let root = temp_dir();
    fs::create_dir_all(&root).expect("create temp dir");
    fs::write(root.join("CLAW.md"), "Always write tests first.").expect("write CLAW.md");

    let prompt = prompt::build_system_prompt(&root);
    assert!(
        prompt.contains("Always write tests first."),
        "prompt should include CLAW.md content"
    );
    assert!(prompt.contains("# Instructions"), "prompt should have instructions section");

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn permission_policy_denies_shell_in_readonly_mode() {
    let agent = RigAgent::with_mode("test-model", PermissionMode::ReadOnly);
    let required = agent.permission_policy.required_mode_for("shell");
    assert!(
        required > PermissionMode::ReadOnly,
        "shell should require more than read-only"
    );
}

#[test]
fn required_permission_for_write_file_is_workspace_write() {
    assert_eq!(
        required_permission_for_tool("write_file"),
        PermissionMode::WorkspaceWrite
    );
}

#[test]
fn compact_history_summarizes_old_messages() {
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

    let mut history = vec![
        user_text(&"x".repeat(10_000)),
        assistant_text(&"y".repeat(10_000)),
        user_text(&"z".repeat(10_000)),
        assistant_text("recent reply"),
    ];

    compact_history(
        &mut history,
        CompactionConfig {
            preserve_recent_messages: 2,
            max_estimated_tokens: 1,
        },
    );

    assert!(
        matches!(history[0], Message::System { .. }),
        "first message should be compaction summary"
    );
    assert_eq!(history.len(), 3);
}
