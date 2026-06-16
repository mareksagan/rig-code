use rig::message::Message;
use std::path::{Path, PathBuf};

const SESSION_VERSION: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Session {
    pub version: u32,
    pub messages: Vec<Message>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: SESSION_VERSION,
            messages: Vec::new(),
        }
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let session: Self = serde_json::from_str(&contents)?;
        Ok(session)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[must_use]
pub fn default_session_path() -> PathBuf {
    if let Ok(path) = std::env::var("RIG_CODE_SESSION") {
        return PathBuf::from(path);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".rig-code")
        .join("session.json")
}

#[cfg(test)]
mod tests {
    use super::{default_session_path, Session};
    use rig::message::{Message, Text};
    use rig::OneOrMany;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("rig-code-session-{nanos}.json"))
    }

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
    fn saves_and_loads_session() {
        let path = temp_path();
        let mut session = Session::new();
        session.messages.push(user_text("hello"));
        session.messages.push(assistant_text("hi there"));

        session.save_to_path(&path).expect("save session");
        let loaded = Session::load_from_path(&path).expect("load session");
        std::fs::remove_file(&path).expect("cleanup");

        assert_eq!(loaded, session);
    }

    #[test]
    fn default_session_path_uses_rig_code_session_env() {
        let custom = "/tmp/custom-session.json";
        unsafe {
            std::env::set_var("RIG_CODE_SESSION", custom);
        }
        assert_eq!(default_session_path(), std::path::PathBuf::from(custom));
        unsafe {
            std::env::remove_var("RIG_CODE_SESSION");
        }
    }
}
