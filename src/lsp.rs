use lsp_types::{
    Diagnostic, DiagnosticSeverity, InitializeParams, PublishDiagnosticsParams,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LspServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct LspConfig {
    #[serde(default)]
    pub servers: Vec<LspServerConfig>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct FileDiagnostics {
    pub uri: String,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct WorkspaceDiagnostics {
    pub files: Vec<FileDiagnostics>,
}

impl WorkspaceDiagnostics {
    #[must_use]
    pub fn total_diagnostics(&self) -> usize {
        self.files.iter().map(|f| f.diagnostics.len()).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_diagnostics() == 0
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct LspContextEnrichment {
    pub diagnostics: WorkspaceDiagnostics,
}

impl LspContextEnrichment {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    #[must_use]
    pub fn render_prompt_section(&self) -> String {
        const MAX_DIAGNOSTICS: usize = 12;
        let mut lines = vec!["# LSP context".to_string()];
        lines.push(format!(
            " - Workspace diagnostics: {} across {} file(s)",
            self.diagnostics.total_diagnostics(),
            self.diagnostics.files.len()
        ));

        if !self.diagnostics.files.is_empty() {
            lines.push(String::new());
            lines.push("Diagnostics:".to_string());
            let mut count = 0;
            for file in &self.diagnostics.files {
                for diagnostic in &file.diagnostics {
                    if count >= MAX_DIAGNOSTICS {
                        lines.push(" - ...".to_string());
                        return lines.join("\n");
                    }
                    let pos = diagnostic.range.start;
                    let (line, col) = (pos.line + 1, pos.character + 1);
                    let severity = diagnostic_severity_label(diagnostic.severity);
                    let message = diagnostic.message.lines().next().unwrap_or("");
                    lines.push(format!(
                        " - {}:{}:{} [{}] {}",
                        file.uri, line, col, severity, message
                    ));
                    count += 1;
                }
            }
        }

        lines.join("\n")
    }
}

fn diagnostic_severity_label(severity: Option<DiagnosticSeverity>) -> &'static str {
    match severity {
        Some(DiagnosticSeverity::ERROR) => "error",
        Some(DiagnosticSeverity::WARNING) => "warning",
        Some(DiagnosticSeverity::INFORMATION) => "info",
        Some(DiagnosticSeverity::HINT) => "hint",
        _ => "unknown",
    }
}

#[derive(Debug)]
struct LspClient {
    #[allow(dead_code)]
    process: Child,
    stdin: ChildStdin,
    diagnostics: Arc<Mutex<BTreeMap<String, Vec<Diagnostic>>>>,
    request_id: Arc<Mutex<u64>>,
    #[allow(dead_code)]
    shutdown_tx: mpsc::Sender<()>,
}

impl LspClient {
    async fn spawn(config: &LspServerConfig, workspace_root: &Path) -> anyhow::Result<Self> {
        let mut process = Command::new(&config.command)
            .args(&config.args)
            .current_dir(workspace_root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        let stdin = process
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to capture lsp stdin"))?;
        let stdout = process
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to capture lsp stdout"))?;

        let diagnostics: Arc<Mutex<BTreeMap<String, Vec<Diagnostic>>>> =
            Arc::new(Mutex::new(BTreeMap::new()));
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

        let diagnostics_clone = diagnostics.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut content_length: Option<usize> = None;

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    result = read_lsp_message(&mut reader, &mut content_length) => {
                        match result {
                            Ok(Some(text)) => handle_lsp_message(&text, &diagnostics_clone),
                            Ok(None) => continue,
                            Err(_) => break,
                        }
                    }
                }
            }
        });

        let mut client = Self {
            process,
            stdin,
            diagnostics,
            request_id: Arc::new(Mutex::new(0)),
            shutdown_tx,
        };

        client.initialize(workspace_root).await?;
        Ok(client)
    }

    async fn initialize(&mut self, workspace_root: &Path) -> anyhow::Result<()> {
        let uri = lsp_types::Url::from_file_path(workspace_root)
            .map_err(|_| anyhow::anyhow!("failed to convert workspace root to URI"))?;
        let name = workspace_root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let params = InitializeParams {
            workspace_folders: Some(vec![lsp_types::WorkspaceFolder { uri, name }]),
            capabilities: lsp_types::ClientCapabilities {
                text_document: Some(lsp_types::TextDocumentClientCapabilities {
                    publish_diagnostics: Some(lsp_types::PublishDiagnosticsClientCapabilities {
                        related_information: Some(true),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };

        let _ = self
            .request::<lsp_types::request::Initialize>(params)
            .await?;
        self.notify("initialized", json!({})).await?;
        Ok(())
    }

    async fn did_open(&mut self, path: &Path, text: &str) -> anyhow::Result<()> {
        let uri = lsp_types::Url::from_file_path(path)
            .map_err(|_| anyhow::anyhow!("failed to convert path to URI"))?
            .to_string();
        let language_id = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("text")
            .to_string();

        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text
                }
            }),
        )
        .await?;
        Ok(())
    }

    async fn request<R: lsp_types::request::Request>(
        &mut self,
        params: R::Params,
    ) -> anyhow::Result<R::Result>
    where
        R::Params: serde::Serialize,
        R::Result: serde::de::DeserializeOwned,
    {
        let id = {
            let mut lock = self.request_id.lock().unwrap();
            *lock += 1;
            *lock
        };
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": R::METHOD,
            "params": serde_json::to_value(params)?
        });
        self.send_raw(&message).await?;

        // Simplified: we don't wait for the response; return default for initialize.
        // A real implementation would correlate responses by id.
        serde_json::from_value(json!({})).map_err(|e| anyhow::anyhow!(e))
    }

    async fn notify(&mut self, method: &str, params: Value) -> anyhow::Result<()> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        self.send_raw(&message).await
    }

    async fn send_raw(&mut self, message: &Value) -> anyhow::Result<()> {
        let body = message.to_string();
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        self.stdin.write_all(header.as_bytes()).await?;
        self.stdin.write_all(body.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    fn collect_diagnostics(&self) -> WorkspaceDiagnostics {
        let lock = self.diagnostics.lock().unwrap();
        let files = lock
            .iter()
            .map(|(uri, diagnostics)| FileDiagnostics {
                uri: uri.clone(),
                diagnostics: diagnostics.clone(),
            })
            .collect();
        WorkspaceDiagnostics { files }
    }
}

async fn read_lsp_message(
    reader: &mut BufReader<ChildStdout>,
    content_length: &mut Option<usize>,
) -> std::io::Result<Option<String>> {
    let mut header_line = String::new();

    loop {
        header_line.clear();
        let bytes_read = reader.read_line(&mut header_line).await?;
        if bytes_read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "lsp stream closed",
            ));
        }

        let line = header_line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            if let Some(len) = content_length.take() {
                let mut chunk = vec![0u8; len];
                reader.read_exact(&mut chunk).await?;
                return String::from_utf8(chunk)
                    .map(Some)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e));
            }
        } else if let Some(rest) = line.strip_prefix("Content-Length: ") {
            *content_length = rest.parse().ok();
        }
    }
}

fn handle_lsp_message(text: &str, diagnostics: &Arc<Mutex<BTreeMap<String, Vec<Diagnostic>>>>) {
    let Ok(message) = serde_json::from_str::<Value>(text) else {
        return;
    };

    if message.get("method").and_then(|m| m.as_str()) != Some("textDocument/publishDiagnostics") {
        return;
    }

    let Some(params) = message.get("params") else {
        return;
    };

    if let Ok(params) = serde_json::from_value::<PublishDiagnosticsParams>(params.clone()) {
        let mut lock = diagnostics.lock().unwrap();
        if params.diagnostics.is_empty() {
            lock.remove(&params.uri.to_string());
        } else {
            lock.insert(params.uri.to_string(), params.diagnostics);
        }
    }
}

#[derive(Debug)]
pub struct LspManager {
    configs: Vec<LspServerConfig>,
    clients: Arc<Mutex<HashMap<String, LspClient>>>,
    workspace_root: PathBuf,
}

impl LspManager {
    pub fn new(workspace_root: impl Into<PathBuf>, configs: Vec<LspServerConfig>) -> Self {
        Self {
            configs,
            clients: Arc::new(Mutex::new(HashMap::new())),
            workspace_root: workspace_root.into(),
        }
    }

    pub async fn open_document(&self, path: &Path) {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if extension.is_empty() {
            return;
        }

        let config = self
            .configs
            .iter()
            .find(|c| c.extensions.iter().any(|e| e.to_lowercase() == extension));

        let Some(config) = config else {
            return;
        };

        let text = match tokio::fs::read_to_string(path).await {
            Ok(text) => text,
            Err(_) => return,
        };

        let should_spawn = {
            let clients = self.clients.lock().unwrap();
            !clients.contains_key(&config.name)
        };

        if should_spawn {
            match LspClient::spawn(config, &self.workspace_root).await {
                Ok(client) => {
                    let mut clients = self.clients.lock().unwrap();
                    clients.insert(config.name.clone(), client);
                }
                Err(_) => return,
            }
        }

        let mut clients = self.clients.lock().unwrap();
        if let Some(client) = clients.get_mut(&config.name) {
            let _ = client.did_open(path, &text).await;
        }
    }

    pub fn collect_diagnostics(&self) -> WorkspaceDiagnostics {
        let clients = self.clients.lock().unwrap();
        let mut all = WorkspaceDiagnostics::default();
        for client in clients.values() {
            let diag = client.collect_diagnostics();
            all.files.extend(diag.files);
        }
        all
    }

    pub fn enrichment(&self) -> LspContextEnrichment {
        LspContextEnrichment {
            diagnostics: self.collect_diagnostics(),
        }
    }
}

pub fn load_lsp_config(cwd: impl AsRef<Path>) -> std::io::Result<LspConfig> {
    let path = cwd.as_ref().join(".rig-code").join("lsp.json");
    if !path.exists() {
        return Ok(LspConfig::default());
    }
    let contents = std::fs::read_to_string(&path)?;
    let config: LspConfig = serde_json::from_str(&contents)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::{diagnostic_severity_label, LspConfig, LspContextEnrichment, WorkspaceDiagnostics};
    use lsp_types::DiagnosticSeverity;

    #[test]
    fn renders_lsp_context_section() {
        let enrichment = LspContextEnrichment {
            diagnostics: WorkspaceDiagnostics {
                files: vec![super::FileDiagnostics {
                    uri: "file:///tmp/main.rs".to_string(),
                    diagnostics: vec![lsp_types::Diagnostic {
                        range: lsp_types::Range::new(
                            lsp_types::Position::new(0, 0),
                            lsp_types::Position::new(0, 5),
                        ),
                        severity: Some(DiagnosticSeverity::ERROR),
                        message: "mock error".to_string(),
                        ..Default::default()
                    }],
                }],
            },
        };

        let rendered = enrichment.render_prompt_section();
        assert!(rendered.contains("# LSP context"));
        assert!(rendered.contains("Workspace diagnostics: 1 across 1 file(s)"));
        assert!(rendered.contains("Diagnostics:"));
        assert!(rendered.contains("mock error"));
        assert!(rendered.contains("[error]"));
    }

    #[test]
    fn parses_lsp_config() {
        let json = r#"{"servers":[{"name":"rust-analyzer","command":"rust-analyzer","extensions":["rs"]}]}"#;
        let config: LspConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "rust-analyzer");
    }

    #[test]
    fn severity_labels() {
        assert_eq!(
            diagnostic_severity_label(Some(DiagnosticSeverity::ERROR)),
            "error"
        );
        assert_eq!(
            diagnostic_severity_label(Some(DiagnosticSeverity::WARNING)),
            "warning"
        );
    }
}
