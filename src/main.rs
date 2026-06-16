use anyhow::Result;
use clap::Parser;
use rig_code::agent::RigAgent;
use rig_code::hooks::{HookConfig, HookRunner};
use rig_code::lsp::{load_lsp_config, LspManager};
use rig_code::permissions::PermissionMode;
use rig_code::prompt;
use rig_code::session::default_session_path;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(name = "rig-code")]
#[command(about = "A Rig Code CLI agent powered by Ollama + rig")]
struct Args {
    /// Single prompt to execute (non-interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Ollama model to use
    #[arg(short, long, default_value = "qwen2.5:3b")]
    model: String,

    /// Auto-approve destructive operations (use with caution)
    #[arg(long)]
    auto_approve: bool,

    /// Permission mode for tool execution: read-only, workspace-write, danger-full-access, prompt, allow
    #[arg(long, default_value = "prompt")]
    permission_mode: String,

    /// Path to the session file (default: .rig-code/session.json)
    #[arg(long)]
    session: Option<PathBuf>,

    /// Do not load or save the session
    #[arg(long)]
    no_session: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    if args.auto_approve {
        unsafe { std::env::set_var("RIG_CODE_AUTO_APPROVE", "1"); }
    }

    let mode = if args.auto_approve {
        PermissionMode::Allow
    } else {
        PermissionMode::from_str(&args.permission_mode).map_err(|e| anyhow::anyhow!(e))?
    };

    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let session_path = args.session.unwrap_or_else(default_session_path);
    let system_prompt = prompt::build_system_prompt(&cwd);
    let hook_config = HookConfig::load(&cwd).unwrap_or_default();
    let lsp_manager = load_lsp_config(&cwd)
        .ok()
        .filter(|config| !config.servers.is_empty())
        .map(|config| LspManager::new(&cwd, config.servers));

    let agent = RigAgent::with_options(
        &args.model,
        mode,
        system_prompt,
        session_path,
        args.no_session,
        HookRunner::new(hook_config),
        lsp_manager,
        None,
    );

    if let Some(prompt) = args.prompt {
        let response = agent.run_once(&prompt).await?;
        println!("{}", response);
    } else {
        agent.run_interactive().await?;
    }

    Ok(())
}
