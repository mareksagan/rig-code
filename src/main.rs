use anyhow::Result;
use clap::Parser;
use rig_code::agent::RigAgent;

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
    let agent = RigAgent::new(&args.model);

    if let Some(prompt) = args.prompt {
        let response = agent.run_once(&prompt).await?;
        println!("{}", response);
    } else {
        agent.run_interactive().await?;
    }

    Ok(())
}
