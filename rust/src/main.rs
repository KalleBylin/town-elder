//! Clap binary entrypoint for town_elder
//!
//! This provides a standalone CLI binary that can be used
//! for testing the Rust core functionality.
//!
//! Build and run with:
//!   cargo build --manifest-path rust/Cargo.toml
//!   cargo run --manifest-path rust/Cargo.toml -- --help

use clap::Parser;

/// Minimal CLI for testing the clap binary entrypoint.
#[derive(Parser, Debug)]
#[command(name = "te")]
#[command(version = "0.1.0-scaffold")]
#[command(about = "Town Elder CLI (Rust scaffolding)", long_about = None)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Subcommand to run
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Run the health check
    Health,
    /// Print version info
    Version,
    /// Run the placeholder command
    Placeholder,
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .init();
    }

    match cli.command {
        Some(Commands::Health) => {
            println!("te-core: OK");
        }
        Some(Commands::Version) => {
            println!("0.1.0-scaffold");
        }
        Some(Commands::Placeholder) => {
            println!("42");
        }
        None => {
            // No subcommand: print help
            println!("te CLI (Rust scaffolding)");
            println!("Use --help for more information");
        }
    }
}
