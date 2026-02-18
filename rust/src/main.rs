//! Native Rust CLI for Town Elder.

use clap::{Args, Parser, Subcommand};
use std::collections::HashSet;
use std::path::PathBuf;
use te_core::{
    build_documents_from_files, get_version, health_check, BackendConfig, EmbeddingBackend,
    InMemoryBackend,
};

#[derive(Parser, Debug)]
#[command(name = "te-rs")]
#[command(version = "0.1.0")]
#[command(about = "Town Elder native Rust CLI")]
struct Cli {
    /// Enable verbose logging.
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Health check for the shared core.
    Health,
    /// Print te-core version.
    Version,
    /// File indexing commands.
    Index {
        #[command(subcommand)]
        command: IndexCommand,
    },
}

#[derive(Subcommand, Debug)]
enum IndexCommand {
    /// Index source files using te-core scanning/parsing logic.
    Files(IndexFilesArgs),
}

#[derive(Args, Debug)]
struct IndexFilesArgs {
    /// Root path to scan for files.
    path: PathBuf,

    /// Additional exclude patterns (additive with defaults).
    #[arg(short = 'e', long = "exclude")]
    exclude: Vec<String>,

    /// Explicit file extensions to index (default: .py,.md,.rst).
    #[arg(long = "ext")]
    extensions: Vec<String>,

    /// Optional query to execute after indexing (end-to-end workflow).
    #[arg(long)]
    query: Option<String>,

    /// Number of results for the optional query.
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Embedding dimensions for in-memory backend.
    #[arg(long, default_value_t = 384)]
    embedding_dimensions: usize,
}

fn normalize_extension(extension: &str) -> String {
    if extension.starts_with('.') {
        extension.to_string()
    } else {
        format!(".{extension}")
    }
}

fn run_index_files(args: IndexFilesArgs) -> Result<(), String> {
    let extension_set: Option<HashSet<String>> = if args.extensions.is_empty() {
        None
    } else {
        Some(
            args.extensions
                .iter()
                .map(|value| normalize_extension(value))
                .collect(),
        )
    };

    let exclude_set: Option<HashSet<String>> = if args.exclude.is_empty() {
        None
    } else {
        Some(args.exclude.iter().cloned().collect())
    };

    let documents = build_documents_from_files(
        &args.path,
        extension_set.as_ref(),
        exclude_set.as_ref(),
    )
    .map_err(|error| format!("Failed to index files from {}: {error}", args.path.display()))?;

    let document_count = documents.len();
    let backend = InMemoryBackend::from_config(&BackendConfig {
        embedding_dimensions: args.embedding_dimensions,
        ..Default::default()
    });

    backend
        .embed_and_store(documents)
        .map_err(|error| format!("Failed to embed/store indexed documents: {error}"))?;

    println!(
        "Indexed {document_count} document chunks from {}",
        args.path.display()
    );

    if let Some(query) = args.query {
        let results = backend
            .search_text(&query, args.top_k)
            .map_err(|error| format!("Query failed: {error}"))?;

        println!("Query: {query}");
        for (index, result) in results.iter().enumerate() {
            let source = result
                .document
                .metadata
                .get("source")
                .and_then(|value| value.as_str())
                .unwrap_or("<unknown>");
            println!("{}. {:.4} {}", index + 1, result.score, source);
        }
    }

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    }

    let result = match cli.command {
        Commands::Health => {
            println!("{}", health_check());
            Ok(())
        }
        Commands::Version => {
            println!("{}", get_version());
            Ok(())
        }
        Commands::Index { command } => match command {
            IndexCommand::Files(args) => run_index_files(args),
        },
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
