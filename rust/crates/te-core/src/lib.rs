//! te-core: Shared core logic for Town Elder
//!
//! This crate provides shared primitives for file scanning, git operations,
//! and document parsing. It can be compiled as:
//! - A native Rust library
//! - A Python extension module via PyO3
//! - A standalone CLI binary via clap

use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Minimal Compile-Only Functions (Scaffolding)
// =============================================================================

/// Returns the version of the te-core library.
/// This is a minimal compile-only function to verify the build works.
pub fn get_version() -> &'static str {
    "0.1.0-scaffold"
}

/// Health check function that returns a greeting.
/// Used to verify PyO3 and clap wiring are functional.
pub fn health_check() -> String {
    "te-core: OK".to_string()
}

/// Placeholder function demonstrating the module structure.
/// This will be expanded in subsequent tickets.
pub fn placeholder() -> u32 {
    42
}

// =============================================================================
// Deterministic Doc-ID Generation
// =============================================================================

/// Build deterministic doc ID for file content chunks.
///
/// For chunk_index == 0, uses the path directly.
/// For chunk_index > 0, appends "#chunk:{chunk_index}" to the path.
///
/// Returns the first 16 hex characters of SHA256 hash.
pub fn build_file_doc_id(path: &str, chunk_index: u32) -> String {
    use sha2::{Sha256, Digest};

    let doc_id_input = if chunk_index == 0 {
        path.to_string()
    } else {
        format!("{}#chunk:{}", path, chunk_index)
    };

    let mut hasher = Sha256::new();
    hasher.update(doc_id_input.as_bytes());
    let result = hasher.finalize();

    // Return first 16 hex characters
    format!("{:x}", result)[..16].to_string()
}

/// Return canonical and legacy ID input strings for a path.
///
/// This is used for deletion safety to ensure we can find docs
/// regardless of whether they were indexed with relative or absolute paths.
pub fn get_doc_id_inputs(path: &str, repo_root: &Path) -> Vec<String> {
    let mut doc_id_inputs = vec![path.to_string()];

    // If path is not absolute, also try the resolved absolute path
    let path_obj = Path::new(path);
    if !path_obj.is_absolute() {
        let absolute_path = repo_root.join(path_obj);
        if let Some(resolved) = absolute_path.to_str() {
            doc_id_inputs.push(resolved.to_string());
        }
    }

    doc_id_inputs
}

/// Normalize chunk metadata by merging base and chunk metadata,
/// and validate/normalize the chunk_index field.
///
/// Returns a tuple of (normalized_metadata, chunk_index).
pub fn normalize_chunk_metadata(
    base_metadata: &HashMap<String, serde_json::Value>,
    chunk_metadata: &HashMap<String, serde_json::Value>,
    fallback_chunk_index: u32,
) -> (HashMap<String, serde_json::Value>, u32) {
    use serde_json::Value;

    let mut metadata = base_metadata.clone();
    metadata.extend(chunk_metadata.clone());

    let chunk_index_value = metadata.get("chunk_index");

    let chunk_index = match chunk_index_value {
        Some(Value::Number(n)) => {
            if let Some(idx) = n.as_u64() {
                // idx is always >= 0 since it's u64
                idx as u32
            } else {
                fallback_chunk_index
            }
        }
        // Bool is treated as invalid (matches Python isinstance(..., bool) check)
        Some(Value::Bool(_)) | None => fallback_chunk_index,
        _ => fallback_chunk_index,
    };

    // Always ensure chunk_index is set in metadata
    metadata.insert("chunk_index".to_string(), Value::Number(chunk_index.into()));

    (metadata, chunk_index)
}

// =============================================================================
// PyO3 Python Bindings
// =============================================================================

#[cfg(feature = "python")]
mod pyo3_bindings {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Returns the te-core version as a Python string.
    #[pyfunction]
    pub fn version() -> String {
        get_version().to_string()
    }

    /// Health check function callable from Python.
    #[pyfunction]
    pub fn health() -> String {
        health_check()
    }

    /// Placeholder function callable from Python.
    #[pyfunction]
    pub fn placeholder_fn() -> u32 {
        placeholder()
    }

    /// Build deterministic doc ID for file content chunks.
    ///
    /// For chunk_index == 0, uses the path directly.
    /// For chunk_index > 0, appends "#chunk:{chunk_index}" to the path.
    ///
    /// Returns the first 16 hex characters of SHA256 hash.
    #[pyfunction]
    pub fn build_file_doc_id(path: &str, chunk_index: u32) -> String {
        super::build_file_doc_id(path, chunk_index)
    }

    /// Return canonical and legacy ID input strings for a path.
    ///
    /// This is used for deletion safety to ensure we can find docs
    /// regardless of whether they were indexed with relative or absolute paths.
    #[pyfunction]
    pub fn get_doc_id_inputs(path: &str, repo_root: &str) -> Vec<String> {
        super::get_doc_id_inputs(path, &PathBuf::from(repo_root))
    }

    /// Normalize chunk metadata by merging base and chunk metadata,
    /// and validate/normalize the chunk_index field.
    ///
    /// Returns a tuple of (normalized_metadata, chunk_index).
    #[pyfunction]
    pub fn normalize_chunk_metadata(
        base_metadata: HashMap<String, serde_json::Value>,
        chunk_metadata: HashMap<String, serde_json::Value>,
        fallback_chunk_index: u32,
    ) -> (HashMap<String, serde_json::Value>, u32) {
        super::normalize_chunk_metadata(&base_metadata, &chunk_metadata, fallback_chunk_index)
    }

    /// Define the Python module "town_elder._te_core".
    pub fn create_module(py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyModule>> {
        let module = pyo3::Bound::new(py, "town_elder._te_core")?;
        module.add_function(pyo3::wrap_pyfunction!(version, &module)?)?;
        module.add_function(pyo3::wrap_pyfunction!(health, &module)?)?;
        module.add_function(pyo3::wrap_pyfunction!(placeholder_fn, &module)?)?;
        module.add_function(pyo3::wrap_pyfunction!(build_file_doc_id, &module)?)?;
        module.add_function(pyo3::wrap_pyfunction!(get_doc_id_inputs, &module)?)?;
        module.add_function(pyo3::wrap_pyfunction!(normalize_chunk_metadata, &module)?)?;
        module.add("__version__", get_version())?;
        Ok(module)
    }
}

#[cfg(feature = "python")]
use pyo3_bindings::*;

// =============================================================================
// Clap CLI Bindings
// =============================================================================

use clap::Parser;

/// Minimal CLI for testing the clap binary entrypoint.
#[derive(Parser, Debug)]
#[command(name = "te-core")]
#[command(version = "0.1.0-scaffold")]
#[command(about = "Town Elder Core CLI (scaffolding)", long_about = None)]
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

impl Cli {
    /// Execute the CLI logic.
    pub fn run(self) {
        if self.verbose {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
                .init();
        }

        match self.command {
            Some(Commands::Health) => {
                println!("{}", health_check());
            }
            Some(Commands::Version) => {
                println!("{}", get_version());
            }
            Some(Commands::Placeholder) => {
                println!("{}", placeholder());
            }
            None => {
                // No subcommand: print help
                println!("te-core CLI (scaffolding)");
                println!("Use --help for more information");
            }
        }
    }
}

/// Entry point for the clap binary.
/// Can be run via: `cargo run --manifest-path rust/Cargo.toml --bin te-core`
pub fn main() {
    let cli = Cli::parse();
    cli.run();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(get_version(), "0.1.0-scaffold");
    }

    #[test]
    fn test_health_check() {
        assert_eq!(health_check(), "te-core: OK");
    }

    #[test]
    fn test_placeholder() {
        assert_eq!(placeholder(), 42);
    }
}
