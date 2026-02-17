//! te-core: Shared core logic for Town Elder
//!
//! This crate provides shared primitives for file scanning, git operations,
//! and document parsing. It can be compiled as:
//! - A native Rust library
//! - A Python extension module via PyO3
//! - A standalone CLI binary via clap

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
// PyO3 Python Bindings
// =============================================================================

#[cfg(feature = "python")]
mod pyo3_bindings {
    use super::*;

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

    /// Define the Python module "town_elder._te_core".
    pub fn create_module(py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyModule>> {
        let module = pyo3::Bound::new(py, "town_elder._te_core")?;
        module.add_function(pyo3::wrap_pyfunction!(version, &module)?)?;
        module.add_function(pyo3::wrap_pyfunction!(health, &module)?)?;
        module.add_function(pyo3::wrap_pyfunction!(placeholder_fn, &module)?)?;
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
