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
// Git Blob Parsing (git ls-files --stage)
// =============================================================================

/// Valid file modes for git-tracked files
const VALID_FILE_MODES: &[&str] = &["100644", "100755"];

/// Represents a git-tracked file with its blob hash.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackedFile {
    /// Relative path from repo root
    pub path: String,
    /// 40-character SHA-1 hex digest
    pub blob_hash: String,
    /// File mode (e.g., "100644")
    pub mode: String,
}

/// Parse a single line from `git ls-files --stage` output.
///
/// Format: `<mode> <blob_hash> <stage>\t<path>`
/// Example: `100644 0acd8c61f0d9dc1e5db7ad7c2dbce2bb16b8d6de 0\t.beads/.gitignore`
///
/// Returns Some(TrackedFile) if parsing succeeds and mode is valid (100644 or 100755).
/// Returns None if:
/// - Line is empty or malformed
/// - Mode is not a valid file mode (skips symlinks, etc.)
pub fn parse_git_blob_line(line: &str) -> Option<TrackedFile> {
    if line.is_empty() {
        return None;
    }

    // Format: <mode> <blob_hash> <stage>\t<path>
    // Split on tab to preserve path (may contain spaces)
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() != 2 {
        return None;
    }

    let metadata = parts[0];
    let file_path = parts[1];

    // Parse metadata: mode, blob_hash, stage
    let meta_parts: Vec<&str> = metadata.split_whitespace().collect();
    if meta_parts.len() != 3 {
        return None;
    }

    let mode = meta_parts[0];
    let blob_hash = meta_parts[1];
    let _stage = meta_parts[2]; // Stage is not needed for our use case

    // Skip symlinks (mode 120000) and other special files
    if !VALID_FILE_MODES.contains(&mode) {
        return None;
    }

    // Validate blob_hash is a valid SHA-1 hex (40 characters)
    if blob_hash.len() != 40 || !blob_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }

    Some(TrackedFile {
        path: file_path.to_string(),
        blob_hash: blob_hash.to_string(),
        mode: mode.to_string(),
    })
}

/// Parse git ls-files --stage output into a dictionary of TrackedFile.
///
/// Input: output from `git ls-files --stage`
/// Returns: HashMap mapping file path to TrackedFile
pub fn parse_git_blobs(output: &str) -> HashMap<String, TrackedFile> {
    let mut tracked_files: HashMap<String, TrackedFile> = HashMap::new();

    for line in output.lines() {
        if let Some(tracked_file) = parse_git_blob_line(line) {
            tracked_files.insert(tracked_file.path.clone(), tracked_file);
        }
    }

    tracked_files
}

// =============================================================================
// Git Diff Parsing
// =============================================================================

/// Git diff header prefix
const GIT_DIFF_HEADER_PREFIX: &str = "diff --git";

/// Extract the path from a git diff header component.
///
/// Handles both:
/// - Unquoted: b/path or a/path
/// - Quoted: "b/path" or "a/path"
///
/// Returns the file path without the a/ or b/ prefix.
pub fn parse_git_path(path: &str) -> String {
    let mut path = path.to_string();

    // Remove leading a/ or b/ prefix
    if path.starts_with("a/") || path.starts_with("b/") {
        path = path[2..].to_string();
    }

    // Handle quoted paths - remove surrounding quotes
    if path.starts_with('"') && path.ends_with('"') && path.len() >= 2 {
        path = path[1..path.len() - 1].to_string();
    }

    path
}

/// Extract the 'b/' path from a 'diff --git' line.
///
/// Handles both quoted and unquoted formats:
/// - diff --git a/path b/path
/// - diff --git "a/path with spaces" "b/path with spaces"
///
/// Returns the b/ path component (with or without quotes), or None if parsing fails.
pub fn extract_b_path(line: &str) -> Option<String> {
    // Remove the 'diff --git ' prefix
    if !line.starts_with(GIT_DIFF_HEADER_PREFIX) {
        return None;
    }

    let remainder = line[GIT_DIFF_HEADER_PREFIX.len()..].trim_start();
    if remainder.is_empty() {
        return None;
    }

    // Check if paths are quoted
    if remainder.starts_with('"') {
        // Quoted format: "a/path" "b/path"
        // Find the closing quote of the first quoted string
        let first_quote_end = remainder[1..].find('"')? + 1; // Adjust for slice offset

        // Find the opening quote of the second quoted string
        let second_quote_start = remainder[first_quote_end + 1..].find('"')? + first_quote_end + 1;

        // Find the closing quote of the second quoted string
        let second_quote_end = remainder[second_quote_start + 1..].find('"');
        let second_quote_end = match second_quote_end {
            Some(end) => end + second_quote_start + 1,
            None => remainder.len(), // No closing quote - take rest of line
        };

        // Extract the b/ path (second quoted string)
        Some(remainder[second_quote_start + 1..second_quote_end].to_string())
    } else {
        // Unquoted format: a/path b/path
        let parts: Vec<&str> = remainder.split_whitespace().collect();
        if parts.len() >= 2 {
            Some(parts[1].to_string()) // Last part is the b/ path
        } else {
            None
        }
    }
}

/// Represents a file change in a diff.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffFile {
    /// File path
    pub path: String,
    /// Status: "added", "modified", "deleted"
    pub status: String,
    /// List of hunk texts
    pub hunks: Vec<String>,
}

/// Parser for git diff output.
#[derive(Debug, Clone)]
pub struct DiffParser {
    /// If true, track parse error count
    warn_on_parse_error: bool,
}

impl DiffParser {
    /// Create a new DiffParser.
    pub fn new(warn_on_parse_error: bool) -> Self {
        Self {
            warn_on_parse_error,
        }
    }

    /// Parse git diff output into file changes.
    pub fn parse(&self, diff_output: &str) -> Vec<DiffFile> {
        let mut files: Vec<DiffFile> = Vec::new();

        let mut current_file: Option<String> = None;
        let mut current_status: Option<String> = None;
        let mut current_hunks: Vec<String> = Vec::new();
        let mut current_hunk_lines: Vec<String> = Vec::new();
        let mut _parse_error_count: usize = 0;

        for line in diff_output.lines() {
            // New file start
            if line.starts_with("diff --git") {
                // Yield previous file if exists
                if let Some(ref file_path) = current_file {
                    if !current_hunk_lines.is_empty() {
                        current_hunks.push(current_hunk_lines.join("\n"));
                    }
                    files.push(DiffFile {
                        path: file_path.clone(),
                        status: current_status.clone().unwrap_or_else(|| "modified".to_string()),
                        hunks: current_hunks.clone(),
                    });
                }

                // Parse the file path from "diff --git a/path b/path" or quoted variant
                let b_path = extract_b_path(line);
                if let Some(ref path) = b_path {
                    current_file = Some(parse_git_path(path));
                } else {
                    if self.warn_on_parse_error {
                        eprintln!(
                            "Warning: Failed to parse diff header: {}...",
                            &line[..line.len().min(60)]
                        );
                    }
                    _parse_error_count += 1;
                    current_file = None; // Explicitly set to None to prevent content association
                }
                current_status = None;
                current_hunks = Vec::new();
                current_hunk_lines = Vec::new();
            }
            // File status
            else if line.starts_with("new file") {
                current_status = Some("added".to_string());
            } else if line.starts_with("deleted file") {
                current_status = Some("deleted".to_string());
            } else if line.starts_with("old mode") || line.starts_with("new mode") {
                // Ignore mode changes
            }
            // Hunk header
            else if line.starts_with("@@") {
                if !current_hunk_lines.is_empty() {
                    current_hunks.push(current_hunk_lines.join("\n"));
                }
                current_hunk_lines = vec![line.to_string()];
            }
            // Regular diff content
            else if current_file.is_some() {
                current_hunk_lines.push(line.to_string());
            }
        }

        // Yield the last file
        if let Some(ref file_path) = current_file {
            if !current_hunk_lines.is_empty() {
                current_hunks.push(current_hunk_lines.join("\n"));
            }
            files.push(DiffFile {
                path: file_path.clone(),
                status: current_status.unwrap_or_else(|| "modified".to_string()),
                hunks: current_hunks,
            });
        }

        files
    }

    /// Convert a diff to plain text for embedding.
    pub fn parse_diff_to_text(&self, diff_output: &str) -> String {
        let parts: Vec<String> = self
            .parse(diff_output)
            .iter()
            .map(|diff_file| {
                let mut file_parts = vec![format!("File: {} ({})", diff_file.path, diff_file.status)];
                for hunk in &diff_file.hunks {
                    file_parts.push(hunk.clone());
                }
                file_parts.join("\n")
            })
            .collect();
        parts.join("\n\n")
    }
}

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
    // PyO3 bindings are defined in rust/src/lib.rs instead
    // This module re-exports te-core functions for the Python module
}

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
