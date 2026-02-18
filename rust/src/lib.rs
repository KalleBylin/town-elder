#![cfg(feature = "python")]

//! PyO3 module entrypoint for town_elder
//!
//! This module provides Python bindings for the te-core Rust crate.
//! The module is exposed as `town_elder._te_core`.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Python representation of a tracked file.
#[pyclass(get_all)]
pub struct PyTrackedFile {
    pub path: String,
    pub blob_hash: String,
    pub mode: String,
}

#[pymethods]
impl PyTrackedFile {
    #[new]
    fn new(path: String, blob_hash: String, mode: String) -> Self {
        Self {
            path,
            blob_hash,
            mode,
        }
    }
}

/// Python representation of a diff file.
#[pyclass(get_all)]
pub struct PyDiffFile {
    pub path: String,
    pub status: String,
    pub hunks: Vec<String>,
}

#[pymethods]
impl PyDiffFile {
    #[new]
    fn new(path: String, status: String, hunks: Vec<String>) -> Self {
        Self {
            path,
            status,
            hunks,
        }
    }
}

/// Python representation of a commit.
#[pyclass(get_all)]
pub struct PyCommit {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub date: String,
    pub files_changed: Vec<String>,
}

#[pymethods]
impl PyCommit {
    #[new]
    fn new(hash: String, message: String, author: String, date: String, files_changed: Vec<String>) -> Self {
        Self {
            hash,
            message,
            author,
            date,
            files_changed,
        }
    }
}

/// Python DiffParser class.
#[pyclass]
pub struct PyDiffParser {
    parser: te_core::DiffParser,
}

#[pymethods]
impl PyDiffParser {
    #[new]
    fn new(warn_on_parse_error: bool) -> Self {
        Self {
            parser: te_core::DiffParser::new(warn_on_parse_error),
        }
    }

    /// Parse git diff output into list of PyDiffFile.
    fn parse(&self, diff_output: &str) -> Vec<PyDiffFile> {
        self.parser
            .parse(diff_output)
            .into_iter()
            .map(|df| PyDiffFile {
                path: df.path,
                status: df.status,
                hunks: df.hunks,
            })
            .collect()
    }

    /// Convert diff to plain text for embedding.
    fn parse_diff_to_text(&self, diff_output: &str) -> String {
        self.parser.parse_diff_to_text(diff_output)
    }
}

/// Initialize the PyO3 module with te-core functions.
#[pymodule]
pub fn _te_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Re-export functions from te-core
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(health, m)?)?;
    m.add_function(wrap_pyfunction!(placeholder_fn, m)?)?;
    m.add_function(wrap_pyfunction!(build_file_doc_id, m)?)?;
    m.add_function(wrap_pyfunction!(get_doc_id_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_chunk_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(scan_files, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_extensions, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_excludes, m)?)?;

    // Git blob parsing
    m.add_function(wrap_pyfunction!(parse_git_blob_line, m)?)?;
    m.add_function(wrap_pyfunction!(parse_git_blobs, m)?)?;

    // Git diff parsing
    m.add_function(wrap_pyfunction!(parse_git_path, m)?)?;
    m.add_function(wrap_pyfunction!(extract_b_path, m)?)?;

    // Commit log parsing
    m.add_function(wrap_pyfunction!(parse_commit_header, m)?)?;
    m.add_function(wrap_pyfunction!(parse_numstat_line, m)?)?;
    m.add_function(wrap_pyfunction!(parse_commits_with_numstat, m)?)?;
    m.add_function(wrap_pyfunction!(is_commit_header_line, m)?)?;

    // Diff truncation
    m.add_function(wrap_pyfunction!(truncate_diff, m)?)?;
    m.add_function(wrap_pyfunction!(is_diff_truncated, m)?)?;

    // Commit text assembly
    m.add_function(wrap_pyfunction!(assemble_commit_text, m)?)?;
    m.add_function(wrap_pyfunction!(append_truncation_note, m)?)?;

    // Constants
    m.add("DEFAULT_MAX_DIFF_SIZE", te_core::DEFAULT_MAX_DIFF_SIZE)?;

    // RST parsing
    m.add_function(wrap_pyfunction!(parse_rst_content, m)?)?;
    m.add_function(wrap_pyfunction!(find_section_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(get_heading_level, m)?)?;
    m.add_function(wrap_pyfunction!(extract_directives, m)?)?;
    m.add_function(wrap_pyfunction!(check_temporal_tags, m)?)?;
    m.add_function(wrap_pyfunction!(get_chunk_metadata, m)?)?;

    // Add Python classes
    m.add_class::<PyTrackedFile>()?;
    m.add_class::<PyDiffFile>()?;
    m.add_class::<PyDiffParser>()?;
    m.add_class::<PyCommit>()?;
    m.add_class::<PyRSTChunk>()?;

    m.add("__version__", te_core::get_version())?;
    Ok(())
}

// =============================================================================
// Git Blob Parsing Functions
// =============================================================================

/// Return te-core version string.
#[pyfunction]
fn version() -> &'static str {
    te_core::get_version()
}

/// Return te-core health check string.
#[pyfunction]
fn health() -> String {
    te_core::health_check()
}

/// Return placeholder integer.
#[pyfunction]
fn placeholder_fn() -> u32 {
    te_core::placeholder()
}

/// Build deterministic doc ID for file content chunks.
#[pyfunction]
fn build_file_doc_id(path: &str, chunk_index: u32) -> String {
    te_core::build_file_doc_id(path, chunk_index)
}

/// Return canonical and legacy ID input strings for a path.
#[pyfunction]
fn get_doc_id_inputs(path: &str, repo_root: &str) -> Vec<String> {
    te_core::get_doc_id_inputs(path, Path::new(repo_root))
}

/// Return scanner default file extensions.
#[pyfunction]
fn get_default_extensions() -> Vec<String> {
    te_core::DEFAULT_FILE_EXTENSIONS
        .iter()
        .map(|value| value.to_string())
        .collect()
}

/// Return scanner default exclude patterns.
#[pyfunction]
fn get_default_excludes() -> Vec<String> {
    te_core::DEFAULT_FILE_EXCLUDES
        .iter()
        .map(|value| value.to_string())
        .collect()
}

/// Scan files under a root directory and return deterministic path strings.
#[pyfunction]
#[pyo3(signature = (root_path, extensions=None, exclude_patterns=None))]
fn scan_files(
    root_path: &str,
    extensions: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
) -> Vec<String> {
    let extension_set = extensions.map(|values| values.into_iter().collect::<HashSet<_>>());
    let exclude_set = exclude_patterns.map(|values| values.into_iter().collect::<HashSet<_>>());

    te_core::scan_files(
        Path::new(root_path),
        extension_set.as_ref(),
        exclude_set.as_ref(),
    )
    .into_iter()
    .map(|path| path.to_string_lossy().to_string())
    .collect()
}

/// Normalize chunk metadata and return (metadata, chunk_index).
#[pyfunction]
fn normalize_chunk_metadata(
    py: Python<'_>,
    base_metadata: HashMap<String, PyObject>,
    chunk_metadata: HashMap<String, PyObject>,
    fallback_chunk_index: i64,
) -> PyResult<(HashMap<String, PyObject>, i64)> {
    let mut metadata = base_metadata;
    metadata.extend(chunk_metadata);

    let chunk_index = match metadata.get("chunk_index") {
        Some(value) => {
            let bound = value.bind(py);
            if bound.is_instance_of::<PyBool>() {
                fallback_chunk_index
            } else {
                match bound.extract::<i64>() {
                    Ok(index) if index >= 0 => index,
                    _ => fallback_chunk_index,
                }
            }
        }
        None => fallback_chunk_index,
    };

    metadata.insert("chunk_index".to_string(), chunk_index.into_py(py));
    Ok((metadata, chunk_index))
}

/// Parse a single line from `git ls-files --stage` output.
#[pyfunction]
fn parse_git_blob_line(line: &str) -> Option<PyTrackedFile> {
    te_core::parse_git_blob_line(line).map(|tf| PyTrackedFile {
        path: tf.path,
        blob_hash: tf.blob_hash,
        mode: tf.mode,
    })
}

/// Parse git ls-files --stage output.
#[pyfunction]
fn parse_git_blobs(output: &str) -> HashMap<String, PyTrackedFile> {
    te_core::parse_git_blobs(output)
        .into_iter()
        .map(|(k, v)| (k, PyTrackedFile {
            path: v.path,
            blob_hash: v.blob_hash,
            mode: v.mode,
        }))
        .collect()
}

// =============================================================================
// Git Diff Parsing Functions
// =============================================================================

/// Parse a git path from a diff header component.
#[pyfunction]
fn parse_git_path(path: &str) -> String {
    te_core::parse_git_path(path)
}

/// Extract the b/ path from a diff --git header line.
#[pyfunction]
fn extract_b_path(line: &str) -> Option<String> {
    te_core::extract_b_path(line)
}

// =============================================================================
// RST Parsing Functions
// =============================================================================

/// Python representation of an RST chunk.
#[pyclass(get_all, set_all)]
pub struct PyRSTChunk {
    pub text: String,
    pub section_path: Vec<String>,
    pub directives: std::collections::HashMap<String, Vec<String>>,
    pub temporal_tags: Vec<String>,
    pub chunk_index: u32,
}

#[pymethods]
impl PyRSTChunk {
    #[new]
    fn new(
        text: String,
        section_path: Vec<String>,
        directives: std::collections::HashMap<String, Vec<String>>,
        temporal_tags: Vec<String>,
        chunk_index: u32,
    ) -> Self {
        Self {
            text,
            section_path,
            directives,
            temporal_tags,
            chunk_index,
        }
    }
}

/// Parse RST content and return list of chunks.
#[pyfunction]
fn parse_rst_content(content: &str) -> Vec<PyRSTChunk> {
    te_core::parse_rst_content(content)
        .into_iter()
        .map(|chunk| PyRSTChunk {
            text: chunk.text,
            section_path: chunk.section_path,
            directives: chunk.directives,
            temporal_tags: chunk.temporal_tags,
            chunk_index: chunk.chunk_index,
        })
        .collect()
}

/// Find section boundaries in RST content.
#[pyfunction]
fn find_section_boundaries(content: &str) -> Vec<(usize, String, String)> {
    te_core::find_section_boundaries(content)
}

/// Get heading level from underline character.
#[pyfunction]
fn get_heading_level(underline: &str) -> u32 {
    te_core::get_heading_level(underline)
}

/// Extract directives from chunk text.
#[pyfunction]
fn extract_directives(chunk_text: &str) -> (std::collections::HashMap<String, Vec<String>>, Vec<String>) {
    te_core::extract_directives(chunk_text)
}

/// Check for temporal tags in text.
#[pyfunction]
fn check_temporal_tags(text: &str) -> Vec<String> {
    te_core::check_temporal_tags(text)
}

/// Get chunk metadata as a dictionary.
#[pyfunction]
fn get_chunk_metadata(
    py: Python<'_>,
    chunk: &PyRSTChunk,
) -> PyResult<std::collections::HashMap<String, PyObject>> {
    let te_chunk = te_core::RSTChunk {
        text: chunk.text.clone(),
        section_path: chunk.section_path.clone(),
        directives: chunk.directives.clone(),
        temporal_tags: chunk.temporal_tags.clone(),
        chunk_index: chunk.chunk_index,
    };

    let metadata = te_core::get_chunk_metadata(&te_chunk);
    let mut python_metadata: std::collections::HashMap<String, PyObject> =
        std::collections::HashMap::new();
    for (key, value) in metadata {
        python_metadata.insert(key, json_value_to_pyobject(py, &value)?);
    }
    Ok(python_metadata)
}

fn json_value_to_pyobject(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    let object = match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(inner) => inner.into_py(py),
        serde_json::Value::Number(inner) => {
            if let Some(inner) = inner.as_i64() {
                inner.into_py(py)
            } else if let Some(inner) = inner.as_u64() {
                inner.into_py(py)
            } else if let Some(inner) = inner.as_f64() {
                inner.into_py(py)
            } else {
                py.None()
            }
        }
        serde_json::Value::String(inner) => inner.into_py(py),
        serde_json::Value::Array(inner) => inner
            .iter()
            .map(|item| json_value_to_pyobject(py, item))
            .collect::<PyResult<Vec<_>>>()?
            .into_py(py),
        serde_json::Value::Object(inner) => {
            let dict = PyDict::new_bound(py);
            for (key, item) in inner {
                dict.set_item(key, json_value_to_pyobject(py, item)?)?;
            }
            dict.into_py(py)
        }
    };
    Ok(object)
}

// =============================================================================
// Commit Log Parsing Functions
// =============================================================================

/// Parse a git log commit header line into a PyCommit.
#[pyfunction]
fn parse_commit_header(line: &str) -> Option<PyCommit> {
    te_core::parse_commit_header(line).map(|c| PyCommit {
        hash: c.hash,
        message: c.message,
        author: c.author,
        date: c.date,
        files_changed: c.files_changed,
    })
}

/// Parse a numstat line to extract file path.
#[pyfunction]
fn parse_numstat_line(line: &str) -> Option<String> {
    te_core::parse_numstat_line(line)
}

/// Parse git log output with numstat into PyCommit objects.
#[pyfunction]
fn parse_commits_with_numstat(output: &str) -> Vec<PyCommit> {
    te_core::parse_commits_with_numstat(output)
        .into_iter()
        .map(|c| PyCommit {
            hash: c.hash,
            message: c.message,
            author: c.author,
            date: c.date,
            files_changed: c.files_changed,
        })
        .collect()
}

/// Check if a line contains the commit record separator.
#[pyfunction]
fn is_commit_header_line(line: &str) -> bool {
    te_core::is_commit_header_line(line)
}

// =============================================================================
// Diff Truncation Functions
// =============================================================================

/// Truncate diff text if it exceeds the maximum size.
#[pyfunction]
fn truncate_diff(diff: &str, max_size: usize) -> String {
    te_core::truncate_diff(diff, max_size)
}

/// Check if diff text contains the truncation marker.
#[pyfunction]
fn is_diff_truncated(diff: &str) -> bool {
    te_core::is_diff_truncated(diff)
}

// =============================================================================
// Commit Text Assembly Functions
// =============================================================================

/// Assemble commit text for embedding.
#[pyfunction]
fn assemble_commit_text(message: &str, diff_text: &str) -> String {
    te_core::assemble_commit_text(message, diff_text)
}

/// Check if the original diff was truncated and append truncation note.
#[pyfunction]
fn append_truncation_note(text: &str, diff: &str) -> String {
    te_core::append_truncation_note(text, diff)
}
