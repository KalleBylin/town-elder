#![cfg(feature = "python")]

//! PyO3 module entrypoint for town_elder
//!
//! This module provides Python bindings for the te-core Rust crate.
//! The module is exposed as `town_elder._te_core`.

use pyo3::prelude::*;
use pyo3::types::PyBool;
use std::collections::HashMap;
use std::path::Path;

/// Python representation of a tracked file.
#[pyclass]
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
#[pyclass]
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

    // Git blob parsing
    m.add_function(wrap_pyfunction!(parse_git_blob_line, m)?)?;
    m.add_function(wrap_pyfunction!(parse_git_blobs, m)?)?;

    // Git diff parsing
    m.add_function(wrap_pyfunction!(parse_git_path, m)?)?;
    m.add_function(wrap_pyfunction!(extract_b_path, m)?)?;

    // Add Python classes
    m.add_class::<PyTrackedFile>()?;
    m.add_class::<PyDiffFile>()?;
    m.add_class::<PyDiffParser>()?;

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
