//! PyO3 module entrypoint for town_elder
//!
//! This module provides Python bindings for the te-core Rust crate.
//! The module is exposed as `town_elder._te_core`.

use pyo3::prelude::*;

/// Initialize the PyO3 module with te-core functions.
#[pymodule]
pub fn _te_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Re-export functions from te-core with the python feature
    m.add_function(wrap_pyfunction!(te_core::version, m)?)?;
    m.add_function(wrap_pyfunction!(te_core::health, m)?)?;
    m.add_function(wrap_pyfunction!(te_core::placeholder_fn, m)?)?;
    m.add("__version__", te_core::get_version())?;
    Ok(())
}
