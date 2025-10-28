mod agent;
mod environment;
pub mod geometry;
pub mod state;
mod utils;

pub use agent::Agent;
pub use environment::Environment;

use pyo3::prelude::*;

/// Register spatial classes to the provided Python module
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Environment>()?;
    m.add_class::<Agent>()?;
    Ok(())
}
