mod recon;
mod sims;
pub mod utils;

use crate::recon::{symmetry_eigenvalue, Operator, Reconstruction};
use crate::sims::*;
use crate::utils::*;
use pyo3::prelude::*;
use pyo3::Python;

#[pymodule]
fn shadow_reconstruction(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Sample>()?;
    m.add_class::<Samples>()?;
    m.add_class::<BitString>()?;
    m.add_class::<DensityMatrix>()?;
    m.add_class::<Experiment>()?;
    m.add_class::<Reconstruction>()?;
    m.add_class::<Operator>()?;
    m.add_wrapped(wrap_pyfunction!(symmetry_eigenvalue))?;
    Ok(())
}
