mod recon;
mod samples;
#[cfg(feature = "sampling")]
mod sims;
pub mod utils;

use crate::recon::*;
use crate::samples::*;
#[cfg(feature = "sampling")]
use crate::sims::*;
use crate::utils::*;
use pyo3::prelude::*;
use pyo3::Python;

#[pymodule]
fn shadow_reconstruction(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Sample>()?;
    m.add_class::<Samples>()?;
    m.add_class::<BitString>()?;
    m.add_class::<Reconstruction>()?;
    m.add_class::<Operator>()?;
    m.add_wrapped(wrap_pyfunction!(symmetry_eigenvalue))?;
    m.add_wrapped(wrap_pyfunction!(get_g_matrix))?;
    #[cfg(feature = "sampling")]
    m.add_class::<DensityMatrix>()?;
    #[cfg(feature = "sampling")]
    m.add_class::<Experiment>()?;
    Ok(())
}
