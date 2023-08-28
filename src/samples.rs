use ndarray::Array3;
use num_complex::Complex;
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::utils::{get_pauli_ops, BitString};
use pyo3::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct Samples {
    pub l: usize,
    pub ops: Array3<Complex<f64>>,
    pub samples: Vec<Sample>,
}

impl Samples {
    pub fn new_raw(l: usize, ops: Array3<Complex<f64>>) -> Self {
        Self {
            l,
            ops,
            samples: vec![],
        }
    }
}

#[pymethods]
impl Samples {
    #[new]
    pub fn new(l: usize, ops: Option<PyReadonlyArray3<Complex<f64>>>) -> Self {
        let ops = ops
            .map(|ops| ops.as_array().to_owned())
            .unwrap_or_else(get_pauli_ops);
        Self::new_raw(l, ops)
    }

    pub fn add(&mut self, gates: Vec<((usize, usize), usize)>, measurement: Vec<bool>) {
        let sample = Sample::new(gates, BitString::from(measurement));
        self.add_sample(sample)
    }

    pub fn subset(&self, n: usize) -> Self {
        let mut rng = thread_rng();
        let subset = self.samples.choose_multiple(&mut rng, n).cloned().collect();
        Self {
            l: self.l,
            ops: self.ops.clone(),
            samples: subset,
        }
    }

    pub fn add_from(&mut self, other: &Samples) {
        self.samples.extend(other.samples.iter().cloned());
    }

    pub fn combine(&self, other: &Samples) -> Self {
        let mut samples = self.samples.clone();
        samples.extend(other.samples.iter().cloned());
        Self {
            l: self.l,
            ops: self.ops.clone(),
            samples,
        }
    }

    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn add_sample(&mut self, sample: Sample) {
        self.samples.push(sample)
    }

    pub fn get_sample(&self, index: usize) -> Sample {
        self.samples[index].clone()
    }

    pub fn save_to(&self, filename: &str) -> PyResult<()> {
        let filepath = Path::new(filename);
        let mut file =
            File::create(filepath).map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        let encoded =
            bincode::serialize(self).map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        file.write_all(&encoded)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load_from(filename: &str) -> PyResult<Self> {
        let filepath = Path::new(filename);
        let mut buf = vec![];
        let mut file =
            File::open(filepath).map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        file.read_to_end(&mut buf)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        bincode::deserialize(&buf).map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub gates: Vec<((usize, usize), usize)>,
    pub measurement: BitString,
}

#[pymethods]
impl Sample {
    #[new]
    pub fn new(gates: Vec<((usize, usize), usize)>, measurement: BitString) -> Self {
        Self { gates, measurement }
    }

    fn get_gates(&self) -> Vec<((usize, usize), usize)> {
        self.gates.clone()
    }

    fn get_measurement(&self) -> BitString {
        self.measurement.clone()
    }
}
