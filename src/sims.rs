use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{One, Zero};
use pyo3::{pyclass, pymethods, PyResult};
use std::fs::File;
use std::io::{Read, Write};
use std::mem::swap;
use std::ops::Mul;
use std::path::Path;

use crate::recon::Operator;
use crate::sims::DensityType::{MixedSparse, PureDense, PureSparse};
use crate::utils::{BitString, OperatorString};
use num_complex::Complex;
use numpy::ndarray::{Array3, Axis};
use numpy::{ndarray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use qip_iterators::iterators::MatrixOp;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sprs::*;

#[pyclass]
pub struct Experiment {
    qubits: usize,
    pairwise_ops: Array3<Complex<f64>>,
}

impl Experiment {
    fn new_raw(qubits: usize, ops: Array3<Complex<f64>>) -> Self {
        Self {
            qubits,
            pairwise_ops: ops,
        }
    }
}

impl Experiment {
    fn sample_raw(&self, rho: &DensityMatrix, samples: usize) -> Result<Samples, String> {
        let n = match &rho.mat {
            MixedSparse(m) => {
                let (a, b) = m.shape();
                if a != b {
                    return Err(format!(
                        "Expected square density matrix but found {:?}",
                        m.shape()
                    ));
                }
                a
            }
            PureSparse(v) => v.dim(),
            PureDense(v) => v.shape()[0],
        };

        if n != 1 << self.qubits {
            return Err(format!(
                "Expected shape {:?} but found {:?}",
                1 << self.qubits,
                n
            ));
        }

        let samples = (0..samples)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let mut perm: Vec<_> = (0..self.qubits).collect();
                perm.shuffle(&mut rng);

                let opis = (0..self.qubits >> 1)
                    .map(|i| {
                        (
                            (perm[2 * i], perm[2 * i + 1]),
                            rng.gen_range(0..self.pairwise_ops.shape()[0]),
                        )
                    })
                    .collect::<Vec<_>>();

                // Now we have our permutation and our operator.
                // Lets compute (U.U.U.U)S = US
                let f = rng.gen();
                let i = match &rho.mat {
                    PureDense(v) => {
                        let ops_it = opis.iter().map(|((i, j), opi)| {
                            let op = self.pairwise_ops.index_axis(Axis(0), *opi);
                            ((*i, *j), op)
                        });

                        let i = measure_channel_pure_dense(self.qubits, ops_it, v.view(), f);
                        i
                    }
                    _ => unimplemented!(),
                };

                Sample::new(opis, BitString::new_short(i, self.qubits).make_long())
            })
            .collect::<Vec<_>>();
        Ok(Samples {
            l: self.qubits,
            ops: self.pairwise_ops.clone(),
            samples,
        })
    }
}

#[pymethods]
impl Experiment {
    #[new]
    fn new(qubits: usize, ops: PyReadonlyArray3<Complex<f64>>) -> Self {
        let ops = ops.as_array().to_owned();
        Self::new_raw(qubits, ops)
    }

    fn sample(&self, rho: &DensityMatrix, samples: usize) -> PyResult<Samples> {
        self.sample_raw(rho, samples).map_err(PyValueError::new_err)
    }
}

// Performs best when u and s are csr
fn measure_channel_pure_dense<'a, It>(
    qubits: usize,
    ops: It,
    rho: ArrayView1<Complex<f64>>,
    mut random_float: f64,
) -> usize
where
    It: IntoIterator<Item = ((usize, usize), ArrayView2<'a, Complex<f64>>)>,
{
    // let u = kron_helper(ops.iter().map(|op| make_sprs(*op)));
    let ops = ops
        .into_iter()
        .map(|((i, j), op)| MatrixOp::new_matrix([i, j], op.iter().copied().collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);
    let mut t_sp = rho.to_owned();
    let mut t_usp = Array1::zeros((t_sp.len(),));

    // b is the "input" to the ops, a is the arena.
    let mut b = t_sp.as_slice_mut().unwrap();
    let mut a = t_usp.as_slice_mut().unwrap();

    for op in ops.iter() {
        // b is the "input", the last thing written to.
        swap(&mut a, &mut b);
        // Now a is the input, we write to b
        qip_iterators::matrix_ops::apply_op(qubits, op, a, b, 0, 0);
        // a is cleared, all needed data in b.
        a.iter_mut().for_each(|x| *x = Complex::zero());
        // b is last thing written to, "input" for next stage.
    }

    // Value in b represents application of ops to sp.
    let usp = b;

    for (i, busp) in usp.iter().enumerate() {
        let p = busp.norm_sqr();

        random_float -= p;
        if random_float <= 0.0 {
            return i;
        }
    }
    (1 << qubits) - 1
}

fn apply_ops<'a, OPS>(
    qubits: usize,
    ops: OPS,
    mut a: &'a mut [Complex<f64>],
    mut b: &'a mut [Complex<f64>],
) where
    OPS: IntoIterator<Item = MatrixOp<Complex<f64>>>,
{
    for op in ops.into_iter() {
        // a is the input, we write to b
        qip_iterators::matrix_ops::apply_op(qubits, &op, a, b, 0, 0);
        // b is the input to the next stage, so call it a.
        swap(&mut a, &mut b);
        // b is cleared, all needed data in a.
        b.iter_mut().for_each(|x| *x = Complex::zero());
        // a is last thing written to, "input" for next stage.
    }
    // copy a to b
    a.iter().zip(b.iter_mut()).for_each(|(a, b)| *b = *a);
}

pub enum DensityType {
    MixedSparse(CsMat<Complex<f64>>),
    PureSparse(CsVec<Complex<f64>>),
    PureDense(Array1<Complex<f64>>),
}

#[pyclass]
pub struct DensityMatrix {
    mat: DensityType,
}

#[pymethods]
impl DensityMatrix {
    #[new]
    fn new(
        n: usize,
        rows: PyReadonlyArray1<usize>,
        cols: PyReadonlyArray1<usize>,
        coefs: PyReadonlyArray1<Complex<f64>>,
    ) -> Self {
        let row_slice = rows.as_slice().unwrap();
        let col_slice = cols.as_slice().unwrap();
        let coe_slice = coefs.as_slice().unwrap();
        let ri = row_slice.iter();
        let ci = col_slice.iter();
        let vi = coe_slice.iter();
        let i = ri.zip(ci.zip(vi)).map(|(a, (b, c))| (*a, *b, *c));
        let mat = Self::make_sprs(n, i);
        Self {
            mat: MixedSparse(mat),
        }
    }

    #[staticmethod]
    fn new_pure_sparse(arr: PyReadonlyArray1<Complex<f64>>, tol: Option<f64>) -> Self {
        let tol = tol.unwrap_or(1e-10);
        let arr = arr.as_array();
        let n = arr.len();
        let (indices, data) = arr
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, c)| c.norm() >= tol)
            .unzip();

        let v = CsVec::new(n, indices, data);
        Self { mat: PureSparse(v) }
    }

    #[staticmethod]
    fn new_pure_dense(arr: PyReadonlyArray1<Complex<f64>>) -> Self {
        let arr = arr.as_array().to_owned();
        Self {
            mat: PureDense(arr),
        }
    }

    #[staticmethod]
    fn new_mixed_sparse(arr: PyReadonlyArray2<Complex<f64>>, tol: Option<f64>) -> Self {
        let tol = tol.unwrap_or(1e-10);
        let arr = arr.as_array();

        let mut a = TriMat::new((arr.shape()[0], arr.shape()[1]));
        ndarray::Zip::indexed(arr).for_each(|(row, col), v| {
            if v.norm() > tol {
                a.add_triplet(row, col, *v)
            }
        });
        Self {
            mat: MixedSparse(a.to_csr()),
        }
    }

    fn expectation(&self, op: &Operator) -> PyResult<Complex<f64>> {
        op.opstrings
            .iter()
            .try_fold(
                Complex::zero(),
                |acc, (c, ps)| -> Result<Complex<f64>, String> {
                    let exp = self.expectation_opstring(ps)?;
                    Ok(acc + c * exp)
                },
            )
            .map_err(PyValueError::new_err)
    }

    fn expectation_string(&self, opstring: String) -> PyResult<Complex<f64>> {
        OperatorString::try_from(opstring)
            .and_then(|x| self.expectation_opstring(&x))
            .map_err(PyValueError::new_err)
    }

    // Helper constructors
    #[staticmethod]
    fn new_uniform_in_sector_sprs(qubits: usize, sector: u32) -> Self {
        let indices = (0..1usize << qubits)
            .into_par_iter()
            .filter(|i| i.count_ones() == sector)
            .collect::<Vec<_>>();
        let n = indices.len();
        let f = (n as f64).sqrt();
        let v = CsVec::new(1 << qubits, indices, vec![Complex::one() / f; n]);
        Self { mat: PureSparse(v) }
    }

    #[staticmethod]
    fn new_uniform_in_sector_dense(qubits: usize, sector: u32) -> Self {
        let mut v = Array1::zeros((1 << qubits,));
        let indices = (0..1usize << qubits)
            .into_par_iter()
            .filter(|i| i.count_ones() == sector)
            .collect::<Vec<_>>();
        let f = (indices.len() as f64).sqrt();
        indices.into_iter().for_each(|i| v[i] = Complex::one() / f);
        Self { mat: PureDense(v) }
    }
}

impl DensityMatrix {
    fn expectation_opstring(&self, opstring: &OperatorString) -> Result<Complex<f64>, String> {
        match &self.mat {
            MixedSparse(m) => {
                let opmat = opstring.make_matrix();
                if opmat.shape() != m.shape() {
                    Err(format!(
                        "Expected operator of size {:?} but found {:?}",
                        m.shape(),
                        opmat.shape()
                    ))
                } else {
                    let s = opmat
                        .mul(m)
                        .diag()
                        .iter()
                        .map(|(_, c)| c)
                        .sum::<Complex<f64>>();
                    Ok(s)
                }
            }
            PureSparse(v) => {
                let opmat = opstring.make_matrix();
                if opmat.shape().0 != v.dim() {
                    Err(format!(
                        "Expected operator of size ({}, {}) but found {:?}",
                        v.dim(),
                        v.dim(),
                        opmat.shape()
                    ))
                } else {
                    let mut cv = v.clone();
                    cv.iter_mut().for_each(|(_, c)| *c = c.conj());
                    let s = opmat.mul(v).dot(&cv);
                    Ok(s)
                }
            }
            PureDense(v) => {
                let nqubits = opstring.opstring.len();
                if 1 << nqubits != v.shape()[0] {
                    Err(format!(
                        "Expected operator of size ({}, {}) but found {:?}",
                        v.dim(),
                        v.dim(),
                        1 << nqubits
                    ))
                } else {
                    let ops = opstring
                        .make_matrices_skip_ident()
                        .into_iter()
                        .map(|(i, op)| {
                            MatrixOp::new_matrix([i], op.into_iter().collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>();

                    let mut cv = v.clone();
                    let mut cv_arena = Array1::zeros((v.shape()[0],));
                    apply_ops(
                        nqubits,
                        ops,
                        cv.as_slice_mut().unwrap(),
                        cv_arena.as_slice_mut().unwrap(),
                    );

                    let mut cv = v.clone();
                    cv.iter_mut().zip(v.iter()).for_each(|(a, b)| *a = b.conj());
                    let s = cv_arena.dot(&cv);

                    debug_assert!({
                        let opmat = opstring.make_matrix();
                        let mut cv = v.clone();
                        cv.iter_mut().for_each(|c| *c = c.conj());
                        let news = opmat.mul(v).dot(&cv);
                        (s - news).norm() < f64::EPSILON
                    });

                    Ok(s)
                }
            }
        }
    }

    fn make_sprs<It>(n: usize, items: It) -> CsMat<Complex<f64>>
    where
        It: IntoIterator<Item = (usize, usize, Complex<f64>)>,
    {
        let mut a = TriMat::new((1 << n, 1 << n));
        for (row, col, val) in items.into_iter() {
            a.add_triplet(row, col, val);
        }
        a.to_csr()
    }
}

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
    pub fn new(l: usize, ops: PyReadonlyArray3<Complex<f64>>) -> Self {
        let ops = ops.as_array().to_owned();
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
    fn new(gates: Vec<((usize, usize), usize)>, measurement: BitString) -> Self {
        Self { gates, measurement }
    }

    fn get_gates(&self) -> Vec<((usize, usize), usize)> {
        self.gates.clone()
    }

    fn get_measurement(&self) -> BitString {
        self.measurement.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{make_numcons_pauli_pairs, make_sprs_onehot};
    use num_complex::Complex;
    use num_traits::One;

    impl DensityMatrix {
        fn new_raw_pure_dense(mat: Array1<Complex<f64>>) -> Self {
            Self {
                mat: PureDense(mat),
            }
        }
    }

    fn x_flip_num() -> CsMat<Complex<f64>> {
        let mut ub: TriMat<Complex<f64>> = TriMat::new((1 << 2, 1 << 2));
        ub.add_triplet(0, 0, Complex::one());
        ub.add_triplet(1, 2, Complex::one());
        ub.add_triplet(2, 1, Complex::one());
        ub.add_triplet(3, 3, Complex::one());
        ub.to_csr()
    }

    #[test]
    fn test_sprs_matmul() {
        let qubits = 4;
        let ua = CsMat::eye(1 << 2);
        let ub = x_flip_num();
        let u = kronecker_product(ua.view(), ub.view());
        let b = make_sprs_onehot::<Complex<f64>>(0b0001, 1 << qubits);
        let u_on_b = u.mul(&b).to_dense();
        let bb = make_sprs_onehot::<Complex<f64>>(0b0010, 1 << qubits).to_dense();
        assert_eq!(bb, u_on_b);

        let b = make_sprs_onehot::<Complex<f64>>(0b1000, 1 << qubits);
        let u_on_b = u.mul(&b).to_dense();
        let bb = make_sprs_onehot::<Complex<f64>>(0b1000, 1 << qubits).to_dense();
        assert_eq!(bb, u_on_b);

        let u = kronecker_product(ub.view(), ua.view());
        let b = make_sprs_onehot::<Complex<f64>>(0b0001, 1 << qubits);
        let u_on_b = u.mul(&b).to_dense();
        let bb = make_sprs_onehot::<Complex<f64>>(0b0001, 1 << qubits).to_dense();
        assert_eq!(bb, u_on_b);

        let b = make_sprs_onehot::<Complex<f64>>(0b1000, 1 << qubits);
        let u_on_b = u.mul(&b).to_dense();
        let bb = make_sprs_onehot::<Complex<f64>>(0b0100, 1 << qubits).to_dense();
        assert_eq!(bb, u_on_b);
    }

    #[test]
    fn test_sampling_pure_dense() -> Result<(), String> {
        let qubits = 4;
        let rho = CsVec::new(
            1 << qubits,
            (0..1 << qubits).collect(),
            vec![Complex::one(); 1 << qubits],
        );
        let rho = rho.to_dense();

        let pauli_pairs = make_numcons_pauli_pairs();
        let flat_paulis = pauli_pairs.into_shape((16, 4, 4)).unwrap();
        let exp = Experiment::new_raw(qubits, flat_paulis);

        let rho = DensityMatrix::new_raw_pure_dense(rho);
        let _samples = exp.sample_raw(&rho, 1000)?;

        Ok(())
    }

    #[test]
    fn test_sampling_pure_dense_larger() -> Result<(), String> {
        let qubits = 6;
        let rho = Array1::ones((1 << qubits,));

        let pauli_pairs = make_numcons_pauli_pairs();
        let flat_paulis = pauli_pairs.into_shape((16, 4, 4)).unwrap();
        let exp = Experiment::new_raw(qubits, flat_paulis);

        let rho = DensityMatrix::new_raw_pure_dense(rho);
        let _samples = exp.sample_raw(&rho, 1000)?;

        Ok(())
    }
}
