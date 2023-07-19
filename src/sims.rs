use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{One, Zero};
use pyo3::{pyclass, pymethods, PyAny, PyResult, Python};
use std::fs::File;
use std::io::{Read, Write};
use std::mem::swap;
use std::ops::Mul;
use std::path::Path;

use crate::recon::Operator;
use crate::sims::DensityType::{MixedSparse, PureDense, PureSparse};
use crate::utils::{
    kron_helper, make_perm, make_sprs, make_sprs_onehot, scipy_mat, OperatorString,
};
use num_complex::Complex;
use numpy::ndarray::{Array2, Array3, Axis};
use numpy::{ndarray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use qip_iterators::iterators::MatrixOp;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sprs::vec::IntoSparseVecIter;
use sprs::*;

#[pyclass]
pub struct Experiment {
    qubits: usize,
    pairwise_ops: Array3<Complex<f64>>,
    perms: Option<Array2<usize>>,
}

impl Experiment {
    fn new_raw(
        qubits: usize,
        ops: Option<Array3<Complex<f64>>>,
        perms: Option<Array2<usize>>,
    ) -> Self {
        Self {
            qubits,
            pairwise_ops: ops.unwrap(),
            perms,
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
                if let Some(perms) = &self.perms {
                    let i = rng.gen_range(0..perms.shape()[0]);
                    perm.iter_mut()
                        .zip(perms.index_axis(Axis(0), i).iter())
                        .for_each(|(a, b)| {
                            *a = *b;
                        });
                } else {
                    perm.shuffle(&mut rng);
                };
                let opis = (0..self.qubits >> 1)
                    .map(|_| rng.gen_range(0..self.pairwise_ops.shape()[0]))
                    .collect::<Vec<_>>();

                // Now we have our permutation and our operator.
                // Lets compute (U.U.U.U)S = US
                let ops = opis
                    .iter()
                    .map(|opi| {
                        let op = self.pairwise_ops.index_axis(Axis(0), *opi);
                        op
                    })
                    .collect::<Vec<_>>();

                let f = rng.gen();
                let i = match &rho.mat {
                    MixedSparse(m) => {
                        let i = measure_channel(self.qubits, &ops, &perm, m, f);
                        // Check same as dumb
                        debug_assert_eq!(i, measure_channel_dumb(&ops, &perm, m, f));
                        i
                    }
                    PureSparse(v) => {
                        let i = measure_channel_pure(self.qubits, &ops, &perm, v, f);
                        // Check same as dumb
                        debug_assert_eq!(i, {
                            let vv = v.to_dense();
                            measure_channel_pure_dense(self.qubits, &ops, &perm, vv.view(), f)
                        }, "Output from pure sparse measurement not the same as from dense measurement.");
                        debug_assert!({
                            let mut a = TriMat::new((v.dim(), v.dim()));
                            v.into_sparse_vec_iter().for_each(|(i, ci)| {
                                v.into_sparse_vec_iter().for_each(|(j, cj)| {
                                    a.add_triplet(i, j, ci * cj.conj());
                                });
                            });
                            let m = a.to_csr();
                            let newi = measure_channel_dumb(&ops, &perm, &m, f);
                            i == newi
                        } , "Output from pure sparse measurement not the same as from dumb measurement.");
                        i
                    }
                    PureDense(v) => {
                        let i = measure_channel_pure_dense(self.qubits, &ops, &perm, v.view(), f);
                        // Check same as dumb
                        debug_assert!({
                            let mut a = TriMat::new((v.shape()[0], v.shape()[0]));
                            v.iter().enumerate().for_each(|(i, ci)| {
                                v.iter().enumerate().for_each(|(j, cj)| {
                                    if (ci * cj.conj()).norm() > 1e-10 {
                                        a.add_triplet(i, j, ci * cj.conj());
                                    }
                                });
                            });
                            let m = a.to_csr();
                            let newi = measure_channel_dumb(&ops, &perm, &m, f);
                            i == newi
                        }, "Output from pure dense measurement not the same as from dumb measurement.");
                        i
                    }
                };

                Sample::new(opis, perm, i)
            })
            .collect::<Vec<_>>();
        Ok(Samples { samples })
    }
}

#[pymethods]
impl Experiment {
    #[new]
    fn new(
        qubits: usize,
        ops: Option<PyReadonlyArray3<Complex<f64>>>,
        perms: Option<PyReadonlyArray2<usize>>,
    ) -> Self {
        let ops = ops.unwrap().as_array().to_owned();
        let perms = perms.map(|x| x.as_array().to_owned());

        Self::new_raw(qubits, Some(ops), perms)
    }

    fn sample(&self, rho: &DensityMatrix, samples: usize) -> PyResult<Samples> {
        self.sample_raw(rho, samples).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn make_perm_mat(py: Python<'_>, perm: Vec<usize>) -> PyResult<&PyAny> {
        let permmat = make_perm::<f64>(&perm);
        scipy_mat(py, &permmat).map_err(PyValueError::new_err)
    }
}

// Performs best when u and s are csr
fn measure_channel_pure(
    qubits: usize,
    ops: &[ArrayView2<Complex<f64>>],
    perm: &[usize],
    rho: &CsVec<Complex<f64>>,
    mut random_float: f64,
) -> usize {
    let u = kron_helper(ops.iter().map(|op| make_sprs(*op)));
    let s = make_perm::<Complex<f64>>(&perm);
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);

    let sp = s.mul(rho);
    let usp = u.mul(&sp);
    for i in 0..1 << qubits {
        // let b = make_sprs_onehot(i, 1 << qubits);
        // let bus = b.mul(&u).mul(&s);
        // let busp = bus.dot(rho);

        let b = make_sprs_onehot(i, 1 << qubits);
        let busp = b.dot(&usp);

        let p = busp.norm_sqr();

        random_float -= p;
        if random_float <= 0.0 {
            return i;
        }
    }
    (1 << qubits) - 1
}

// Performs best when u and s are csr
fn measure_channel_pure_dense(
    qubits: usize,
    ops: &[ArrayView2<Complex<f64>>],
    perm: &[usize],
    rho: ArrayView1<Complex<f64>>,
    mut random_float: f64,
) -> usize {
    // let u = kron_helper(ops.iter().map(|op| make_sprs(*op)));
    let ops = ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            MatrixOp::new_matrix([2 * i, 2 * i + 1], op.iter().copied().collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    let s = make_perm::<Complex<f64>>(&perm);
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);
    let mut t_sp = s.mul(&rho);
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

    for i in 0..1 << qubits {
        let busp = usp[i];
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
    // b is the "input" to the ops, a is the arena.
    for op in ops.into_iter() {
        // b is the "input", the last thing written to.
        swap(&mut a, &mut b);
        // Now a is the input, we write to b
        qip_iterators::matrix_ops::apply_op(qubits, &op, a, b, 0, 0);
        // a is cleared, all needed data in b.
        a.iter_mut().for_each(|x| *x = Complex::zero());
        // b is last thing written to, "input" for next stage.
    }
}

// Performs best when u, s, rho are csr
fn measure_channel(
    qubits: usize,
    ops: &[ArrayView2<Complex<f64>>],
    perm: &[usize],
    rho: &CsMat<Complex<f64>>,
    mut random_float: f64,
) -> usize {
    let u = kron_helper(ops.iter().map(|op| make_sprs(*op)));
    let s = make_perm::<Complex<f64>>(&perm);
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);
    for i in 0..1 << qubits {
        let b = make_sprs_onehot(i, 1 << qubits);
        let bus = b.mul(&u).mul(&s);
        let mut bust = bus.clone();
        bust.iter_mut().for_each(|(_, c)| *c = c.conj());
        let buspsub = bus.mul(rho).dot(&bust);
        debug_assert!(buspsub.im < 1e-10);
        random_float -= buspsub.re;
        if random_float <= 0.0 {
            return i;
        }
    }
    (1 << qubits) - 1
}

fn measure_channel_dumb(
    ops: &[ArrayView2<Complex<f64>>],
    perm: &[usize],
    rho: &CsMat<Complex<f64>>,
    mut random_float: f64,
) -> usize {
    let u = kron_helper(ops.iter().map(|op| make_sprs(*op)));
    let s = make_perm::<Complex<f64>>(&perm);
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);
    let mut udag = u.clone().transpose_into();
    udag.data_mut().iter_mut().for_each(|x| *x = x.conj());

    let usosu = u.mul(&s).mul(rho).mul(&s.transpose_view()).mul(&udag);
    let diag = usosu.diag().to_dense();
    let mut i = 0;
    while i < diag.len() {
        debug_assert!(diag[i].im.abs() < f64::EPSILON);
        random_float -= diag[i].re;
        if random_float < 0.0 {
            return i;
        }
        i += 1;
    }
    diag.len() - 1
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
                        .make_matrices()
                        .iter()
                        .enumerate()
                        .map(|(i, op)| {
                            MatrixOp::new_matrix([i], op.iter().copied().collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>();

                    let mut cv = v.clone();
                    let mut cv_arena = v.clone();
                    apply_ops(
                        nqubits,
                        ops,
                        cv.as_slice_mut().unwrap(),
                        cv_arena.as_slice_mut().unwrap(),
                    );

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
#[derive(Serialize, Deserialize, Default)]
pub struct Samples {
    pub samples: Vec<Sample>,
}

#[pymethods]
impl Samples {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn get_perm_sizes(&self) -> Vec<usize> {
        self.samples.iter().map(|x| x.perm.len()).collect()
    }

    fn subset(&self, n: usize) -> Self {
        let mut rng = thread_rng();
        let subset = self.samples.choose_multiple(&mut rng, n).cloned().collect();
        Self { samples: subset }
    }

    fn add_from(&mut self, other: &Samples) {
        self.samples.extend(other.samples.iter().cloned());
    }

    fn combine(&self, other: &Samples) -> Self {
        let mut samples = self.samples.clone();
        samples.extend(other.samples.iter().cloned());
        Self { samples }
    }

    fn num_samples(&self) -> usize {
        self.samples.len()
    }

    fn add_sample(&mut self, sample: Sample) {
        self.samples.push(sample)
    }

    fn get_sample(&self, index: usize) -> Sample {
        self.samples[index].clone()
    }

    fn save_to(&self, filename: &str) -> PyResult<()> {
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
    fn load_from(filename: &str) -> PyResult<Self> {
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
    pub gates: Vec<usize>,
    pub perm: Vec<usize>,
    pub measurement: usize,
}

#[pymethods]
impl Sample {
    #[new]
    fn new(gates: Vec<usize>, perm: Vec<usize>, measurement: usize) -> Self {
        Self {
            gates,
            perm,
            measurement,
        }
    }

    fn get_gates(&self) -> Vec<usize> {
        self.gates.clone()
    }

    fn get_perm(&self) -> Vec<usize> {
        self.perm.clone()
    }

    fn get_measurement(&self) -> usize {
        self.measurement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::make_numcons_pauli_pairs;
    use num_complex::Complex;
    use num_traits::One;

    impl DensityMatrix {
        fn new_raw(mat: CsMat<Complex<f64>>) -> Self {
            Self {
                mat: MixedSparse(mat),
            }
        }

        fn new_raw_pure(mat: CsVec<Complex<f64>>) -> Self {
            Self {
                mat: PureSparse(mat),
            }
        }

        fn new_raw_pure_dense(mat: Array1<Complex<f64>>) -> Self {
            Self {
                mat: PureDense(mat),
            }
        }
    }

    fn make_simple_rho(qubits: usize, i: usize) -> CsMat<Complex<f64>> {
        let mut a = TriMat::new((1 << qubits, 1 << qubits));
        a.add_triplet(i, i, Complex::<f64>::one());
        a.to_csr()
    }
    fn make_simple_rho_dense(qubits: usize, i: usize) -> Array1<Complex<f64>> {
        let mut arr = Array1::zeros((1 << qubits,));
        arr[i] = Complex::one();
        arr
    }

    fn make_mixed_rho(qubits: usize) -> CsMat<Complex<f64>> {
        let mut c = CsMat::eye(1 << qubits);
        c.diag_iter_mut().for_each(|x| {
            if let Some(x) = x {
                *x /= (1 << qubits) as f64
            }
        });
        c
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
    fn check_sim_easy() {
        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [0, 1, 2, 3];
        let rho = make_simple_rho(qubits, 0);
        let mut rng = thread_rng();
        let i = measure_channel(qubits, &ops, &s, &rho, rng.gen());
        assert_eq!(i, 0b0000);
    }

    #[test]
    fn check_sim_shift() {
        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [1, 2, 3, 0];
        let rho = make_simple_rho(qubits, 0b0000);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(i, 0b0000);

        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [1, 2, 3, 0];
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(i, 0b1000);
    }

    #[test]
    fn check_sim_one() {
        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [0, 1, 2, 3];
        let rho = make_simple_rho(qubits, 1);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(i, 0b0001);
    }

    #[test]
    fn check_sim_shift_one() {
        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [1, 2, 3, 0];
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(i, 0b1000);
    }

    #[test]
    fn check_sim_invshift_one() {
        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [3, 0, 1, 2];
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(i, 0b0010);
    }

    #[test]
    fn check_sim_mixed() {
        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [3, 0, 1, 2];
        let rho = make_mixed_rho(qubits);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&ops, &s, &rho, f));
        println!("{}", i);
    }

    #[test]
    fn check_trivials() {
        let qubits = 4;
        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2).map(|_| eye.view()).collect::<Vec<_>>();
        let s = [0, 1, 2, 3];
        let mut rng = thread_rng();

        for j in 0..qubits {
            let mut rho = TriMat::new((1 << qubits, 1 << qubits));
            rho.add_triplet(j, j, Complex::one());
            let rho = rho.to_csr();

            let f = rng.gen();
            let newi = measure_channel(qubits, &ops, &s, &rho, f);
            assert_eq!(newi, measure_channel_dumb(&ops, &s, &rho, f));
            assert_eq!(newi, j);
        }
    }

    #[test]
    fn check_spin_flip_first() {
        let qubits = 4;
        let s = [0, 1, 2, 3];

        let mut x: TriMat<Complex<f64>> = TriMat::new((2, 2));
        x.add_triplet(0, 1, Complex::one());
        x.add_triplet(1, 0, Complex::one());
        let x = x.to_csr();

        let eye = CsMat::eye(2);
        let x = kronecker_product(x.view(), eye.view()).to_dense();

        let eye = Array2::eye(1 << 2);
        let ops = Some(x.view())
            .iter()
            .copied()
            .chain((0..qubits / 2 - 1).map(|_| eye.view()))
            .collect::<Vec<_>>();

        let mut rng = thread_rng();

        for j in 0..qubits {
            let mut rho = TriMat::new((1 << qubits, 1 << qubits));
            rho.add_triplet(j, j, Complex::one());
            let rho = rho.to_csr();

            let f = rng.gen();
            let newi = measure_channel(qubits, &ops, &s, &rho, f);
            assert_eq!(newi, measure_channel_dumb(&ops, &s, &rho, f));
            assert_eq!(newi, 0b1000 ^ j);
        }
    }

    #[test]
    fn check_spin_flip_last() {
        let qubits = 4;
        let s = [0, 1, 2, 3];

        let mut x: TriMat<Complex<f64>> = TriMat::new((2, 2));
        x.add_triplet(0, 1, Complex::one());
        x.add_triplet(1, 0, Complex::one());
        let x = x.to_csr();

        let eye = CsMat::eye(2);
        let x = kronecker_product(eye.view(), x.view()).to_dense();

        let eye = Array2::eye(1 << 2);
        let ops = (0..qubits / 2 - 1)
            .map(|_| eye.view())
            .chain(Some(x.view()).iter().copied())
            .collect::<Vec<_>>();
        let mut rng = thread_rng();

        for j in 0..qubits {
            let mut rho = TriMat::new((1 << qubits, 1 << qubits));
            rho.add_triplet(j, j, Complex::one());
            let rho = rho.to_csr();

            let f = rng.gen();
            let newi = measure_channel(qubits, &ops, &s, &rho, f);
            assert_eq!(newi, measure_channel_dumb(&ops, &s, &rho, f));
            assert_eq!(newi, 0b0001 ^ j);
        }
    }

    #[test]
    fn check_sim_flip() {
        let qubits = 4;
        let ua = Array2::eye(1 << 2);
        let ub = x_flip_num().to_dense();
        let ops = [ua.view(), ub.view()];
        let s = [0, 1, 2, 3];
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(newi, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(newi, 0b0010);

        let ops = [ub.view(), ua.view()];
        let rho = make_simple_rho(qubits, 0b1000);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(newi, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(newi, 0b0100);

        let s = [1, 0, 2, 3];

        let ops = [ua.view(), ua.view()];
        let rho = make_simple_rho(qubits, 0b1010);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel(qubits, &ops, &s, &rho, f);
        assert_eq!(newi, measure_channel_dumb(&ops, &s, &rho, f));
        assert_eq!(newi, 0b0110);
    }

    #[test]
    fn check_sim_flip_dense() {
        let qubits = 4;
        let ua = Array2::eye(1 << 2);
        let ub = x_flip_num().to_dense();
        let ops = [ua.view(), ub.view()];
        let s = [0, 1, 2, 3];
        let rho = make_simple_rho_dense(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel_pure_dense(qubits, &ops, &s, rho.view(), f);
        assert_eq!(newi, 0b0010);

        let ops = [ub.view(), ua.view()];
        let rho = make_simple_rho_dense(qubits, 0b1000);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel_pure_dense(qubits, &ops, &s, rho.view(), f);
        assert_eq!(newi, 0b0100);

        let s = [1, 0, 2, 3];

        let ops = [ua.view(), ua.view()];
        let rho = make_simple_rho_dense(qubits, 0b1010);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel_pure_dense(qubits, &ops, &s, rho.view(), f);
        assert_eq!(newi, 0b0110);
    }

    #[test]
    fn test_expectation() -> Result<(), String> {
        let qubits = 4;
        let rho = DensityMatrix::new_raw_pure_dense(Array1::ones((1 << qubits,)));

        let opstring = OperatorString::try_new("ZZZZ".chars())?;
        rho.expectation_opstring(&opstring)?;

        Ok(())
    }

    #[test]
    fn test_sampling() -> Result<(), String> {
        let qubits = 4;
        let mut rho = TriMat::<Complex<f64>>::new((1 << qubits, 1 << qubits));
        rho.add_triplet(0b0001, 0b0001, Complex::one());
        rho.add_triplet(0b1000, 0b0001, Complex::one());
        rho.add_triplet(0b0001, 0b1000, Complex::one());
        rho.add_triplet(0b1000, 0b1000, Complex::one());
        let rho: CsMat<Complex<f64>> = rho.to_csr();

        let pauli_pairs = make_numcons_pauli_pairs();
        let flat_paulis = pauli_pairs.into_shape((16, 4, 4)).unwrap();
        let exp = Experiment::new_raw(qubits, Some(flat_paulis), None);

        let rho = DensityMatrix::new_raw(rho);
        let _samples = exp.sample_raw(&rho, 1000)?;

        Ok(())
    }

    #[test]
    fn test_sampling_pure() -> Result<(), String> {
        let qubits = 4;
        let rho = CsVec::new(
            1 << qubits,
            (0..1 << qubits).collect(),
            vec![Complex::one(); 1 << qubits],
        );

        let pauli_pairs = make_numcons_pauli_pairs();
        let flat_paulis = pauli_pairs.into_shape((16, 4, 4)).unwrap();
        let exp = Experiment::new_raw(qubits, Some(flat_paulis), None);

        let rho = DensityMatrix::new_raw_pure(rho);
        let _samples = exp.sample_raw(&rho, 1000)?;

        Ok(())
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
        let exp = Experiment::new_raw(qubits, Some(flat_paulis), None);

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
        let exp = Experiment::new_raw(qubits, Some(flat_paulis), None);

        let rho = DensityMatrix::new_raw_pure_dense(rho);
        let _samples = exp.sample_raw(&rho, 1000)?;

        Ok(())
    }
}
