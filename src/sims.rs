use ndarray::indices;
use num_traits::Zero;
use pyo3::{pyclass, pymethods, PyAny, PyResult, Python};
use std::fs::File;
use std::io::{Read, Write};
use std::ops::Mul;
use std::path::Path;

use crate::recon::Operator;
use crate::sims::DensityType::{Mixed, Pure};
use crate::utils::{
    kron_helper, make_perm, make_sprs, make_sprs_onehot, scipy_mat, OperatorString,
};
use num_complex::Complex;
use numpy::ndarray::{Array2, Array3, Axis};
use numpy::{ndarray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
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
            Mixed(m) => {
                let (a, b) = m.shape();
                if a != b {
                    return Err(format!(
                        "Expected square density matrix but found {:?}",
                        m.shape()
                    ));
                }
                a
            }
            Pure(v) => v.dim(),
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
                let mut rng = rand::thread_rng();
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
                let ops = opis.iter().map(|opi| {
                    let op = self.pairwise_ops.index_axis(Axis(0), *opi);
                    make_sprs(op)
                });

                let u = kron_helper(ops);
                let s = make_perm::<Complex<f64>>(&perm);
                let f = rng.gen();
                let i = match &rho.mat {
                    Mixed(m) => {
                        let i = measure_channel(self.qubits, &u, &s, m, f);
                        // Check same as dumb
                        debug_assert_eq!(i, measure_channel_dumb(&u, &s, m, f));
                        i
                    }
                    Pure(v) => {
                        let i = measure_channel_pure(self.qubits, &u, &s, v, f);
                        // Check same as dumb
                        debug_assert!({
                            let mut a = TriMat::new((v.dim(), v.dim()));
                            v.into_sparse_vec_iter().for_each(|(i, ci)| {
                                v.into_sparse_vec_iter().for_each(|(j, cj)| {
                                    a.add_triplet(i, j, ci * cj.conj());
                                });
                            });
                            let m = a.to_csr();
                            let newi = measure_channel_dumb(&u, &s, &m, f);
                            i == newi
                        });
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
    u: &CsMat<Complex<f64>>,
    s: &CsMat<Complex<f64>>,
    rho: &CsVec<Complex<f64>>,
    mut random_float: f64,
) -> usize {
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);
    for i in 0..1 << qubits {
        let b = make_sprs_onehot(i, 1 << qubits);
        let bus = b.mul(u).mul(s);
        let busp = bus.dot(rho);
        let p = busp.norm_sqr();

        random_float -= p;
        if random_float <= 0.0 {
            return i;
        }
    }
    (1 << qubits) - 1
}

// Performs best when u, s, rho are csr
fn measure_channel(
    qubits: usize,
    u: &CsMat<Complex<f64>>,
    s: &CsMat<Complex<f64>>,
    rho: &CsMat<Complex<f64>>,
    mut random_float: f64,
) -> usize {
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);
    for i in 0..1 << qubits {
        let b = make_sprs_onehot(i, 1 << qubits);
        let bus = b.mul(u).mul(s);
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
    u: &CsMat<Complex<f64>>,
    s: &CsMat<Complex<f64>>,
    rho: &CsMat<Complex<f64>>,
    mut random_float: f64,
) -> usize {
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);
    let mut udag = u.clone().transpose_into();
    udag.data_mut().iter_mut().for_each(|x| *x = x.conj());

    let usosu = u.mul(s).mul(rho).mul(&s.transpose_view()).mul(&udag);
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
    Mixed(CsMat<Complex<f64>>),
    Pure(CsVec<Complex<f64>>),
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
        Self { mat: Mixed(mat) }
    }

    #[staticmethod]
    fn new_pure_dense(arr: PyReadonlyArray1<Complex<f64>>, tol: Option<f64>) -> Self {
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
        Self { mat: Pure(v) }
    }

    #[staticmethod]
    fn from_dense(arr: PyReadonlyArray2<Complex<f64>>, tol: Option<f64>) -> Self {
        let tol = tol.unwrap_or(1e-10);
        let arr = arr.as_array();

        let mut a = TriMat::new((arr.shape()[0], arr.shape()[1]));
        ndarray::Zip::indexed(arr).for_each(|(row, col), v| {
            if v.norm() > tol {
                a.add_triplet(row, col, *v)
            }
        });
        Self {
            mat: Mixed(a.to_csr()),
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
}

impl DensityMatrix {
    fn expectation_opstring(&self, opstring: &OperatorString) -> Result<Complex<f64>, String> {
        let opmat = opstring.make_matrix();
        match &self.mat {
            Mixed(m) => {
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
            Pure(v) => {
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
            Self { mat: Mixed(mat) }
        }

        fn new_raw_pure(mat: CsVec<Complex<f64>>) -> Self {
            Self { mat: Pure(mat) }
        }
    }

    fn make_simple_rho(qubits: usize, i: usize) -> CsMat<Complex<f64>> {
        let mut a = TriMat::new((1 << qubits, 1 << qubits));
        a.add_triplet(i, i, Complex::<f64>::one());
        a.to_csr()
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
        let u = CsMat::eye(1 << qubits);
        let s = make_perm(&[0, 1, 2, 3]);
        let rho = make_simple_rho(qubits, 0);
        let mut rng = thread_rng();
        let i = measure_channel(qubits, &u, &s, &rho, rng.gen());
        assert_eq!(i, 0b0000);
    }

    #[test]
    fn check_sim_shift() {
        let qubits = 4;
        let u = CsMat::eye(1 << qubits);
        let s = make_perm(&[1, 2, 3, 0]);
        let rho = make_simple_rho(qubits, 0b0000);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(i, 0b0000);

        let qubits = 4;
        let u = CsMat::eye(1 << qubits);
        let s = make_perm(&[1, 2, 3, 0]);
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(i, 0b1000);
    }

    #[test]
    fn check_sim_one() {
        let qubits = 4;
        let u = CsMat::eye(1 << qubits);
        let s = make_perm(&[0, 1, 2, 3]);
        let rho = make_simple_rho(qubits, 1);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(i, 0b0001);
    }

    #[test]
    fn check_sim_shift_one() {
        let qubits = 4;
        let u = CsMat::eye(1 << qubits);
        let s = make_perm(&[1, 2, 3, 0]);
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(i, 0b1000);
    }

    #[test]
    fn check_sim_invshift_one() {
        let qubits = 4;
        let u = CsMat::eye(1 << qubits);
        let s = make_perm(&[3, 0, 1, 2]);
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(i, 0b0010);
    }

    #[test]
    fn check_sim_mixed() {
        let qubits = 4;
        let u = CsMat::eye(1 << qubits);
        let s = make_perm(&[3, 0, 1, 2]);
        let rho = make_mixed_rho(qubits);
        let mut rng = thread_rng();
        let f = rng.gen();
        let i = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(i, measure_channel_dumb(&u, &s, &rho, f));
        println!("{}", i);
    }

    #[test]
    fn check_trivials() {
        let qubits = 4;
        let s = make_perm(&[0, 1, 2, 3]);
        let u = CsMat::eye(1 << qubits);
        let mut rng = thread_rng();

        for j in 0..qubits {
            let mut rho = TriMat::new((1 << qubits, 1 << qubits));
            rho.add_triplet(j, j, Complex::one());
            let rho = rho.to_csr();

            let f = rng.gen();
            let newi = measure_channel(qubits, &u, &s, &rho, f);
            assert_eq!(newi, measure_channel_dumb(&u, &s, &rho, f));
            assert_eq!(newi, j);
        }
    }

    #[test]
    fn check_spin_flip_first() {
        let qubits = 4;
        let s = make_perm(&[0, 1, 2, 3]);

        let mut x: TriMat<Complex<f64>> = TriMat::new((2, 2));
        x.add_triplet(0, 1, Complex::one());
        x.add_triplet(1, 0, Complex::one());
        let x = x.to_csr();

        let rest = CsMat::eye(1 << (qubits - 1));
        let u = kronecker_product(x.view(), rest.view());
        let mut rng = thread_rng();

        for j in 0..qubits {
            let mut rho = TriMat::new((1 << qubits, 1 << qubits));
            rho.add_triplet(j, j, Complex::one());
            let rho = rho.to_csr();

            let f = rng.gen();
            let newi = measure_channel(qubits, &u, &s, &rho, f);
            assert_eq!(newi, measure_channel_dumb(&u, &s, &rho, f));
            assert_eq!(newi, 0b1000 ^ j);
        }
    }

    #[test]
    fn check_spin_flip_last() {
        let qubits = 4;
        let s = make_perm(&[0, 1, 2, 3]);

        let mut x: TriMat<Complex<f64>> = TriMat::new((2, 2));
        x.add_triplet(0, 1, Complex::one());
        x.add_triplet(1, 0, Complex::one());
        let x = x.to_csr();

        let rest = CsMat::eye(1 << (qubits - 1));
        let u = kronecker_product(rest.view(), x.view());
        let mut rng = thread_rng();

        for j in 0..qubits {
            let mut rho = TriMat::new((1 << qubits, 1 << qubits));
            rho.add_triplet(j, j, Complex::one());
            let rho = rho.to_csr();

            let f = rng.gen();
            let newi = measure_channel(qubits, &u, &s, &rho, f);
            assert_eq!(newi, measure_channel_dumb(&u, &s, &rho, f));
            assert_eq!(newi, 0b0001 ^ j);
        }
    }

    #[test]
    fn check_sim_flip() {
        let qubits = 4;
        let ua = CsMat::eye(1 << 2);
        let ub = x_flip_num();
        let u = kronecker_product(ua.view(), ub.view());
        let s = make_perm(&[0, 1, 2, 3]);
        let rho = make_simple_rho(qubits, 0b0001);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(newi, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(newi, 0b0010);

        let u = kronecker_product(ub.view(), ua.view());
        let rho = make_simple_rho(qubits, 0b1000);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(newi, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(newi, 0b0100);

        let s = make_perm(&[1, 0, 2, 3]);
        let u = kronecker_product(ua.view(), ua.view());
        let rho = make_simple_rho(qubits, 0b1010);
        let mut rng = thread_rng();
        let f = rng.gen();
        let newi = measure_channel(qubits, &u, &s, &rho, f);
        assert_eq!(newi, measure_channel_dumb(&u, &s, &rho, f));
        assert_eq!(newi, 0b0110);
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
}
