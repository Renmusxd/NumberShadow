use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{One, Zero};
use pyo3::prelude::*;
use std::mem::swap;
use std::ops::Mul;

use crate::recon::Operator;
use crate::samples::{Sample, Samples};
use crate::sims::DensityType::{MixedSparse, PureDense, PureSparse};
use crate::utils::{get_pauli_ops, BitString};
use crate::utils::{OperatorString, SparseVec};
use num_complex::Complex;
use numpy::ndarray::{Array3, Axis};
use numpy::{ndarray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use qip_iterators::iterators::MatrixOp;
use rand::prelude::*;
use rayon::prelude::*;
use sprs::*;

#[pyclass]
pub struct Experiment {
    qubits: usize,
    pairwise_ops: Array3<Complex<f64>>,
}

impl Experiment {
    fn new_raw(qubits: usize, ops: Option<Array3<Complex<f64>>>) -> Self {
        Self {
            qubits,
            pairwise_ops: ops.unwrap_or_else(get_pauli_ops),
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
                    PureSparse(v) => {
                        let ops_it = opis.iter().map(|((i, j), opi)| {
                            let op = self.pairwise_ops.index_axis(Axis(0), *opi);
                            ((*i, *j), op)
                        });

                        measure_channel_pure_sparse(self.qubits, ops_it, v, f)
                    }
                    _ => unimplemented!(),
                };

                Sample::new(opis, i)
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
    fn new(qubits: usize, ops: Option<PyReadonlyArray3<Complex<f64>>>) -> Self {
        let ops = ops.map(|ops| ops.as_array().to_owned());
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
) -> BitString
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
            return BitString::new_short(i, qubits);
        }
    }
    BitString::new_short((1 << qubits) - 1, qubits)
}

// Performs best when u and s are csr
fn measure_channel_pure_sparse<'a, It>(
    qubits: usize,
    ops: It,
    rho: &SparseVec,
    mut random_float: f64,
) -> BitString
where
    It: IntoIterator<Item = ((usize, usize), ArrayView2<'a, Complex<f64>>)>,
{
    debug_assert!(random_float >= 0.0);
    debug_assert!(random_float <= 1.0);

    let rho = ops.into_iter().fold(rho.clone(), |rho, ((i, j), op)| {
        rho.apply_twobody_op(i, j, op)
    });

    for (i, busp) in rho.iter() {
        let p = busp.norm_sqr();

        random_float -= p;
        if random_float <= 0.0 {
            return i.clone();
        }
    }
    BitString::new_short((1 << qubits) - 1, qubits)
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
    PureSparse(SparseVec),
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
    fn new_pure_sparse_indices(
        num_qubits: usize,
        indices: PyReadonlyArray1<usize>,
        data: PyReadonlyArray1<Complex<f64>>,
    ) -> Self {
        let indices = indices.as_array().to_vec();
        let data = data.as_array().to_vec();
        let mut v = SparseVec::new(num_qubits);
        indices
            .into_iter()
            .zip(data)
            .for_each(|(i, c)| v.overwrite(BitString::new_short(i, num_qubits), c));
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
    fn new_uniform_in_sector_sprs(_qubits: usize, _sector: u32) -> Self {
        todo!()
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
            PureSparse(v) => Ok(v.expectation(opstring)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::OpChar;
    use ndarray::Array4;
    use num_complex::Complex;

    pub fn make_numcons_pauli_pairs() -> Array4<Complex<f64>> {
        let mut pauli_pairs = Array4::<Complex<f64>>::zeros((4, 4, 4, 4));
        pauli_pairs
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut pauli_pairs)| {
                let opchar_a = OpChar::try_from(i).unwrap().get_matrix();
                pauli_pairs
                    .axis_iter_mut(Axis(0))
                    .enumerate()
                    .for_each(|(j, mut pauli_pairs)| {
                        // Kron them
                        let opchar_b = OpChar::try_from(j).unwrap().get_matrix();
                        let kron_prod = ndarray::linalg::kron(&opchar_a, &opchar_b);
                        pauli_pairs
                            .iter_mut()
                            .zip(kron_prod)
                            .for_each(|(x, y)| *x = y);
                    });
            });
        pauli_pairs
    }

    impl DensityMatrix {
        fn new_raw_pure_dense(mat: Array1<Complex<f64>>) -> Self {
            Self {
                mat: PureDense(mat),
            }
        }
    }

    #[test]
    fn test_sampling_pure_dense_larger() -> Result<(), String> {
        let qubits = 6;
        let rho = Array1::ones((1 << qubits,));

        let pauli_pairs = make_numcons_pauli_pairs();
        let flat_paulis = pauli_pairs.into_shape((16, 4, 4)).unwrap();
        let exp = Experiment::new_raw(qubits, Some(flat_paulis));

        let rho = DensityMatrix::new_raw_pure_dense(rho);
        let _samples = exp.sample_raw(&rho, 1000)?;

        Ok(())
    }
}
