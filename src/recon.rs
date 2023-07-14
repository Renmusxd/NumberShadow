use crate::sims::{Sample, Samples};
use crate::utils::{
    fact, fact2, kron_helper, make_numcons_pauli_pairs, reverse_n_bits, scipy_mat, OpChar,
    OperatorString,
};
use num_complex::Complex;
use num_traits::{One, ToPrimitive, Zero};
use numpy::ndarray::{s, Array1, Array2, Array3, ArrayView3, ArrayView4, Axis};
use numpy::{PyArray1, PyReadonlyArray3, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyComplex;
use pyo3::{pyclass, pymethods, Py, PyResult, Python};
use rayon::prelude::*;
use sprs::{CsMat, TriMat};
use std::any::Any;

#[pyclass]
pub struct Reconstruction {
    qubits: usize,
    pairwise_ops: Array3<Complex<f64>>,
}

#[pymethods]
impl Reconstruction {
    #[new]
    fn new(qubits: usize, ops: Option<PyReadonlyArray3<Complex<f64>>>) -> Self {
        let ops = ops.unwrap().as_array().to_owned();

        Self {
            qubits,
            pairwise_ops: ops,
        }
    }

    fn estimate_string_for_each_sample(
        &self,
        py: Python,
        op: String,
        samples: &Samples,
    ) -> PyResult<Py<PyArray1<Complex<f64>>>> {
        let ps = OperatorString::try_new(op.chars()).map_err(PyValueError::new_err)?;

        let mut samples_estimates = Array1::zeros(samples.samples.len());
        let pauli_pairs = make_numcons_pauli_pairs();

        let enumerating_indices = (0..ps.opstring.len()).collect::<Vec<_>>();
        let available_indices = ps.indices.as_ref();
        let iter_indices = if let Some(indices) = available_indices {
            indices
        } else {
            &enumerating_indices
        };

        let (noni_indices, noni_substring): (Vec<_>, Vec<_>) = iter_indices
            .iter()
            .zip(ps.opstring.iter())
            .filter(|(_, op)| OpChar::I.ne(op))
            .map(|(a, b)| (a.clone(), b.clone()))
            .unzip();

        let subop: OperatorString = noni_substring.clone().into();
        let cw = channel_weight(&subop);
        if cw.abs() > f64::EPSILON {
            samples_estimates
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(samples.samples.par_iter())
                .filter(|(_, sample)| filter_permutations(&sample.perm, &noni_indices))
                .for_each(|(mut x, sample)| {
                    *x.get_mut([]).unwrap() = estimate_string_for_sample(
                        &ps,
                        sample,
                        pauli_pairs.view(),
                        self.pairwise_ops.view(),
                    );
                });
        }

        Ok(samples_estimates.to_pyarray(py).to_owned())
    }

    fn estimate_operator_string(&self, op: String, samples: &Samples) -> Complex<f64> {
        let mut new_op = Operator::new();
        new_op.add_string_rust(Complex::<f64>::one(), op, None);
        self.estimate_operator(&new_op, samples)
    }

    fn estimate_operator(&self, op: &Operator, samples: &Samples) -> Complex<f64> {
        let pauli_pairs = make_numcons_pauli_pairs();

        let acc = op
            .opstrings
            .iter()
            .map(|(op_weight, ps)| -> Complex<f64> {
                // For each operator substring, estimate it using the relevant permutations.
                let enumerating_indices = (0..ps.opstring.len()).collect::<Vec<_>>();
                let available_indices = ps.indices.as_ref();
                let iter_indices = if let Some(indices) = available_indices {
                    indices
                } else {
                    &enumerating_indices
                };

                let (noni_indices, noni_substring): (Vec<_>, Vec<_>) = iter_indices
                    .iter()
                    .zip(ps.opstring.iter())
                    .filter(|(_, op)| OpChar::I.ne(op))
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .unzip();

                let subop: OperatorString = noni_substring.clone().into();
                let cw = channel_weight(&subop);
                if cw.abs() < f64::EPSILON {
                    Complex::<f64>::zero()
                } else {
                    let (tot, count) = samples
                        .samples
                        .par_iter()
                        .filter(|sample| filter_permutations(&sample.perm, &noni_indices))
                        .map(|sample| -> (Complex<f64>, usize) {
                            let sub_acc = estimate_string_for_sample(
                                ps,
                                sample,
                                pauli_pairs.view(),
                                self.pairwise_ops.view(),
                            );
                            (sub_acc, 1)
                        })
                        .reduce(
                            || (Complex::<f64>::zero(), 0),
                            |(ac, ai), (bc, bi)| (ac + bc, ai + bi),
                        );

                    (*op_weight) * tot / (cw * count as f64)
                }
            })
            .sum::<Complex<f64>>();
        acc
        // (acc.re, acc.im)
    }
}

fn estimate_string_for_sample(
    ps: &OperatorString,
    sample: &Sample,
    pauli_pairs: ArrayView4<Complex<f64>>,
    pairwise_ops: ArrayView3<Complex<f64>>,
) -> Complex<f64> {
    let ordered_meas = reverse_n_bits(sample.measurement, sample.perm.len() as u32);

    // Number of sites spanned by operator string.
    let op_support = sample.perm.len();

    let mut perm_inv = sample.perm.clone();
    for (i, k) in sample.perm.iter().enumerate() {
        perm_inv[*k] = i;
    }

    (0..op_support / 2)
        .map(|pair_index| {
            // let site_a = sample.perm[2 * pair_index];
            // let site_b = sample.perm[2 * pair_index + 1];

            let site_a = perm_inv[2 * pair_index];
            let site_b = perm_inv[2 * pair_index + 1];

            let pauli_a = ps.get(site_a);
            let pauli_b = ps.get(site_b);

            if pauli_a == OpChar::I && pauli_b == OpChar::I {
                Complex::<f64>::one()
            } else {
                // Get the 'local' measurement result.
                // TODO check that we are indexing in the correct order.
                // let bit_a = (sample.measurement >> (2 * pair_index)) & 1;
                // let bit_b = (sample.measurement >> (2 * pair_index + 1)) & 1;

                let bit_a = (ordered_meas >> (2 * pair_index)) & 1;
                let bit_b = (ordered_meas >> (2 * pair_index + 1)) & 1;

                let submeas = (bit_a << 1) | bit_b;

                // Create <b|
                let mut b = Array2::zeros((1, 4));
                b[(0, submeas)] = Complex::<f64>::one();

                // Compute <b|U
                let pairwise_opi = sample.gates[pair_index];
                let u = pairwise_ops.slice(s![pairwise_opi, .., ..]);
                let bu = b.dot(&u);
                debug_assert_eq!(b.shape(), bu.shape());

                // Compute <b|USp
                let pauli_ai: usize = pauli_a.into();
                let pauli_bi: usize = pauli_b.into();
                let pauli_pair_mat = pauli_pairs.slice(s![pauli_ai, pauli_bi, .., ..]);
                let buso = bu.dot(&pauli_pair_mat);
                debug_assert_eq!(b.shape(), buso.shape());

                // Compute <b|USpSU|b>
                let busosub = buso
                    .into_iter()
                    .zip(bu.into_iter())
                    .map(|(a, b)| a * b.conj())
                    .sum::<Complex<f64>>();

                busosub
            }
        })
        .product::<Complex<f64>>()
}

fn filter_permutations(perm: &[usize], noni_indices: &[usize]) -> bool {
    let mut perm_inv = perm.to_vec();
    for (i, k) in perm.iter().enumerate() {
        perm_inv[*k] = i;
    }

    (0..perm.len() / 2).all(|pair_index| {
        // let sia = perm[2 * pair_index];
        // let sib = perm[2 * pair_index + 1];

        let sia = perm_inv[2 * pair_index];
        let sib = perm_inv[2 * pair_index + 1];

        noni_indices.contains(&sia) == noni_indices.contains(&sib)
    })
}

fn channel_weight(opstring: &OperatorString) -> f64 {
    let [count_z, count_p, count_m, count_i] = opstring.count_terms();
    let qubits = count_z + count_p + count_m + count_i;
    if count_i != 0 {
        unimplemented!()
    }

    // Not number conserving
    if count_p != count_m {
        return 0.0;
    }

    // All Z
    if count_p == 0 {
        return 1.0;
    }

    let prefactor = (1. / 3.0f64).powi(count_p as i32);
    let places_for_pluses = fact2(qubits) / fact2(qubits - 2 * count_p);
    let places_for_minuses = fact(count_m);
    let places_for_rest = fact(qubits - (count_m + count_p));
    let total_arrangements = fact(qubits);

    return prefactor
        * ((places_for_pluses * places_for_minuses * places_for_rest) as f64
            / total_arrangements as f64);
}

#[pyclass]
pub struct Operator {
    opstrings: Vec<(Complex<f64>, OperatorString)>,
}

impl Default for Operator {
    fn default() -> Self {
        Self { opstrings: vec![] }
    }
}

#[pymethods]
impl Operator {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn add_string(
        &mut self,
        c: &PyComplex,
        opstring: String,
        indices: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let cr = c.real().to_f64().unwrap();
        let ci = c.imag().to_f64().unwrap();
        let c = Complex::<f64>::new(cr, ci);
        self.add_string_rust(c, opstring, indices)
            .map_err(PyValueError::new_err)
    }
}

impl Operator {
    pub(crate) fn add_string_rust(
        &mut self,
        c: Complex<f64>,
        opstring: String,
        indices: Option<Vec<usize>>,
    ) -> Result<(), String> {
        let opstring = OperatorString::try_new_indices(indices, opstring.chars())?;
        self.opstrings.push((c, opstring));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_weight_simple() -> Result<(), String> {
        let opstring = "ZZZZ".try_into()?;
        let w = channel_weight(&opstring);

        assert!((w - 1.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn check_weight_pmpmpm() -> Result<(), String> {
        let opstring = "+-+-+-".try_into()?;
        let w = channel_weight(&opstring);
        assert!((w - 0.014814814814814815).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_recon() -> Result<(), String> {
        let opstring = "ZZII".try_into()?;

        let pairwise_ops = Array2::eye(1 << 2).insert_axis(Axis(0));
        let pauli_pairs = make_numcons_pauli_pairs();

        let sample = Sample {
            gates: vec![0, 0],
            perm: vec![0, 1, 2, 3],
            measurement: 0b1000,
        };

        let measured =
            estimate_string_for_sample(&opstring, &sample, pauli_pairs.view(), pairwise_ops.view());

        println!("measured: {:?}", measured);

        Ok(())
    }

    #[test]
    fn test_recon_allz() -> Result<(), String> {
        let opstring = "ZZZZ".try_into()?;

        let pairwise_ops = Array2::eye(1 << 2).insert_axis(Axis(0));
        let pauli_pairs = make_numcons_pauli_pairs();

        let sample = Sample {
            gates: vec![0, 0],
            perm: vec![0, 1, 2, 3],
            measurement: 0b1000,
        };

        let measured =
            estimate_string_for_sample(&opstring, &sample, pauli_pairs.view(), pairwise_ops.view());

        assert!((measured.re + 1.0).abs() < f64::EPSILON);
        assert!((measured.im).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_recon_farpm() -> Result<(), String> {
        let opstring = "+II-".try_into()?;

        let pairwise_ops = Array2::eye(1 << 2).insert_axis(Axis(0));
        let pauli_pairs = make_numcons_pauli_pairs();

        let sample = Sample {
            gates: vec![0, 0],
            perm: vec![1, 3, 2, 0],
            measurement: 0b1000,
        };

        let measured =
            estimate_string_for_sample(&opstring, &sample, pauli_pairs.view(), pairwise_ops.view());

        dbg!(measured);

        Ok(())
    }
}
