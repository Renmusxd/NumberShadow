use crate::samples::{Sample, Samples};
use crate::utils::*;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::Inverse;
use num_bigint::BigInt;
use num_complex::{Complex, ComplexFloat};
use num_rational::BigRational;
use num_traits::{Inv, One, Pow, ToPrimitive, Zero};
use numpy::ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyComplex;
use pyo3::{pyclass, pyfunction, pymethods, Py, PyResult, Python};
use rayon::prelude::*;

#[derive(Copy, Clone, Default)]
enum EstimatorType {
    #[default]
    GlobalSmartResum,
    GlobalSumAllStrings,
    InversionByParts,
}

#[pyclass]
pub struct Reconstruction {
    estimator_type: EstimatorType,
    default_filtered: bool,
}

impl Default for Reconstruction {
    fn default() -> Self {
        Self {
            default_filtered: true,
            estimator_type: EstimatorType::default(),
        }
    }
}

#[pymethods]
impl Reconstruction {
    #[new]
    fn new(filtered: Option<bool>) -> Self {
        Self {
            default_filtered: filtered.unwrap_or(true),
            estimator_type: EstimatorType::default(),
        }
    }

    fn use_inversion_by_parts_estimator(&mut self) {
        self.estimator_type = EstimatorType::InversionByParts
    }
    fn use_smart_estimator(&mut self) {
        self.estimator_type = EstimatorType::GlobalSmartResum
    }
    fn use_dumb_estimator(&mut self) {
        self.estimator_type = EstimatorType::GlobalSumAllStrings
    }

    fn estimate_string_for_each_sample(
        &self,
        py: Python,
        op: String,
        samples: &Samples,
        filtered: Option<bool>,
    ) -> PyResult<Py<PyArray1<Complex<f64>>>> {
        let filtered = filtered.unwrap_or(self.default_filtered);
        let opstring =
            OperatorString::try_from(op).map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        let v = self
            .estimate_operator_string_iterator(&opstring, samples, filtered)
            .map_err(PyValueError::new_err)?
            .collect::<Vec<_>>();
        Ok(Array1::from_vec(v).into_pyarray(py).to_owned())
    }

    fn estimate_string(
        &self,
        op: String,
        samples: &Samples,
        filtered: Option<bool>,
    ) -> PyResult<Complex<f64>> {
        let filtered = filtered.unwrap_or(self.default_filtered);
        let opstring =
            OperatorString::try_from(op).map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        let (total, count) = self
            .estimate_operator_string_iterator(&opstring, samples, filtered)
            .map_err(PyValueError::new_err)?
            .map(|c| (c, 1usize))
            .reduce(|| (Complex::zero(), 0), |(a, ac), (b, bc)| (a + b, ac + bc));
        Ok(total / count as f64)
    }
}

impl Reconstruction {
    fn estimate_operator_string_iterator<'a>(
        &self,
        opstring: &'a OperatorString,
        samples: &'a Samples,
        filtered: bool,
    ) -> Result<impl ParallelIterator<Item = Complex<f64>> + 'a, String> {
        let l = samples.l;
        if opstring.opstring.len() != l {
            return Err(format!(
                "Operator String is only defined on {} out of {} sites.",
                opstring.opstring.len(),
                l
            ));
        }

        let [np, nm, nz, ni] = opstring
            .opstring
            .iter()
            .copied()
            .map(|op| match op {
                OpChar::Plus => 0,
                OpChar::Minus => 1,
                OpChar::Z => 2,
                OpChar::I => 3,
            })
            .fold([0; 4], |mut acc, i| {
                acc[i] += 1usize;
                acc
            });
        let mut ops_and_meas = (0..l).collect::<Vec<_>>();
        ops_and_meas.sort_unstable_by_key(|i| {
            let op = opstring.opstring.get(*i).copied().unwrap_or(OpChar::I);
            match op {
                OpChar::Plus => 0,
                OpChar::Minus => 1,
                OpChar::Z => 2,
                OpChar::I => 3,
            }
        });
        let z_indices = &ops_and_meas[np + nm..np + nm + nz];
        let i_indices = &ops_and_meas[np + nm + nz..];
        let opinfo = OpInfo {
            l,
            nz,
            np,
            nm,
            z_indices: z_indices.to_vec(),
            i_indices: i_indices.to_vec(),
        };

        if nz > ni {
            return Err(format!(
                "Unimplemented: Operator String has more Z operators than I ({} vs {})",
                nz, ni
            ));
        }

        let gmat = get_g_mat(l, nz);
        let gmat_inv = gmat.inv().map_err(|e| format!("{:?}", e))?;
        let inv_c = Array1::from_iter((0..=nz.min(l - nz)).map(|a| {
            symmetry_sector_eigenvalue(l, a)
                .to_f64()
                .expect("Couldn't convert to f64")
                .inv()
        }));
        let betas = gmat_inv.dot(&inv_c);

        // For each operator substring, estimate it using the relevant permutations.
        let pairwise_ops = samples.ops.view();
        let estimator = self.estimator_type;
        Ok(samples
            .samples
            .par_iter()
            .filter(move |sample| {
                if filtered {
                    filter_pm(opstring, sample)
                } else {
                    true
                }
            })
            .map(move |sample: &Sample| -> Complex<f64> {
                estimate_op_string(
                    opstring,
                    &opinfo,
                    sample,
                    pairwise_ops,
                    betas.as_slice().unwrap(),
                    estimator,
                    filtered,
                )
            }))
    }
}

fn filter_pm(opstring: &OperatorString, sample: &Sample) -> bool {
    sample
        .gates
        .iter()
        .copied()
        .all(|((i, j), _)| match (opstring.get(i), opstring.get(j)) {
            (OpChar::Plus, x) | (x, OpChar::Plus) => x.eq(&OpChar::Minus),
            (OpChar::Minus, x) | (x, OpChar::Minus) => x.eq(&OpChar::Plus),
            _ => true,
        })
}

struct OpInfo {
    l: usize,
    nz: usize,
    np: usize,
    nm: usize,
    z_indices: Vec<usize>,
    i_indices: Vec<usize>,
}

fn estimate_op_string(
    opstring: &OperatorString,
    opinfo: &OpInfo,
    sample: &Sample,
    pairwise_ops: ArrayView3<Complex<f64>>,
    betas: &[f64],
    estimator: EstimatorType,
    filtered: bool,
) -> Complex<f64> {
    // If not number conserving.
    if opinfo.nm != opinfo.np {
        return Complex::zero();
    }
    // Check joined.
    let pm_us = sample
        .gates
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, ((a, b), _))| {
            let oa = opstring.opstring.get(*a).copied().unwrap_or(OpChar::I);
            let ob = opstring.opstring.get(*b).copied().unwrap_or(OpChar::I);
            let pm_matched = matches!(
                (oa, ob),
                (OpChar::Plus, OpChar::Minus) | (OpChar::Minus, OpChar::Plus)
            );
            pm_matched
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let num_plus_joined_to_minus = pm_us.len();
    debug_assert!(num_plus_joined_to_minus <= opinfo.np);

    // If not all pluses are joined to minuses.
    if filtered {
        debug_assert!(num_plus_joined_to_minus == opinfo.np);
    }
    if num_plus_joined_to_minus != opinfo.np {
        return Complex::zero();
    }

    // Now do the +- estimate
    let pm_prod = pm_estimate(
        opinfo.l,
        pm_us.len(),
        pm_us.iter().copied().map(|iu| sample.gates[iu]),
        &opstring.opstring,
        &sample.measurement,
        pairwise_ops,
        filtered,
    );

    if pm_prod.is_zero() {
        return Complex::zero();
    }

    debug_assert_eq!(
        opinfo.l - (opinfo.np + opinfo.nm),
        opinfo.z_indices.len() + opinfo.i_indices.len()
    );
    let mut binstring = BitString::new_long(opinfo.l - (opinfo.np + opinfo.nm));
    let mut mapping = vec![usize::MAX; opinfo.l];
    opinfo
        .z_indices
        .iter()
        .copied()
        .chain(opinfo.i_indices.iter().copied())
        .enumerate()
        .for_each(|(new_index, old_index)| {
            binstring.set_bit(new_index, sample.measurement.get_bit(old_index));
            mapping[old_index] = new_index;
        });

    let remap_gates = sample
        .gates
        .iter()
        .copied()
        .filter_map(|((a, b), ui)| {
            let new_a = mapping[a];
            let new_b = mapping[b];
            match (new_a, new_b) {
                (usize::MAX, usize::MAX) => None,
                (usize::MAX, _) | (_, usize::MAX) => {
                    unreachable!()
                }
                (a, b) => Some(((a, b), ui)),
            }
        })
        .collect::<Vec<_>>();
    let subsample = Sample {
        gates: remap_gates,
        measurement: binstring,
    };

    let z_estimate = match estimator {
        EstimatorType::InversionByParts => {
            estimate_canonical_z_string_by_parts(opinfo.nz, &subsample, pairwise_ops)
        }
        EstimatorType::GlobalSmartResum => estimate_canonical_z_string(
            opinfo.l - (opinfo.np + opinfo.nm),
            opinfo.nz,
            &subsample,
            pairwise_ops,
            betas,
        ),
        EstimatorType::GlobalSumAllStrings => dumb_estimate_canonical_z_string(
            opinfo.l - (opinfo.np + opinfo.nm),
            opinfo.nz,
            &subsample,
            pairwise_ops,
            betas,
        ),
    };

    pm_prod * z_estimate
}

fn pm_estimate<It>(
    l: usize,
    np: usize,
    it: It,
    ops: &[OpChar],
    measurement: &BitString,
    pairwise_ops: ArrayView3<Complex<f64>>,
    filtered: bool,
) -> Complex<f64>
where
    It: IntoIterator<Item = ((usize, usize), usize)>,
{
    let expectation = it
        .into_iter()
        .map(|((i, j), ui)| {
            let u = pairwise_ops.index_axis(Axis(0), ui);
            let bi = measurement.get_bit(i);
            let bj = measurement.get_bit(j);
            let a_bstar = u[(1, 1)] * u[(1, 2)].conj();
            let c_dstar = u[(2, 1)] * u[(2, 2)].conj();

            let (z1, z2) = match (ops[i], ops[j]) {
                (OpChar::Minus, OpChar::Plus) => (a_bstar, c_dstar),
                (OpChar::Plus, OpChar::Minus) => (a_bstar.conj(), c_dstar.conj()),
                _ => unreachable!(),
            };

            match (bi, bj) {
                (false, true) => z1,
                (true, false) => z2,
                _ => Complex::zero(),
            }
        })
        .product::<Complex<f64>>();

    if filtered {
        expectation * (3.0).pow(np as i32)
    } else {
        let top = pairings_product_list(l).expect("Can't make pairings");
        let bot = pairings_product_list(l - 2 * np)
            .expect("Can't make pairings")
            .chain(permutation_product_list(np));
        let eigenvalue_ratio =
            rational_quotient_of_products(top, bot) * BigInt::from(3).pow(np as u32);
        let eigenvalue = eigenvalue_ratio.to_f64().unwrap_or_default();

        expectation * eigenvalue
    }
}

fn dumb_estimate_canonical_z_string(
    l: usize,
    k: usize,
    sample: &Sample,
    pairwise_ops: ArrayView3<Complex<f64>>,
    betas: &[f64],
) -> Complex<f64> {
    fold_over_choices(l - 1, k, Complex::zero(), |acc, z_spots| {
        let mut zmask = vec![false; l];
        z_spots.iter().for_each(|x| zmask[*x] = true);
        let d = k - zmask[..k].iter().copied().filter(|x| *x).count();

        let expectation = prod_zs(
            sample.gates.iter().copied(),
            &zmask,
            &sample.measurement,
            pairwise_ops,
        );

        debug_assert_eq!(
            expectation,
            sample
                .gates
                .iter()
                .copied()
                .map(|((i, j), iu)| {
                    let bi = sample.measurement.get_bit(i);
                    let bj = sample.measurement.get_bit(j);
                    let zi = zmask[i];
                    let zj = zmask[j];

                    compute_single_inner(bi, bj, zi, zj, pairwise_ops.index_axis(Axis(0), iu))
                })
                .product::<Complex<f64>>()
        );

        acc + betas[d] * expectation
    })
}

fn estimate_canonical_z_string(
    l: usize,
    k: usize,
    sample: &Sample,
    pairwise_ops: ArrayView3<Complex<f64>>,
    betas: &[f64],
) -> Complex<f64> {
    // Indices 0...(k-1) have Z operators.
    let connected_to_zs =
        get_connected_to_zs(k, sample.gates.iter().map(|(x, _)| *x)).collect::<Vec<_>>();

    // Get the counts of binary pairs away from the operator.
    let it = sample
        .gates
        .iter()
        .map(|(x, _)| *x)
        .filter(|(a, b)| *a >= k && *b >= k);
    let (n00, n01, n11) = count_pairs_for_us(it, &sample.measurement);
    debug_assert_eq!(
        2 * (n00 + n01 + n11),
        l - (k + connected_to_zs.len()),
        "Miscalculation: 2({}+{}+{}) is not {}-({} + {})",
        n00,
        n01,
        n11,
        l,
        k,
        connected_to_zs.len()
    );
    // Use the counts to compute all the delocalized inner products we may need.
    let hs = (0..=k)
        .map(|nz| {
            delocalized_z_calculation(nz, n00, n01, n11)
                .to_f64()
                .expect("Error converting big rational to f64")
        })
        .collect::<Vec<_>>();

    // Lets track backwards which Unitaries connect to site i
    let mut lookup = vec![0; l];
    let mut us_in_op_region = vec![];
    sample
        .gates
        .iter()
        .copied()
        .enumerate()
        .for_each(|(ui, ((a, b), _))| {
            lookup[a] = ui;
            lookup[b] = ui;
            us_in_op_region.push(ui);
        });

    // Iterate over all swap-distances d
    let res = (0..=k.min(l - k))
        .map(|d| {
            // Iterate over internal vs external swaps.
            betas[d]
                * (0..=d.min(connected_to_zs.len()))
                    .map(|a| -> Complex<f64> {
                        let b = d - a;
                        // Within specified nearby-operator indices:
                        // start with k Zs, remove b of them, swap a of them
                        // (k-d) of the Zs are still in place.
                        // if k = 0 then d = 0 but we still want to do one iteration.
                        fold_over_choices(
                            k.max(1) - 1,
                            k - d,
                            Complex::zero(),
                            |acc, stay_in_place| {
                                fold_over_choices(
                                    connected_to_zs.len().max(1) - 1,
                                    a,
                                    acc,
                                    |acc, moved_to_rel| {
                                        debug_assert_eq!(
                                            stay_in_place.len() + moved_to_rel.len(),
                                            k - b
                                        );
                                        let mut zlookup = vec![false; l];
                                        stay_in_place
                                            .iter()
                                            .copied()
                                            .chain(
                                                moved_to_rel
                                                    .iter()
                                                    .copied()
                                                    .map(|i| connected_to_zs[i]),
                                            )
                                            .for_each(|x| zlookup[x] = true);
                                        let prod = prod_zs(
                                            us_in_op_region.iter().map(|ui| sample.gates[*ui]),
                                            &zlookup,
                                            &sample.measurement,
                                            pairwise_ops,
                                        );

                                        acc + prod
                                    },
                                )
                            },
                        ) * hs[b]
                    })
                    .sum::<Complex<f64>>()
        })
        .sum::<Complex<f64>>();

    let check = dumb_estimate_canonical_z_string(l, k, sample, pairwise_ops, betas);
    debug_assert!(
        (res - check).abs() < 1e-10,
        "{:?} too far from {:?} ({:?})",
        res,
        check,
        (res - check).abs()
    );
    res
}

fn estimate_canonical_z_string_by_parts(
    k: usize,
    sample: &Sample,
    pairwise_ops: ArrayView3<Complex<f64>>,
) -> Complex<f64> {
    // The local channel is Mij = 1/2 ( 1/3 ( I - S ) + (I + S) )
    // The local inverse is therefore iMij = 1/2 ( 3 (I - S) + (I + S) ) = 2I - S
    sample
        .gates
        .iter()
        .copied()
        .map(|((i, j), iu)| {
            let zi = i < k;
            let zj = j < k;
            let bi = sample.measurement.get_bit(i);
            let bj = sample.measurement.get_bit(j);
            let u = pairwise_ops.index_axis(Axis(0), iu);
            2. * compute_single_inner(bi, bj, zi, zj, u) - compute_single_inner(bi, bj, zj, zi, u)
        })
        .product::<Complex<f64>>()
}

fn prod_zs<It>(
    ops: It,
    zmask: &[bool],
    measurement: &BitString,
    pairwise_ops: ArrayView3<Complex<f64>>,
) -> Complex<f64>
where
    It: IntoIterator<Item = ((usize, usize), usize)>,
{
    ops.into_iter()
        .map(|((i, j), iu)| {
            let bi = measurement.get_bit(i);
            let bj = measurement.get_bit(j);
            let zi = zmask[i];
            let zj = zmask[j];

            compute_single_inner(bi, bj, zi, zj, pairwise_ops.index_axis(Axis(0), iu))
        })
        .product::<Complex<f64>>()
}

fn count_pairs_for_us<It>(it: It, measurement: &BitString) -> (usize, usize, usize)
where
    It: IntoIterator<Item = (usize, usize)>,
{
    it.into_iter().fold(
        (0usize, 0usize, 0usize),
        |(mut n00, mut n01, mut n11), (a, b)| {
            let bh = measurement.get_bit(a);
            let bt = measurement.get_bit(b);
            match (bh, bt) {
                (false, false) => {
                    n00 += 1;
                }
                (true, false) | (false, true) => {
                    n01 += 1;
                }
                (true, true) => {
                    n11 += 1;
                }
            }
            (n00, n01, n11)
        },
    )
}

fn get_connected_to_zs<It>(k: usize, it: It) -> impl Iterator<Item = usize>
where
    It: IntoIterator<Item = (usize, usize)>,
{
    it.into_iter().filter_map(move |(a, b)| {
        let head = a.min(b);
        let tail = a.max(b);
        if head < k && tail >= k {
            Some(tail)
        } else {
            None
        }
    })
}

fn compute_single_inner(
    bi: bool,
    bj: bool,
    zi: bool,
    zj: bool,
    u: ArrayView2<Complex<f64>>,
) -> Complex<f64> {
    // If bi==bj then we are in the full or empty sectors where U is trivial.
    match (bi, bj) {
        (false, false) => Complex::one(),
        (true, true) => {
            if zi == zj {
                Complex::one()
            } else {
                -Complex::one()
            }
        }
        (bi, _) => {
            // We want <b| U (Pi Pj) U^\dagger |b>
            // First get <b| U
            let bu = u.slice(s![1 + (bi as usize), 1..3]).to_owned();
            // Then get U^\dagger |b>
            let mut ub = bu.clone();
            ub.iter_mut().for_each(|x| *x = x.conj());

            if zi {
                ub[1] *= -1.0;
            }
            if zj {
                ub[0] *= -1.0;
            }

            bu.dot(&ub)
        }
    }
}

fn delocalized_z_calculation(nz: usize, n00: usize, n01: usize, n11: usize) -> BigRational {
    // From the condition in the pairs sum:
    // nz - 2m <= n00 + n11
    // (nz - (n00 + n11)) / 2 <= m
    let minval = if n00 + n11 >= nz {
        0
    } else {
        (nz - n00 - n11 + 1) / 2
    };

    (minval..=nz / 2)
        .map(|m| {
            let first_sum = (0..=m)
                .map(|a| -> BigRational {
                    let b = m - a;
                    let neg = if b % 2 == 0 {
                        BigRational::one()
                    } else {
                        -BigRational::one()
                    };
                    neg * rational_choose(n00 + n11 - (nz - 2 * m), a) * rational_choose(n01, b)
                })
                .sum::<BigRational>();
            let second_sum = (0..=(nz - 2 * m))
                .map(|x| -> BigRational {
                    let y = (nz - 2 * m) - x;

                    let neg = if y % 2 == 0 {
                        BigRational::one()
                    } else {
                        -BigRational::one()
                    };
                    neg * BigInt::from(2).pow((nz - 2 * m) as u32)
                        * rational_choose(n00, x)
                        * rational_choose(n11, y)
                })
                .sum::<BigRational>();
            first_sum * second_sum
        })
        .sum::<BigRational>()
}

#[pyfunction]
pub fn symmetry_eigenvalue(l: usize, a: usize) -> f64 {
    symmetry_sector_eigenvalue(l, a)
        .to_f64()
        .unwrap_or_default()
}

fn symmetry_sector_eigenvalue(l: usize, a: usize) -> BigRational {
    let one_third = BigRational::new(BigInt::from(-1), BigInt::from(3));
    let two_third = BigRational::new(BigInt::from(2), BigInt::from(3));

    (0..=a)
        .map(|d| {
            let pref = (one_third.clone()).pow(d as i32);
            (0..=(a - d))
                .filter(|m| m % 2 == (a - d) % 2)
                .map(|m| {
                    let pref = two_third.clone().pow(m as i32);
                    let choose = rational_choose(l - d - a, m);
                    let top = permutation_product_list(a)
                        .chain(pairings_product_list(a - d - m).expect("Must be even"))
                        .chain(pairings_product_list(l - d - a - m).expect("Must be even"));
                    let bot = permutation_product_list(a - d - m)
                        .chain(pairings_product_list(l).expect("Must be even"));
                    pref * choose * rational_quotient_of_products(top, bot)
                })
                .sum::<BigRational>()
                * pref
        })
        .sum::<BigRational>()
}

#[pyfunction]
pub fn get_g_matrix(py: Python, l: usize, k: usize) -> Py<PyArray2<f64>> {
    get_g_mat(l, k).to_pyarray(py).to_owned()
}

fn get_g_mat(l: usize, k: usize) -> Array2<f64> {
    let k = k.min(l - k);
    let mut g = Array2::zeros((k + 1, k + 1));
    ndarray::Zip::indexed(&mut g)
        .for_each(|(a, d), x| *x = get_g_mat_entry(l, k, a, d).to_f64().unwrap_or_default());
    g
}

fn get_g_mat_entry(l: usize, k: usize, a: usize, d: usize) -> BigRational {
    (0..=d)
        .map(|x| {
            let y = d - x;
            let anti_symm = rational_choose(a, x) * BigInt::from(-1).pow(x as u32);
            let symm = rational_choose(k - a, y) * rational_choose(l - k - a, y);
            anti_symm * symm
        })
        .sum::<BigRational>()
}

#[pyclass]
#[derive(Default)]
pub struct Operator {
    pub opstrings: Vec<(Complex<f64>, OperatorString)>,
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
    use ndarray::{Array2, Array3};

    fn make_pairwise_eye() -> Array3<Complex<f64>> {
        let mut arr = Array3::zeros((1, 4, 4));
        ndarray::Zip::indexed(&mut arr).for_each(|(_, i, j), x| {
            if i == j {
                *x = Complex::one();
            } else {
                *x = Complex::zero();
            }
        });
        arr
    }

    fn make_sample(l: usize, m: usize) -> Sample {
        let bm = BitString::new_short(m, l);
        let gates = (0..l / 2).map(|i| ((2 * i, 2 * i + 1), 0));
        Sample {
            gates: gates.collect(),
            measurement: bm,
        }
    }

    #[test]
    fn test_eigenvalue() {
        for i in 0..=3 {
            let res = symmetry_sector_eigenvalue(6, i);
            println!("{:?}", res);
        }
    }

    #[test]
    fn check_deloc_simple() {
        let l = 8;
        // Assume we have all 0s, deloc just counts strings.
        for i in 0..l / 2 {
            assert_eq!(
                delocalized_z_calculation(i, l / 2, 0, 0),
                rational_choose(l, i)
            );
        }
    }

    #[test]
    fn check_deloc_simple_ones() {
        let l = 8;
        // All 1s identical to all 0s but with a global (-1)^nz
        for i in 0..l / 2 {
            assert_eq!(
                delocalized_z_calculation(i, l / 2, 0, 0),
                delocalized_z_calculation(i, 0, 0, l / 2) * BigInt::from(-1).pow(i as u32),
            );
        }
    }

    #[test]
    fn test_deloc_dumb() {
        let bm = BitString::new_short(0b00011011, 8);
        let gates = (0..4).map(|i| ((2 * i, 2 * i + 1), 0)).collect::<Vec<_>>();
        let (n00, n01, n11) = count_pairs_for_us(gates.iter().copied().map(|(x, _)| x), &bm);
        assert_eq!((n00, n01, n11), (1, 2, 1));

        let sample = Sample {
            gates,
            measurement: bm,
        };
        let pairwise = make_pairwise_eye();

        for k in 0..8 {
            let betas = vec![1.0; k + 1];
            let res = dumb_estimate_canonical_z_string(8, k, &sample, pairwise.view(), &betas);
            let deloc_res = delocalized_z_calculation(k, n00, n01, n11)
                .to_f64()
                .map(Complex::from)
                .unwrap_or_default();
            assert_eq!(res, deloc_res)
        }
    }

    #[test]
    fn test_canonical_across() {
        let bm = BitString::new_short(0b00011011, 8);
        let gates = (0..4).map(|i| ((2 * i, 2 * i + 1), 0)).collect::<Vec<_>>();

        let sample = Sample {
            gates,
            measurement: bm,
        };
        let pairwise = make_pairwise_eye();

        for k in 0..8 {
            let mut betas = vec![0.0; k + 1];
            for d in 0..(k + 1) {
                betas[d] = 1.0;
                let resa = dumb_estimate_canonical_z_string(8, k, &sample, pairwise.view(), &betas);
                let resb = estimate_canonical_z_string(8, k, &sample, pairwise.view(), &betas);
                assert_eq!(resa, resb);
                betas[d] = 0.0;
            }
        }
    }

    #[test]
    fn test_estimate_full() -> Result<(), String> {
        let l = 8;
        let pairwise = make_pairwise_eye();
        let bm = BitString::new_short(0b00011011, l);
        let gates = (0..l / 2)
            .map(|i| ((2 * i, 2 * i + 1), 0))
            .collect::<Vec<_>>();

        let sample = Sample {
            gates,
            measurement: bm,
        };
        let mut samples = Samples::new_raw(l, pairwise);
        samples.add_sample(sample);

        let opstring = OperatorString::new(
            (0..l / 2)
                .map(|_| OpChar::Z)
                .chain((0..l / 2).map(|_| OpChar::I))
                .collect::<Vec<_>>(),
        );

        let recon = Reconstruction::new(None);
        let it = recon.estimate_operator_string_iterator(&opstring, &samples, false)?;
        it.for_each(|_| {});
        Ok(())
    }

    #[test]
    fn test_count_pairs() {
        let samples = make_sample(8, 0);
        let counts =
            count_pairs_for_us(samples.gates.iter().map(|(x, _)| *x), &samples.measurement);
        assert_eq!(counts, (4, 0, 0));

        let samples = make_sample(8, 0b11111111);
        let counts =
            count_pairs_for_us(samples.gates.iter().map(|(x, _)| *x), &samples.measurement);
        assert_eq!(counts, (0, 0, 4));

        let samples = make_sample(8, 0b00011011);
        let counts =
            count_pairs_for_us(samples.gates.iter().map(|(x, _)| *x), &samples.measurement);
        assert_eq!(counts, (1, 2, 1));
    }

    #[test]
    fn test_overlaps() {
        let eye = Array2::eye(4);
        [false, true].iter().copied().for_each(|bi| {
            [false, true].iter().copied().for_each(|bj| {
                [false, true].iter().copied().for_each(|zi| {
                    [false, true].iter().copied().for_each(|zj| {
                        let mut x = Complex::one();
                        if bi && zi {
                            x *= -1.0;
                        }
                        if bj && zj {
                            x *= -1.0;
                        }
                        assert_eq!(
                            x,
                            compute_single_inner(bi, bj, zi, zj, eye.view()),
                            "Failed to estimate for state |{},{}> with Zs: {},{}",
                            bi,
                            bj,
                            zi,
                            zj
                        );
                    })
                });
            })
        });

        let mut arr = Array2::zeros((4, 4));
        arr[(0, 0)] = Complex::one();
        arr[(1, 2)] = Complex::one();
        arr[(2, 1)] = Complex::one();
        arr[(3, 3)] = Complex::one();
        [false, true].iter().copied().for_each(|bi| {
            [false, true].iter().copied().for_each(|bj| {
                [false, true].iter().copied().for_each(|zi| {
                    [false, true].iter().copied().for_each(|zj| {
                        let mut x = Complex::one();
                        if bi && zj {
                            x *= -1.0;
                        }
                        if bj && zi {
                            x *= -1.0;
                        }
                        assert_eq!(
                            x,
                            compute_single_inner(bi, bj, zi, zj, arr.view()),
                            "Failed to estimate for state |{},{}> with Zs: {},{}",
                            bi,
                            bj,
                            zi,
                            zj
                        );
                    })
                });
            })
        });
    }
}
