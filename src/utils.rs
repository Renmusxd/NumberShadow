use ndarray::Array1;
use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::BigRational;
use num_traits::{One, Zero};
use numpy::ndarray;
use numpy::ndarray::{Array2, Array4, ArrayView2, Axis};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::Python;
use serde::{Deserialize, Serialize};
use sprs::{kronecker_product, CsMat, CsVec, TriMat};
use std::ops::Add;

pub fn permute_bits(input: usize, perm: &[usize]) -> usize {
    let mut acc = 0;

    for (i, j) in perm.iter().copied().enumerate() {
        acc |= ((input >> (perm.len() - 1 - i)) & 1) << (perm.len() - 1 - j);
    }
    acc
}

pub fn make_sprs<P>(m: ArrayView2<P>) -> CsMat<P>
where
    P: Clone + Add<Output = P>,
{
    let mut a = TriMat::new((m.shape()[0], m.shape()[1]));
    ndarray::Zip::indexed(m).for_each(|(row, col), val| {
        a.add_triplet(row, col, val.clone());
    });
    a.to_csr()
}

pub fn make_sprs_onehot<P>(i: usize, n: usize) -> CsVec<P>
where
    P: Clone + Add<Output = P> + One,
{
    CsVec::new(n, vec![i], vec![P::one()])
}

pub fn make_dense_onehot<P>(i: usize, n: usize) -> Array1<P>
where
    P: One + Clone + Zero,
{
    let mut arr = Array1::zeros((n,));
    arr[i] = P::one();
    arr
}

pub fn make_perm<P>(perm: &[usize]) -> CsMat<P>
where
    P: Clone + Add<Output = P> + One,
{
    let mut a = TriMat::new((1 << perm.len(), 1 << perm.len()));
    (0..1 << perm.len()).for_each(|i| a.add_triplet(permute_bits(i, perm), i, P::one()));
    // (0..1 << perm.len()).for_each(|i| a.add_triplet(i, permute_bits(i, perm), P::one()));
    a.to_csr()
}

pub fn fact(mut n: usize) -> usize {
    let mut acc = 1;
    while n > 1 {
        acc *= n;
        n -= 1;
    }
    acc
}

pub fn fact2(mut n: usize) -> usize {
    let mut acc = 1;
    while n > 1 {
        acc *= n;
        n -= 2;
    }
    acc
}

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

/// Reverse last n bits of a
pub fn reverse_n_bits(a: usize, n: u32) -> usize {
    a.reverse_bits() >> (usize::BITS - n)
}

pub fn kron_helper<It>(mats: It) -> CsMat<Complex<f64>>
where
    It: IntoIterator<Item = CsMat<Complex<f64>>>,
{
    mats.into_iter()
        .fold(None, |acc: Option<CsMat<Complex<f64>>>, op| {
            if let Some(acc) = acc {
                let acc = kronecker_product(acc.view(), op.view());
                Some(acc)
            } else {
                Some(op)
            }
        })
        .unwrap()
}

pub struct OperatorString {
    pub(crate) indices: Option<Vec<usize>>,
    pub(crate) opstring: Vec<OpChar>,
}

impl OperatorString {
    pub fn make_matrices(&self) -> Vec<Array2<Complex<f64>>> {
        self.opstring.iter().map(|c| c.get_matrix()).collect()
    }

    pub fn make_matrices_skip_ident(&self) -> Vec<(usize, Array2<Complex<f64>>)> {
        self.opstring
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, c)| OpChar::I.ne(c))
            .map(|(i, c)| (i, c.get_matrix()))
            .collect()
    }

    pub fn make_matrix(&self) -> CsMat<Complex<f64>> {
        let cmats = self.opstring.iter().map(|c| {
            let m = c.get_matrix();
            let mut a = TriMat::new((m.shape()[0], m.shape()[1]));
            ndarray::Zip::indexed(&m).for_each(|(i, j), c| {
                if c.norm() > f64::EPSILON {
                    a.add_triplet(i, j, *c);
                }
            });
            a.to_csr()
        });
        kron_helper(cmats)
    }
}

impl From<Vec<OpChar>> for OperatorString {
    fn from(value: Vec<OpChar>) -> Self {
        OperatorString::new(value)
    }
}

impl TryFrom<String> for OperatorString {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        OperatorString::try_from(value.as_str())
    }
}

impl TryFrom<&str> for OperatorString {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_new(value.chars())
    }
}

impl OperatorString {
    pub(crate) fn get(&self, index: usize) -> OpChar {
        if let Some(indices) = &self.indices {
            indices
                .iter()
                .copied()
                .zip(self.opstring.iter())
                .find(|(i, _)| index == *i)
                .map(|(_, b)| *b)
                .unwrap_or(OpChar::I)
        } else {
            self.opstring[index]
        }
    }

    pub fn new<VC>(chars: VC) -> Self
    where
        VC: Into<Vec<OpChar>>,
    {
        Self {
            indices: None,
            opstring: chars.into(),
        }
    }

    pub fn new_indices<VI, VC>(indices: Option<VI>, chars: VC) -> Self
    where
        VI: Into<Vec<usize>>,
        VC: Into<Vec<OpChar>>,
    {
        Self {
            indices: indices.map(|indices| indices.into()),
            opstring: chars.into(),
        }
    }

    pub(crate) fn try_new<Itb, OC, E>(chars: Itb) -> Result<Self, String>
    where
        Itb: IntoIterator<Item = OC>,
        OC: TryInto<OpChar, Error = E>,
        E: Into<String>,
    {
        let opstring =
            chars
                .into_iter()
                .try_fold(vec![], |mut acc, x| -> Result<Vec<OpChar>, String> {
                    let x = x.try_into().map_err(|e| e.into())?;
                    acc.push(x);
                    Ok(acc)
                })?;
        Ok(Self {
            indices: None,
            opstring,
        })
    }

    pub(crate) fn try_new_indices<Iti, Itb, OC, E>(
        indices: Option<Iti>,
        chars: Itb,
    ) -> Result<Self, String>
    where
        Iti: IntoIterator<Item = usize>,
        Itb: IntoIterator<Item = OC>,
        OC: TryInto<OpChar, Error = E>,
        E: Into<String>,
    {
        let indices = indices.map(|indices| indices.into_iter().collect());

        let opstring =
            chars
                .into_iter()
                .try_fold(vec![], |mut acc, x| -> Result<Vec<OpChar>, String> {
                    let x = x.try_into().map_err(|e| e.into())?;
                    acc.push(x);
                    Ok(acc)
                })?;
        Ok(Self { indices, opstring })
    }

    pub(crate) fn count_terms(&self) -> [usize; 4] {
        let mut counts = [0, 0, 0, 0];
        for op in &self.opstring {
            let index: usize = (*op).into();
            counts[index] += 1;
        }
        counts
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpChar {
    Z,
    Plus,
    Minus,
    I,
}

impl OpChar {
    pub fn get_matrix(&self) -> Array2<Complex<f64>> {
        let o = Complex::<f64>::zero();
        let l = Complex::<f64>::one();
        match self {
            OpChar::Z => Array2::from_shape_vec((2, 2), vec![l, o, o, -l]).unwrap(),
            OpChar::Plus => Array2::from_shape_vec((2, 2), vec![o, o, l, o]).unwrap(),
            OpChar::Minus => Array2::from_shape_vec((2, 2), vec![o, l, o, o]).unwrap(),
            OpChar::I => Array2::from_shape_vec((2, 2), vec![l, o, o, l]).unwrap(),
        }
    }
}

impl From<OpChar> for usize {
    fn from(val: OpChar) -> Self {
        match val {
            OpChar::Z => 0,
            OpChar::Plus => 1,
            OpChar::Minus => 2,
            OpChar::I => 3,
        }
    }
}

impl TryFrom<usize> for OpChar {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Z),
            1 => Ok(Self::Plus),
            2 => Ok(Self::Minus),
            3 => Ok(Self::I),
            _ => Err(()),
        }
    }
}

impl TryFrom<char> for OpChar {
    type Error = String;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'Z' => Ok(OpChar::Z),
            '+' => Ok(OpChar::Plus),
            '-' => Ok(OpChar::Minus),
            'I' => Ok(OpChar::I),
            _ => Err("Not a valid operator string".to_string()),
        }
    }
}

pub fn scipy_mat<'a, P>(py: Python<'a>, mat: &CsMat<P>) -> Result<&'a PyAny, String>
where
    P: Clone + IntoPy<Py<PyAny>>,
{
    let scipy_sparse = PyModule::import(py, "scipy.sparse").map_err(|e| {
        let res = format!("Python error: {e:?}");
        e.print_and_set_sys_last_vars(py);
        res
    })?;
    let indptr = mat.indptr().to_proper().to_vec();
    scipy_sparse
        .call_method(
            "csr_matrix",
            ((mat.data().to_vec(), mat.indices().to_vec(), indptr),),
            Some([("shape", mat.shape())].into_py_dict(py)),
        )
        .map_err(|e| {
            let res = format!("Python error: {e:?}");
            e.print_and_set_sys_last_vars(py);
            res
        })
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BitString {
    size: usize,
    data: BitStringEnum,
}

#[pymethods]
impl BitString {
    #[new]
    fn new(bits: Vec<bool>) -> Self {
        Self {
            size: bits.len(),
            data: BitStringEnum::Large(bits),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum BitStringEnum {
    Small(usize),
    Large(Vec<bool>),
}

impl BitString {
    pub fn new_long(num_bits: usize) -> Self {
        Self {
            size: num_bits,
            data: BitStringEnum::Large(vec![false; num_bits]),
        }
    }

    pub fn new_short(i: usize, num_bits: usize) -> Self {
        Self {
            size: num_bits,
            data: BitStringEnum::Small(i),
        }
    }

    pub fn make_long(&self) -> Self {
        match &self.data {
            BitStringEnum::Large(_) => self.clone(),
            BitStringEnum::Small(i) => {
                let mut i = reverse_n_bits(*i, self.size as u32);
                let mut v = vec![];
                for _ in 0..self.size {
                    v.push((i & 1) == 1);
                    i >>= 1;
                }
                Self {
                    size: self.size,
                    data: BitStringEnum::Large(v),
                }
            }
        }
    }

    pub fn num_bits(&self) -> usize {
        self.size
    }

    pub fn get_bit(&self, i: usize) -> bool {
        match &self.data {
            BitStringEnum::Large(v) => v[i],
            BitStringEnum::Small(b) => {
                let bit = (b >> (self.size - i - 1)) & 1;
                bit == 1
            }
        }
    }

    pub fn set_bit(&mut self, i: usize, b: bool) {
        match &mut self.data {
            BitStringEnum::Large(v) => v[i] = b,
            BitStringEnum::Small(_) => {
                todo!()
            }
        }
    }
}

impl From<Vec<bool>> for BitString {
    fn from(value: Vec<bool>) -> Self {
        Self {
            size: value.len(),
            data: BitStringEnum::Large(value),
        }
    }
}

pub fn fold_over_choices<T, F>(maxval: usize, length: usize, init: T, f: F) -> T
where
    F: Fn(T, &[usize]) -> T,
{
    if length > maxval + 1 {
        return init;
    }
    if length == 0 {
        return f(init, &[]);
    }
    fold_over_choices_rec(&mut vec![], 0, maxval, length, &f, init)
}

pub fn fold_over_choices_rec<T, F>(
    prefix: &mut Vec<usize>,
    minval: usize,
    maxval: usize,
    length: usize,
    f: F,
    init: T,
) -> T
where
    F: Copy + Fn(T, &[usize]) -> T,
{
    (minval..=maxval - (length - (prefix.len() + 1))).fold(init, |acc, a| -> T {
        prefix.push(a);
        let ret = if prefix.len() >= length {
            f(acc, prefix)
        } else {
            fold_over_choices_rec(prefix, a + 1, maxval, length, f, acc)
        };
        prefix.pop();
        ret
    })
}

pub fn number_conserving_callback<F>(
    prefix: &mut Vec<usize>,
    minval: usize,
    maxval: usize,
    length: usize,
    callback: &F,
) where
    F: Fn(&[usize]),
{
    for a in minval..=maxval {
        prefix.push(a);
        if prefix.len() == length {
            callback(prefix)
        } else {
            number_conserving_callback(prefix, a + 1, maxval, length, callback)
        }
        prefix.pop();
    }
}

pub fn rational_choose(n: usize, m: usize) -> BigRational {
    if m > n {
        return BigRational::zero();
    }
    if m == 0 || m == n {
        return BigRational::one();
    }

    let top = (m + 1)..=n;
    let bot = 1..=(n - m);
    let res = rational_quotient_of_products(top, bot);
    debug_assert_eq!(res, {
        let top = 1..=n;
        let bot = (1..=m).chain(1..=(n - m));
        rational_quotient_of_products(top, bot)
    });
    res
}

pub fn rational_quotient_of_products<It1, It2>(top: It1, bot: It2) -> BigRational
where
    It1: IntoIterator<Item = usize>,
    It2: IntoIterator<Item = usize>,
{
    let mut top = top.into_iter();
    let mut bot = bot.into_iter();

    let mut acc = BigRational::one();
    loop {
        let t = top.next();
        let b = bot.next();
        if t.is_none() && b.is_none() {
            return acc;
        }
        if let Some(t) = t {
            acc *= BigInt::from(t);
        }
        if let Some(b) = b {
            acc /= BigInt::from(b);
        }
    }
}

pub fn permutation_product_list(l: usize) -> impl Iterator<Item = usize> {
    1..=l
}

pub fn pairings_product_list(l: usize) -> Result<impl Iterator<Item = usize>, String> {
    if l % 2 == 1 {
        return Err("Pairings only valid for even.".to_string());
    }
    Ok((0..l / 2).map(|i| 2 * i + 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    #[test]
    fn test_choose() {
        for i in 0..=30 {
            let mut acc = BigRational::zero();
            for j in 0..=i {
                acc += rational_choose(i, j);
            }
            assert!((acc.to_f64().expect("Error") - (1 << i) as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fold_num_cons() {
        let v = fold_over_choices(5, 3, vec![], |mut v, x| {
            v.push(x.to_vec());
            v
        });
        // (5+1) choose 3
        assert_eq!(v.len(), 20);
    }

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_n_bits(0b0001, 4), 0b1000);
        assert_eq!(reverse_n_bits(0b001, 3), 0b100);
        assert_eq!(reverse_n_bits(0b01, 2), 0b10);
        assert_eq!(reverse_n_bits(0b1, 1), 0b1);
    }

    #[test]
    fn easy_perm() {
        for i in 0..1 << 4 {
            let result = permute_bits(i, &[0, 1, 2, 3]);
            assert_eq!(result, i)
        }
    }

    #[test]
    fn test_bitstring() {
        let bs = BitString::new_short(0b00011011, 8);
        let bl = bs.make_long();
        for i in 0..8 {
            assert_eq!(bs.get_bit(i), bl.get_bit(i));
        }
    }

    #[test]
    fn swap_bits() {
        let result = permute_bits(0b0101, &[0, 1, 2, 3]);
        assert_eq!(result, 0b0101);

        let result = permute_bits(0b0101, &[1, 0, 2, 3]);
        assert_eq!(result, 0b1001);

        let result = permute_bits(0b0101, &[0, 1, 3, 2]);
        assert_eq!(result, 0b0110);
    }

    #[test]
    fn shift_perm() {
        let result = permute_bits(0b0001, &[1, 2, 3, 0]);
        assert_eq!(result, 0b1000);

        let result = permute_bits(0b0010, &[1, 2, 3, 0]);
        assert_eq!(result, 0b0001);

        let result = permute_bits(0b0100, &[1, 2, 3, 0]);
        assert_eq!(result, 0b0010);

        let result = permute_bits(0b1000, &[1, 2, 3, 0]);
        assert_eq!(result, 0b0100);

        let result = permute_bits(0b0011, &[1, 2, 3, 0]);
        assert_eq!(result, 0b1001);

        let result = permute_bits(0b0110, &[1, 2, 3, 0]);
        assert_eq!(result, 0b0011);

        let result = permute_bits(0b1100, &[1, 2, 3, 0]);
        assert_eq!(result, 0b0110);
    }
}
