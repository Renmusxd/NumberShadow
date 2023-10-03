use ndarray::{array, s, Array3, ArrayView2};
use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::BigRational;
use num_traits::{One, Zero};
use numpy::ndarray::Array2;
use numpy::{ndarray, IntoPyArray, PyArray3};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "sampling")]
use sprs::{kronecker_product, CsMat, TriMat};
use std::collections::HashMap;

/// Reverse last n bits of a
pub(crate) fn reverse_n_bits(a: usize, n: u32) -> usize {
    a.reverse_bits() >> (usize::BITS - n)
}

pub(crate) fn kron_helper<It>(mats: It) -> CsMat<Complex<f64>>
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
    pub(crate) fn make_matrices_skip_ident(&self) -> Vec<(usize, Array2<Complex<f64>>)> {
        self.opstring
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, c)| OpChar::I.ne(c))
            .map(|(i, c)| (i, c.get_matrix()))
            .collect()
    }

    pub(crate) fn make_matrix(&self) -> CsMat<Complex<f64>> {
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

    pub(crate) fn new<VC>(chars: VC) -> Self
    where
        VC: Into<Vec<OpChar>>,
    {
        Self {
            indices: None,
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OpChar {
    Z,
    Plus,
    Minus,
    I,
}

impl OpChar {
    pub(crate) fn get_matrix(&self) -> Array2<Complex<f64>> {
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

#[pyclass]
#[derive(Clone, Serialize, Deserialize, Debug, Hash, Eq, PartialEq)]
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
            BitStringEnum::Small(bb) => {
                if b {
                    *bb |= 1 << (self.size - i - 1);
                } else {
                    *bb &= !(1 << (self.size - i - 1));
                }
            }
        }
        debug_assert_eq!(self.get_bit(i), b);
    }

    fn get_bits(&self) -> Vec<bool> {
        match self.clone().make_long().data {
            BitStringEnum::Small(_) => unreachable!(),
            BitStringEnum::Large(v) => v,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Hash, Eq, PartialEq)]
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

    pub fn make_short(&self) -> Self {
        let mut x = Self::new_short(0, self.size);
        for i in 0..self.size {
            x.set_bit(i, self.get_bit(i));
        }
        x
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

    pub fn iter_bits(&self) -> impl Iterator<Item = bool> + '_ {
        let v = (0..self.size).map(|i| self.get_bit(i)).collect::<Vec<_>>();
        v.into_iter()
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

pub(crate) fn fold_over_choices<T, F>(maxval: usize, length: usize, init: T, f: F) -> T
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

pub(crate) fn fold_over_choices_rec<T, F>(
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

pub(crate) fn rational_choose(n: usize, m: usize) -> BigRational {
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

pub(crate) fn rational_quotient_of_products<It1, It2>(top: It1, bot: It2) -> BigRational
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

pub(crate) fn permutation_product_list(l: usize) -> impl Iterator<Item = usize> {
    1..=l
}

pub(crate) fn pairings_product_list(l: usize) -> Result<impl Iterator<Item = usize>, String> {
    if l % 2 == 1 {
        return Err("Pairings only valid for even.".to_string());
    }
    Ok((0..l / 2).map(|i| 2 * i + 1))
}

pub(crate) fn get_pauli_ops() -> Array3<Complex<f64>> {
    let l = Complex::one();
    let o = Complex::zero();
    let rx = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let ix = Complex::new(0.0, std::f64::consts::FRAC_1_SQRT_2);
    let mids = array![
        [[l, o], [o, l]],
        [[rx, -rx], [rx, rx],],
        [[rx, -ix], [-ix, rx]],
    ];
    let mut res = Array3::zeros((3, 4, 4));
    let mut s = res.slice_mut(s![.., 0, 0]);
    s.iter_mut().for_each(|x| *x = Complex::one());
    let mut s = res.slice_mut(s![.., 3, 3]);
    s.iter_mut().for_each(|x| *x = Complex::one());
    let mut s = res.slice_mut(s![.., 1..3, 1..3]);
    s.iter_mut().zip(mids).for_each(|(a, b)| *a = b);
    res
}

#[pyfunction]
pub fn make_pauli_ops(py: Python) -> Py<PyArray3<Complex<f64>>> {
    let ops = get_pauli_ops();
    ops.into_pyarray(py).to_owned()
}

#[derive(Debug, Clone)]
pub struct SparseVec {
    n_qubits: usize,
    data: HashMap<BitString, Complex<f64>>,
}

impl SparseVec {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            data: Default::default(),
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn dim(&self) -> usize {
        1 << self.n_qubits
    }

    pub fn overwrite<K>(&mut self, row: K, val: Complex<f64>)
    where
        K: Into<BitString>,
    {
        self.data.insert(row.into(), val);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&BitString, &Complex<f64>)> {
        self.data.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&BitString, &mut Complex<f64>)> {
        self.data.iter_mut()
    }

    pub fn apply_twobody_op(&self, i: usize, j: usize, op: ArrayView2<Complex<f64>>) -> Self {
        assert_eq!(op.shape(), &[4, 4]);
        let mut v = HashMap::<BitString, Complex<f64>>::default();
        self.data.iter().for_each(|(index, value)| {
            if Complex::zero().eq(value) {
                return;
            }
            let ib = index.get_bit(i);
            let jb = index.get_bit(j);
            let subcol = ((ib as usize) << 1) | (jb as usize);

            let mut new_index = index.clone();
            for io in [0, 1] {
                new_index.set_bit(i, io == 1);
                for jo in [0, 1] {
                    new_index.set_bit(j, jo == 1);
                    let subrow = (io << 1) | jo;
                    if let Some(c) = op.get((subrow, subcol)) {
                        if Complex::zero().ne(c) {
                            *v.entry(new_index.clone()).or_insert(Complex::zero()) += c * value;
                        }
                    }
                }
            }
        });
        Self {
            n_qubits: self.n_qubits,
            data: v,
        }
    }

    pub fn expectation(&self, opstring: &OperatorString) -> Complex<f64> {
        let mut v = HashMap::<BitString, Complex<f64>>::default();
        self.data.iter().for_each(|(index, value)| {
            let mut new_index = index.clone();
            let mut new_val = *value;
            let mut insert = true;
            for (i, op) in opstring.opstring.iter().enumerate() {
                let b = index.get_bit(i);
                match op {
                    OpChar::Z if b => new_val *= -1.0,
                    OpChar::Plus => {
                        if !b {
                            new_index.set_bit(i, true)
                        } else {
                            insert = false;
                            break;
                        }
                    }
                    OpChar::Minus => {
                        if b {
                            new_index.set_bit(i, false)
                        } else {
                            insert = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if insert {
                v.insert(new_index, new_val);
            }
        });
        v.into_iter()
            .filter_map(|(index, val)| self.data.get(&index).map(|c| c.conj() * val))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;
    pub fn permute_bits(input: usize, perm: &[usize]) -> usize {
        let mut acc = 0;

        for (i, j) in perm.iter().copied().enumerate() {
            acc |= ((input >> (perm.len() - 1 - i)) & 1) << (perm.len() - 1 - j);
        }
        acc
    }

    #[test]
    fn test_pauli_ops() {
        let x = get_pauli_ops();
        println!("{:?}", x)
    }

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
        let num_bits = 8;
        for i in 0..1 << num_bits {
            let bs = BitString::new_short(i, num_bits);
            let bl = bs.make_long();
            for i in 0..num_bits {
                assert_eq!(bs.get_bit(i), bl.get_bit(i));
            }
        }
    }

    #[test]
    fn test_bitstring_set_long() {
        let bs = BitString::new_short(0b00000000, 8);
        let mut bl = bs.make_long();
        for i in 0..8 {
            bl.set_bit(i, true);
            for j in 0..8 {
                assert_eq!(bl.get_bit(j), i == j);
            }
            bl.set_bit(i, false);
        }
    }
    #[test]
    fn test_bitstring_set() {
        let mut bs = BitString::new_short(0b00000000, 8);
        for i in 0..8 {
            bs.set_bit(i, true);
            for j in 0..8 {
                assert_eq!(bs.get_bit(j), i == j);
            }
            bs.set_bit(i, false);
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
