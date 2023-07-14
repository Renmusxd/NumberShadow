use num_complex::Complex;
use num_traits::{One, Zero};
use numpy::ndarray;
use numpy::ndarray::{Array2, Array4, ArrayView2, Axis};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::Python;
use sprs::{kronecker_product, CsMat, TriMat};
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

pub fn make_sprs_onehot<P>(i: usize, n: usize) -> CsMat<P>
where
    P: Clone + Add<Output = P> + One,
{
    let mut a = TriMat::new((1, n));
    a.add_triplet(0, i, P::one());
    a.to_csr()
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
                        .zip(kron_prod.into_iter())
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
                Some(op.clone())
            }
        })
        .unwrap()
}

pub struct OperatorString {
    pub(crate) indices: Option<Vec<usize>>,
    pub(crate) opstring: Vec<OpChar>,
}

impl OperatorString {
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

    fn new<VC>(chars: VC) -> Self
    where
        VC: Into<Vec<OpChar>>,
    {
        Self {
            indices: None,
            opstring: chars.into(),
        }
    }

    fn new_indices<VI, VC>(indices: Option<VI>, chars: VC) -> Self
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
            let index: usize = op.clone().into();
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

impl Into<usize> for OpChar {
    fn into(self) -> usize {
        match self {
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

#[cfg(test)]
mod tests {
    use super::*;

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
