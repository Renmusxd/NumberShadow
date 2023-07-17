#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use num_traits::identities::One;
    use sprs::*;
    use std::ops::{Add, Mul};
    use test::Bencher;

    fn make_sprs_onehot<P>(i: usize, n: usize) -> TriMat<P>
    where
        P: Clone + Add<Output = P> + One,
    {
        let mut a = TriMat::new((1, n));
        a.add_triplet(0, i, P::one());
        a
    }

    fn make_ones_trimat(D: usize) -> TriMat<Complex<f64>> {
        let mut b = TriMat::new((D, D));
        for i in 0..D {
            for j in 0..D {
                b.add_triplet(i, j, Complex::one());
            }
        }
        b
    }

    #[bench]
    fn vec_mul_csr(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let v = CsVec::new(D, vec![0usize], vec![Complex::one()]);
        let m = make_ones_trimat(D).to_csr::<usize>();
        b.iter(|| v.mul(&m));
    }

    #[bench]
    fn vec_mul_vec(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let va = CsVec::new(D, vec![0usize], vec![Complex::<f64>::one()]);
        let vb = CsVec::new(D, (0..D).collect(), vec![Complex::<f64>::one(); D]);
        b.iter(|| va.dot(&vb));
    }

    #[bench]
    fn vec_mul_csc(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let v = CsVec::new(D, vec![0usize], vec![Complex::one()]);
        let m = make_ones_trimat(D).to_csc::<usize>();
        b.iter(|| v.mul(&m));
    }

    #[bench]
    fn onehot_csr_mul_csr(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let v = make_sprs_onehot(0, D).to_csr();
        let m = make_ones_trimat(D).to_csr::<usize>();
        b.iter(|| v.mul(&m));
    }

    #[bench]
    fn onehot_csc_mul_csr(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let v = make_sprs_onehot(0, D).to_csc();
        let m = make_ones_trimat(D).to_csr::<usize>();
        b.iter(|| v.mul(&m));
    }

    #[bench]
    fn onehot_csr_mul_csc(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let v = make_sprs_onehot(0, D).to_csr();
        let m = make_ones_trimat(D).to_csc::<usize>();
        b.iter(|| v.mul(&m));
    }

    #[bench]
    fn onehot_csc_mul_csc(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let v = make_sprs_onehot(0, D).to_csc();
        let m = make_ones_trimat(D).to_csc::<usize>();
        b.iter(|| v.mul(&m));
    }

    #[bench]
    fn csc_mul_csc(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let ma = make_ones_trimat(D).to_csc::<usize>();
        let mb = ma.clone();
        b.iter(|| ma.mul(&mb));
    }

    #[bench]
    fn csr_mul_csr(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let ma = make_ones_trimat(D).to_csr::<usize>();
        let mb = ma.clone();
        b.iter(|| ma.mul(&mb));
    }

    #[bench]
    fn csr_mul_csc(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let ma = make_ones_trimat(D).to_csr::<usize>();
        let mb = make_ones_trimat(D).to_csc::<usize>();
        b.iter(|| ma.mul(&mb));
    }

    #[bench]
    fn csc_mul_csr(b: &mut Bencher) {
        let qubits = 8;
        let D = 1 << qubits;
        let ma = make_ones_trimat(D).to_csc::<usize>();
        let mb = make_ones_trimat(D).to_csr::<usize>();
        b.iter(|| ma.mul(&mb));
    }
}
