#![allow(incomplete_features)]

use num_traits::{Zero, One, Float};
use std::ops::Range;

//#region FloatNum
pub trait FloatNum: 
    Float + Zero + One 
    + std::iter::Sum<Self>
    + std::iter::Product<Self>
    + std::ops::AddAssign + std::ops::SubAssign 
    + std::ops::MulAssign + std::ops::DivAssign
    + std::fmt::Debug
{
    const ZERO: Self;
    const ONE: Self;
}

impl FloatNum for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl FloatNum for f32 {
    const ZERO: Self = 0.0_f32;
    const ONE: Self = 1.0_f32;
}
//#endregion


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TMat<T: FloatNum, const R: usize, const C: usize> {
    pub cols: [[T; R]; C],
}

pub type TMat00<T> = TMat<T, 0, 0>;
pub type TMat11<T> = TMat<T, 1, 1>;
pub type TMat22<T> = TMat<T, 2, 2>;
pub type TMat33<T> = TMat<T, 3, 3>;
pub type TMat44<T> = TMat<T, 4, 4>;
pub type TVec<T, const N: usize> = TMat<T, N, 1>;
pub type TVec2<T> = TVec<T, 2>;
pub type TVec3<T> = TVec<T, 3>;
pub type TVec4<T> = TVec<T, 4>;

impl<T: FloatNum, const R: usize, const C: usize> TMat<T, R, C> {
    pub const ZEROS: Self = TMat { cols: [[T::ZERO; R]; C] };
    pub const ONES: Self = TMat { cols: [[T::ONE; R]; C] };
    pub const COLS: usize = C;
    pub const ROWS: usize = R;
}

/// 不定个参数构造向量，例如：`mvec!(1.0, 2.0)`、`mvec!(1.0, 2.0, 3.0)`。
#[macro_export]
macro_rules! mvec {
    ($($x:expr),* $(,)?) => {
        $crate::math::TVec { cols: [[$($x),*]] }
    };
}

impl<T: FloatNum, const N: usize> TVec<T, N> {
    pub fn new(data: [T; N]) -> Self {
        TMat::<T, N, 1> { cols: [data] }
    }
    pub fn norm(self) -> T {
        (self.cols[0].iter().map(|x: &T| *x * *x).sum::<T>()).sqrt()
    }
    pub fn normalize(self) -> Self {
        self * (T::ONE / self.norm())
    }
    #[inline(always)]
    pub fn x(&self) -> T {
        self.cols[0][0]
    }
    #[inline(always)]
    pub fn y(&self) -> T {
        self.cols[0][1]
    }
    #[inline(always)]
    pub fn z(&self) -> T {
        self.cols[0][2]
    }
    #[inline(always)]
    pub fn w(&self) -> T {
        self.cols[0][3]
    }
    #[inline(always)]
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.cols[0][0]
    }
    #[inline(always)]
    pub fn y_mut(&mut self) -> &mut T {
        &mut self.cols[0][1]
    }
    #[inline(always)]
    pub fn z_mut(&mut self) -> &mut T {
        &mut self.cols[0][2]
    }
    #[inline(always)]
    pub fn w_mut(&mut self) -> &mut T {
        &mut self.cols[0][3]
    }

    pub fn unit_x() -> Self {
        let mut data = [T::ZERO; N];
        data[0] = T::ONE;
        TVec::<T, N>::new(data)
    }

    pub fn unit_y() -> Self {
        let mut data = [T::ZERO; N];
        data[1] = T::ONE;
        TVec::<T, N>::new(data)
    }

    pub fn unit_z() -> Self {
        let mut data = [T::ZERO; N];
        data[2] = T::ONE;
        TVec::<T, N>::new(data)
    }

    pub fn unit_w() -> Self {
        let mut data = [T::ZERO; N];
        data[3] = T::ONE;
        TVec::<T, N>::new(data)
    }
}

impl<T: FloatNum> TVec2<T> {
    pub fn rotate(self, angle: T) -> Self {
        let sin = angle.sin();
        let cos = angle.cos();
        TVec2::<T>::new(
            [self.x() * cos - self.y() * sin, self.x() * sin + self.y() * cos]
        )
    }

    pub fn angle(self) -> T {
        let normalized = self.normalize();
        normalized.y().atan2(normalized.x())
    }
}

impl<T: FloatNum, const R: usize, const C: usize> std::ops::Add for TMat<T, R, C> {
    type Output = TMat<T, R, C>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut mat = TMat::<T, R, C>::ZEROS;
        for i in 0..R {
            for j in 0..C {
                mat.cols[j][i] = self.cols[j][i] + rhs.cols[j][i];
            }
        }
        mat
    }
}

impl<T: FloatNum, const R: usize, const C: usize> std::ops::AddAssign for TMat<T, R, C> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..R {
            for j in 0..C {
                self.cols[j][i] += rhs.cols[j][i];
            }
        }
    }
}

impl<T: FloatNum, const R: usize, const C: usize> std::ops::Sub for TMat<T, R, C> {
    type Output = TMat<T, R, C>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut mat = TMat::<T, R, C>::ZEROS;
        for i in 0..R {
            for j in 0..C {
                *mat.v_mut(i, j) = self.v(i, j) - rhs.v(i, j);
            }
        }
        mat
    }
}

impl<T: FloatNum, const R: usize, const C: usize> std::ops::SubAssign for TMat<T, R, C> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..R {
            for j in 0..C {
                self.cols[j][i] -= rhs.cols[j][i];
            }
        }
    }
}

impl<T: FloatNum, const R1: usize, const C1: usize, const N: usize> std::ops::Mul<TMat<T, C1, N>> for TMat<T, R1, C1> {
    type Output = TMat<T, R1, N>;
    fn mul(self, rhs: TMat<T, C1, N>) -> Self::Output {
        let mut mat = TMat::<T, R1, N>::ZEROS;
        for i in 0..R1 {
            for j in 0..N {
                mat.cols[j][i] = self.row(i).dot(rhs.col(j));
            }
        }
        mat
    }
}

impl<T: FloatNum, const R: usize, const C: usize> std::ops::Mul<T> for TMat<T, R, C> {
    type Output = TMat<T, R, C>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut mat = TMat::<T, R, C>::ZEROS;
        for i in 0..R {
            for j in 0..C {
                mat.cols[j][i] = self.cols[j][i] * rhs;
            }
        }
        mat
    }
}

impl<T: FloatNum, const R: usize, const C: usize> std::ops::MulAssign<T> for TMat<T, R, C> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..R {
            for j in 0..C {
                self.cols[j][i] *= rhs;
            }
        }
    }
}

impl<T: FloatNum, const R: usize, const C: usize> std::ops::Neg for TMat<T, R, C> {
    type Output = TMat<T, R, C>;
    fn neg(self) -> Self::Output {
        let mut mat = TMat::<T, R, C>::ZEROS;
        for i in 0..R {
            for j in 0..C {
                mat.cols[j][i] = -self.cols[j][i];
            }
        }
        mat
    }
}

pub trait Dot<RHS> {
    type Output;
    fn dot(self, rhs: RHS) -> Self::Output;
}

impl<T: FloatNum, const D: usize> Dot<TVec<T, D>> for TVec<T, D> {
    type Output = T;
    fn dot(self, rhs: TVec<T, D>) -> Self::Output {
        self.cols[0].iter().zip(rhs.cols[0].iter()).map(|(a, b)| *a * *b).sum()
    }
}

pub trait Cross<RHS> {
    type Output;
    fn cross(self, rhs: RHS) -> Self::Output;
}

impl<T: FloatNum> Cross<TVec<T, 2>> for TVec<T, 2> {
    type Output = T;
    fn cross(self, rhs: TVec<T, 2>) -> Self::Output {
        self.x() * rhs.y() - self.y() * rhs.x()
    }
}

impl<T: FloatNum> Cross<T> for TVec<T, 2> {
    type Output = TVec<T, 2>;
    fn cross(self, rhs: T) -> Self::Output {
        TVec::<T, 2>::new([self.y() * rhs, -self.x() * rhs])
    }
}

impl<T: FloatNum> Cross<TVec<T, 2>> for T {
    type Output = TVec<T, 2>;
    fn cross(self, rhs: TVec<T, 2>) -> Self::Output {
        TVec::<T, 2>::new([-self * rhs.y(), self * rhs.x()])
    }
}
//#endregion


impl<T: FloatNum, const R: usize, const C: usize> TMat<T, R, C> {
    pub fn from_cols(cols: [[T; R]; C]) -> Self {
        TMat { cols: cols }
    }

    pub fn from_rows(rows: [[T; C]; R]) -> Self {
        let mut cols = [[T::ZERO; R]; C];
        for i in 0..R {
            for j in 0..C {
                cols[j][i] = rows[i][j];
            }
        }
        TMat::from_cols(cols)
    }

    pub fn zeros() -> Self {
        TMat::ZEROS
    }

    pub fn row(&self, i: usize) -> TVec<T, C> {
        TVec::<T, C>::new(std::array::from_fn(|j| self.cols[j][i]))
    }

    pub fn col(&self, j: usize) -> TVec<T, R> {
        TVec::<T, R>::new(self.cols[j])
    }

    pub fn T(&self) -> TMat<T, C, R> {
        let rows = self.cols;
        TMat::<T, C, R>::from_rows(rows)
    }

    pub fn v(&self, i: usize, j: usize) -> T {
        self.cols[j][i]
    }

    pub fn v_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.cols[j][i]
    }
}

impl<T: FloatNum, const R: usize, const C1: usize> TMat<T, R, C1> {
    pub fn h_con<const C2: usize>(self, rhs: TMat<T, R, C2>) -> TMat<T, R, {C1 + C2}> {
        let cols: [[T; R]; {C1 + C2}] = std::array::from_fn(|j| {
            if j < C1 {
                self.cols[j]
            } else {
                rhs.cols[j - C1]
            }
        });
        TMat::<T, R, {C1 + C2}>::from_cols(cols)
    }
}


impl<T: FloatNum, const R1: usize, const C: usize> TMat<T, R1, C> {
    pub fn v_con<const R2: usize>(self, rhs: TMat<T, R2, C>) -> TMat<T, {R1 + R2}, C> {
        let cols: [[T; {R1 + R2}]; C] = std::array::from_fn(|i| {
            arr_concat(self.cols[i], rhs.cols[i])
        });
        TMat::<T, {R1 + R2}, C>::from_cols(cols)
    }
}

fn arr_concat<T: FloatNum, const N1: usize, const N2: usize>(arr1: [T; N1], arr2: [T; N2]) -> [T; {N1+N2}] {
    let mut arr = [T::ZERO; {N1 + N2}];
    arr[..N1].copy_from_slice(&arr1);
    arr[N1..].copy_from_slice(&arr2);
    arr
}

pub trait AsMat<T: FloatNum, const R: usize, const C: usize> {
    fn as_mat(&self) -> TMat<T, R, C>;
}

impl<T: FloatNum> AsMat<T, 1, 1> for T {
    fn as_mat(&self) -> TMat<T, 1, 1> {
        TMat::<T, 1, 1>::from_cols([[*self]])
    }
}

impl<T: FloatNum, const R: usize, const C: usize> AsMat<T, R, C> for TMat<T, R, C> {
    fn as_mat(&self) -> TMat<T, R, C> {
        *self
    }
}

pub trait AsFloat<T: FloatNum> {
    fn as_float(&self) -> T;
}


impl<T: FloatNum> AsFloat<T> for TMat<T, 1, 1> {
    fn as_float(&self) -> T {
        self.cols[0][0]
    }
}

#[macro_export]
macro_rules! h_concat {
    ($mat:expr) => { $mat };
    
    ($first:expr, $second:expr) => {{ $first.as_mat().h_con($second.as_mat()) }};
    
    ($first:expr, $second:expr, $($rest:expr),+) => {{
        let merged = $first.as_mat().h_con($second.as_mat());
        h_concat!(merged, $($rest),+)
    }};
}

#[macro_export]
macro_rules! v_concat {
    ($mat:expr) => { $mat };
    
    ($first:expr, $second:expr) => {{ $first.as_mat().v_con($second.as_mat()) }};
    
    ($first:expr, $second:expr, $($rest:expr),+) => {{ 
        let merged = $first.as_mat().v_con($second.as_mat());
        v_concat!(merged, $($rest),+)
    }};
}


impl<T: FloatNum, const N: usize> TMat<T, N, N> {
    pub fn eye() -> Self {
        let mut cols = [[T::ZERO; N]; N];
        for j in 0..N {
            cols[j][j] = T::ONE;
        }
        TMat::<T, N, N>::from_cols(cols)
    }

    pub fn diag(v: [T; N]) -> Self {
        let mut cols = [[T::ZERO; N]; N];
        for j in 0..N {
            cols[j][j] = v[j];
        }
        TMat::<T, N, N>::from_cols(cols)
    }
}

impl<T: FloatNum, const N1: usize> TMat<T, N1, N1> {
    pub fn d_con<const N2: usize>(self, rhs: TMat<T, N2, N2>) -> TMat<T, {N1 + N2}, {N1 + N2}> {
        let cols: [[T; {N1 + N2}]; {N1 + N2}] = std::array::from_fn(|j| {
            if j < N1 {
                arr_concat(self.cols[j], [T::ZERO; N2])
            } else {
                arr_concat([T::ZERO; N1], rhs.cols[j - N1])
            }
        });
        TMat::<T, {N1 + N2}, {N1 + N2}>::from_cols(cols)
    }
}

#[macro_export]
macro_rules! d_concat {
    ($mat:expr) => { $mat };
    
    ($first:expr, $second:expr) => {{ $first.as_mat().d_con($second.as_mat()) }};

    ($first:expr, $second:expr, $($rest:expr),+) => {{
        let merged = $first.as_mat().d_con($second.as_mat());
        d_concat!(merged, $($rest),+)
    }};
}

impl<T: FloatNum, const R: usize, const C: usize> TMat<T, R, C> {
    pub fn slice<const R1: usize, const C1: usize>(
        &self,
        r: usize,
        c: usize,
    ) -> TMat<T, R1, C1> {
        assert!(r + R1 <= R && c + C1 <= C, "slice out of bounds");

        let mut mat = TMat::<T, R1, C1>::ZEROS;
        for i in r..(r+R1) {
            for j in c..(c+C1) {
                *mat.v_mut(i, j) = self.v(i, j);
            }
        }
        mat
    }

    pub fn h_slice<const C1: usize>(&self, r: usize, c: usize) -> TMat<T, 1, C1> {
        self.slice::<1, C1>(r, c)
    }

    pub fn v_slice<const R1: usize>(&self, r: usize, c: usize) -> TMat<T, R1, 1> {
        self.slice::<R1, 1>(r, c)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TDynMat<T: FloatNum> {
    pub cols: Vec<Vec<T>>,
    pub r: usize,
    pub c: usize,
}

impl<T: FloatNum> TDynMat<T> {

    pub fn from_mat<const R: usize, const C: usize>(mat: TMat<T, R, C>) -> Self {
        TDynMat { 
            cols: mat.cols.iter().map(|row| row.to_vec()).collect(), 
            r: R, 
            c: C }
    }
    
    pub fn zeros(r: usize, c: usize) -> Self {
        TDynMat { cols: vec![vec![T::ZERO; r]; c], r, c }
    }

    pub fn v(&self, i: usize, j: usize) -> T {
        self.cols[j][i]
    }

    pub fn v_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.cols[j][i]
    }

    pub fn n_rows(&self) -> usize {
        self.r
    }

    pub fn n_cols(&self) -> usize {
        self.c
    }

    pub fn row(&self, i: usize) -> TDynMat<T> {
        TDynMat { cols: self.cols.iter().map(|row| vec![row[i]]).collect(), r: self.n_rows(), c: 1 }
    }

    pub fn col(&self, i: usize) -> TDynMat<T> {
        TDynMat { cols: vec![self.cols[i].clone()], r: 1, c: self.n_cols() }
    }

    pub fn T(&self) -> TDynMat<T> {
        let mut mat = TDynMat::<T>::zeros(self.n_cols(), self.n_rows());
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *mat.v_mut(j, i) = self.v(i, j);
            }
        }
        mat
    }

    pub fn slice(
        &self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> TDynMat<T> {
        let r = rows.len();
        let c = cols.len();
        TDynMat {
            cols: if r == 0 || c == 0 {
                vec![]
            } else {
                self.cols[cols].iter().map(|row| row[rows.clone()].to_vec()).collect()
            },
            r,
            c,
        }
    }

    pub fn h_slice(&self, rows: usize, cols: Range<usize>) -> TDynMat<T> {
        self.slice(rows..rows+1, cols)
    }

    pub fn v_slice(&self, rows: Range<usize>, cols: usize) -> TDynMat<T> {
        self.slice(rows, cols..cols+1)
    }

    pub fn norm(&self) -> T {
        assert_eq!(self.n_cols(), 1, "matrix is not vector");
        let mut norm = T::ZERO;
        for i in 0..self.n_rows() {
            norm += self.v(i, 0) * self.v(i, 0);
        }
        norm.sqrt()
    }
}


impl<T: FloatNum> std::ops::Add<&TDynMat<T>> for &TDynMat<T> {
    type Output = TDynMat<T>;
    fn add(self, rhs: &TDynMat<T>) -> Self::Output {
        assert_eq!(self.n_rows(), rhs.n_rows(), "matrix dimensions do not match");
        assert_eq!(self.n_cols(), rhs.n_cols(), "matrix dimensions do not match");
        
        let mut mat = TDynMat::<T>::zeros(self.n_rows(), self.n_cols());
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *mat.v_mut(i, j) = self.v(i, j) + rhs.v(i, j);
            }
        }
        mat
    }
}

impl<T: FloatNum> std::ops::Add<TDynMat<T>> for TDynMat<T> {
    type Output = TDynMat<T>;
    fn add(self, rhs: TDynMat<T>) -> Self::Output {
        (&self) - (&rhs)
    }
}

impl<T: FloatNum> std::ops::Add<&TDynMat<T>> for TDynMat<T> {
    type Output = TDynMat<T>;
    fn add(self, rhs: &TDynMat<T>) -> Self::Output {
        (&self) - rhs
    }
}

impl<T: FloatNum> std::ops::Add<TDynMat<T>> for &TDynMat<T> {
    type Output = TDynMat<T>;
    fn add(self, rhs: TDynMat<T>) -> Self::Output {
        self - (&rhs)
    }
}

impl<T: FloatNum> std::ops::AddAssign for TDynMat<T> {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.n_rows(), rhs.n_rows(), "matrix dimensions do not match");
        assert_eq!(self.n_cols(), rhs.n_cols(), "matrix dimensions do not match");
        
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *self.v_mut(i, j) += rhs.v(i, j);
            }
        }
    }
}

impl<T: FloatNum> std::ops::Sub<&TDynMat<T>> for &TDynMat<T> {
    type Output = TDynMat<T>;
    fn sub(self, rhs: &TDynMat<T>) -> Self::Output {
        assert_eq!(self.n_rows(), rhs.n_rows(), "matrix dimensions do not match");
        assert_eq!(self.n_cols(), rhs.n_cols(), "matrix dimensions do not match");
        
        let mut mat = TDynMat::<T>::zeros(self.n_rows(), self.n_cols());
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *mat.v_mut(i, j) = self.v(i, j) - rhs.v(i, j);
            }
        }
        mat
    }
}

impl<T: FloatNum> std::ops::Sub<TDynMat<T>> for TDynMat<T> {
    type Output = TDynMat<T>;
    fn sub(self, rhs: TDynMat<T>) -> Self::Output {
        (&self) - (&rhs)
    }
}

impl<T: FloatNum> std::ops::Sub<&TDynMat<T>> for TDynMat<T> {
    type Output = TDynMat<T>;
    fn sub(self, rhs: &TDynMat<T>) -> Self::Output {
        (&self) - rhs
    }
}

impl<T: FloatNum> std::ops::Sub<TDynMat<T>> for &TDynMat<T> {
    type Output = TDynMat<T>;
    fn sub(self, rhs: TDynMat<T>) -> Self::Output {
        self - (&rhs)
    }
}

impl<T: FloatNum> std::ops::SubAssign for TDynMat<T> {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.n_rows(), rhs.n_rows(), "matrix dimensions do not match");
        assert_eq!(self.n_cols(), rhs.n_cols(), "matrix dimensions do not match");
        
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *self.v_mut(i, j) -= rhs.v(i, j);
            }
        }
    }
}

impl<T: FloatNum> std::ops::Mul<&TDynMat<T>> for &TDynMat<T> {
    type Output = TDynMat<T>;
    fn mul(self, rhs: &TDynMat<T>) -> Self::Output {
        assert_eq!(self.n_cols(), rhs.n_rows(), "matrix dimensions do not match");
        
        let mut mat = TDynMat::<T>::zeros(self.n_rows(), rhs.n_cols());
        for i in 0..self.n_rows() {
            for j in 0..rhs.n_cols() {
                let mut d = T::ZERO;
                for k in 0..self.n_cols() {
                    d += self.v(i, k) * rhs.v(k, j);
                }
                *mat.v_mut(i, j) = d;
            }
        }
        mat
    }
}

impl<T: FloatNum> std::ops::Mul<TDynMat<T>> for TDynMat<T> {
    type Output = TDynMat<T>;
    fn mul(self, rhs: TDynMat<T>) -> Self::Output {
        (&self) * (&rhs)
    }
}

impl<T: FloatNum> std::ops::Mul<&TDynMat<T>> for TDynMat<T> {
    type Output = TDynMat<T>;
    fn mul(self, rhs: &TDynMat<T>) -> Self::Output {
        (&self) * rhs
    }
}

impl<T: FloatNum> std::ops::Mul<TDynMat<T>> for &TDynMat<T> {
    type Output = TDynMat<T>;
    fn mul(self, rhs: TDynMat<T>) -> Self::Output {
        self * (&rhs)
    }
}

impl<T: FloatNum> std::ops::Mul<T> for TDynMat<T> {
    type Output = TDynMat<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut mat = TDynMat::<T>::zeros(self.n_rows(), self.n_cols());
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *mat.v_mut(i, j) = self.v(i, j) * rhs;
            }
        }
        mat
    }
}

impl<T: FloatNum> std::ops::MulAssign<T> for TDynMat<T> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *self.v_mut(i, j) *= rhs;
            }
        }
    }
}


impl<T: FloatNum> std::ops::Neg for &TDynMat<T> {
    type Output = TDynMat<T>;
    fn neg(self) -> Self::Output {
        let mut mat = TDynMat::<T>::zeros(self.n_rows(), self.n_cols());
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                *mat.v_mut(i, j) = -self.v(i, j);
            }
        }
        mat
    }
}


impl<T: FloatNum> std::ops::Neg for TDynMat<T> {
    type Output = TDynMat<T>;
    fn neg(self) -> Self::Output {
        -(&self)
    }
}

impl<T: FloatNum> AsFloat<T> for TDynMat<T> {
    fn as_float(&self) -> T {
        assert_eq!(self.n_rows(), 1, "matrix is not 1x1");
        assert_eq!(self.n_cols(), 1, "matrix is not 1x1");
        self.v(0, 0)
    }
}



// --- TTransform2d ---
#[derive(Clone, Copy, Debug)]
pub struct TTransform2d<T: FloatNum> {
    pub origin: TVec2<T>,
    pub angle: T,
}

impl<T: FloatNum> TTransform2d<T> {
    pub const IDENTITY: Self = TTransform2d { origin: TVec2::ZEROS, angle: T::ZERO };

    pub fn new(origin: TVec2<T>, angle: T) -> Self {
        TTransform2d::<T> { origin, angle }
    }

    pub fn inv(&self) -> Self {
        TTransform2d::<T> {
            origin: -self.origin.rotate(-self.angle),
            angle: -self.angle,
        }
    }

    pub fn transform_vector(&self, v: TVec2<T>) -> TVec2<T> {
        v.rotate(self.angle)
    }

    pub fn transform_position(&self, p: TVec2<T>) -> TVec2<T> {
        p.rotate(self.angle) + self.origin
    }

    pub fn as_mat33(&self) -> TMat33<T> {
        let c = self.angle.cos();
        let s = self.angle.sin();
        TMat33::from_rows([
            [c, -s, T::ZERO],
            [s, c, T::ZERO],
            [T::ZERO, T::ZERO, T::ONE],
        ])
    }
}

impl<T: FloatNum> std::ops::Mul<TTransform2d<T>> for TTransform2d<T> {
    type Output = TTransform2d::<T>;
    fn mul(self, rhs: TTransform2d<T>) -> Self::Output {
        TTransform2d::<T> {
            origin: rhs.origin.rotate(self.angle) + self.origin,
            angle: self.angle + rhs.angle,
        }
    }
}


// #[macro_export]
// macro_rules! slice {
//     ($mat:expr, $rows:expr, $cols:expr) => {{
//         let rows = $rows;
//         let cols = $cols;
//         match rows, cols {
//             (usize, std::ops::Range<usize>) => {
//                 $mat.h_slice(rows, cols)
//             }
//             (std::ops::Range<usize>, usize) => {
//                 $mat.v_slice(cols, rows)
//             }
//             (std::ops::Range<usize>, std::ops::Range<usize>) => {
//                 $mat.slice(rows, cols)
//             }
//             _ => {
//                 panic!("slice: invalid range");
//             }
//         }
//     }};

//     // 向量切片: vec.slice!(1..3)
//     ($vec:expr, $rows:expr) => {{
//         let rows = $rows;
//         $vec.v_slice(rows, 0)
//     }};
// }


//#region
pub type Vec2 = TVec2<f64>;
pub type Vec3 = TVec3<f64>;
pub type Vec4 = TVec4<f64>;
pub type Mat11 = TMat11<f64>;
pub type Mat22 = TMat22<f64>;
pub type Mat33 = TMat33<f64>;
pub type Mat44 = TMat44<f64>;
pub type Transform2d = TTransform2d<f64>;
//#endregion

pub fn solve_gauss_seidel<T: FloatNum, const R: usize, const C: usize>(
    A: TMat<T, R, C>,
    b: TVec<T, R>,
    max_iterations: usize,
    tolerance: T,
) -> TVec<T, C> {

    let A_dyn = TDynMat::from_mat(A);
    let b_dyn = TDynMat::from_mat(b);

    let x_dyn = solve_gauss_seidel_dyn(A_dyn, b_dyn, max_iterations, tolerance);
    assert_eq!(x_dyn.n_rows(), C);

    let mut x = TVec::<T, C>::zeros();
    for i in 0..C {
        *x.v_mut(i, 0) = x_dyn.v(i, 0);
    }
    x
}


pub fn solve_gauss_seidel_dyn<T: FloatNum>(
    A: TDynMat<T>,
    b: TDynMat<T>,
    max_iterations: usize,
    tolerance: T,
) -> TDynMat<T> {

    let r = A.n_rows();
    let c = A.n_cols();
    assert_eq!(A.n_rows(), b.n_rows(), "A and b have different number of rows");

    // A = D + L + U
    // x_next = (D + L)^{-1} @ (-U @ x_prev + b)

    let mut x = TDynMat::<T>::zeros(c, 1);
    let mut iterations = 0;
    while iterations < max_iterations {
        let mut x_new = TDynMat::<T>::zeros(c, 1);
        for i in 0..r {
            *x_new.v_mut(i, 0) = (b.v(i, 0) 
                - (A.h_slice(i, 0..i) * x_new.v_slice(0..i, 0)).as_float()
                - (A.h_slice(i, i+1..c) * x.v_slice(i+1..c, 0)).as_float()) / A.v(i, i);
        }
        x = x_new;
        let err = ((A.clone() * x.clone()) - b.clone()).norm();
        if err < tolerance {
            break;
        }
        iterations += 1;
    }
    x
}


#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec2_approx_eq(a: TVec2<f64>, b: TVec2<f64>) -> bool {
        approx_eq(a.x(), b.x()) && approx_eq(a.y(), b.y())
    }

    // --- FloatNum ---
    #[test]
    fn float_num_f64_constants() {
        assert_eq!(f64::ZERO, 0.0);
        assert_eq!(f64::ONE, 1.0);
    }

    #[test]
    fn float_num_f32_constants() {
        assert_eq!(f32::ZERO, 0.0_f32);
        assert_eq!(f32::ONE, 1.0_f32);
    }

    // --- TVec2 constructors & ZERO ---
    #[test]
    fn vec2_new_and_zero() {
        let v = mvec!(3.0, 4.0);
        assert_eq!(v.x(), 3.0);
        assert_eq!(v.y(), 4.0);
        assert_eq!(TVec2::<f64>::ZEROS.x(), 0.0);
        assert_eq!(TVec2::<f64>::ZEROS.y(), 0.0);
    }

    // --- TVec2::norm ---
    #[test]
    fn vec2_norm() {
        assert!(approx_eq(mvec!(3.0, 4.0).norm(), 5.0));
        assert!(approx_eq(mvec!(0.0, 0.0).norm(), 0.0));
        assert!(approx_eq(mvec!(1.0, 0.0).norm(), 1.0));
    }

    // --- TVec2::rotate ---
    #[test]
    fn vec2_rotate_90() {
        let v = mvec!(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!(vec2_approx_eq(r, mvec!(0.0, 1.0)));
    }

    #[test]
    fn vec2_rotate_identity() {
        let v = mvec!(1.0, 2.0);
        let r = v.rotate(0.0);
        assert!(vec2_approx_eq(r, v));
    }

    // --- Add / Sub / Neg ---
    #[test]
    fn vec2_add_sub_neg() {
        let a = mvec!(1.0, 2.0);
        let b = mvec!(3.0, 4.0);
        assert!(vec2_approx_eq(a + b, mvec!(4.0, 6.0)));
        assert!(vec2_approx_eq(b - a, mvec!(2.0, 2.0)));
        assert!(vec2_approx_eq(-a, mvec!(-1.0, -2.0)));
    }

    // --- AddAssign / SubAssign / MulAssign ---
    #[test]
    fn vec2_assign_ops() {
        let mut v = mvec!(1.0, 2.0);
        v += mvec!(1.0, 1.0);
        assert!(vec2_approx_eq(v, mvec!(2.0, 3.0)));
        v -= mvec!(0.0, 1.0);
        assert!(vec2_approx_eq(v, mvec!(2.0, 2.0)));
        v *= 2.0;
        assert!(vec2_approx_eq(v, mvec!(4.0, 4.0)));
    }

    // --- Mul scalar ---
    #[test]
    fn vec2_mul_scalar() {
        let v = mvec!(1.0, 2.0);
        assert!(vec2_approx_eq(v * 3.0, mvec!(3.0, 6.0)));
    }

    // --- Dot ---
    #[test]
    fn vec2_dot() {
        let a = mvec!(1.0, 0.0);
        let b = mvec!(1.0, 0.0);
        assert!(approx_eq(a.dot(b), 1.0));
        let c = mvec!(3.0, 4.0);
        assert!(approx_eq(a.dot(c), 3.0));
        assert!(approx_eq(c.dot(c), 25.0));
    }

    // --- Cross (TVec2 x TVec2 -> scalar) ---
    #[test]
    fn vec2_cross_vec2() {
        let a = mvec!(1.0, 0.0);
        let b = mvec!(0.0, 1.0);
        assert!(approx_eq(a.cross(b), 1.0));
        assert!(approx_eq(b.cross(a), -1.0));
    }

    // --- Cross (TVec2 x T -> TVec2) ---
    #[test]
    fn vec2_cross_scalar() {
        let v = mvec!(1.0, 0.0);
        let r = v.cross(2.0);
        assert!(vec2_approx_eq(r, mvec!(0.0, -2.0)));
    }

    // --- Cross (T x TVec2 -> TVec2) ---
    #[test]
    fn scalar_cross_vec2() {
        let v = mvec!(1.0, 0.0);
        let r = 2.0_f64.cross(v);
        assert!(vec2_approx_eq(r, mvec!(0.0, 2.0)));
    }

    // --- TTransform2d ---
    fn transform2d_approx_eq(a: TTransform2d<f64>, b: TTransform2d<f64>) -> bool {
        vec2_approx_eq(a.origin, b.origin) && approx_eq(a.angle, b.angle)
    }

    fn transform2d_identity() -> TTransform2d<f64> {
        TTransform2d::new(TVec2::ZEROS, 0.0)
    }

    #[test]
    fn transform2d_new() {
        let o = mvec!(1.0, 2.0);
        let t = TTransform2d::new(o, std::f64::consts::FRAC_PI_2);
        assert!(vec2_approx_eq(t.origin, o));
        assert!(approx_eq(t.angle, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn transform2d_identity_mul_left() {
        let t = TTransform2d::new(mvec!(3.0, 4.0), 0.5);
        let id = transform2d_identity();
        assert!(transform2d_approx_eq(id * t, t));
    }

    #[test]
    fn transform2d_identity_mul_right() {
        let t = TTransform2d::new(mvec!(3.0, 4.0), 0.5);
        let id = transform2d_identity();
        assert!(transform2d_approx_eq(t * id, t));
    }

    #[test]
    fn transform2d_invert_roundtrip() {
        let t = TTransform2d::new(mvec!(1.0, 2.0), 0.7);
        let inv = t.inv();
        assert!(transform2d_approx_eq(t * inv, transform2d_identity()));
        assert!(transform2d_approx_eq(inv * t, transform2d_identity()));
    }

    #[test]
    fn transform2d_invert_twice() {
        let t = TTransform2d::new(mvec!(-1.0, 3.0), std::f64::consts::PI);
        assert!(transform2d_approx_eq(t.inv().inv(), t));
    }

    #[test]
    fn transform2d_mul_composition() {
        let a = TTransform2d::new(mvec!(1.0, 0.0), 0.0);
        let b = TTransform2d::new(mvec!(0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let ab = a * b;
        // a * b: first apply b (translate (0,1) then rotate 90°), then a (translate (1,0))
        // ab.origin = b.origin.rotate(a.angle) + a.origin = (0,1).rotate(0) + (1,0) = (1,1)
        assert!(vec2_approx_eq(ab.origin, mvec!(1.0, 1.0)));
        assert!(approx_eq(ab.angle, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn transform2d_mul_associativity() {
        let a = TTransform2d::new(mvec!(1.0, 0.0), 0.3);
        let b = TTransform2d::new(mvec!(0.0, 1.0), 0.5);
        let c = TTransform2d::new(mvec!(2.0, -1.0), -0.2);
        assert!(transform2d_approx_eq((a * b) * c, a * (b * c)));
    }

    // --- h_concat / v_concat ---
    fn mat2x2_approx_eq(a: TMat22<f64>, b: TMat22<f64>) -> bool {
        for j in 0..2 {
            for i in 0..2 {
                if !approx_eq(a.cols[j][i], b.cols[j][i]) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn h_concat_two_vec2_to_mat2x2() {
        // 两个 2x1 列向量水平拼接成 2x2
        let a = mvec!(1.0, 2.0);  // 2x1
        let b = mvec!(3.0, 4.0);  // 2x1
        let c = a.h_con(b);
        let expected = TMat22::from_cols([[1.0, 2.0], [3.0, 4.0]]);
        assert!(mat2x2_approx_eq(c, expected));
    }

    #[test]
    fn h_concat_2x2_plus_2x1() {
        // 2x2 与 2x1 水平拼接成 2x3
        let a = TMat::from_cols([[1.0, 2.0], [3.0, 4.0]]);  // 2x2
        let b = mvec!(5.0, 6.0);  // 2x1
        let c: TMat<f64, 2, 3> = a.h_con(b);
        assert!(approx_eq(c.cols[0][0], 1.0) && approx_eq(c.cols[0][1], 2.0));
        assert!(approx_eq(c.cols[1][0], 3.0) && approx_eq(c.cols[1][1], 4.0));
        assert!(approx_eq(c.cols[2][0], 5.0) && approx_eq(c.cols[2][1], 6.0));
    }

    #[test]
    fn v_concat_two_row_vectors() {
        // 两个 1x2 行向量垂直拼接成 2x2
        let a = TMat::from_cols([[1.0], [2.0]]);  // 1x2 即一行 [1, 2]
        let b = TMat::from_cols([[3.0], [4.0]]);  // 1x2 即一行 [3, 4]
        let c: TMat22<f64> = a.v_con(b);
        let expected = TMat22::from_cols([[1.0, 3.0], [2.0, 4.0]]);
        assert!(mat2x2_approx_eq(c, expected));
    }

    #[test]
    fn v_concat_2x2_plus_1x2() {
        // 2x2 与 1x2 垂直拼接成 3x2
        let a = TMat::from_cols([[1.0, 2.0], [3.0, 4.0]]);  // 2x2
        let b = TMat::from_cols([[5.0], [6.0]]);  // 1x2
        let c: TMat<f64, 3, 2> = a.v_con(b);
        assert!(approx_eq(c.cols[0][0], 1.0) && approx_eq(c.cols[0][1], 2.0) && approx_eq(c.cols[0][2], 5.0));
        assert!(approx_eq(c.cols[1][0], 3.0) && approx_eq(c.cols[1][1], 4.0) && approx_eq(c.cols[1][2], 6.0));
    }

    #[test]
    fn h_concat_macro_two_matrices() {
        let a = mvec!(1.0, 0.0);
        let b = mvec!(0.0, 1.0);
        let m: TMat22<f64> = a.h_con(b);
        assert!(mat2x2_approx_eq(m, TMat22::from_cols([[1.0, 0.0], [0.0, 1.0]])));
    }

    #[test]
    fn h_concat_macro_three_matrices() {
        let a = mvec!(1.0, 0.0);
        let b = mvec!(0.0, 1.0);
        let c = mvec!(-1.0, -1.0);
        let m: TMat<f64, 2, 3> = {
            let temp: TMat22<f64> = a.h_con(b);
            temp.h_con(c)
        };
        assert!(approx_eq(m.cols[0][0], 1.0) && approx_eq(m.cols[0][1], 0.0));
        assert!(approx_eq(m.cols[1][0], 0.0) && approx_eq(m.cols[1][1], 1.0));
        assert!(approx_eq(m.cols[2][0], -1.0) && approx_eq(m.cols[2][1], -1.0));
    }

    #[test]
    fn v_concat_macro_two_matrices() {
        let a = TMat::from_cols([[1.0], [0.0]]);  // 1x2
        let b = TMat::from_cols([[0.0], [1.0]]);
        let m: TMat22<f64> = a.v_con(b);
        assert!(mat2x2_approx_eq(m, TMat22::from_cols([[1.0, 0.0], [0.0, 1.0]])));
    }

    #[test]
    fn v_concat_macro_three_matrices() {
        let a = TMat::from_cols([[1.0], [0.0]]);  // 1x2
        let b = TMat::from_cols([[0.0], [1.0]]);
        let c = TMat::from_cols([[1.0], [1.0]]);
        let m: TMat<f64, 3, 2> = {
            let temp: TMat22<f64> = a.v_con(b);
            temp.v_con(c)
        };
        assert!(approx_eq(m.cols[0][0], 1.0) && approx_eq(m.cols[0][1], 0.0) && approx_eq(m.cols[0][2], 1.0));
        assert!(approx_eq(m.cols[1][0], 0.0) && approx_eq(m.cols[1][1], 1.0) && approx_eq(m.cols[1][2], 1.0));
    }
}