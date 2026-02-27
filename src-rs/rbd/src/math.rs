#![allow(unused)]

use num_traits::{Zero, One, Float};

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
}

//#region TVec
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// pub struct TVec<T: FloatNum, const D: usize> {
//     pub data: [T; D],
// }

// pub type TVec2<T> = TVec<T, 2>;
// pub type TVec3<T> = TVec<T, 3>;
// pub type TVec4<T> = TVec<T, 4>;

/// 不定个参数构造向量，例如：`mvec!(1.0, 2.0)`、`mvec!(1.0, 2.0, 3.0)`。
#[macro_export]
macro_rules! mvec {
    ($($x:expr),* $(,)?) => {
        $crate::math::TVec { cols: [[$($x),*]] }
    };
}

impl<T: FloatNum, const D: usize> TVec<T, D> {
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
}

impl<T: FloatNum, const N: usize> TVec<T, N> {
    pub fn new(data: [T; N]) -> Self {
        TMat::<T, N, 1> { cols: [data] }
    }

    pub fn norm(self) -> T {
        (self.cols[0].iter().map(|x: &T| *x * *x).sum::<T>()).sqrt()
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
                mat.cols[j][i] = self.cols[j][i] - rhs.cols[j][i];
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
}

impl<T: FloatNum, const D: usize> TMat<T, D, D> {
    pub fn eye() -> Self {
        let mut cols = [[T::ZERO; D]; D];
        for j in 0..D {
            cols[j][j] = T::ONE;
        }
        TMat::<T, D, D>::from_cols(cols)
    }
}

impl<T: FloatNum, const N: usize> TVec<T, N> {
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


//#region
pub type Vec2 = TVec2<f64>;
pub type Vec3 = TVec3<f64>;
pub type Vec4 = TVec4<f64>;
pub type Mat22 = TMat22<f64>;
pub type Mat33 = TMat33<f64>;
pub type Mat44 = TMat44<f64>;
pub type Transform2d = TTransform2d<f64>;
//#endregion

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
}