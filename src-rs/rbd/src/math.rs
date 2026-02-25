#![allow(unused)]

use num_traits::{Zero, One, Float};

//#region FloatNum
pub trait FloatNum: 
    Float + Zero + One 
    + std::iter::Sum<Self>
    + std::iter::Product<Self>
    + std::ops::AddAssign + std::ops::SubAssign 
    + std::ops::MulAssign + std::ops::DivAssign {
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

//#region TVec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TVec<T: FloatNum, const D: usize> {
    pub data: [T; D],
}

pub type TVec2<T> = TVec<T, 2>;
pub type TVec3<T> = TVec<T, 3>;
pub type TVec4<T> = TVec<T, 4>;

/// 不定个参数构造向量，例如：`vec!(1.0, 2.0)`、`vec!(1.0, 2.0, 3.0)`。
#[macro_export]
macro_rules! vec {
    ($($x:expr),* $(,)?) => {
        $crate::math::TVec { data: [$($x),*] }
    };
}

impl<T: FloatNum, const D: usize> TVec<T, D> {
    #[inline(always)]
    pub fn x(&self) -> T {
        self.data[0]
    }
    #[inline(always)]
    pub fn y(&self) -> T {
        self.data[1]
    }
    #[inline(always)]
    pub fn z(&self) -> T {
        self.data[2]
    }
    #[inline(always)]
    pub fn w(&self) -> T {
        self.data[3]
    }
    #[inline(always)]
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.data[0]
    }
    #[inline(always)]
    pub fn y_mut(&mut self) -> &mut T {
        &mut self.data[1]
    }
    #[inline(always)]
    pub fn z_mut(&mut self) -> &mut T {
        &mut self.data[2]
    }
    #[inline(always)]
    pub fn w_mut(&mut self) -> &mut T {
        &mut self.data[3]
    }
}

impl<T: FloatNum, const D: usize> TVec<T, D> {
    pub const ZERO: Self = TVec { data: [T::ZERO; D] };

    pub fn new(data: [T; D]) -> Self {
        TVec { data }
    }

    pub fn norm(self) -> T {
        (self.data.iter().map(|x: &T| *x * *x).sum::<T>()).sqrt()
    }
}

impl<T: FloatNum> TVec<T, 2> {

    pub fn rotate(self, angle: T) -> Self {
        let sin = angle.sin();
        let cos = angle.cos();
        TVec::<T, 2>::new(
            [self.x() * cos - self.y() * sin, self.x() * sin + self.y() * cos]
        )
    }
}

impl<T: FloatNum> std::ops::Add for TVec<T, 2> {
    type Output = TVec<T, 2>;
    fn add(self, rhs: Self) -> Self::Output {
        TVec {
            data: [self.x() + rhs.x(), self.y() + rhs.y()],
        }
    }
}

impl<T: FloatNum + std::ops::AddAssign> std::ops::AddAssign for TVec<T, 2> {
    fn add_assign(&mut self, rhs: Self) {
        *self.x_mut() += rhs.x();
        *self.y_mut() += rhs.y();
    }
}

impl<T: FloatNum> std::ops::Sub for TVec<T, 2> {
    type Output = TVec<T, 2>;
    fn sub(self, rhs: Self) -> Self::Output {
        TVec {
            data: [self.x() - rhs.x(), self.y() - rhs.y()],
        }
    }
}

impl<T: FloatNum + std::ops::SubAssign> std::ops::SubAssign for TVec<T, 2> {
    fn sub_assign(&mut self, rhs: Self) {
        *self.x_mut() -= rhs.x();
        *self.y_mut() -= rhs.y();
    }
}

impl<T: FloatNum> std::ops::Mul<T> for TVec<T, 2> {
    type Output = TVec<T, 2>;
    fn mul(self, rhs: T) -> Self::Output {
        TVec {
            data: [self.x() * rhs, self.y() * rhs],
        }
    }
}


impl<T: FloatNum + std::ops::MulAssign> std::ops::MulAssign<T> for TVec<T, 2> {
    fn mul_assign(&mut self, rhs: T) {
        *self.x_mut() *= rhs;
        *self.y_mut() *= rhs;
    }
}

impl<T: FloatNum> std::ops::Neg for TVec<T, 2> {
    type Output = TVec<T, 2>;
    fn neg(self) -> Self::Output {
        TVec {
            data: [-self.x(), -self.y()],
        }
    }
}

pub trait Dot<RHS> {
    type Output;
    fn dot(self, rhs: RHS) -> Self::Output;
}

impl<T: FloatNum> Dot<TVec<T, 2>> for TVec<T, 2> {
    type Output = T;
    fn dot(self, rhs: TVec<T, 2>) -> Self::Output {
        self.x() * rhs.x() + self.y() * rhs.y()
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


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TMat<T: FloatNum, const R: usize, const C: usize> {
    pub cols: [TVec<T, R>; C],
}

pub type TMat22<T> = TMat<T, 2, 2>;
pub type TMat33<T> = TMat<T, 3, 3>;
pub type TMat44<T> = TMat<T, 4, 4>;

impl<T: FloatNum, const R: usize, const C: usize> TMat<T, R, C> {
    pub fn from_cols(cols: [TVec<T, R>; C]) -> Self {
        TMat { cols: cols }
    }

    pub fn zeros() -> Self {
        TMat::from_cols([TVec::<T, R>::new([T::ZERO; R]); C])
    }
}

impl<T: FloatNum, const D: usize> TMat<T, D, D> {
    pub fn identity() -> Self
    {
        let mut cols = [TVec::<T, D>::new([T::ZERO; D]); D];
        for j in 0..D {
            cols[j].data[j] = T::ONE;
        }
        TMat::from_cols(cols)
    }
}

// --- TTransform2d ---
#[derive(Clone, Copy, Debug)]
pub struct TTransform2d<T: FloatNum> {
    pub origin: TVec2<T>,
    pub angle: T,
}

impl<T: FloatNum> TTransform2d<T> {
    pub const IDENTITY: Self = TTransform2d { origin: TVec2::ZERO, angle: T::ZERO };

    pub fn new(origin: TVec2<T>, angle: T) -> Self {
        TTransform2d::<T> { origin, angle }
    }

    pub fn invert(&self) -> Self {
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
        let v = vec!(3.0, 4.0);
        assert_eq!(v.x(), 3.0);
        assert_eq!(v.y(), 4.0);
        assert_eq!(TVec2::<f64>::ZERO.x(), 0.0);
        assert_eq!(TVec2::<f64>::ZERO.y(), 0.0);
    }

    // --- TVec2::norm ---
    #[test]
    fn vec2_norm() {
        assert!(approx_eq(vec!(3.0, 4.0).norm(), 5.0));
        assert!(approx_eq(vec!(0.0, 0.0).norm(), 0.0));
        assert!(approx_eq(vec!(1.0, 0.0).norm(), 1.0));
    }

    // --- TVec2::rotate ---
    #[test]
    fn vec2_rotate_90() {
        let v = vec!(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!(vec2_approx_eq(r, vec!(0.0, 1.0)));
    }

    #[test]
    fn vec2_rotate_identity() {
        let v = vec!(1.0, 2.0);
        let r = v.rotate(0.0);
        assert!(vec2_approx_eq(r, v));
    }

    // --- Add / Sub / Neg ---
    #[test]
    fn vec2_add_sub_neg() {
        let a = vec!(1.0, 2.0);
        let b = vec!(3.0, 4.0);
        assert!(vec2_approx_eq(a + b, vec!(4.0, 6.0)));
        assert!(vec2_approx_eq(b - a, vec!(2.0, 2.0)));
        assert!(vec2_approx_eq(-a, vec!(-1.0, -2.0)));
    }

    // --- AddAssign / SubAssign / MulAssign ---
    #[test]
    fn vec2_assign_ops() {
        let mut v = vec!(1.0, 2.0);
        v += vec!(1.0, 1.0);
        assert!(vec2_approx_eq(v, vec!(2.0, 3.0)));
        v -= vec!(0.0, 1.0);
        assert!(vec2_approx_eq(v, vec!(2.0, 2.0)));
        v *= 2.0;
        assert!(vec2_approx_eq(v, vec!(4.0, 4.0)));
    }

    // --- Mul scalar ---
    #[test]
    fn vec2_mul_scalar() {
        let v = vec!(1.0, 2.0);
        assert!(vec2_approx_eq(v * 3.0, vec!(3.0, 6.0)));
    }

    // --- Dot ---
    #[test]
    fn vec2_dot() {
        let a = vec!(1.0, 0.0);
        let b = vec!(1.0, 0.0);
        assert!(approx_eq(a.dot(b), 1.0));
        let c = vec!(3.0, 4.0);
        assert!(approx_eq(a.dot(c), 3.0));
        assert!(approx_eq(c.dot(c), 25.0));
    }

    // --- Cross (TVec2 x TVec2 -> scalar) ---
    #[test]
    fn vec2_cross_vec2() {
        let a = vec!(1.0, 0.0);
        let b = vec!(0.0, 1.0);
        assert!(approx_eq(a.cross(b), 1.0));
        assert!(approx_eq(b.cross(a), -1.0));
    }

    // --- Cross (TVec2 x T -> TVec2) ---
    #[test]
    fn vec2_cross_scalar() {
        let v = vec!(1.0, 0.0);
        let r = v.cross(2.0);
        assert!(vec2_approx_eq(r, vec!(0.0, -2.0)));
    }

    // --- Cross (T x TVec2 -> TVec2) ---
    #[test]
    fn scalar_cross_vec2() {
        let v = vec!(1.0, 0.0);
        let r = 2.0_f64.cross(v);
        assert!(vec2_approx_eq(r, vec!(0.0, 2.0)));
    }

    // --- TTransform2d ---
    fn transform2d_approx_eq(a: TTransform2d<f64>, b: TTransform2d<f64>) -> bool {
        vec2_approx_eq(a.origin, b.origin) && approx_eq(a.angle, b.angle)
    }

    fn transform2d_identity() -> TTransform2d<f64> {
        TTransform2d::new(TVec2::ZERO, 0.0)
    }

    #[test]
    fn transform2d_new() {
        let o = vec!(1.0, 2.0);
        let t = TTransform2d::new(o, std::f64::consts::FRAC_PI_2);
        assert!(vec2_approx_eq(t.origin, o));
        assert!(approx_eq(t.angle, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn transform2d_identity_mul_left() {
        let t = TTransform2d::new(vec!(3.0, 4.0), 0.5);
        let id = transform2d_identity();
        assert!(transform2d_approx_eq(id * t, t));
    }

    #[test]
    fn transform2d_identity_mul_right() {
        let t = TTransform2d::new(vec!(3.0, 4.0), 0.5);
        let id = transform2d_identity();
        assert!(transform2d_approx_eq(t * id, t));
    }

    #[test]
    fn transform2d_invert_roundtrip() {
        let t = TTransform2d::new(vec!(1.0, 2.0), 0.7);
        let inv = t.invert();
        assert!(transform2d_approx_eq(t * inv, transform2d_identity()));
        assert!(transform2d_approx_eq(inv * t, transform2d_identity()));
    }

    #[test]
    fn transform2d_invert_twice() {
        let t = TTransform2d::new(vec!(-1.0, 3.0), std::f64::consts::PI);
        assert!(transform2d_approx_eq(t.invert().invert(), t));
    }

    #[test]
    fn transform2d_mul_composition() {
        let a = TTransform2d::new(vec!(1.0, 0.0), 0.0);
        let b = TTransform2d::new(vec!(0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let ab = a * b;
        // a * b: first apply b (translate (0,1) then rotate 90°), then a (translate (1,0))
        // ab.origin = b.origin.rotate(a.angle) + a.origin = (0,1).rotate(0) + (1,0) = (1,1)
        assert!(vec2_approx_eq(ab.origin, vec!(1.0, 1.0)));
        assert!(approx_eq(ab.angle, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn transform2d_mul_associativity() {
        let a = TTransform2d::new(vec!(1.0, 0.0), 0.3);
        let b = TTransform2d::new(vec!(0.0, 1.0), 0.5);
        let c = TTransform2d::new(vec!(2.0, -1.0), -0.2);
        assert!(transform2d_approx_eq((a * b) * c, a * (b * c)));
    }
}