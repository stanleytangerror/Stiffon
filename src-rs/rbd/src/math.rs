#![allow(unused)]

use num_traits::{Zero, One, Float};

//#region FloatNum
pub trait FloatNum: 
    Float + Zero + One 
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

//#region TVec2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TVec2<T> {
    pub x: T,
    pub y: T,
}

impl<T: FloatNum> TVec2<T> {
    pub const ZERO: Self = TVec2 { x: T::ZERO, y: T::ZERO };

    pub fn new(x: T, y: T) -> Self {
        TVec2 { x, y }
    }

    pub fn norm(self) -> T {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn rotate(self, angle: T) -> Self {
        let sin = angle.sin();
        let cos = angle.cos();
        TVec2 { 
            x: self.x * cos - self.y * sin, 
            y: self.x * sin + self.y * cos }
    }
}

impl<T: FloatNum> std::ops::Add for TVec2<T> {
    type Output = TVec2<T>;
    fn add(self, rhs: Self) -> Self::Output {
        TVec2 { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl<T: FloatNum + std::ops::AddAssign> std::ops::AddAssign for TVec2<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: FloatNum> std::ops::Sub for TVec2<T> {
    type Output = TVec2<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        TVec2 { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl<T: FloatNum + std::ops::SubAssign> std::ops::SubAssign for TVec2<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: FloatNum> std::ops::Mul<T> for TVec2<T> {
    type Output = TVec2<T>;
    fn mul(self, rhs: T) -> Self::Output {
        TVec2 { x: self.x * rhs, y: self.y * rhs }
    }
}


impl<T: FloatNum + std::ops::MulAssign> std::ops::MulAssign<T> for TVec2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: FloatNum> std::ops::Neg for TVec2<T> {
    type Output = TVec2<T>;
    fn neg(self) -> Self::Output {
        TVec2 { x: -self.x, y: -self.y }
    }
}

pub trait Dot<RHS> {
    type Output;
    fn dot(self, rhs: RHS) -> Self::Output;
}

impl<T: FloatNum> Dot<TVec2<T>> for TVec2<T> {
    type Output = T;
    fn dot(self, rhs: TVec2<T>) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y
    }
}

pub trait Cross<RHS> {
    type Output;
    fn cross(self, rhs: RHS) -> Self::Output;
}

impl<T: FloatNum> Cross<TVec2<T>> for TVec2<T> {
    type Output = T;
    fn cross(self, rhs: TVec2<T>) -> Self::Output {
        self.x * rhs.y - self.y * rhs.x
    }
}

impl<T: FloatNum> Cross<T> for TVec2<T> {
    type Output = TVec2<T>;
    fn cross(self, rhs: T) -> Self::Output {
        TVec2::<T>::new(self.y * rhs, -self.x * rhs)
    }
}

impl<T: FloatNum> Cross<TVec2<T>> for T {
    type Output = TVec2<T>;
    fn cross(self, rhs: TVec2<T>) -> Self::Output {
        TVec2::<T>::new(-self * rhs.y, self * rhs.x)
    }
}
//#endregion

// --- TTransform2d ---
#[derive(Clone, Copy, Debug)]
pub struct TTransform2d<T> {
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec2_approx_eq(a: TVec2<f64>, b: TVec2<f64>) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y)
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
        let v = TVec2::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(TVec2::<f64>::ZERO.x, 0.0);
        assert_eq!(TVec2::<f64>::ZERO.y, 0.0);
    }

    // --- TVec2::norm ---
    #[test]
    fn vec2_norm() {
        assert!(approx_eq(TVec2::new(3.0, 4.0).norm(), 5.0));
        assert!(approx_eq(TVec2::new(0.0, 0.0).norm(), 0.0));
        assert!(approx_eq(TVec2::new(1.0, 0.0).norm(), 1.0));
    }

    // --- TVec2::rotate ---
    #[test]
    fn vec2_rotate_90() {
        let v = TVec2::new(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!(vec2_approx_eq(r, TVec2::new(0.0, 1.0)));
    }

    #[test]
    fn vec2_rotate_identity() {
        let v = TVec2::new(1.0, 2.0);
        let r = v.rotate(0.0);
        assert!(vec2_approx_eq(r, v));
    }

    // --- Add / Sub / Neg ---
    #[test]
    fn vec2_add_sub_neg() {
        let a = TVec2::new(1.0, 2.0);
        let b = TVec2::new(3.0, 4.0);
        assert!(vec2_approx_eq(a + b, TVec2::new(4.0, 6.0)));
        assert!(vec2_approx_eq(b - a, TVec2::new(2.0, 2.0)));
        assert!(vec2_approx_eq(-a, TVec2::new(-1.0, -2.0)));
    }

    // --- AddAssign / SubAssign / MulAssign ---
    #[test]
    fn vec2_assign_ops() {
        let mut v = TVec2::new(1.0, 2.0);
        v += TVec2::new(1.0, 1.0);
        assert!(vec2_approx_eq(v, TVec2::new(2.0, 3.0)));
        v -= TVec2::new(0.0, 1.0);
        assert!(vec2_approx_eq(v, TVec2::new(2.0, 2.0)));
        v *= 2.0;
        assert!(vec2_approx_eq(v, TVec2::new(4.0, 4.0)));
    }

    // --- Mul scalar ---
    #[test]
    fn vec2_mul_scalar() {
        let v = TVec2::new(1.0, 2.0);
        assert!(vec2_approx_eq(v * 3.0, TVec2::new(3.0, 6.0)));
    }

    // --- Dot ---
    #[test]
    fn vec2_dot() {
        let a = TVec2::new(1.0, 0.0);
        let b = TVec2::new(1.0, 0.0);
        assert!(approx_eq(a.dot(b), 1.0));
        let c = TVec2::new(3.0, 4.0);
        assert!(approx_eq(a.dot(c), 3.0));
        assert!(approx_eq(c.dot(c), 25.0));
    }

    // --- Cross (TVec2 x TVec2 -> scalar) ---
    #[test]
    fn vec2_cross_vec2() {
        let a = TVec2::new(1.0, 0.0);
        let b = TVec2::new(0.0, 1.0);
        assert!(approx_eq(a.cross(b), 1.0));
        assert!(approx_eq(b.cross(a), -1.0));
    }

    // --- Cross (TVec2 x T -> TVec2) ---
    #[test]
    fn vec2_cross_scalar() {
        let v = TVec2::new(1.0, 0.0);
        let r = v.cross(2.0);
        assert!(vec2_approx_eq(r, TVec2::new(0.0, -2.0)));
    }

    // --- Cross (T x TVec2 -> TVec2) ---
    #[test]
    fn scalar_cross_vec2() {
        let v = TVec2::new(1.0, 0.0);
        let r = 2.0_f64.cross(v);
        assert!(vec2_approx_eq(r, TVec2::new(0.0, 2.0)));
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
        let o = TVec2::new(1.0, 2.0);
        let t = TTransform2d::new(o, std::f64::consts::FRAC_PI_2);
        assert!(vec2_approx_eq(t.origin, o));
        assert!(approx_eq(t.angle, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn transform2d_identity_mul_left() {
        let t = TTransform2d::new(TVec2::new(3.0, 4.0), 0.5);
        let id = transform2d_identity();
        assert!(transform2d_approx_eq(id * t, t));
    }

    #[test]
    fn transform2d_identity_mul_right() {
        let t = TTransform2d::new(TVec2::new(3.0, 4.0), 0.5);
        let id = transform2d_identity();
        assert!(transform2d_approx_eq(t * id, t));
    }

    #[test]
    fn transform2d_invert_roundtrip() {
        let t = TTransform2d::new(TVec2::new(1.0, 2.0), 0.7);
        let inv = t.invert();
        assert!(transform2d_approx_eq(t * inv, transform2d_identity()));
        assert!(transform2d_approx_eq(inv * t, transform2d_identity()));
    }

    #[test]
    fn transform2d_invert_twice() {
        let t = TTransform2d::new(TVec2::new(-1.0, 3.0), std::f64::consts::PI);
        assert!(transform2d_approx_eq(t.invert().invert(), t));
    }

    #[test]
    fn transform2d_mul_composition() {
        let a = TTransform2d::new(TVec2::new(1.0, 0.0), 0.0);
        let b = TTransform2d::new(TVec2::new(0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let ab = a * b;
        // a * b: first apply b (translate (0,1) then rotate 90°), then a (translate (1,0))
        // ab.origin = b.origin.rotate(a.angle) + a.origin = (0,1).rotate(0) + (1,0) = (1,1)
        assert!(vec2_approx_eq(ab.origin, TVec2::new(1.0, 1.0)));
        assert!(approx_eq(ab.angle, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn transform2d_mul_associativity() {
        let a = TTransform2d::new(TVec2::new(1.0, 0.0), 0.3);
        let b = TTransform2d::new(TVec2::new(0.0, 1.0), 0.5);
        let c = TTransform2d::new(TVec2::new(2.0, -1.0), -0.2);
        assert!(transform2d_approx_eq((a * b) * c, a * (b * c)));
    }
}