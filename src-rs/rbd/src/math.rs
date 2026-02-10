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

//#region Vec2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T: FloatNum> Vec2<T> {
    pub const ZERO: Self = Vec2 { x: T::ZERO, y: T::ZERO };

    pub fn new(x: T, y: T) -> Self {
        Vec2 { x, y }
    }

    pub fn norm(self) -> T {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn rotate(self, angle: T) -> Self {
        let sin = angle.sin();
        let cos = angle.cos();
        Vec2 { 
            x: self.x * cos - self.y * sin, 
            y: self.x * sin + self.y * cos }
    }
}

impl<T: FloatNum> std::ops::Add for Vec2<T> {
    type Output = Vec2<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Vec2 { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl<T: FloatNum + std::ops::AddAssign> std::ops::AddAssign for Vec2<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: FloatNum> std::ops::Sub for Vec2<T> {
    type Output = Vec2<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2 { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl<T: FloatNum + std::ops::SubAssign> std::ops::SubAssign for Vec2<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: FloatNum> std::ops::Mul<T> for Vec2<T> {
    type Output = Vec2<T>;
    fn mul(self, rhs: T) -> Self::Output {
        Vec2 { x: self.x * rhs, y: self.y * rhs }
    }
}

impl<T: FloatNum + std::ops::MulAssign> std::ops::MulAssign<T> for Vec2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: FloatNum> std::ops::Neg for Vec2<T> {
    type Output = Vec2<T>;
    fn neg(self) -> Self::Output {
        Vec2 { x: -self.x, y: -self.y }
    }
}

trait Dot<RHS> {
    type Output;
    fn dot(self, rhs: RHS) -> Self::Output;
}

impl<T: FloatNum> Dot<Vec2<T>> for Vec2<T> {
    type Output = T;
    fn dot(self, rhs: Vec2<T>) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y
    }
}

pub trait Cross<RHS> {
    type Output;
    fn cross(self, rhs: RHS) -> Self::Output;
}

impl<T: FloatNum> Cross<Vec2<T>> for Vec2<T> {
    type Output = T;
    fn cross(self, rhs: Vec2<T>) -> Self::Output {
        self.x * rhs.y - self.y * rhs.x
    }
}

impl<T: FloatNum> Cross<T> for Vec2<T> {
    type Output = Vec2<T>;
    fn cross(self, rhs: T) -> Self::Output {
        Vec2::<T>::new(self.y * rhs, -self.x * rhs)
    }
}

impl<T: FloatNum> Cross<Vec2<T>> for T {
    type Output = Vec2<T>;
    fn cross(self, rhs: Vec2<T>) -> Self::Output {
        Vec2::<T>::new(-self * rhs.y, self * rhs.x)
    }
}
//#endregion

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec2_approx_eq(a: Vec2<f64>, b: Vec2<f64>) -> bool {
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

    // --- Vec2 constructors & ZERO ---
    #[test]
    fn vec2_new_and_zero() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(Vec2::<f64>::ZERO.x, 0.0);
        assert_eq!(Vec2::<f64>::ZERO.y, 0.0);
    }

    // --- Vec2::norm ---
    #[test]
    fn vec2_norm() {
        assert!(approx_eq(Vec2::new(3.0, 4.0).norm(), 5.0));
        assert!(approx_eq(Vec2::new(0.0, 0.0).norm(), 0.0));
        assert!(approx_eq(Vec2::new(1.0, 0.0).norm(), 1.0));
    }

    // --- Vec2::rotate ---
    #[test]
    fn vec2_rotate_90() {
        let v = Vec2::new(1.0, 0.0);
        let r = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!(vec2_approx_eq(r, Vec2::new(0.0, 1.0)));
    }

    #[test]
    fn vec2_rotate_identity() {
        let v = Vec2::new(1.0, 2.0);
        let r = v.rotate(0.0);
        assert!(vec2_approx_eq(r, v));
    }

    // --- Add / Sub / Neg ---
    #[test]
    fn vec2_add_sub_neg() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!(vec2_approx_eq(a + b, Vec2::new(4.0, 6.0)));
        assert!(vec2_approx_eq(b - a, Vec2::new(2.0, 2.0)));
        assert!(vec2_approx_eq(-a, Vec2::new(-1.0, -2.0)));
    }

    // --- AddAssign / SubAssign / MulAssign ---
    #[test]
    fn vec2_assign_ops() {
        let mut v = Vec2::new(1.0, 2.0);
        v += Vec2::new(1.0, 1.0);
        assert!(vec2_approx_eq(v, Vec2::new(2.0, 3.0)));
        v -= Vec2::new(0.0, 1.0);
        assert!(vec2_approx_eq(v, Vec2::new(2.0, 2.0)));
        v *= 2.0;
        assert!(vec2_approx_eq(v, Vec2::new(4.0, 4.0)));
    }

    // --- Mul scalar ---
    #[test]
    fn vec2_mul_scalar() {
        let v = Vec2::new(1.0, 2.0);
        assert!(vec2_approx_eq(v * 3.0, Vec2::new(3.0, 6.0)));
    }

    // --- Dot ---
    #[test]
    fn vec2_dot() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        assert!(approx_eq(a.dot(b), 1.0));
        let c = Vec2::new(3.0, 4.0);
        assert!(approx_eq(a.dot(c), 3.0));
        assert!(approx_eq(c.dot(c), 25.0));
    }

    // --- Cross (Vec2 x Vec2 -> scalar) ---
    #[test]
    fn vec2_cross_vec2() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!(approx_eq(a.cross(b), 1.0));
        assert!(approx_eq(b.cross(a), -1.0));
    }

    // --- Cross (Vec2 x T -> Vec2) ---
    #[test]
    fn vec2_cross_scalar() {
        let v = Vec2::new(1.0, 0.0);
        let r = v.cross(2.0);
        assert!(vec2_approx_eq(r, Vec2::new(0.0, -2.0)));
    }

    // --- Cross (T x Vec2 -> Vec2) ---
    #[test]
    fn scalar_cross_vec2() {
        let v = Vec2::new(1.0, 0.0);
        let r = 2.0_f64.cross(v);
        assert!(vec2_approx_eq(r, Vec2::new(0.0, 2.0)));
    }
}