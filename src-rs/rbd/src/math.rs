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