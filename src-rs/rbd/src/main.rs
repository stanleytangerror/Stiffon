mod math;

use std::f64;
use math::Cross;
use ndarray::Array2;

type Vec2 = math::Vec2<f64>;

/// 2×6 矩阵，基于 ndarray
pub type Mat2x6 = Array2<f64>;

/// 构造 2×6 零矩阵
pub fn mat2x6_zeros() -> Mat2x6 {
    Array2::zeros((2, 6))
}

/// 6×6 矩阵，基于 ndarray
pub type Mat6x6 = Array2<f64>;

/// 构造 6×6 零矩阵
pub fn mat6x6_zeros() -> Mat6x6 {
    Array2::zeros((6, 6))
}

// --- Transform2d ---
#[derive(Clone, Copy, Debug)]
struct Transform2d {
    origin: Vec2,
    angle: f64,
}

impl Transform2d {
    const fn new(origin: Vec2, angle: f64) -> Self {
        Transform2d { origin, angle }
    }

    fn invert(&self) -> Transform2d {
        Transform2d {
            origin: -self.origin.rotate(-self.angle),
            angle: -self.angle,
        }
    }
}

impl std::ops::Mul<Transform2d> for Transform2d {
    type Output = Transform2d;
    fn mul(self, rhs: Transform2d) -> Transform2d {
        Transform2d {
            origin: rhs.origin.rotate(self.angle) + self.origin,
            angle: self.angle + rhs.angle,
        }
    }
}

// --- Geometry ---
#[derive(Clone, Debug)]
enum Geometry {
    Rectangle { half_extents: Vec2 },
    Circle { radius: f64 },
}

impl Geometry {
    fn rectangle(half_extents: Vec2) -> Self {
        Geometry::Rectangle { half_extents }
    }
    fn circle(radius: f64) -> Self {
        Geometry::Circle { radius }
    }
}

// --- Body2d ---
struct Body2d {
    mass: f64,
    inv_mass: f64,
    inertia: f64,
    inv_inertia: f64,
    inv_inertia_world: f64,
    linear_velocity: Vec2,
    angular_velocity: f64,
    pose: Transform2d,
    geometry: Geometry,
    total_force: Vec2,
    total_torque: f64,
    delta_linear_velocity: Vec2,
    delta_angular_velocity: f64,
}

/// Like Python: 1/m if m != inf else 0
fn safe_inv_mass(m: f64) -> f64 {
    if m != f64::INFINITY && m.is_finite() && m != 0.0 {
        1.0 / m
    } else {
        0.0
    }
}

/// Like Python: 1/i if i != inf else 0
fn safe_inv_inertia(i: f64) -> f64 {
    if i != f64::INFINITY && i.is_finite() && i != 0.0 {
        1.0 / i
    } else {
        0.0
    }
}

impl Body2d {
    fn new(
        mass: f64,
        inertia: f64,
        linear_velocity: Vec2,
        angular_velocity: f64,
        pose: Transform2d,
        geometry: Geometry,
    ) -> Self {
        let inv_mass = safe_inv_mass(mass);
        let inv_inertia = safe_inv_inertia(inertia);
        Body2d {
            mass,
            inv_mass,
            inertia,
            inv_inertia,
            inv_inertia_world: inv_inertia,
            linear_velocity,
            angular_velocity,
            pose,
            geometry,
            total_force: Vec2::ZERO,
            total_torque: 0.0,
            delta_linear_velocity: Vec2::ZERO,
            delta_angular_velocity: 0.0,
        }
    }

    /// Apply a force at a point. In 2D, torque is a scalar (cross product z-component).
    fn apply_force(&mut self, force: Vec2, point: Vec2) {
        self.total_force += force;
        let r = point - self.pose.origin;
        self.total_torque += r.cross(force);
    }
}

struct Solver {
    bodies: Vec<Body2d>,
}

impl Solver {
    fn new() -> Self {
        Solver {
            bodies: Vec::new(),
        }
    }
    fn add_body(&mut self, body: Body2d) {
        self.bodies.push(body);
    }
}

// struct ConstraintConnection {
//     body_A: &mut Body2d,
//     body_B: &mut Body2d,
//     local_frame_body_A: Transform2d,
//     local_frame_body_B: Transform2d,
// }

// impl ConstraintConnection {
//     fn world_transform_A(&self) -> Transform2d {
//         self.body_A.pose * self.local_frame_body_A
//     }

//     fn world_transform_B(&self) -> Transform2d {
//         self.body_B.pose * self.local_frame_body_B
//     }
// }

// impl ConstraintConnection {
//     fn generic_inv_mass(&self) -> Mat6x6 {
//         let mut generic_inv_mass = mat6x6_zeros();
//         generic_inv_mass[[0, 0]] = self.body_A.inv_mass;
//         generic_inv_mass[[1, 1]] = self.body_A.inv_mass;
//         generic_inv_mass[[2, 2]] = self.body_A.inv_inertia_world;
//         generic_inv_mass[[3, 3]] = self.body_B.inv_mass;
//         generic_inv_mass[[4, 4]] = self.body_B.inv_mass;
//         generic_inv_mass[[5, 5]] = self.body_B.inv_inertia_world;
//         generic_inv_mass
//     }
// }

// struct BallJointConstraint {
//     connection: ConstraintConnection
// }

// impl BallJointConstraint {
//     fn setup(&self) -> (Mat2x6, Vec2) {
//         // C = v_A + ω_A × r_A - v_B - ω_B × r_B in R^2
//         // J = [ I_2, -[r_A]x, -I_2, [r_B]x ] in R^2x6
//         let world_transform_A = self.world_transform_A();
//         let world_transform_B = self.world_transform_B();

//         let c_init = world_transform_A.origin - world_transform_B.origin;

//         let r_A = world_transform_A.origin - self.connection.body_A.pose.origin;
//         let r_B = world_transform_B.origin - self.connection.body_B.pose.origin;
//         let mut jacobian = mat2x6_zeros();
//         jacobian[.., ..2].fill(1.0);
//         jacobian[.., 2] = -r_A.cross(&1.0);
//         jacobian[.., 3:5].fill(-1.0);
//         jacobian[.., 5] = r_B.cross(&1.0);
        
//         (jacobian, c_init)
//     }
// }


fn main() {
    let origin = Vec2::new(0.0, 0.0);
    let pose = Transform2d::new(origin, 0.0);
    let geometry = Geometry::rectangle(Vec2::new(0.5, 0.5));
    let mut body = Body2d::new(
        1.0,
        1.0,
        Vec2::ZERO,
        0.0,
        pose,
        geometry,
    );
    let mut solver = Solver::new();
    solver.add_body(body);
    println!("{:?}", solver.bodies.len());
}
