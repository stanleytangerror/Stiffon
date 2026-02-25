#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]

use std::f64;
use crate::math::*;
use crate::vec;

// --- Geometry ---
#[derive(Clone, Debug)]
pub enum Geometry {
    Rectangle { half_extents: Vec2 },
    Circle { radius: f64 },
}

impl Geometry {
    pub fn rectangle(half_extents: Vec2) -> Self {
        Geometry::Rectangle { half_extents }
    }
    pub fn circle(radius: f64) -> Self {
        Geometry::Circle { radius }
    }
}

// --- Body2d ---
pub struct Body2d {
    mass: f64,
    inv_mass: f64,
    inertia: f64,
    inv_inertia: f64,
    inv_inertia_world: f64,
    v: Vec2,
    𝜔: f64,
    delta_v: Vec2,
    delta_𝜔: f64,
    pose: Transform2d,
    geometry: Geometry,
    f_ext: Vec2,
    τ_ext: f64,
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
    pub fn new(
        mass: f64,
        inertia: f64,
        v: Vec2,
        𝜔: f64,
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
            v,
            𝜔,
            delta_v: Vec2::ZERO,
            delta_𝜔: 0.0,
            pose,
            geometry,
            f_ext: Vec2::ZERO,
            τ_ext: 0.0,
            delta_linear_velocity: Vec2::ZERO,
            delta_angular_velocity: 0.0,
        }
    }

    /// Apply a force at a point. In 2D, torque is a scalar (cross product z-component).
    pub fn apply_local_force(&mut self, f: Vec2, p: Vec2) {
        let f_ws = self.pose.transform_vector(f);
        let r = self.pose.transform_vector(p);
        self.f_ext += f_ws;
        self.τ_ext += r.cross(f_ws);
    }
    
    pub fn apply_gravity(&mut self, gravity: Vec2) {
        self.f_ext += gravity;
    }

    pub fn post_solve(&mut self, dt: f64) {
        let v_new = self.v + self.delta_v + self.f_ext * self.inv_mass * dt;
        let 𝜔_new = self.𝜔 + self.delta_𝜔 + self.inv_inertia_world * self.τ_ext * dt;

        self.pose = Transform2d::new(
            self.pose.origin + v_new * dt, 
            self.pose.angle + 𝜔_new * dt);
        
        self.delta_v = Vec2::ZERO;
        self.delta_𝜔 = 0.0;
    }

    pub fn pose(&self) -> Transform2d {
        self.pose
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
}

pub struct Solver {
    pub bodies: Vec<Body2d>,
    gravity: Vec2,
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            bodies: Vec::new(),
            gravity: vec!(0.0, -9.8),
        }
    }

    pub fn add_body(&mut self, body: Body2d) {
        self.bodies.push(body);
    }

    pub fn bodies(&self) -> &Vec<Body2d> {
        &self.bodies
    }

    pub fn step(&mut self, dt: f64) {
        for body in &mut self.bodies {
            if body.inv_mass != 0.0 {
                body.apply_gravity(self.gravity);
            }
        }

        for body in &mut self.bodies {
            body.post_solve(dt);
        }
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

