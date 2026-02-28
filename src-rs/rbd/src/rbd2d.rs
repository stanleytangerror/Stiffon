#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]

use std::f64;
use crate::math::*;
use crate::mvec;
use crate::h_concat;
use crate::v_concat;

// --- Geometry ---
#[derive(Clone, Debug)]
pub enum Geometry2d {
    Rectangle { half_extents: Vec2 },
    Circle { radius: f64 },
}

impl Geometry2d {
    pub fn rectangle(half_extents: Vec2) -> Self {
        Geometry2d::Rectangle { half_extents }
    }
    pub fn circle(radius: f64) -> Self {
        Geometry2d::Circle { radius }
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
    geometry: Geometry2d,
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
        geometry: Geometry2d,
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
            delta_v: Vec2::ZEROS,
            delta_𝜔: 0.0,
            pose,
            geometry,
            f_ext: Vec2::ZEROS,
            τ_ext: 0.0,
            delta_linear_velocity: Vec2::ZEROS,
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
        
        self.delta_v = Vec2::ZEROS;
        self.delta_𝜔 = 0.0;
    }

    pub fn pose(&self) -> Transform2d {
        self.pose
    }

    pub fn geometry(&self) -> &Geometry2d {
        &self.geometry
    }
}

pub struct Solver2d {
    pub bodies: Vec<Body2d>,
    gravity: Vec2,
}

impl Solver2d {
    pub fn new() -> Self {
        Solver2d {
            bodies: Vec::new(),
            gravity: mvec!(0.0, -9.8),
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

struct BallJointConstraint2d {
    body_A: usize,
    body_B: usize,
    local_frame_body_A: Transform2d,
    local_frame_body_B: Transform2d,
}

impl BallJointConstraint2d {


    fn setup(&self, solver: &Solver2d) -> (TMat<f64, 1, 8>, Vec2) {
        // C = v_A + ω_A × r_A - v_B - ω_B × r_B in R^2
        // J = [ I_2, -[r_A]x, -I_2, [r_B]x ] in R^2x6
        let world_transform_A = solver.bodies[self.body_A].pose * self.local_frame_body_A;
        let world_transform_B = solver.bodies[self.body_B].pose * self.local_frame_body_B;

        let c_init = world_transform_A.origin - world_transform_B.origin;

        let r_A = world_transform_A.origin - world_transform_A.origin;
        let r_B = world_transform_B.origin - world_transform_B.origin;
        let jacobian = h_concat!(
            Vec2::ONES.T(),
            -r_A.cross(1.0).T(), 
            -Vec2::ONES.T(), 
            r_B.cross(1.0).T()
        );

        let jacobian1 = v_concat!(
            Vec2::ONES.T(),
            -r_A.cross(1.0).T(), 
            -Vec2::ONES.T(), 
            r_B.cross(1.0).T()
        );
        
        // (jacobian, c_init)
        (TMat::<f64, 1, 8>::ZEROS, Vec2::ZEROS)
    }
}

