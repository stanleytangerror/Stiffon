#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]

use std::f64;
use crate::math::*;
use crate::mvec;
#[macro_use]
use crate::h_concat;
#[macro_use]
use crate::v_concat;
#[macro_use]
use crate::d_concat;

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

    pub fn pre_solve(&mut self, dt: f64) {
        self.delta_v = self.f_ext * self.inv_mass * dt;
        self.delta_𝜔 = self.inv_inertia * self.τ_ext * dt;

        self.f_ext = Vec2::ZEROS;
        self.τ_ext = 0.0;
    }

    pub fn post_solve(&mut self, dt: f64) {
        let v_new = self.v + self.delta_v;
        let 𝜔_new = self.𝜔 + self.delta_𝜔;

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
    constraints: Vec<BallJoint2d>,
    iterations: usize,
}

impl Solver2d {
    pub fn new() -> Self {
        Solver2d {
            bodies: Vec::new(),
            gravity: mvec!(0.0, -9.8),
            constraints: Vec::new(),
            iterations: 1,
        }
    }

    pub fn add_body(&mut self, body: Body2d) -> usize {
        let index = self.bodies.len();
        self.bodies.push(body);
        index
    }

    pub fn add_constraint(&mut self, constraint: BallJoint2d) -> usize {
        let index = self.constraints.len();
        self.constraints.push(constraint);
        index
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
            body.pre_solve(dt);
        }

        for constraint in &mut self.constraints {
            constraint.setup(&self.bodies[constraint.body_A], &self.bodies[constraint.body_B], dt);
        }

        for i in 0..self.iterations {
            for constraint in &mut self.constraints {
                let [mut body_A, mut body_B] = self.bodies
                    .get_disjoint_mut([constraint.body_A, constraint.body_B])
                    .expect("constraint body indices out of bounds or equal");
                constraint.iteration(body_A, body_B, dt);
            }
        }

        for body in &mut self.bodies {
            body.post_solve(dt);
        }
    }
}

#[derive(Copy, Clone)]
struct Cons1d {
    eff_mass: MFloat,
    jacobian: TMat<f64, 1, 6>,
    bias: MFloat,
}

impl Cons1d {
    pub fn new() -> Self {
        Cons1d {
            eff_mass: MFloat::ZEROS,
            jacobian: TMat::ZEROS,
            bias: MFloat::ZEROS,
        }
    }
}

pub struct BallJoint2d {
    body_A: usize,
    body_B: usize,
    local_frame_body_A: Transform2d,
    local_frame_body_B: Transform2d,
    
    inv_m: TMat<f64, 6, 6>,
    Cons1d: [Cons1d; 2],
}

impl BallJoint2d {

    pub fn new(body_A: usize, body_B: usize, local_frame_body_A: Transform2d, local_frame_body_B: Transform2d) -> Self {
        BallJoint2d {
            body_A,
            body_B,
            local_frame_body_A,
            local_frame_body_B,

            inv_m: TMat::ZEROS,
            Cons1d: [Cons1d::new(); 2],
        }
    }

    fn setup(&mut self, body_A: &Body2d, body_B: &Body2d, dt: f64) {
        
        let world_transform_A = body_A.pose * self.local_frame_body_A;
        let world_transform_B = body_B.pose * self.local_frame_body_B;

        let r_A = world_transform_A.origin - body_A.pose.origin;
        let r_B = world_transform_B.origin - body_B.pose.origin;

        self.inv_m = d_concat!(
            Mat22::diag([body_A.inv_mass; 2]),
            body_A.inv_inertia,
            Mat22::diag([body_B.inv_mass; 2]),
            body_B.inv_inertia
        );

        let axis = [ Vec2::unit_x(), Vec2::unit_y() ];
        
        for i in 0..2 {
            let n = axis[i];

            // C = n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) in R
            // J = [ n^T, (r_A x n)^T, -n^T, -(r_B x n)^T ] in R^1x6
            self.Cons1d[i].jacobian = h_concat!(
                n.T(), 
                r_A.cross(n), 
                -n.T(), 
                -r_B.cross(n)
            );

            let c_init = n.T() * (world_transform_A.origin - world_transform_B.origin);
            let erp = 0.2;
            self.Cons1d[i].bias = -c_init * (erp / dt);

            self.Cons1d[i].eff_mass = self.Cons1d[i].jacobian * self.inv_m * self.Cons1d[i].jacobian.T();
        }       
    }

    fn iteration(&mut self, body_A: &mut Body2d, body_B: &mut Body2d, dt: f64) {

        let v = v_concat!(
            body_A.v,
            body_A.𝜔,
            body_B.v,
            body_B.𝜔
        );

        let mut dv = v_concat!(
            body_A.delta_v,
            body_A.delta_𝜔,
            body_B.delta_v,
            body_B.delta_𝜔
        );
        
        for i in 0..2 {
            let rhs = -self.Cons1d[i].jacobian * (v + dv) - self.Cons1d[i].bias;
            let lambda = self.Cons1d[i].eff_mass * rhs;
            let impulse = self.Cons1d[i].jacobian.T() * lambda;
            dv += self.inv_m * impulse;
        }

        body_A.delta_v = dv.sub(0..2, 0..1);
        body_A.delta_𝜔 = dv.sub(2..3, 0..1).as_float();
        body_B.delta_v = dv.sub(3..5, 0..1);
        body_B.delta_𝜔 = dv.sub(5..6, 0..1).as_float();
    }
}

