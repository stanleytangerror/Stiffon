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
    ext_force_dv: Vec2,
    ext_torque_d𝜔: f64,
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
            ext_force_dv: Vec2::ZEROS,
            ext_torque_d𝜔: 0.0,
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
        self.ext_force_dv = self.f_ext * self.inv_mass * dt;
        self.ext_torque_d𝜔 = self.inv_inertia * self.τ_ext * dt;

        self.f_ext = Vec2::ZEROS;
        self.τ_ext = 0.0;
    }

    pub fn post_pos_solve(&mut self, dt: f64) {
        let v_new = self.v + self.delta_v + self.ext_force_dv;
        let 𝜔_new = self.𝜔 + self.delta_𝜔 + self.ext_torque_d𝜔;

        self.pose.origin += v_new * dt;
        self.pose.angle += 𝜔_new * dt;

        self.delta_v = Vec2::ZEROS;
        self.delta_𝜔 = 0.0;
    }

    pub fn post_vel_solve(&mut self, dt: f64) {
        self.v = self.v + self.delta_v + self.ext_force_dv;
        self.𝜔 = self.𝜔 + self.delta_𝜔 + self.ext_torque_d𝜔;
        
        self.delta_v = Vec2::ZEROS;
        self.delta_𝜔 = 0.0;
        self.ext_force_dv = Vec2::ZEROS;
        self.ext_torque_d𝜔 = 0.0;
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
    constraints: Vec<Box<dyn Constraint>>,
    pos_iter_count: usize,
    vel_iter_count: usize,
}

impl Solver2d {
    pub fn new() -> Self {
        Solver2d {
            bodies: Vec::new(),
            gravity: mvec!(0.0, -9.8),
            constraints: Vec::new(),
            pos_iter_count: 1,
            vel_iter_count: 1,
        }
    }

    pub fn add_body(&mut self, body: Body2d) -> usize {
        let index = self.bodies.len();
        self.bodies.push(body);
        index
    }

    pub fn add_point_joint(&mut self, body_A_id: usize, body_B_id: usize, pos_world_A: Vec2, pos_world_B: Vec2) -> usize {
        let index = self.constraints.len();

        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];
        let local_frame_body_A = body_A.pose.inv() * Transform2d::new(pos_world_A, 0.0);
        let local_frame_body_B = body_B.pose.inv() * Transform2d::new(pos_world_B, 0.0);

        let mut constraint = PointJoint2d::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B);
        self.constraints.push(Box::new(constraint));

        index
    }

    pub fn add_angular_joint(&mut self, body_A_id: usize, body_B_id: usize, angle: f64) -> usize {
        let index = self.constraints.len();
        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];
        let local_frame_body_A = body_A.pose.inv() * Transform2d::new(Vec2::ZEROS, angle.to_radians());
        let local_frame_body_B = body_B.pose.inv() * Transform2d::new(Vec2::ZEROS, 0.0);
        let mut constraint = AngularJoint2d::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B);
        self.constraints.push(Box::new(constraint));
        index
    }

    pub fn add_angular_motor(&mut self, body_A_id: usize, body_B_id: usize, 𝜔: f64) -> usize {
        let index = self.constraints.len();
        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];
        let local_frame_body_A = body_A.pose.inv();
        let local_frame_body_B = body_B.pose.inv();
        let mut constraint = AngularMotor2d::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, 𝜔.to_radians());
        self.constraints.push(Box::new(constraint));
        index
    }

    pub fn bodies(&self) -> &Vec<Body2d> {
        &self.bodies
    }

    pub fn step(&mut self, dt: f64) {
        for body in &mut self.bodies {
            body.apply_gravity(self.gravity);
        }

        for body in &mut self.bodies {
            body.pre_solve(dt);
        }

        for constraint in &mut self.constraints {
            constraint.setup(&self.bodies[constraint.body_A_id()], &self.bodies[constraint.body_B_id()], dt);
        }

        for i in 0..self.pos_iter_count {
            for constraint in &mut self.constraints {
                let [mut body_A, mut body_B] = self.bodies
                    .get_disjoint_mut([constraint.body_A_id(), constraint.body_B_id()])
                    .expect("constraint body indices out of bounds or equal");
                constraint.iteration(body_A, body_B, true);
            }
        }

        for body in &mut self.bodies {
            body.post_pos_solve(dt);
        }

        for i in 0..self.vel_iter_count {
            for constraint in &mut self.constraints {
                let [mut body_A, mut body_B] = self.bodies
                    .get_disjoint_mut([constraint.body_A_id(), constraint.body_B_id()])
                    .expect("constraint body indices out of bounds or equal");
                constraint.iteration(body_A, body_B, false);
            }
        }

        for body in &mut self.bodies {
            body.post_vel_solve(dt);
        }
    }
}

trait Constraint {
    fn body_A_id(&self) -> usize;
    fn body_B_id(&self) -> usize;
    fn warm_up(&mut self, body_A: &mut Body2d, body_B: &mut Body2d);
    fn setup(&mut self, body_A: &Body2d, body_B: &Body2d, dt: f64);
    fn iteration(&mut self, body_A: &mut Body2d, body_B: &mut Body2d, is_pos_iter: bool);
}

#[derive(Copy, Clone)]
struct Constraint1d {
    inv_eff_mass: f64,
    jacobian: TMat<f64, 1, 6>,
    bias: f64,
    impulse_mag: f64,
}

impl Constraint1d {
    pub fn new() -> Self {
        Constraint1d {
            inv_eff_mass: 0.0,
            jacobian: TMat::ZEROS,
            bias: 0.0,
            impulse_mag: 0.0,
        }
    }
}

pub struct PointJoint2d {
    body_A: usize,
    body_B: usize,
    local_frame_body_A: Transform2d,
    local_frame_body_B: Transform2d,
    
    inv_m: TMat<f64, 6, 6>,
    constraint_1d: [Constraint1d; 2],

    // Cons2d: Cons2d,
}

impl PointJoint2d {
    pub fn new(body_A: usize, body_B: usize, local_frame_body_A: Transform2d, local_frame_body_B: Transform2d) -> Self {
        PointJoint2d {
            body_A,
            body_B,
            local_frame_body_A,
            local_frame_body_B,

            inv_m: TMat::ZEROS,
            constraint_1d: [Constraint1d::new(); 2],
        }
    }
}

impl Constraint for PointJoint2d {
    fn body_A_id(&self) -> usize {
        self.body_A
    }
    fn body_B_id(&self) -> usize {
        self.body_B
    }
    fn warm_up(&mut self, body_A: &mut Body2d, body_B: &mut Body2d) {
        let dv = 
            self.inv_m * self.constraint_1d[0].jacobian.T() * self.constraint_1d[0].impulse_mag +
            self.inv_m * self.constraint_1d[1].jacobian.T() * self.constraint_1d[1].impulse_mag;

        body_A.delta_v += dv.v_slice(0..2, 0).into();
        body_A.delta_𝜔 += Mat11::from(dv.v_slice(2..3, 0)).as_float();
        body_B.delta_v += dv.v_slice(3..5, 0).into();
        body_B.delta_𝜔 += Mat11::from(dv.v_slice(5..6, 0)).as_float();
    }

    fn setup(&mut self, body_A: &Body2d, body_B: &Body2d, dt: f64) {
        
        let p_A = body_A.pose * self.local_frame_body_A;
        let p_B = body_B.pose * self.local_frame_body_B;

        let r_A = p_A.origin - body_A.pose.origin;
        let r_B = p_B.origin - body_B.pose.origin;

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
            self.constraint_1d[i].jacobian = h_concat!(
                n.T(), 
                r_A.cross(n), 
                -n.T(), 
                -r_B.cross(n)
            );

            let c_init = (n.T() * (p_A.origin - p_B.origin)).as_float();
            let erp = 0.2;
            self.constraint_1d[i].bias = c_init * (erp / dt);

            self.constraint_1d[i].inv_eff_mass = 1.0 / (self.constraint_1d[i].jacobian * self.inv_m * self.constraint_1d[i].jacobian.T()).as_float();
        }  
    }

    fn iteration(&mut self, body_A: &mut Body2d, body_B: &mut Body2d, is_pos_iter: bool) {

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

        let ext_dv = v_concat!(
            body_A.ext_force_dv,
            body_A.ext_torque_d𝜔,
            body_B.ext_force_dv,
            body_B.ext_torque_d𝜔
        );
        
        for i in 0..2 {
            let cons = &self.constraint_1d[i];

            let jv = (cons.jacobian * (v + dv + ext_dv)).as_float();
            let rhs = if is_pos_iter { -jv - cons.bias } else { -jv };
            let lambda = cons.inv_eff_mass * rhs;
            let impulse = cons.jacobian.T() * lambda;
            dv += self.inv_m * impulse;
        }

        body_A.delta_v = dv.v_slice(0..2, 0).into();
        body_A.delta_𝜔 = Mat11::from(dv.v_slice(2..3, 0)).as_float();
        body_B.delta_v = dv.v_slice(3..5, 0).into();
        body_B.delta_𝜔 = Mat11::from(dv.v_slice(5..6, 0)).as_float();
    }
}


pub struct AngularJoint2d {
    body_A: usize,
    body_B: usize,
    local_frame_body_A: Transform2d,
    local_frame_body_B: Transform2d,
    
    inv_m: TMat<f64, 6, 6>,
    constraint_1d: Constraint1d,

    // Cons2d: Cons2d,
}

impl AngularJoint2d {

    pub fn new(body_A: usize, body_B: usize, local_frame_body_A: Transform2d, local_frame_body_B: Transform2d) -> Self {
        AngularJoint2d {
            body_A,
            body_B,
            local_frame_body_A,
            local_frame_body_B,

            inv_m: TMat::ZEROS,
            constraint_1d: Constraint1d::new(),
        }
    }
}

impl Constraint for AngularJoint2d {
    fn body_A_id(&self) -> usize {
        self.body_A
    }
    fn body_B_id(&self) -> usize {
        self.body_B
    }

    fn warm_up(&mut self, body_A: &mut Body2d, body_B: &mut Body2d) {
        let dv = self.inv_m * self.constraint_1d.jacobian.T() * self.constraint_1d.impulse_mag;

        body_A.delta_v += dv.v_slice(0..2, 0).into();
        body_A.delta_𝜔 += Mat11::from(dv.v_slice(2..3, 0)).as_float();
        body_B.delta_v += dv.v_slice(3..5, 0).into();
        body_B.delta_𝜔 += Mat11::from(dv.v_slice(5..6, 0)).as_float();
    }

    fn setup(&mut self, body_A: &Body2d, body_B: &Body2d, dt: f64) {
        
        let p_A = body_A.pose * self.local_frame_body_A;
        let p_B = body_B.pose * self.local_frame_body_B;

        let r_A = p_A.origin - body_A.pose.origin;
        let r_B = p_B.origin - body_B.pose.origin;

        self.inv_m = d_concat!(
            Mat22::diag([body_A.inv_mass; 2]),
            body_A.inv_inertia,
            Mat22::diag([body_B.inv_mass; 2]),
            body_B.inv_inertia
        );
        
        // C = o_A - o_B = 0
        // J = [ 0, 1, 0, -1 ] in R^1x6
        self.constraint_1d.jacobian = h_concat!(
            Vec2::ZEROS.T(), 
            1.0, 
            Vec2::ZEROS.T(), 
            -1.0
        );

        let c_init = p_A.angle - p_B.angle;
        let erp = 0.2;
        self.constraint_1d.bias = c_init * (erp / dt);

        self.constraint_1d.inv_eff_mass = 1.0 / (self.constraint_1d.jacobian * self.inv_m * self.constraint_1d.jacobian.T()).as_float();
    }

    fn iteration(&mut self, body_A: &mut Body2d, body_B: &mut Body2d, is_pos_iter: bool) {

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

        let ext_dv = v_concat!(
            body_A.ext_force_dv,
            body_A.ext_torque_d𝜔,
            body_B.ext_force_dv,
            body_B.ext_torque_d𝜔
        );
        
        let cons = &self.constraint_1d;

        let jv = (cons.jacobian * (v + dv + ext_dv)).as_float();
        let rhs = if is_pos_iter { -jv - cons.bias } else { -jv };
        let lambda = cons.inv_eff_mass * rhs;
        let impulse = cons.jacobian.T() * lambda;
        dv += self.inv_m * impulse;

        body_A.delta_v = dv.v_slice(0..2, 0).into();
        body_A.delta_𝜔 = Mat11::from(dv.v_slice(2..3, 0)).as_float();
        body_B.delta_v = dv.v_slice(3..5, 0).into();
        body_B.delta_𝜔 = Mat11::from(dv.v_slice(5..6, 0)).as_float();
    }
}


pub struct AngularMotor2d {
    body_A: usize,
    body_B: usize,
    local_frame_body_A: Transform2d,
    local_frame_body_B: Transform2d,
    
    𝜔: f64,
    inv_m: TMat<f64, 6, 6>,
    constraint_1d: Constraint1d,

    // Cons2d: Cons2d,
}

impl AngularMotor2d {

    pub fn new(body_A: usize, body_B: usize, local_frame_body_A: Transform2d, local_frame_body_B: Transform2d, 𝜔: f64) -> Self {
        AngularMotor2d {
            body_A,
            body_B,
            local_frame_body_A,
            local_frame_body_B,

            𝜔,
            inv_m: TMat::ZEROS,
            constraint_1d: Constraint1d::new(),
        }
    }
}

impl Constraint for AngularMotor2d {
    fn body_A_id(&self) -> usize {
        self.body_A
    }
    fn body_B_id(&self) -> usize {
        self.body_B
    }

    fn warm_up(&mut self, body_A: &mut Body2d, body_B: &mut Body2d) {}

    fn setup(&mut self, body_A: &Body2d, body_B: &Body2d, dt: f64) {
        
        let p_A = body_A.pose * self.local_frame_body_A;
        let p_B = body_B.pose * self.local_frame_body_B;

        let r_A = p_A.origin - body_A.pose.origin;
        let r_B = p_B.origin - body_B.pose.origin;

        let angle_diff = p_A.angle - p_B.angle;

        self.inv_m = d_concat!(
            Mat22::diag([body_A.inv_mass; 2]),
            body_A.inv_inertia,
            Mat22::diag([body_B.inv_mass; 2]),
            body_B.inv_inertia
        );
        
        // C = (o_A - o_B) - angle_diff - 𝜔*dt = 0
        // J = [ 0, 1, 0, -1 ] in R^1x6
        self.constraint_1d.jacobian = h_concat!(
            Vec2::ZEROS.T(), 
            1.0, 
            Vec2::ZEROS.T(), 
            -1.0
        );

        let c_init = (p_A.angle - p_B.angle) - angle_diff - self.𝜔*dt;
        let erp = 1.0;
        self.constraint_1d.bias = c_init * (erp / dt);

        self.constraint_1d.inv_eff_mass = 1.0 / (self.constraint_1d.jacobian * self.inv_m * self.constraint_1d.jacobian.T()).as_float();
    }

    fn iteration(&mut self, body_A: &mut Body2d, body_B: &mut Body2d, is_pos_iter: bool) {

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

        let ext_dv = v_concat!(
            body_A.ext_force_dv,
            body_A.ext_torque_d𝜔,
            body_B.ext_force_dv,
            body_B.ext_torque_d𝜔
        );
        
        let cons = &self.constraint_1d;

        let jv = (cons.jacobian * (v + dv + ext_dv)).as_float();
        let rhs = if is_pos_iter { -jv - cons.bias } else { -jv };
        let lambda = cons.inv_eff_mass * rhs;
        let impulse = cons.jacobian.T() * lambda;
        dv += self.inv_m * impulse;

        body_A.delta_v = dv.v_slice(0..2, 0).into();
        body_A.delta_𝜔 = Mat11::from(dv.v_slice(2..3, 0)).as_float();
        body_B.delta_v = dv.v_slice(3..5, 0).into();
        body_B.delta_𝜔 = Mat11::from(dv.v_slice(5..6, 0)).as_float();
    }
}
