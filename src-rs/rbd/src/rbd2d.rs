#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]
#![allow(non_snake_case)]

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

        let mut constraint = EqualConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [Pos1dEqConsFunc{ n_local: Vec2::unit_x() }, Pos1dEqConsFunc{ n_local: Vec2::unit_y() }]);
        self.constraints.push(Box::new(constraint));

        index
    }

    pub fn add_angular_joint(&mut self, body_A_id: usize, body_B_id: usize, angle: f64) -> usize {
        let index = self.constraints.len();
        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];
        let local_frame_body_A = body_A.pose.inv() * Transform2d::new(Vec2::ZEROS, angle.to_radians());
        let local_frame_body_B = body_B.pose.inv() * Transform2d::new(Vec2::ZEROS, 0.0);
        let mut constraint = EqualConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [RotEqConsFunc{}]);
        self.constraints.push(Box::new(constraint));
        index
    }

    pub fn add_angular_motor(&mut self, body_A_id: usize, body_B_id: usize, torque_max: f64, 𝜔: f64) -> usize {
        let index = self.constraints.len();
        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];
        let local_frame_body_A = body_A.pose.inv();
        let local_frame_body_B = body_B.pose.inv();
        let mut constraint = EqualConstraints::<RotMotorConsFunc, 1>::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [RotMotorConsFunc{ torque_max, 𝜔: 𝜔.to_radians() }]);
        self.constraints.push(Box::new(constraint));
        index
    }

    pub fn add_prismatic_joint(&mut self, 
        body_A_id: usize, body_B_id: usize, 
        pos_world_A: Vec2, pos_world_B: Vec2, 
        dir_world_A: Vec2, angle_A_minus_B: f64,
        has_motor: bool, force_max: Option<f64>, v: Option<f64>,
        has_limit: bool, dist_min: Option<f64>, dist_max: Option<f64>,
    ) -> usize {
        let index = self.constraints.len();
        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];

        let angle_local_A = body_A.pose.inv().transform_vector(dir_world_A).angle();
        let angle_local_B = angle_local_A + angle_A_minus_B.to_radians();
        let pos_local_A = body_A.pose.inv().transform_position(pos_world_A);
        let pos_local_B = body_B.pose.inv().transform_position(pos_world_B);
        

        let local_frame_body_A = Transform2d::new(pos_local_A, angle_local_A);
        let local_frame_body_B = Transform2d::new(pos_local_B, angle_local_B);

        self.constraints.push(Box::new(EqualConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [Pos1dEqConsFunc{ n_local: Vec2::unit_y() }])));
        self.constraints.push(Box::new(EqualConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [RotEqConsFunc{}])));
        
        if has_motor {
            self.constraints.push(Box::new(EqualConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [Pos1dMotorFunc{ n_local: Vec2::unit_x(), force_max: force_max.unwrap(), v: v.unwrap() }])));
        }

        if has_limit {
            self.constraints.push(Box::new(InequalConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [Pos1dMinConsFunc{ n_local: Vec2::unit_x(), dist_min: dist_min.unwrap() }])));
            self.constraints.push(Box::new(InequalConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [Pos1dMaxConsFunc{ n_local: Vec2::unit_x(), dist_max: dist_max.unwrap() }])));
        }

        index
    }

    pub fn add_revolute_joint(&mut self, 
        body_A_id: usize, body_B_id: usize, 
        pos_world_A: Vec2, pos_world_B: Vec2, 
        has_motor: bool, torque_max: Option<f64>, 𝜔: Option<f64>,
        has_limit: bool, limit_min: Option<f64>, limit_max: Option<f64>,
    ) -> usize {
        let index = self.constraints.len();
        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];

        let pos_local_A = body_A.pose.inv().transform_position(pos_world_A);
        let pos_local_B = body_B.pose.inv().transform_position(pos_world_B);
        

        let local_frame_body_A = Transform2d::new(pos_local_A, 0.0);
        let local_frame_body_B = Transform2d::new(pos_local_B, 0.0);

        self.constraints.push(Box::new(EqualConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [Pos1dEqConsFunc{ n_local: Vec2::unit_x() }, Pos1dEqConsFunc{ n_local: Vec2::unit_y() }])));
        
        if has_motor {
            self.constraints.push(Box::new(EqualConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [RotMotorConsFunc{ torque_max: torque_max.unwrap(), 𝜔: 𝜔.unwrap().to_radians() }])));
        }
        if has_limit {
            self.constraints.push(Box::new(InequalConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [AngleMinConsFunc{ angle_min: limit_min.unwrap().to_radians() }])));
            self.constraints.push(Box::new(InequalConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [AngleMaxConsFunc{ angle_max: limit_max.unwrap().to_radians() }])));
        }
        index
    }


    pub fn add_angular_limit(&mut self, body_A_id: usize, body_B_id: usize, angle_min: f64, angle_max: f64) -> usize {
        let index = self.constraints.len();
        let body_A = &self.bodies[body_A_id];
        let body_B = &self.bodies[body_B_id];
        let local_frame_body_A = body_A.pose.inv();
        let local_frame_body_B = body_B.pose.inv();
        let mut constraint1 = InequalConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [AngleMinConsFunc{ angle_min: angle_min.to_radians() }]);
        let mut constraint2 = InequalConstraints::new(body_A_id, body_B_id, local_frame_body_A, local_frame_body_B, [AngleMaxConsFunc{ angle_max: angle_max.to_radians() }]);
        self.constraints.push(Box::new(constraint1));
        self.constraints.push(Box::new(constraint2));
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

pub trait EqualConsFunc {
    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6>;
    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64;
    fn max_accum_lambda(&self, dt: f64) -> f64;
}

#[derive(Copy, Clone)]
struct EqualConsData {
    inv_eff_mass: f64,
    jacobian: TMat<f64, 1, 6>,
    bias: f64,
    accum_lambda: f64,
    max_accum_lambda: f64,
    need_resolve: bool,
}

impl EqualConsData {
    pub fn new() -> Self {
        EqualConsData {
            inv_eff_mass: 1.0,
            jacobian: TMat::ZEROS,
            bias: 0.0,
            accum_lambda: 0.0,
            max_accum_lambda: f64::INFINITY,
            need_resolve: true,
        }
    }

    pub fn reset(&mut self) {
        self.inv_eff_mass = 1.0;
        self.jacobian = TMat::ZEROS;
        self.bias = 0.0;
        self.accum_lambda = 0.0;
        self.max_accum_lambda = f64::INFINITY;
        self.need_resolve = true;
    }
}

pub struct EqualConstraints<T: EqualConsFunc, const N: usize> {
    body_A: usize,
    body_B: usize,
    local_frame_body_A: Transform2d,
    local_frame_body_B: Transform2d,
    
    inv_m: TMat<f64, 6, 6>,
    constraint_1d: [EqualConsData; N],
    pub cons_funcs: [T; N],
}

impl <T: EqualConsFunc, const N: usize> EqualConstraints<T, N> {
    pub fn new(body_A: usize, body_B: usize, local_frame_body_A: Transform2d, local_frame_body_B: Transform2d, cons_funcs: [T; N]) -> Self {
        EqualConstraints {
            body_A,
            body_B,
            local_frame_body_A,
            local_frame_body_B,

            inv_m: TMat::ZEROS,
            constraint_1d: [EqualConsData::new(); N],
            cons_funcs,
        }
    }
}

impl <T: EqualConsFunc, const N: usize> Constraint for EqualConstraints<T, N> {
    fn body_A_id(&self) -> usize {
        self.body_A
    }
    fn body_B_id(&self) -> usize {
        self.body_B
    }
    fn warm_up(&mut self, body_A: &mut Body2d, body_B: &mut Body2d) {
        let dv = 
            self.inv_m * self.constraint_1d[0].jacobian.T() * self.constraint_1d[0].accum_lambda +
            self.inv_m * self.constraint_1d[1].jacobian.T() * self.constraint_1d[1].accum_lambda;

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

        for i in 0..N {
            self.constraint_1d[i].reset();
            self.constraint_1d[i].jacobian = self.cons_funcs[i].jacobian(body_A, body_B, p_A, p_B);
            let c_init = self.cons_funcs[i].c_init(body_A, body_B, p_A, p_B, dt);
            let erp = 0.2;
            self.constraint_1d[i].bias = c_init * (erp / dt);
            self.constraint_1d[i].inv_eff_mass = 1.0 / (self.constraint_1d[i].jacobian * self.inv_m * self.constraint_1d[i].jacobian.T()).as_float();
            self.constraint_1d[i].max_accum_lambda = self.cons_funcs[i].max_accum_lambda(dt);
            self.constraint_1d[i].accum_lambda = 0.0;
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
        
        for i in 0..N {
            let cons = &mut self.constraint_1d[i];

            let jv = (cons.jacobian * (v + dv + ext_dv)).as_float();
            let rhs = if is_pos_iter { -jv - cons.bias } else { -jv };
            let lambda = cons.inv_eff_mass * rhs;

            let last_accum_lambda = cons.accum_lambda;
            cons.accum_lambda = (last_accum_lambda + lambda).clamp(-cons.max_accum_lambda, cons.max_accum_lambda);
            let new_lambda = cons.accum_lambda - last_accum_lambda;
            let impulse = cons.jacobian.T() * new_lambda;
    
            dv += self.inv_m * impulse;
        }

        body_A.delta_v = dv.v_slice(0..2, 0).into();
        body_A.delta_𝜔 = Mat11::from(dv.v_slice(2..3, 0)).as_float();
        body_B.delta_v = dv.v_slice(3..5, 0).into();
        body_B.delta_𝜔 = Mat11::from(dv.v_slice(5..6, 0)).as_float();
    }
}


struct Pos1dEqConsFunc {
    pub n_local: Vec2,
}

impl EqualConsFunc for Pos1dEqConsFunc {
    // C = n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) in R
    // J = [ n^T, (r_A x n)^T, -n^T, -(r_B x n)^T ] in R^1x6
    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        let r_A = p_A.origin - body_A.pose.origin;
        let r_B = p_B.origin - body_B.pose.origin;

        let n = p_A.transform_vector(self.n_local);

        h_concat!(
            n.T(), 
            r_A.cross(n), 
            -n.T(), 
            -r_B.cross(n)
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        let n = p_A.transform_vector(self.n_local);
        (n.T() * (p_A.origin - p_B.origin)).as_float()
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        f64::INFINITY
    }
}


struct Pos1dMotorFunc {
    pub n_local: Vec2,
    pub force_max: f64,
    pub v: f64,
}

impl EqualConsFunc for Pos1dMotorFunc {
    // C = n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) - n^T pos_diff - v*dt = 0
    // J = [ n^T, (r_A x n)^T, -n^T, -(r_B x n)^T ]
    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        let r_A = p_A.origin - body_A.pose.origin;
        let r_B = p_B.origin - body_B.pose.origin;

        let n = p_A.transform_vector(self.n_local);

        h_concat!(
            n.T(), 
            r_A.cross(n), 
            -n.T(), 
            -r_B.cross(n)
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        let n = p_A.transform_vector(self.n_local);
        let pos_diff = p_A.origin - p_B.origin;
        (n.T() * (p_A.origin - p_B.origin) - n.T() * pos_diff).as_float() - self.v * dt
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        f64::INFINITY
    }
}


struct RotEqConsFunc {}

impl EqualConsFunc for RotEqConsFunc {

    // C = o_A - o_B = 0
    // J = [ 0, 1, 0, -1 ] in R^1x6

    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        h_concat!(
            Vec2::ZEROS.T(), 
            1.0, 
            Vec2::ZEROS.T(), 
            -1.0
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        p_A.angle - p_B.angle
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        f64::INFINITY
    }
}


struct RotMotorConsFunc {
    torque_max: f64,
    𝜔: f64
}

impl EqualConsFunc for RotMotorConsFunc {

    // C = (o_A - o_B) - angle_diff - 𝜔*dt = 0
    // J = [ 0, 1, 0, -1 ] in R^1x6

    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        h_concat!(
            Vec2::ZEROS.T(), 
            1.0, 
            Vec2::ZEROS.T(), 
            -1.0
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        let angle_diff = p_A.angle - p_B.angle;
        (p_A.angle - p_B.angle) - angle_diff - self.𝜔*dt
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        self.torque_max * dt
    }
}

pub trait InequalConsFunc {
    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6>;
    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64;
    fn max_accum_lambda(&self, dt: f64) -> f64;
}

#[derive(Copy, Clone)]
struct InequalConsData {
    inv_eff_mass: f64,
    jacobian: TMat<f64, 1, 6>,
    bias: f64,
    accum_lambda: f64,
    max_accum_lambda: f64,
    need_resolve: bool,
}

impl InequalConsData {
    pub fn new() -> Self {
        InequalConsData {
            inv_eff_mass: 1.0,
            jacobian: TMat::ZEROS,
            bias: 0.0,
            accum_lambda: 0.0,
            max_accum_lambda: f64::INFINITY,
            need_resolve: true,
        }
    }

    pub fn reset(&mut self) {
        self.inv_eff_mass = 1.0;
        self.jacobian = TMat::ZEROS;
        self.bias = 0.0;
        self.accum_lambda = 0.0;
        self.max_accum_lambda = f64::INFINITY;
        self.need_resolve = true;
    }
}

pub struct InequalConstraints<T: InequalConsFunc, const N: usize> {
    body_A: usize,
    body_B: usize,
    local_frame_body_A: Transform2d,
    local_frame_body_B: Transform2d,
    
    inv_m: TMat<f64, 6, 6>,
    constraint_1d: [InequalConsData; N],
    pub cons_funcs: [T; N],
}

impl <T: InequalConsFunc, const N: usize> InequalConstraints<T, N> {
    pub fn new(body_A: usize, body_B: usize, local_frame_body_A: Transform2d, local_frame_body_B: Transform2d, cons_funcs: [T; N]) -> Self {
        InequalConstraints {
            body_A,
            body_B,
            local_frame_body_A,
            local_frame_body_B,

            inv_m: TMat::ZEROS,
            constraint_1d: [InequalConsData::new(); N],
            cons_funcs,
        }
    }
}

impl <T: InequalConsFunc, const N: usize> Constraint for InequalConstraints<T, N> {
    fn body_A_id(&self) -> usize {
        self.body_A
    }
    fn body_B_id(&self) -> usize {
        self.body_B
    }
    fn warm_up(&mut self, body_A: &mut Body2d, body_B: &mut Body2d) {
        let dv = 
            self.inv_m * self.constraint_1d[0].jacobian.T() * self.constraint_1d[0].accum_lambda +
            self.inv_m * self.constraint_1d[1].jacobian.T() * self.constraint_1d[1].accum_lambda;

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

        for i in 0..N {
            self.constraint_1d[i].reset();
            self.constraint_1d[i].jacobian = self.cons_funcs[i].jacobian(body_A, body_B, p_A, p_B);
            let c_init = self.cons_funcs[i].c_init(body_A, body_B, p_A, p_B, dt);
            if c_init >= 0.0 {
                self.constraint_1d[i].need_resolve = false;
                continue;
            }

            let erp = 0.2;
            self.constraint_1d[i].bias = c_init * (erp / dt);
            self.constraint_1d[i].inv_eff_mass = 1.0 / (self.constraint_1d[i].jacobian * self.inv_m * self.constraint_1d[i].jacobian.T()).as_float();
            self.constraint_1d[i].max_accum_lambda = self.cons_funcs[i].max_accum_lambda(dt);
            self.constraint_1d[i].accum_lambda = 0.0;
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
        
        for i in 0..N {
            let cons = &mut self.constraint_1d[i];

            if !cons.need_resolve {
                continue;
            }

            let jv = (cons.jacobian * (v + dv + ext_dv)).as_float();
            let rhs = if is_pos_iter { -jv - cons.bias } else { -jv };
            let lambda = cons.inv_eff_mass * rhs;

            let last_accum_lambda = cons.accum_lambda;
            cons.accum_lambda = (last_accum_lambda + lambda).clamp(-cons.max_accum_lambda, cons.max_accum_lambda);
            let new_lambda = cons.accum_lambda - last_accum_lambda;
            let impulse = cons.jacobian.T() * new_lambda;
    
            dv += self.inv_m * impulse;
        }

        body_A.delta_v = dv.v_slice(0..2, 0).into();
        body_A.delta_𝜔 = Mat11::from(dv.v_slice(2..3, 0)).as_float();
        body_B.delta_v = dv.v_slice(3..5, 0).into();
        body_B.delta_𝜔 = Mat11::from(dv.v_slice(5..6, 0)).as_float();
    }
}


struct Pos1dMinConsFunc {
    n_local: Vec2,
    dist_min: f64,
}

impl InequalConsFunc for Pos1dMinConsFunc {
    // C = n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) - dist_min >= 0
    // J = [ n^T, (r_A x n)^T, -n^T, -(r_B x n)^T ]
    
    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        let r_A = p_A.origin - body_A.pose.origin;
        let r_B = p_B.origin - body_B.pose.origin;

        let n = p_A.transform_vector(self.n_local);

        h_concat!(
            n.T(), 
            r_A.cross(n), 
            -n.T(), 
            -r_B.cross(n)
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        let n = p_A.transform_vector(self.n_local);
        (n.T() * (p_A.origin - p_B.origin)).as_float() - self.dist_min
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        f64::INFINITY
    }
}

struct Pos1dMaxConsFunc {
    n_local: Vec2,
    dist_max: f64,
}

impl InequalConsFunc for Pos1dMaxConsFunc {
    // C = dist_min - n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) >= 0
    // J = [ -n^T, -(r_A x n)^T, n^T, (r_B x n)^T ]
    
    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        let r_A = p_A.origin - body_A.pose.origin;
        let r_B = p_B.origin - body_B.pose.origin;

        let n = p_A.transform_vector(self.n_local);

        h_concat!(
            -n.T(), 
            -r_A.cross(n), 
            n.T(), 
            r_B.cross(n)
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        let n = p_A.transform_vector(self.n_local);
        self.dist_max - (n.T() * (p_A.origin - p_B.origin)).as_float()
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        f64::INFINITY
    }
}


struct AngleMinConsFunc {
    angle_min: f64,
}

impl InequalConsFunc for AngleMinConsFunc {
    // C = (o_A - o_B) - angle_min >= 0
    // J = [ 0, 1, 0, -1 ] in R^1x6
    
    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        h_concat!(
            Vec2::ZEROS.T(), 
            1.0, 
            Vec2::ZEROS.T(), 
            -1.0
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        (p_A.angle - p_B.angle) - self.angle_min
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        f64::INFINITY
    }
}

struct AngleMaxConsFunc {
    angle_max: f64,
}

impl InequalConsFunc for AngleMaxConsFunc {
    // C = angle_max - (o_A - o_B) >= 0
    // J = [ 0, -1, 0, 1 ] in R^1x6

    fn jacobian(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d) -> TMat<f64, 1, 6> {
        h_concat!(
            Vec2::ZEROS.T(), 
            -1.0, 
            Vec2::ZEROS.T(), 
            1.0
        )
    }

    fn c_init(&self, body_A: &Body2d, body_B: &Body2d, p_A: Transform2d, p_B: Transform2d, dt: f64) -> f64 {
        self.angle_max - (p_A.angle - p_B.angle)
    }

    fn max_accum_lambda(&self, dt: f64) -> f64 {
        f64::INFINITY
    }
}
