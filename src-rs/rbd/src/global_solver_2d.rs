#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]
#![allow(non_snake_case)]

use std::f64;
use crate::math::*;
use crate::rbd2d::*;
use crate::mvec;
#[macro_use]
use crate::h_concat;
#[macro_use]
use crate::v_concat;
#[macro_use]
use crate::d_concat;

#[derive(Copy, Clone)]
pub enum Cons2dData {

    Equal {
        body_A_id: usize,
        body_B_id: usize,
        jacobian: TMat<f64, 1, 6>,
        pos_bias: f64,
        vel_bias: f64,
        max_accum_lambda: f64,
    },

    Inequal {    
        body_A_id: usize,
        body_B_id: usize,
        jacobian: TMat<f64, 1, 6>,
        pos_bias: f64,
        vel_bias: f64,
        max_accum_lambda: f64,
        need_resolve: bool,
    },
}

impl Cons2dData {
    pub fn new_equal_data() -> Self {
        Cons2dData::Equal {
            body_A_id: 0,
            body_B_id: 0,
            jacobian: TMat::ZEROS,
            pos_bias: 0.0,
            vel_bias: 0.0,
            max_accum_lambda: f64::INFINITY,
        }
    }

    pub fn new_inequal_data() -> Self {
        Cons2dData::Inequal {
            body_A_id: 0,
            body_B_id: 0,
            jacobian: TMat::ZEROS,
            pos_bias: 0.0,
            vel_bias: 0.0,
            max_accum_lambda: f64::INFINITY,
            need_resolve: true,
        }
    }
}

fn gen_cons_data_pos1d_lock(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, n: Vec2, p_A: Vec2, p_B: Vec2, dt: f64) -> Cons2dData {

    let r_A = p_A - body_A.pose.origin;
    let r_B = p_B - body_B.pose.origin;

    // C = n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) in R
    // J = [ n^T, (r_A x n)^T, -n^T, -(r_B x n)^T ] in R^1x6
    let jacobian = h_concat!(
        n.T(), 
        r_A.cross(n), 
        -n.T(), 
        -r_B.cross(n)
    );
    let c_init = (n.T() * (p_A - p_B)).as_float();

    let erp = 0.2;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = f64::INFINITY;
    let accum_lambda = 0.0;

    Cons2dData::Equal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: 0.0,
        max_accum_lambda: max_accum_lambda,
    }
}

fn gen_cons_data_pos1d_motor(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, n: Vec2, p_A: Vec2, p_B: Vec2, disp_A_miuns_B: f64, force_max: f64, v: f64, dt: f64) -> Cons2dData {

    let r_A = p_A - body_A.pose.origin;
    let r_B = p_B - body_B.pose.origin;

    // C = n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) - n^T pos_diff - v*dt = 0
    // J = [ n^T, (r_A x n)^T, -n^T, -(r_B x n)^T ]
    let jacobian = h_concat!(
        n.T(), 
        r_A.cross(n), 
        -n.T(), 
        -r_B.cross(n)
    );

    let c_init = (n.T() * (p_A - p_B)).as_float() - disp_A_miuns_B - v * dt;

    let erp = 1.0;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = force_max * dt;

    Cons2dData::Equal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: bias,
        max_accum_lambda: max_accum_lambda,
    }
}

fn gen_cons_data_pos1d_limit_min(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, n: Vec2, p_A: Vec2, p_B: Vec2, disp_A_minus_B_min: f64, dt: f64) -> Cons2dData {
    let r_A = p_A - body_A.pose.origin;
    let r_B = p_B - body_B.pose.origin;

    // C = n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) - dist_min >= 0
    // J = [ n^T, (r_A x n)^T, -n^T, -(r_B x n)^T ]
    let jacobian = h_concat!(
        n.T(), 
        r_A.cross(n), 
        -n.T(), 
        -r_B.cross(n)
    );

    let c_init = (n.T() * (p_A - p_B)).as_float() - disp_A_minus_B_min;

    let erp = 0.2;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = f64::INFINITY;
    let need_resolve = c_init < 0.0;

    if need_resolve {
        println!("need_resolve: {}", c_init);
    }

    Cons2dData::Inequal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: 0.0,
        max_accum_lambda: max_accum_lambda,
        need_resolve: need_resolve,
    }
}


fn gen_cons_data_pos1d_limit_max(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, n: Vec2, p_A: Vec2, p_B: Vec2, disp_A_minus_B_max: f64, dt: f64) -> Cons2dData {
    let r_A = p_A - body_A.pose.origin;
    let r_B = p_B - body_B.pose.origin;

    // C = dist_max - n^T (v_A + ω_A × r_A - v_B - ω_B × r_B) >= 0
    // J = [ -n^T, -(r_A x n)^T, n^T, (r_B x n)^T ]
    
    let jacobian = h_concat!(
        -n.T(),
        -r_A.cross(n),
        n.T(),
        r_B.cross(n)
    );

    let c_init = disp_A_minus_B_max - (n.T() * (p_A - p_B)).as_float();

    let erp = 0.2;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = f64::INFINITY;
    let need_resolve = c_init < 0.0;

    if need_resolve {
        println!("need_resolve: {}", c_init);
    }

    Cons2dData::Inequal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: 0.0,
        max_accum_lambda: max_accum_lambda,
        need_resolve: need_resolve,
    }
}


fn gen_cons_data_ang_lock(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, angle_A_minus_B: f64, dt: f64) -> Cons2dData {

    // C = o_A - o_B - o_diff = 0
    // J = [ 0, 1, 0, -1 ] in R^1x6
    let jacobian = h_concat!(
        Vec2::ZEROS.T(), 
        1.0, 
        Vec2::ZEROS.T(), 
        -1.0
    );
    let c_init = body_A.pose.angle - body_B.pose.angle - angle_A_minus_B;

    let erp = 0.2;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = f64::INFINITY;
    let accum_lambda = 0.0;
    
    Cons2dData::Equal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: 0.0,
        max_accum_lambda: max_accum_lambda,
    }
}

fn gen_cons_data_ang_motor(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, angle_A_minus_B: f64, torque_max: f64, 𝜔: f64, dt: f64) -> Cons2dData {

    // C = (o_A - o_B) - angle_diff - 𝜔*dt = 0
    // J = [ 0, 1, 0, -1 ] in R^1x6
    let jacobian = h_concat!(
        Vec2::ZEROS.T(), 
        1.0, 
        Vec2::ZEROS.T(), 
        -1.0
    );

    let c_init = body_A.pose.angle - body_B.pose.angle - angle_A_minus_B - 𝜔*dt;
    let erp = 1.0;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = torque_max * dt; 
    
    Cons2dData::Equal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: bias,
        max_accum_lambda: max_accum_lambda,
    }
}

fn gen_cons_data_ang_limit_min(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, angle_A_minus_B_min: f64, dt: f64) -> Cons2dData {

    // C = (o_A - o_B) - angle_min >= 0
    // J = [ 0, 1, 0, -1 ] in R^1x6
    let jacobian = h_concat!(
        Vec2::ZEROS.T(), 
        1.0, 
        Vec2::ZEROS.T(), 
        -1.0
    );

    let c_init = (body_A.pose.angle - body_B.pose.angle) - angle_A_minus_B_min;

    let erp = 0.2;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = f64::INFINITY;
    let need_resolve = c_init < 0.0;
    
    Cons2dData::Inequal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: bias,
        max_accum_lambda: max_accum_lambda,
        need_resolve: need_resolve,
    }
}

fn gen_cons_data_ang_limit_max(body_A_id: usize, body_B_id: usize, body_A: &Body2d, body_B: &Body2d, angle_A_minus_B_max: f64, dt: f64) -> Cons2dData {

    // C = angle_max - (o_A - o_B) >= 0
    // J = [ 0, 1, 0, -1 ] in R^1x6
    let jacobian = h_concat!(
        Vec2::ZEROS.T(), 
        1.0, 
        Vec2::ZEROS.T(), 
        -1.0
    );

    let c_init = angle_A_minus_B_max - (body_A.pose.angle - body_B.pose.angle);

    let erp = 0.2;
    let bias = c_init * (erp / dt);
    let max_accum_lambda = f64::INFINITY;
    let accum_lambda = 0.0;
    let need_resolve = c_init < 0.0;
    
    Cons2dData::Inequal {
        body_A_id: body_A_id,
        body_B_id: body_B_id,
        jacobian: jacobian,
        pos_bias: bias,
        vel_bias: 0.0,
        max_accum_lambda: max_accum_lambda,
        need_resolve: need_resolve,
    }
}

pub trait ConstraintBasic {
    fn gen_cons_data(&self, bodies: &Vec<Body2d>, dt: f64) -> Vec<Cons2dData>;
    fn step(&mut self, dt: f64);
}

impl ConstraintBasic for PointJoint2d {
    fn gen_cons_data(&self, bodies: &Vec<Body2d>, dt: f64) -> Vec<Cons2dData> {

        let body_A = &bodies[self.body_A_id];
        let body_B = &bodies[self.body_B_id];

        let p_A = body_A.pose.transform_position(self.local_pos_A);
        let p_B = body_B.pose.transform_position(self.local_pos_B);

        let nx = body_A.pose.transform_vector(Vec2::unit_x());
        let ny = body_A.pose.transform_vector(Vec2::unit_y());

        vec![
            gen_cons_data_pos1d_lock(self.body_A_id, self.body_B_id, body_A, body_B, nx, p_A, p_B, dt),
            gen_cons_data_pos1d_lock(self.body_A_id, self.body_B_id, body_A, body_B, ny, p_A, p_B, dt),
        ]
    }

    fn step(&mut self, dt: f64) {}
}

impl ConstraintBasic for AngularJoint2d {
    fn gen_cons_data(&self, bodies: &Vec<Body2d>, dt: f64) -> Vec<Cons2dData> {

        let body_A = &bodies[self.body_A_id];
        let body_B = &bodies[self.body_B_id];

        vec![gen_cons_data_ang_lock(self.body_A_id, self.body_B_id, body_A, body_B, self.angle_A_minus_B, dt)]
    }

    fn step(&mut self, dt: f64) {}
}

impl ConstraintBasic for AngularMotor2d {
    fn gen_cons_data(&self, bodies: &Vec<Body2d>, dt: f64) -> Vec<Cons2dData> {
        let body_A = &bodies[self.body_A_id];
        let body_B = &bodies[self.body_B_id];

        vec![gen_cons_data_ang_motor(self.body_A_id, self.body_B_id, body_A, body_B, self.angle_A_minus_B, self.torque_max, self.𝜔, dt)]
    }

    fn step(&mut self, dt: f64) {
        self.angle_A_minus_B += self.𝜔 * dt;
    }
}

impl ConstraintBasic for AngularLimit2d {
    fn gen_cons_data(&self, bodies: &Vec<Body2d>, dt: f64) -> Vec<Cons2dData> {
        let body_A = &bodies[self.body_A_id];
        let body_B = &bodies[self.body_B_id];

        vec![
            gen_cons_data_ang_limit_min(self.body_A_id, self.body_B_id, body_A, body_B, self.angle_min, dt), 
            gen_cons_data_ang_limit_max(self.body_A_id, self.body_B_id, body_A, body_B, self.angle_max, dt)]
    }

    fn step(&mut self, dt: f64) {}
}

impl ConstraintBasic for PrismaticJoint2d {
    fn gen_cons_data(&self, bodies: &Vec<Body2d>, dt: f64) -> Vec<Cons2dData> {
        let body_A = &bodies[self.body_A_id];
        let body_B = &bodies[self.body_B_id];

        let p_A = body_A.pose.transform_position(self.local_pos_A);
        let p_B = body_B.pose.transform_position(self.local_pos_B);
        let axis_A = body_A.pose.transform_vector(Vec2::from_angle(self.local_angle_A));
        let normal_A = axis_A.rotate(std::f64::consts::PI / 2.0);

        let mut result = Vec::<Cons2dData>::new();

        result.push(gen_cons_data_pos1d_lock(self.body_A_id, self.body_B_id, body_A, body_B, normal_A, p_A, p_B, dt));
        result.push(gen_cons_data_ang_lock(self.body_A_id, self.body_B_id, body_A, body_B, self.local_angle_A - self.local_angle_B, dt));

        if self.has_motor {
            result.push(gen_cons_data_pos1d_motor(self.body_A_id, self.body_B_id, body_A, body_B, axis_A, p_A, p_B, self.disp_A_minus_B, self.force_max.unwrap(), self.v.unwrap(), dt));
        }

        if self.has_limit {
            result.push(gen_cons_data_pos1d_limit_min(self.body_A_id, self.body_B_id, body_A, body_B, axis_A, p_A, p_B, self.dist_min.unwrap(), dt));
            result.push(gen_cons_data_pos1d_limit_max(self.body_A_id, self.body_B_id, body_A, body_B, axis_A, p_A, p_B, self.dist_max.unwrap(), dt));
        }

        result
    }

    fn step(&mut self, dt: f64) {
        if self.has_motor {
            self.disp_A_minus_B += self.v.unwrap() * dt;
        }
    }
}

impl ConstraintBasic for RevoluteJoint2d {
    fn gen_cons_data(&self, bodies: &Vec<Body2d>, dt: f64) -> Vec<Cons2dData> {
        let body_A = &bodies[self.body_A_id];
        let body_B = &bodies[self.body_B_id];

        let p_A = body_A.pose.transform_position(self.local_pos_A);
        let p_B = body_B.pose.transform_position(self.local_pos_B);
        let dir_A = body_A.pose.transform_vector(Vec2::unit_x());
        let dir_B = body_A.pose.transform_vector(Vec2::unit_y());

        let mut result = Vec::<Cons2dData>::new();

        result.push(gen_cons_data_pos1d_lock(self.body_A_id, self.body_B_id, body_A, body_B, dir_A, p_A, p_B, dt));
        result.push(gen_cons_data_pos1d_lock(self.body_A_id, self.body_B_id, body_A, body_B, dir_B, p_A, p_B, dt));

        if self.has_motor {
            result.push(gen_cons_data_ang_motor(self.body_A_id, self.body_B_id, body_A, body_B, self.angle_A_minus_B, self.torque_max.unwrap(), self.𝜔.unwrap(), dt));
        }

        if self.has_limit {
            result.push(gen_cons_data_ang_limit_min(self.body_A_id, self.body_B_id, body_A, body_B, self.angle_min.unwrap(), dt));
            result.push(gen_cons_data_ang_limit_max(self.body_A_id, self.body_B_id, body_A, body_B, self.angle_max.unwrap(), dt));
        }

        result
    }

    fn step(&mut self, dt: f64) {
        if self.has_motor {
            self.angle_A_minus_B += self.𝜔.unwrap() * dt;
        }
    }
}

pub struct SequentialImpulseSolver2d {
    pub pos_iter_count: usize,
    pub vel_iter_count: usize,

    delta_v: Vec<Vec2>,
    delta_𝜔: Vec<f64>,
    accum_lambda: Vec<f64>, // size of constraints
    ext_dv: Vec<Vec2>,
    ext_d𝜔: Vec<f64>,
}

impl SequentialImpulseSolver2d {
    pub fn new() -> Self {
        SequentialImpulseSolver2d { pos_iter_count: 5, vel_iter_count: 2, delta_v: vec![Vec2::ZEROS; 0], delta_𝜔: vec![0.0; 0], accum_lambda: vec![0.0; 0], ext_dv: vec![Vec2::ZEROS; 0], ext_d𝜔: vec![0.0; 0] }
    }

    pub fn pos_iteration(&mut self, bodies: &Vec<Body2d>, constraints: &Vec<Box<dyn ConstraintBasic>>, dt: f64) -> (Vec<Vec2>, Vec<f64>) {

        let mut constraint_data = Vec::<Cons2dData>::new();
        for c in constraints {
            let c_data = c.gen_cons_data(bodies, dt);
            for cd in c_data {
                match cd {
                    Cons2dData::Inequal { need_resolve: false, .. } => {}
                    other => constraint_data.push(other),
                }
            }
        }

        let n_constraints = constraint_data.len();

        self.delta_v = vec![Vec2::ZEROS; bodies.len()];
        self.delta_𝜔 = vec![0.0; bodies.len()];
        self.ext_dv = vec![Vec2::ZEROS; bodies.len()];
        self.ext_d𝜔 = vec![0.0; bodies.len()];
        self.accum_lambda = vec![0.0; n_constraints];

        for i in 0..bodies.len() {
            self.ext_dv[i] = bodies[i].f_ext * bodies[i].inv_mass * dt;
            self.ext_d𝜔[i] = bodies[i].τ_ext * bodies[i].inv_inertia * dt;
        }

        for _ in 0..self.pos_iter_count {
            for (i, c) in constraint_data.iter().enumerate() {
                match c {
                    Cons2dData::Equal { body_A_id, body_B_id, jacobian, pos_bias, max_accum_lambda, .. } => {
                        let body_A = &bodies[*body_A_id];
                        let body_B = &bodies[*body_B_id];

                        let inv_m: TMat<f64, 6, 6> = d_concat!(
                            Mat22::diag([body_A.inv_mass; 2]),
                            body_A.inv_inertia,
                            Mat22::diag([body_B.inv_mass; 2]),
                            body_B.inv_inertia
                        );

                        let v: TMat<f64, _, 1> = v_concat!(
                            body_A.v,
                            body_A.𝜔,
                            body_B.v,
                            body_B.𝜔
                        );

                        let dv = v_concat!(
                            self.delta_v[*body_A_id],
                            self.delta_𝜔[*body_A_id],
                            self.delta_v[*body_B_id],
                            self.delta_𝜔[*body_B_id]
                        );

                        let ext_dv = v_concat!(
                            self.ext_dv[*body_A_id],
                            self.ext_d𝜔[*body_A_id],
                            self.ext_dv[*body_B_id],
                            self.ext_d𝜔[*body_B_id]
                        );

                        let inv_eff_mass = 1.0 / ((*jacobian * inv_m * jacobian.T()).as_float());
                        let rhs = -((*jacobian * (v + dv + ext_dv)).as_float()) - *pos_bias;
                        let lambda = inv_eff_mass * rhs;

                        let last_accum_lambda = self.accum_lambda[i];
                        self.accum_lambda[i] = (self.accum_lambda[i] + lambda).clamp(-*max_accum_lambda, *max_accum_lambda);

                        let clamped_lambda = self.accum_lambda[i] - last_accum_lambda;

                        let impulse = jacobian.T() * clamped_lambda;
                        let d_v = inv_m * impulse;

                        self.delta_v[*body_A_id] += d_v.v_slice::<2>(0, 0);
                        self.delta_𝜔[*body_A_id] += d_v.v(2, 0);
                        self.delta_v[*body_B_id] += d_v.v_slice::<2>(3, 0);
                        self.delta_𝜔[*body_B_id] += d_v.v(5, 0);
                    }
                    Cons2dData::Inequal { need_resolve: true, body_A_id, body_B_id, jacobian, pos_bias, vel_bias, max_accum_lambda, .. } => {
                        let body_A = &bodies[*body_A_id];
                        let body_B = &bodies[*body_B_id];

                        let inv_m = d_concat!(
                            Mat22::diag([body_A.inv_mass; 2]),
                            body_A.inv_inertia,
                            Mat22::diag([body_B.inv_mass; 2]),
                            body_B.inv_inertia
                        );

                        let v = v_concat!(
                            body_A.v,
                            body_A.𝜔,
                            body_B.v,
                            body_B.𝜔
                        );

                        let dv = v_concat!(
                            self.delta_v[*body_A_id],
                            self.delta_𝜔[*body_A_id],
                            self.delta_v[*body_B_id],
                            self.delta_𝜔[*body_B_id]
                        );

                        let ext_dv = v_concat!(
                            self.ext_dv[*body_A_id],
                            self.ext_d𝜔[*body_A_id],
                            self.ext_dv[*body_B_id],
                            self.ext_d𝜔[*body_B_id]
                        );

                        let inv_eff_mass = 1.0 / ((*jacobian * inv_m * jacobian.T()).as_float());
                        let rhs = -((*jacobian * (v + dv + ext_dv)).as_float()) - *pos_bias;
                        let lambda = inv_eff_mass * rhs;

                        let last_accum_lambda = self.accum_lambda[i];
                        self.accum_lambda[i] = (self.accum_lambda[i] + lambda).clamp(-*max_accum_lambda, *max_accum_lambda);

                        let clamped_lambda = self.accum_lambda[i] - last_accum_lambda;

                        let impulse = jacobian.T() * clamped_lambda;
                        let d_v = inv_m * impulse;

                        self.delta_v[*body_A_id] += d_v.v_slice::<2>(0, 0);
                        self.delta_𝜔[*body_A_id] += d_v.v(2, 0);
                        self.delta_v[*body_B_id] += d_v.v_slice::<2>(3, 0);
                        self.delta_𝜔[*body_B_id] += d_v.v(5, 0);
                    }
                    Cons2dData::Inequal { need_resolve: false, .. } => {}
                }
            }
        }

        let mut v_new = vec![Vec2::ZEROS; bodies.len()];
        let mut 𝜔_new = vec![0.0; bodies.len()];
        for i in 0..bodies.len() {
            v_new[i] = bodies[i].v + self.delta_v[i] + self.ext_dv[i];
            𝜔_new[i] = bodies[i].𝜔 + self.delta_𝜔[i] + self.ext_d𝜔[i];
        }

        (v_new, 𝜔_new)
    }

    pub fn vel_iteration(&mut self, bodies: &Vec<Body2d>, constraints: &Vec<Box<dyn ConstraintBasic>>, dt: f64) -> (Vec<Vec2>, Vec<f64>) {

        let mut constraint_data = Vec::<Cons2dData>::new();
        for c in constraints {
            let c_data = c.gen_cons_data(bodies, dt);
            for cd in c_data {
                match cd {
                    Cons2dData::Inequal { need_resolve: false, .. } => {}
                    other => constraint_data.push(other),
                }
            }
        }

        let n_constraints = constraint_data.len();

        self.delta_v = vec![Vec2::ZEROS; bodies.len()];
        self.delta_𝜔 = vec![0.0; bodies.len()];
        self.ext_dv = vec![Vec2::ZEROS; bodies.len()];
        self.ext_d𝜔 = vec![0.0; bodies.len()];
        self.accum_lambda = vec![0.0; n_constraints];

        for i in 0..bodies.len() {
            self.ext_dv[i] = bodies[i].f_ext * bodies[i].inv_mass * dt;
            self.ext_d𝜔[i] = bodies[i].τ_ext * bodies[i].inv_inertia * dt;
        }

        for _ in 0..self.vel_iter_count {
            for (i, c) in constraint_data.iter().enumerate() {
                match c {
                    Cons2dData::Equal { body_A_id, body_B_id, jacobian, vel_bias, max_accum_lambda, .. } => {
                        let body_A = &bodies[*body_A_id];
                        let body_B = &bodies[*body_B_id];

                        let inv_m: TMat<f64, 6, 6> = d_concat!(
                            Mat22::diag([body_A.inv_mass; 2]),
                            body_A.inv_inertia,
                            Mat22::diag([body_B.inv_mass; 2]),
                            body_B.inv_inertia
                        );

                        let v = v_concat!(
                            body_A.v,
                            body_A.𝜔,
                            body_B.v,
                            body_B.𝜔
                        );

                        let dv = v_concat!(
                            self.delta_v[*body_A_id],
                            self.delta_𝜔[*body_A_id],
                            self.delta_v[*body_B_id],
                            self.delta_𝜔[*body_B_id]
                        );

                        let ext_dv = v_concat!(
                            self.ext_dv[*body_A_id],
                            self.ext_d𝜔[*body_A_id],
                            self.ext_dv[*body_B_id],
                            self.ext_d𝜔[*body_B_id]
                        );

                        let inv_eff_mass = 1.0 / ((*jacobian * inv_m * jacobian.T()).as_float());
                        let rhs = -((*jacobian * (v + dv + ext_dv)).as_float()) - *vel_bias;
                        let lambda = inv_eff_mass * rhs;

                        let last_accum_lambda = self.accum_lambda[i];
                        self.accum_lambda[i] = (self.accum_lambda[i] + lambda).clamp(-*max_accum_lambda, *max_accum_lambda);
                        
                        let clamped_lambda = self.accum_lambda[i] - last_accum_lambda;

                        let impulse = jacobian.T() * clamped_lambda;
                        let d_v = inv_m * impulse;

                        self.delta_v[*body_A_id] += d_v.v_slice::<2>(0, 0);
                        self.delta_𝜔[*body_A_id] += d_v.v(2, 0);
                        self.delta_v[*body_B_id] += d_v.v_slice::<2>(3, 0);
                        self.delta_𝜔[*body_B_id] += d_v.v(5, 0);
                    }
                    Cons2dData::Inequal { need_resolve: true, body_A_id, body_B_id, jacobian, vel_bias, max_accum_lambda, .. } => {
                        let body_A = &bodies[*body_A_id];
                        let body_B = &bodies[*body_B_id];

                        let inv_m = d_concat!(
                            Mat22::diag([body_A.inv_mass; 2]),
                            body_A.inv_inertia,
                            Mat22::diag([body_B.inv_mass; 2]),
                            body_B.inv_inertia
                        );

                        let v = v_concat!(
                            body_A.v,
                            body_A.𝜔,
                            body_B.v,
                            body_B.𝜔
                        );

                        let dv = v_concat!(
                            self.delta_v[*body_A_id],
                            self.delta_𝜔[*body_A_id],
                            self.delta_v[*body_B_id],
                            self.delta_𝜔[*body_B_id]
                        );

                        let ext_dv = v_concat!(
                            self.ext_dv[*body_A_id],
                            self.ext_d𝜔[*body_A_id],
                            self.ext_dv[*body_B_id],
                            self.ext_d𝜔[*body_B_id]
                        );

                        let inv_eff_mass = 1.0 / ((*jacobian * inv_m * jacobian.T()).as_float());
                        let rhs = -((*jacobian * (v + dv + ext_dv)).as_float()) - *vel_bias;
                        let lambda = inv_eff_mass * rhs;

                        let last_accum_lambda = self.accum_lambda[i];
                        self.accum_lambda[i] = (self.accum_lambda[i] + lambda).clamp(-*max_accum_lambda, *max_accum_lambda);

                        let clamped_lambda = self.accum_lambda[i] - last_accum_lambda;

                        let impulse = jacobian.T() * clamped_lambda;
                        let d_v = inv_m * impulse;

                        self.delta_v[*body_A_id] += d_v.v_slice::<2>(0, 0);
                        self.delta_𝜔[*body_A_id] += d_v.v(2, 0);
                        self.delta_v[*body_B_id] += d_v.v_slice::<2>(3, 0);
                        self.delta_𝜔[*body_B_id] += d_v.v(5, 0);
                    }
                    Cons2dData::Inequal { need_resolve: false, .. } => {}
                }
            }
        }

        let mut v_new = vec![Vec2::ZEROS; bodies.len()];
        let mut 𝜔_new = vec![0.0; bodies.len()];
        for i in 0..bodies.len() {
            v_new[i] = bodies[i].v + self.delta_v[i] + self.ext_dv[i];
            𝜔_new[i] = bodies[i].𝜔 + self.delta_𝜔[i] + self.ext_d𝜔[i];
        }

        (v_new, 𝜔_new)
    }

    pub fn solve(&mut self, bodies: &mut Vec<Body2d>, constraints: &Vec<Box<dyn ConstraintBasic>>, dt: f64) {
        let (v_new, 𝜔_new) = self.pos_iteration(bodies, constraints, dt);
        for i in 0..bodies.len() {
            bodies[i].pose.origin += v_new[i] * dt;
            bodies[i].pose.angle += 𝜔_new[i] * dt;
        }

        let (v_new, 𝜔_new) = self.vel_iteration(bodies, constraints, dt);
        for i in 0..bodies.len() {
            bodies[i].v = v_new[i];
            bodies[i].𝜔 = 𝜔_new[i];
        }
    }
}

pub struct GlobalImpulseSolver2d {
}

impl GlobalImpulseSolver2d {
    pub fn solve(&mut self, bodies: &Vec<Body2d>, constraints: &Vec<Box<dyn ConstraintBasic>>, dt: f64) -> (Vec<Vec2>, Vec<f64>) {
        let n_bodies = bodies.len();

        let mut dv_ext = TDynMat::<f64>::zeros(n_bodies * 3, 1);

        for i in 0..n_bodies {
            *dv_ext.v_mut(i * 3 + 0, 0) = bodies[i].f_ext.x() * bodies[i].inv_mass * dt;
            *dv_ext.v_mut(i * 3 + 1, 0) = bodies[i].f_ext.y() * bodies[i].inv_mass * dt;
            *dv_ext.v_mut(i * 3 + 2, 0) = bodies[i].τ_ext * bodies[i].inv_inertia * dt;
        }

        let mut constraint_data = Vec::<Cons2dData>::new();
        for c in constraints {
            let c_data = c.gen_cons_data(bodies, dt);
            for cd in c_data {
                match cd {
                    Cons2dData::Inequal { need_resolve: false, .. } => {}
                    other => constraint_data.push(other),
                }
            }
        }
        let n_constraints = constraint_data.len();

        let mut j = TDynMat::<f64>::zeros(n_constraints, 3 * n_bodies);
        let mut inv_m = TDynMat::<f64>::zeros(3 * n_bodies, 3 * n_bodies);
        let mut v = TDynMat::<f64>::zeros(3 * n_bodies, 1);
        let mut b = TDynMat::<f64>::zeros(n_constraints, 1);

        for i in 0..n_bodies {
            let body = &bodies[i];
            *inv_m.v_mut(i * 3, i * 3) = body.inv_mass;
            *inv_m.v_mut(i * 3 + 1, i * 3 + 1) = body.inv_mass;
            *inv_m.v_mut(i * 3 + 2, i * 3 + 2) = body.inv_inertia;

            *v.v_mut(i * 3 + 0, 0) = body.v.x();
            *v.v_mut(i * 3 + 1, 0) = body.v.y();
            *v.v_mut(i * 3 + 2, 0) = body.𝜔;
        }

        for (i, cons_data) in constraint_data.iter().enumerate() {
            let (body_A_id, body_B_id, local_j, pos_bias, vel_bias) = match cons_data {
                Cons2dData::Equal { body_A_id, body_B_id, jacobian, pos_bias, vel_bias, .. }
                | Cons2dData::Inequal { body_A_id, body_B_id, jacobian, pos_bias, vel_bias, .. } => {
                    (*body_A_id, *body_B_id, *jacobian, *pos_bias, *vel_bias)
                }
            };
            *j.v_mut(i, body_A_id * 3 + 0) = local_j.v(0, 0);
            *j.v_mut(i, body_A_id * 3 + 1) = local_j.v(0, 1);
            *j.v_mut(i, body_A_id * 3 + 2) = local_j.v(0, 2);
            *j.v_mut(i, body_B_id * 3 + 0) = local_j.v(0, 3);
            *j.v_mut(i, body_B_id * 3 + 1) = local_j.v(0, 4);
            *j.v_mut(i, body_B_id * 3 + 2) = local_j.v(0, 5);
            *b.v_mut(i, 0) = pos_bias;
        }

        let k = &j * &inv_m * &j.T();
        let b = -(&j * (&v + &dv_ext)) - &b;

        let lambda = solve_gauss_seidel_dyn(k, b, 100, 1e-6);
        let dv = &inv_m * (&j.T() * &lambda) + &dv_ext;

        let mut delta_v = vec![Vec2::ZEROS; n_bodies];
        let mut delta_𝜔 = vec![0.0; n_bodies];

        for i in 0..n_bodies {
            *delta_v[i].x_mut() = dv.v(i*3, 0);
            *delta_v[i].y_mut() = dv.v(i*3+1, 0);
            delta_𝜔[i] = dv.v(i*3+2, 0);
        }

        (delta_v, delta_𝜔)
    }
}

