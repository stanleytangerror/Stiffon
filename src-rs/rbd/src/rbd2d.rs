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
use crate::solver2d::*;

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
    pub mass: f64,
    pub inv_mass: f64,
    pub inertia: f64,
    pub inv_inertia: f64,
    pub v: Vec2,
    pub 𝜔: f64,
    pub pose: Transform2d,
    pub geometry: Geometry2d,
    pub f_ext: Vec2,
    pub τ_ext: f64,
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
            pose,
            geometry,
            f_ext: Vec2::ZEROS,
            τ_ext: 0.0,
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

    pub fn pose(&self) -> Transform2d {
        self.pose
    }

    pub fn geometry(&self) -> &Geometry2d {
        &self.geometry
    }
}

pub struct PointJoint2d {
    pub body_A_id: usize,
    pub body_B_id: usize,
    pub local_pos_A: Vec2,
    pub local_pos_B: Vec2,
}

impl PointJoint2d {
    pub fn new(bodies: &Vec<Body2d>, body_A_id: usize, body_B_id: usize, pos_world_A: Vec2, pos_world_B: Vec2) -> Self {
        let body_A = &bodies[body_A_id];
        let body_B = &bodies[body_B_id];
        let local_pos_A = body_A.pose.inv().transform_position(pos_world_A);
        let local_pos_B = body_B.pose.inv().transform_position(pos_world_B);

        PointJoint2d {
            body_A_id,
            body_B_id,
            local_pos_A,
            local_pos_B,
        }
    }
}

pub struct AngularJoint2d {
    pub body_A_id: usize,
    pub body_B_id: usize,
    pub angle_A_minus_B: f64,
}

impl AngularJoint2d {
    pub fn new(bodies: &Vec<Body2d>, body_A_id: usize, body_B_id: usize, angle_A_minus_B_degrees: f64) -> Self {
        AngularJoint2d {
            body_A_id,
            body_B_id,
            angle_A_minus_B: angle_A_minus_B_degrees.to_radians(),
        }
    }
}

pub struct AngularMotor2d {
    pub body_A_id: usize,
    pub body_B_id: usize,
    pub angle_A_minus_B: f64,
    pub torque_max: f64,
    pub 𝜔: f64,
}

impl AngularMotor2d {

    pub fn new(bodies: &Vec<Body2d>, body_A_id: usize, body_B_id: usize, torque_max: f64, 𝜔_degrees: f64) -> Self {
        AngularMotor2d {
            body_A_id,
            body_B_id,
            angle_A_minus_B: 0.0,
            torque_max,
            𝜔: 𝜔_degrees.to_radians(),
        }
    }
}

pub struct AngularLimit2d {
    pub body_A_id: usize,
    pub body_B_id: usize,
    pub angle_min: f64,
    pub angle_max: f64,
}

impl AngularLimit2d {
    pub fn new(bodies: &Vec<Body2d>, body_A_id: usize, body_B_id: usize, angle_min_degrees: f64, angle_max_degrees: f64) -> Self {
        AngularLimit2d {
            body_A_id,
            body_B_id,
            angle_min: angle_min_degrees.to_radians(),
            angle_max: angle_max_degrees.to_radians(),
        }
    }
}

pub struct PrismaticJoint2d {
    pub body_A_id: usize,
    pub body_B_id: usize,
    pub local_pos_A: Vec2,
    pub local_pos_B: Vec2,
    pub local_angle_A: f64,
    pub local_angle_B: f64,

    pub has_motor: bool,
    pub disp_A_minus_B: f64,
    pub force_max: Option<f64>,
    pub v: Option<f64>,

    pub has_limit: bool,
    pub dist_min: Option<f64>,
    pub dist_max: Option<f64>,
}

impl PrismaticJoint2d {
    pub fn new(bodies: &Vec<Body2d>, 
        body_A_id: usize, body_B_id: usize, 
        pos_world_A: Vec2, pos_world_B: Vec2, 
        dir_world_A: Vec2, angle_A_minus_B_degrees: f64,
        has_motor: bool, force_max: Option<f64>, v: Option<f64>,
        has_limit: bool, dist_min: Option<f64>, dist_max: Option<f64>,
    ) -> Self {
        let body_A = &bodies[body_A_id];
        let body_B = &bodies[body_B_id];

        let local_angle_A = body_A.pose.inv().transform_vector(dir_world_A).angle();
        let local_angle_B = local_angle_A - angle_A_minus_B_degrees.to_radians();
        let local_pos_A = body_A.pose.inv().transform_position(pos_world_A);
        let local_pos_B = body_B.pose.inv().transform_position(pos_world_B);

        PrismaticJoint2d {
            body_A_id,
            body_B_id,
            local_pos_A,
            local_pos_B,
            local_angle_A,
            local_angle_B,
            has_motor,
            disp_A_minus_B: 0.0,
            force_max,
            v,
            has_limit,
            dist_min,
            dist_max,
        }
    }
}

pub struct RevoluteJoint2d {
    pub body_A_id: usize,
    pub body_B_id: usize,
    pub local_pos_A: Vec2,
    pub local_pos_B: Vec2,

    pub has_motor: bool,
    pub torque_max: Option<f64>,
    pub 𝜔: Option<f64>,
    pub angle_A_minus_B: f64,

    pub has_limit: bool,
    pub angle_min: Option<f64>,
    pub angle_max: Option<f64>,
}

impl RevoluteJoint2d {
    pub fn new(bodies: &Vec<Body2d>, 
        body_A_id: usize, body_B_id: usize, 
        pos_world_A: Vec2, pos_world_B: Vec2, 
        has_motor: bool, torque_max: Option<f64>, 𝜔_degrees: Option<f64>,
        has_limit: bool, angle_min_degrees: Option<f64>, angle_max_degrees: Option<f64>
    ) -> Self {
        let body_A = &bodies[body_A_id];
        let body_B = &bodies[body_B_id];

        let local_pos_A = body_A.pose.inv().transform_position(pos_world_A);
        let local_pos_B = body_B.pose.inv().transform_position(pos_world_B);

        RevoluteJoint2d {
            body_A_id,
            body_B_id,
            local_pos_A,
            local_pos_B,

            has_motor,
            torque_max,
            𝜔: if let Some(𝜔_degrees) = 𝜔_degrees { Some(𝜔_degrees.to_radians()) } else { None },
            angle_A_minus_B: 0.0,

            has_limit,
            angle_min: if let Some(angle_min_degrees) = angle_min_degrees { Some(angle_min_degrees.to_radians()) } else { None },
            angle_max: if let Some(angle_max_degrees) = angle_max_degrees { Some(angle_max_degrees.to_radians()) } else { None },
        }
    }
}

pub struct Scene2d {
    pub bodies: Vec<Body2d>,
    gravity: Vec2,
    constraints: Vec<Box<dyn ConstraintBasic>>,
    solver: SequentialImpulseSolver2d,
}

impl Scene2d {
    pub fn new() -> Self {
        Scene2d {
            bodies: Vec::new(),
            gravity: mvec!(0.0, -9.8),
            constraints: Vec::new(),
            solver: SequentialImpulseSolver2d::new(),
        }
    }

    pub fn set_iteration_count(&mut self, pos_iter_count: usize, vel_iter_count: usize) {
    //     self.pos_iter_count = pos_iter_count;
    //     self.vel_iter_count = vel_iter_count;
    }

    pub fn add_body(&mut self, body: Body2d) -> usize {
        let index = self.bodies.len();
        self.bodies.push(body);
        index
    }

    pub fn add_point_joint(&mut self, body_A_id: usize, body_B_id: usize, pos_world_A: Vec2, pos_world_B: Vec2) -> usize {
        let index = self.constraints.len();
        self.constraints.push(Box::new(PointJoint2d::new(&self.bodies, body_A_id, body_B_id, pos_world_A, pos_world_B)));
        index
    }

    pub fn add_angular_joint(&mut self, body_A_id: usize, body_B_id: usize, angle_degrees: f64) -> usize {
        let index = self.constraints.len();
        let mut constraint = AngularJoint2d::new(&self.bodies, body_A_id, body_B_id, angle_degrees);
        self.constraints.push(Box::new(constraint));
        index
    }

    pub fn add_angular_motor(&mut self, body_A_id: usize, body_B_id: usize, torque_max: f64, 𝜔_degrees: f64) -> usize {
        let index = self.constraints.len();
        let mut constraint = AngularMotor2d::new(&self.bodies, body_A_id, body_B_id, torque_max, 𝜔_degrees);
        self.constraints.push(Box::new(constraint));
        index
    }

    pub fn add_prismatic_joint(&mut self, 
        body_A_id: usize, body_B_id: usize, 
        pos_world_A: Vec2, pos_world_B: Vec2, 
        dir_world_A: Vec2, angle_A_minus_B_degrees: f64,
        has_motor: bool, force_max: Option<f64>, v: Option<f64>,
        has_limit: bool, dist_min: Option<f64>, dist_max: Option<f64>,
    ) -> usize {
        let index = self.constraints.len();
        let mut constraint = PrismaticJoint2d::new(&self.bodies, body_A_id, body_B_id, pos_world_A, pos_world_B, dir_world_A, angle_A_minus_B_degrees, has_motor, force_max, v, has_limit, dist_min, dist_max);
        self.constraints.push(Box::new(constraint));
        index
    }

    pub fn add_revolute_joint(&mut self, 
        body_A_id: usize, body_B_id: usize, 
        pos_world_A: Vec2, pos_world_B: Vec2, 
        has_motor: bool, torque_max: Option<f64>, 𝜔_degrees: Option<f64>,
        has_limit: bool, limit_min_degrees: Option<f64>, limit_max_degrees: Option<f64>,
    ) -> usize {
        let index = self.constraints.len();
        let mut constraint = RevoluteJoint2d::new(&self.bodies, body_A_id, body_B_id, pos_world_A, pos_world_B, has_motor, torque_max, 𝜔_degrees, has_limit, limit_min_degrees, limit_max_degrees);
        self.constraints.push(Box::new(constraint));
        index
    }


    pub fn add_angular_limit(&mut self, body_A_id: usize, body_B_id: usize, angle_min_degrees: f64, angle_max_degrees: f64) -> usize {
        let index = self.constraints.len();
        let mut constraint = AngularLimit2d::new(&self.bodies, body_A_id, body_B_id, angle_min_degrees, angle_max_degrees);
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

        for constraint in &mut self.constraints {
            constraint.step(dt);
        }

        self.solver.solve(&mut self.bodies, &self.constraints, dt);

        for body in &mut self.bodies {
            body.f_ext = Vec2::ZEROS;
            body.τ_ext = 0.0;
        }
    }

    // pub fn step_global(&mut self, dt: f64) {
    //     for body in &mut self.bodies {
    //         body.apply_gravity(self.gravity);
    //     }

    //     self.solver.solve(&mut self.bodies, &self.constraints, dt);
        
    //     // for i in 0..self.bodies.len() {
    //     //     let body = &mut self.bodies[i];
            
    //     //     body.v += delta_v[i];
    //     //     body.𝜔 += delta_𝜔[i];

    //     //     body.pose.origin += body.v * dt;
    //     //     body.pose.angle += body.𝜔 * dt;

    //     //     body.f_ext = Vec2::ZEROS;
    //     //     body.τ_ext = 0.0;
    //     // }
        
    // }
}
