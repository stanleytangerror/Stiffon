#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]
#![feature(generic_const_exprs)]
#[allow(incomplete_features)]

mod math;
mod rbd2d;
mod draw2d;
mod global_solver_2d;

use std::f64::INFINITY;

use crate::math::*;
use crate::rbd2d::*;
use crate::draw2d::Draw2d;
use macroquad::prelude as mq;

trait Rbd2dSample {
    fn setup(&self, solver: &mut Scene2d);
    fn step(&self, solver: &mut Scene2d, dt: f64);
}

struct PointJoint2dSample {
}

impl Rbd2dSample for PointJoint2dSample {
    fn setup(&self, solver: &mut Scene2d) {
        let body1 = solver.add_body(Body2d::new(
            INFINITY,
            INFINITY,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(Vec2::ZEROS, 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
        
        let body2 = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(mvec!(3.0, 0.0), 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
    
        let cons1 = solver.add_point_joint(body1, body2, mvec!(1.5, 0.0), mvec!(1.5, 0.0));
    }

    fn step(&self, solver: &mut Scene2d, dt: f64) {
        solver.step_gs(dt);
    }
}

struct AngularJoint2dSample {
}

impl Rbd2dSample for AngularJoint2dSample {
    fn setup(&self, solver: &mut Scene2d) {
        let body1 = solver.add_body(Body2d::new(
            INFINITY,
            INFINITY,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(Vec2::ZEROS, 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
        
        let body2 = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(mvec!(3.0, 0.0), 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
    
        let cons1 = solver.add_point_joint(body1, body2, mvec!(1.5, 0.0), mvec!(1.5, 0.0));
        let cons2 = solver.add_angular_joint(body1, body2, 90.0);
    }

    fn step(&self, solver: &mut Scene2d, dt: f64) {
        solver.step_gs(dt);
    }
}


struct AngularMotor2dSample {
}

impl Rbd2dSample for AngularMotor2dSample {
    fn setup(&self, solver: &mut Scene2d) {
        let body1 = solver.add_body(Body2d::new(
            INFINITY,
            INFINITY,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(Vec2::ZEROS, 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
        
        let body2 = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(mvec!(3.0, 0.0), 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
    
        let cons1 = solver.add_point_joint(body1, body2, mvec!(1.5, 0.0), mvec!(1.5, 0.0));
        let cons2 = solver.add_angular_motor(body1, body2, 500.0,  -100.0);
    }

    fn step(&self, solver: &mut Scene2d, dt: f64) {
        solver.step_gs(dt);
    }
}

struct AngularLimit2dSample {
}

impl Rbd2dSample for AngularLimit2dSample {
    fn setup(&self, solver: &mut Scene2d) {
        let body1 = solver.add_body(Body2d::new(
            INFINITY,
            INFINITY,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(Vec2::ZEROS, 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
        
        let body2 = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(mvec!(3.0, 0.0), 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
    
        let cons1 = solver.add_point_joint(body1, body2, mvec!(1.5, 0.0), mvec!(1.5, 0.0));
        let cons2 = solver.add_angular_motor(body1, body2, 1000.0, 100.0);
        let cons2 = solver.add_angular_limit(body1, body2, -60.0, 30.0);
    }

    fn step(&self, solver: &mut Scene2d, dt: f64) {
        solver.step_gs(dt);
    }
}


struct Prismatic2dSample {
}

impl Rbd2dSample for Prismatic2dSample {
    fn setup(&self, solver: &mut Scene2d) {
        let body1 = solver.add_body(Body2d::new(
            INFINITY,
            INFINITY,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(Vec2::ZEROS, 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
        
        let body2 = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(mvec!(3.0, 0.0), 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
    
        solver.add_prismatic_joint(
            body1, body2, 
            mvec!(1.5, 0.0), mvec!(1.5, 0.0), 
            mvec!(1.0, 0.4), 0.0,
            true, Some(100.0), Some(-3.0), 
            true, Some(-1.0), Some(1.0)
        );
    }

    fn step(&self, solver: &mut Scene2d, dt: f64) {
        solver.step_gs(dt);
    }
}


struct Revolute2dSample {
}

impl Rbd2dSample for Revolute2dSample {
    fn setup(&self, solver: &mut Scene2d) {
        let body1 = solver.add_body(Body2d::new(
            INFINITY,
            INFINITY,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(Vec2::ZEROS, 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
        
        let body2 = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::new(mvec!(3.0, 0.0), 0.0),
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
    
        solver.add_revolute_joint(
            body1, body2, 
            mvec!(1.5, 0.0), mvec!(1.5, 0.0), 
            true, Some(500.0), Some(100.0), 
            true, Some(-40.0), Some(60.0)
        );
    }

    fn step(&self, solver: &mut Scene2d, dt: f64) {
        solver.step_gs(dt);
    }
}


struct Barrier2dSample {
}

impl Rbd2dSample for Barrier2dSample {
    fn setup(&self, solver: &mut Scene2d) {
        solver.set_iteration_count(5, 2);

        let bottom = solver.add_body(Body2d::new(
            INFINITY,
            INFINITY,
            Vec2::ZEROS,
            0.0,
            Transform2d::IDENTITY,
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
        
        let body_right = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::IDENTITY,
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));

        let body_left = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::IDENTITY,
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));
    
        solver.add_prismatic_joint(
            bottom, body_right, 
            mvec!(1.5, 0.0), mvec!(1.5, 0.0), 
            mvec!(1.0, 0.0), 0.0,
            false, Some(100000.0), Some(30.0), 
            true, Some(-1.5), Some(1.5)
        );

        solver.add_prismatic_joint(
            bottom, body_left, 
            mvec!(-1.5, 0.0), mvec!(-1.5, 0.0), 
            mvec!(1.0, 0.0), 0.0,
            false, Some(100000.0), Some(-30.0), 
            true, Some(-1.5), Some(1.5)
        );

        solver.add_point_joint(body_right, body_left, mvec!(0.0, 0.0), mvec!(0.0, 0.0));

        let mut last_left: usize = body_right;
        let mut last_right: usize = body_left;

        for i in 0..2 {
            let next_left: usize = solver.add_body(Body2d::new(
                1.0,
                1.0,
                Vec2::ZEROS,
                0.0,
                Transform2d::IDENTITY,
                Geometry2d::rectangle(mvec!(1.5, 0.2)),
            ));

            let next_right: usize = solver.add_body(Body2d::new(
                1.0,
                1.0,
                Vec2::ZEROS,
                0.0,
                Transform2d::IDENTITY,
                Geometry2d::rectangle(mvec!(1.5, 0.2)),
            ));

            solver.add_point_joint(
                last_right, next_right, 
                mvec!(1.5, 0.0), mvec!(1.5, 0.0), 
            );

            solver.add_point_joint(
                last_left, next_left, 
                mvec!(-1.5, 0.0), mvec!(-1.5, 0.0), 
            );

            solver.add_point_joint(next_left, next_right, mvec!(0.0, 0.0), mvec!(0.0, 0.0));

            last_left = next_right;
            last_right = next_left;
        }

        let top = solver.add_body(Body2d::new(
            1.0,
            1.0,
            Vec2::ZEROS,
            0.0,
            Transform2d::IDENTITY,
            Geometry2d::rectangle(mvec!(1.5, 0.2)),
        ));

        solver.add_prismatic_joint(
            top, last_right, 
            mvec!(1.5, 0.0), mvec!(1.5, 0.0), 
            mvec!(1.0, 0.0), 0.0,
            false, None, None, 
            true, Some(-1.5), Some(1.5)
        );

        solver.add_prismatic_joint(
            top, last_left, 
            mvec!(-1.5, 0.0), mvec!(-1.5, 0.0), 
            mvec!(1.0, 0.0), 0.0,
            false, None, None, 
            true, Some(-1.5), Some(1.5)
        );

    }

    fn step(&self, solver: &mut Scene2d, dt: f64) {
        solver.step_global(dt);
    }
}


#[macroquad::main("rbd2d")]
async fn main() {
    let mut solver = Scene2d::new();
    let sample = Barrier2dSample {};
    sample.setup(&mut solver);

    let mut draw2d = Draw2d::new();
    draw2d.set_camera_width(50.0);

    loop {
        sample.step(&mut solver, 0.01);
    
        draw2d.draw(&solver);

        mq::next_frame().await;
    }
}