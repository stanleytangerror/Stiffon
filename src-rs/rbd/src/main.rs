#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]
#![feature(generic_const_exprs)]
#[allow(incomplete_features)]

mod math;
mod rbd2d;
mod draw2d;

use std::f64::INFINITY;

use crate::math::*;
use crate::rbd2d::*;
use crate::draw2d::Draw2d;
use macroquad::prelude as mq;

#[macroquad::main("rbd2d")]
async fn main() {
    let mut solver = Solver2d::new();

    let body1 = solver.add_body(Body2d::new(
        INFINITY,
        INFINITY,
        Vec2::ZEROS,
        0.0,
        Transform2d::new(Vec2::ZEROS, 0.0),
        Geometry2d::circle(0.5),
    ));
    
    let body2 = solver.add_body(Body2d::new(
        1.0,
        1.0,
        Vec2::ZEROS,
        0.0,
        Transform2d::new(mvec!(3.0, 0.0), 0.0),
        Geometry2d::rectangle(mvec!(0.5, 0.5)),
    ));

    let cons1 = solver.add_constraint(BallJoint2d::new(
        body1, body2, 
        Transform2d::new(mvec!(1.5, 0.0), 0.0),
        Transform2d::new(mvec!(-1.5, 0.0), 0.0)));

    let draw2d = Draw2d::new();

    loop {
        solver.step(0.01);
    
        draw2d.draw(&solver);

        mq::next_frame().await;
    }
}