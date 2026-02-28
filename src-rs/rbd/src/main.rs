#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]

mod math;
mod rbd2d;
mod draw2d;

use crate::math::*;
use crate::rbd2d::*;
use crate::draw2d::Draw2d;
use macroquad::prelude as mq;

#[macroquad::main("rbd2d")]
async fn main() {
    let mut body1 = Body2d::new(
        1.0,
        1.0,
        Vec2::ZEROS,
        0.0,
        Transform2d::IDENTITY,
        Geometry2d::circle(0.5),
    );
    let mut body2 = Body2d::new(
        1.0,
        1.0,
        Vec2::ZEROS,
        0.0,
        Transform2d::IDENTITY,
        Geometry2d::rectangle(mvec!(0.5, 0.5)),
    );
    let mut solver = Solver2d::new();
    solver.add_body(body1);
    solver.add_body(body2);

    let draw2d = Draw2d::new();

    loop {
        solver.step(0.001);
    
        draw2d.draw(&solver);

        mq::next_frame().await;
    }
}