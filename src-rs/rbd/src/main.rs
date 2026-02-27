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
    let origin = mvec!(0.0, 0.0);
    let pose = Transform2d::new(origin, 0.0);
    let geometry = Geometry::rectangle(mvec!(0.5, 0.5));
    let mut body = Body2d::new(
        1.0,
        1.0,
        Vec2::ZEROS,
        0.0,
        pose,
        geometry,
    );
    let mut solver = Solver::new();
    solver.add_body(body);

    let draw2d = Draw2d::new();

    loop {
        solver.step(1.0);
    
        draw2d.draw(&solver);

        mq::next_frame().await;
    }
}