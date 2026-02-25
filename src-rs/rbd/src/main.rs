#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]

mod math;
mod rbd2d;
mod draw2d;

use rbd2d::{Vec2, Transform2d, Geometry, Body2d, Solver};

fn main() {
    let origin = Vec2::new(0.0, 0.0);
    let pose = Transform2d::new(origin, 0.0);
    let geometry = Geometry::rectangle(Vec2::new(0.5, 0.5));
    let mut body = Body2d::new(
        1.0,
        1.0,
        Vec2::ZERO,
        0.0,
        pose,
        geometry,
    );
    let mut solver = Solver::new();
    solver.add_body(body);

    loop {
        solver.step(1.0);
    }
}
