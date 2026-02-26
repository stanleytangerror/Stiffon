#![allow(unused)]

use crate::math::*;
use crate::rbd2d::*;
use macroquad::prelude as mq;

pub struct Camera2d {
    // world -> camera
    pub transform: Transform2d,

    // camera -> ndc
    pub aspect_ratio: f64,
    pub width: f64,

    // ndc -> pixel
    pub pixel_size: Vec2,

    world_to_screen_mat: Mat33,
}

impl Camera2d {
    pub fn new(position: Vec2, up: Vec2, aspect_ratio: f64, width: f64, pixel_size: Vec2) -> Self {
        let angle = up.y().atan2(up.x());
        let transform = Transform2d::new(position, angle);
        let mut camera = Camera2d { transform, aspect_ratio, width, pixel_size, world_to_screen_mat: Mat33::eye() };
        camera.on_changed();
        camera
    }

    pub fn world_to_screen(&self, p: Vec2) -> Vec2 {
        let pn = Vec3::new([p.x(), p.y(), 1.0]);
        let pn_ndc = self.world_to_screen_mat * pn;
        Vec2::new([pn_ndc.x() * self.pixel_size.x(), pn_ndc.y() * self.pixel_size.y()])
    }

    fn on_changed(&mut self) {
        let camera_mat = self.transform.as_mat33();
        let proj_mat = proj_mat_2d(self.aspect_ratio, self.width);
        let pixel_mat = Mat33::from_rows([
            [self.pixel_size.x(), 0.0, 0.0],
            [0.0, self.pixel_size.y(), 0.0],
            [0.0, 0.0, 1.0],
        ]);
        self.world_to_screen_mat = pixel_mat * proj_mat * camera_mat;
    }
}

// pub fn draw_solver(solver: &Solver) {
//     for body in solver.bodies() {
//         let geometry = body.geometry();
//         let pose = body.pose();

//         match geometry {
//             Geometry::Rectangle { half_extents } => {
//                 mq::draw_rectangle(pose.origin, half_extents.x, half_extents.y);
//             }
//             Geometry::Circle { radius } => {
//                 mq::draw_circle(pose.origin, radius);
//             }
//         }
//     }
// }
