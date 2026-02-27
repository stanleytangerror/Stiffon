#![allow(unused)]

use crate::math::*;
use crate::rbd2d::*;
use crate::mvec;
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
    pub fn new(position: Vec2, up: Vec2, width: f64, pixel_size: Vec2) -> Self {
        let angle = up.y().atan2(up.x());
        let transform = Transform2d::new(position, angle);
        let aspect_ratio = pixel_size.x() / pixel_size.y();
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

pub struct Draw2d {
    camera: Camera2d,
    pixel_size: Vec2,
}

impl Draw2d {
    pub fn new() -> Self {
        let pixel_size = mvec!(mq::screen_width() as f64, mq::screen_height() as f64);
        let camera = Camera2d::new(Vec2::ZEROS, Vec2::unit_y(), 10.0, pixel_size);
        Draw2d { camera, pixel_size }
    }

    pub fn draw(&self, solver: &Solver) {
        self.begin_frame();
        // self.draw_test();
        self.draw_solver(solver);
    }

    fn begin_frame(&self) {
        mq::clear_background(mq::BLACK);
    }

    fn draw_test(&self) {
        let rect_x = mq::screen_width() * 0.5 - 50.0;
        let rect_y = mq::screen_height() * 0.5 - 50.0;
        let rect_width = 100.0;
        let rect_height = 80.0;

        mq::draw_rectangle(rect_x, rect_y, rect_width, rect_height, mq::RED);
        mq::draw_rectangle_lines(rect_x, rect_y, rect_width, rect_height, 3.0, mq::WHITE);
        mq::draw_rectangle(rect_x + 150.0, rect_y, rect_width, rect_height, mq::BLUE);
    }

    
    fn draw_solver(&self,solver: &Solver) {
        for body in solver.bodies() {
            let geometry = body.geometry();
            let pose = body.pose();
            self.draw_geometry(geometry, &pose);
        }
    }

    fn draw_geometry(&self, geometry: &Geometry, pose: &Transform2d) {
        match geometry {
            Geometry::Rectangle { half_extents } => {
                let p1 = self.camera.world_to_screen(pose.transform_position(mvec!(-half_extents.x(), -half_extents.y())));
                let p2 = self.camera.world_to_screen(pose.transform_position(mvec!(half_extents.x(), -half_extents.y())));
                let p3 = self.camera.world_to_screen(pose.transform_position(mvec!(half_extents.x(), half_extents.y())));
                let p4 = self.camera.world_to_screen(pose.transform_position(mvec!(-half_extents.x(), half_extents.y())));
                mq::draw_line(p1.x() as f32, p1.y() as f32, p2.x() as f32, p2.y() as f32, 1.0, mq::WHITE);
                mq::draw_line(p2.x() as f32, p2.y() as f32, p3.x() as f32, p3.y() as f32, 1.0, mq::WHITE);
                mq::draw_line(p3.x() as f32, p3.y() as f32, p4.x() as f32, p4.y() as f32, 1.0, mq::WHITE);
                mq::draw_line(p4.x() as f32, p4.y() as f32, p1.x() as f32, p1.y() as f32, 1.0, mq::WHITE);
            }
            Geometry::Circle { radius } => {
                let center = self.camera.world_to_screen(pose.origin);
                mq::draw_circle(center.x() as f32, center.y() as f32, *radius as f32, mq::WHITE);
            }
        }
    }
}


