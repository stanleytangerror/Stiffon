#![allow(unused)]

use crate::rbd2d::{Solver, Geometry, Vec2, Transform2d};
use macroquad::prelude as mq;

pub struct Camera2d {
    pub position: Vec2,
    pub right: Vec2,
    pub up: Vec2,

    pub aspect_ratio: f32,
    pub scale: f32,
}

impl Camera2d {
    pub fn new(transform: Transform2d, aspect_ratio: f32, scale: f32) -> Self {
        Camera2d { transform, aspect_ratio, scale }
    }

    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        let screen = self.transform.transform_vector(world);
        screen * self.scale
    }

    fn camera_matrix(&self) -> Mat4 {

    }
}

pub fn draw_solver(solver: &Solver) {
    for body in solver.bodies() {
        let geometry = body.geometry();
        let pose = body.pose();

        match geometry {
            Geometry::Rectangle { half_extents } => {
                mq::draw_rectangle(pose.origin, half_extents.x, half_extents.y);
            }
            Geometry::Circle { radius } => {
                mq::draw_circle(pose.origin, radius);
            }
        }
    }
}
