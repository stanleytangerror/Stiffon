#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(unused)]
#![allow(non_snake_case)]

use std::f64;
use crate::math::*;
use crate::mvec;
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