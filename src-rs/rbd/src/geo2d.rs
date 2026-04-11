#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(non_snake_case)]

use std::f64;
use std::fmt;

use crate::math::*;

const EPS: f64 = 1e-12;


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

struct Convex2d {
    ccw_points: Vec<Vec2>,
}

impl Convex2d {
    fn build(points: &[Vec2]) -> Self {
        if points.len() < 3 {
            panic!("Convex2d must have at least 3 points");
        }

        for i in 1..points.len() {
            for j in i+1..points.len() {
                if (points[i] - points[j]).norm() < EPS {
                    panic!("Convex2d points must be unique");
                }
            }
        }

        let pivot = points[0];
        let mut remaining_idx: Vec<usize> = (1..points.len()).collect();
        let dirs = remaining_idx
            .iter()
            .map(|&i| Vec2::unit_x().cross((points[i] - pivot).normalize()))
            .collect::<Vec<f64>>();

        println!("remaining_idx: {:?}", remaining_idx);
        println!("dirs: {:?}", dirs);

        remaining_idx.sort_by(|&i, &j| { dirs[i-1].total_cmp(&dirs[j-1]) });

        println!("remaining_idx: {:?}", remaining_idx);

        let mut ccw_points = remaining_idx.iter().map(|&i| points[i]).collect::<Vec<Vec2>>();
        ccw_points.insert(0, pivot);

        for i in 0..ccw_points.len() {
            let pre = ccw_points[(i + ccw_points.len() - 1) % ccw_points.len()];
            let cur = ccw_points[i];
            let next = ccw_points[(i + 1) % ccw_points.len()];
            if (cur - pre).cross(next - cur) < EPS {
                panic!("Convex2d is not convex");
            }
        }

        Self {
            ccw_points: ccw_points,
        }
    }

    fn len(&self) -> usize {
        self.ccw_points.len()
    }

    fn point(&self, index: usize) -> Vec2 {
        self.ccw_points[(index % self.len() + self.len()) % self.len()]
    }

    fn max_support_point_idx(&self, axis: Vec2, open_interval: &PartialConvex2d) -> Option<usize> {

        let (begin, end) = match *open_interval {
            PartialConvex2d::Full => (0, self.len()),
            PartialConvex2d::OpenInterval { min, max } => {
                let mut b = min + 1;
                let mut e = max;
                while e < b { e += self.len(); }
                (b, e)
            }
        };

        if begin == end {
            return None;
        }

        let mut max_dot = f64::NEG_INFINITY;
        let mut max_index = 0;
        for i in begin..end {
            let point = self.point(i);
            let dot = point.dot(axis);
            if dot > max_dot {
                max_dot = dot;
                max_index = i;
            }
        }
        Some(max_index % self.len())
    }
}

enum PartialConvex2d {
    Full,
    OpenInterval {
        min: usize,
        max: usize,
    },
}

impl PartialConvex2d {
    fn inverse(&self) -> Self {
        match self {
            Self::Full => Self::Full,
            Self::OpenInterval { min, max } => Self::OpenInterval { min: *max, max: *min },
        }
    }

    fn min_bound(&self) -> usize {
        match self {
            Self::Full => 0,
            Self::OpenInterval { min, max } => *min,
        }
    }

    fn max_bound(&self) -> usize {
        match self {
            Self::Full => 0,
            Self::OpenInterval { min, max } => *max,
        }
    }
}

#[derive(Clone, Copy)]
struct MinDiffPoint {
    index_a: usize,
    index_b: usize,
    pos: Vec2,
}

impl fmt::Debug for MinDiffPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MinDiffPoint")
            .field("index_a", &self.index_a)
            .field("index_b", &self.index_b)
            .field("pos", &self.pos)
            .finish()
    }
}

impl fmt::Display for MinDiffPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MinDiffPoint(a={}, b={}, pos=({:.6}, {:.6}))",
            self.index_a,
            self.index_b,
            self.pos.x(),
            self.pos.y()
        )
    }
}

fn convex_gjk(convex_a: &Convex2d, convex_b: &Convex2d) -> Option<[MinDiffPoint; 3]> {
    // first point
    let mut axis = Vec2::unit_x();

    let sp0 = max_support_point_of_minkowski_diff(convex_a, convex_b, 
        PartialConvex2d::Full, PartialConvex2d::Full, 
        -axis);
    if sp0.is_none() {
        panic!("cannot find the first support point");
    }
    let sp0 = sp0.unwrap();

    let sp1 = max_support_point_of_minkowski_diff(convex_a, convex_b, 
        PartialConvex2d::Full, PartialConvex2d::Full, 
        axis);
    if sp1.is_none() {
        panic!("cannot find the second support point");
    }
    let sp1 = sp1.unwrap();

    println!("sp0: {}, sp1: {}", sp0, sp1);

    let to_the_left = (sp1.pos - sp0.pos).cross(-sp0.pos) > 0.0;
    if to_the_left {
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp0, sp1);
    } else {
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp1, sp0);
    }
}

fn convex_gjk_expand_triangle_to_the_left(convex_a: &Convex2d, convex_b: &Convex2d, sp0: MinDiffPoint, sp1: MinDiffPoint) -> Option<[MinDiffPoint; 3]> {

    // origin is to the left of the edge sp0-sp1, i.e., to the left of ray sp0->sp1
    // so the axis is perpendicular to the edge sp0-sp1
    let axis = (sp1.pos - sp0.pos).normalize().rotate(std::f64::consts::PI / 2.0);

    let sp2 = max_support_point_of_minkowski_diff(convex_a, convex_b, 
        // convex points ccw order, so the left part of convex_a is opposite of sp0..sp1
        PartialConvex2d::OpenInterval { min: sp0.index_a, max: sp1.index_a }.inverse(),
        // convex points ccw order, so the right part of convex_b is opposite of sp0..sp1 
        PartialConvex2d::OpenInterval { min: sp0.index_b, max: sp1.index_b }.inverse(),
        axis);
    if sp2.is_none() {
        return None;
    }
    let sp2 = sp2.unwrap();

    println!("expand sp2: {}", sp2);

    // sp0, sp1, sp2 same line check
    if (sp2.pos - sp0.pos).cross(sp1.pos - sp0.pos).abs() < EPS {
        // same line, no intersection
        println!("same line, no intersection");
        return None;
    }

    if (sp2.pos - sp1.pos).cross(-sp1.pos) < 0.0 {
        // origin is outside of edge sp1-sp2
        // i.e., to the right of ray sp1->sp2
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp2, sp1);
    } else if (sp0.pos - sp2.pos).cross(-sp2.pos) < 0.0 {
        // origin is outside of edge sp2-sp0
        // i.e., to the right of ray sp2->sp0
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp0, sp2);
    } else {
        // origin is inside of all three edges, intersection found
        // i.e., to the left of all rays sp0->sp1, sp1->sp2, sp2->sp0
        return Some([sp0, sp1, sp2]);
    }
}

// find the maximum support point of the Minkowski difference between part of the two convex shapes
fn max_support_point_of_minkowski_diff(convex_a: &Convex2d, convex_b: &Convex2d, index_part_a: PartialConvex2d, index_part_b: PartialConvex2d, axis: Vec2) -> Option<MinDiffPoint> {
    let sp_idx_a = convex_a.max_support_point_idx(axis, &index_part_a);
    let sp_idx_b = convex_b.max_support_point_idx(-axis, &index_part_b);

    if sp_idx_a.is_none() && sp_idx_b.is_none() {
        // no more points in minkowski difference
        return None;
    }

    let sp_idx_a = if sp_idx_a.is_none() { index_part_a.min_bound() } else { sp_idx_a.unwrap() };
    let sp_idx_b = if sp_idx_b.is_none() { index_part_b.min_bound() } else { sp_idx_b.unwrap() };

    Some(MinDiffPoint {
        index_a: sp_idx_a,
        index_b: sp_idx_b,
        pos: convex_a.point(sp_idx_a) - convex_b.point(sp_idx_b),
    })
}

fn point_of_minkowski_diff(convex_a: &Convex2d, convex_b: &Convex2d, index_a: usize, index_b: usize) -> Vec2 {
    convex_a.point(index_a) - convex_b.point(index_b)
}

#[cfg(test)]
mod convex_gjk_tests {
    use super::*;
    use crate::{mvec, rbd2d::PointJoint2d};

    fn unit_square_ccw() -> Convex2d {
        Convex2d::build(&[
            mvec!(-1.0, -1.0),
            mvec!(-1.0, 1.0),
            mvec!(1.0, 1.0),
            mvec!(1.0, -1.0),
        ])
    }

    fn triangle_ccw() -> Convex2d {
        Convex2d::build(&[
            mvec!(0.0, 0.0),
            mvec!(4.0, 0.0),
            mvec!(2.0, 3.0),
        ])
    }

    fn assert_terminating_origin_inside(convex_a: &Convex2d, convex_b: &Convex2d, tri: &[MinDiffPoint; 3]) {
        let sp0 = point_of_minkowski_diff(convex_a, convex_b, tri[0].index_a, tri[0].index_b);
        let sp1 = point_of_minkowski_diff(convex_a, convex_b, tri[1].index_a, tri[1].index_b);
        let sp2 = point_of_minkowski_diff(convex_a, convex_b, tri[2].index_a, tri[2].index_b);
        assert!((sp1 - sp0).cross(sp2 - sp1) > 0.0);
        assert!((sp2 - sp1).cross(sp0 - sp2) > 0.0);
        assert!((sp0 - sp2).cross(sp1 - sp0) > 0.0);
    }

    fn assert_support_points_consistent(convex_a: &Convex2d, convex_b: &Convex2d, tri: &[MinDiffPoint; 3]) {
        for p in tri {
            assert!(p.index_a < convex_a.len());
            assert!(p.index_b < convex_b.len());
            let diff = convex_a.point(p.index_a) - convex_b.point(p.index_b);
            assert!(
                (diff - p.pos).norm() < 1e-9,
                "Minkowski point mismatch: {:?} vs {:?}",
                diff,
                p.pos
            );
        }
    }

    #[test]
    fn convex_gjk_identical_squares_returns_triangle() {
        let a = unit_square_ccw();
        let b = unit_square_ccw();
        let tri = convex_gjk(&a, &b);
        assert!(!tri.is_none(), "overlapping shapes should yield a triangle");
        assert_terminating_origin_inside(&a, &b, &tri.unwrap());
    }

    #[test]
    fn convex_gjk_offset_overlapping_squares_returns_triangle() {
        let a = unit_square_ccw();
        let b = Convex2d::build(&[
            mvec!(0.0, 0.0),
            mvec!(0.0, 2.0),
            mvec!(2.0, 2.0),
            mvec!(2.0, 0.0),
        ]);
        let tri = convex_gjk(&a, &b);
        assert!(!tri.is_none(), "overlapping shapes should yield a triangle");
        assert_terminating_origin_inside(&a, &b, &tri.unwrap());
    }

    #[test]
    fn convex_gjk_triangle_vs_square_overlap() {
        let a = triangle_ccw();
        let b = unit_square_ccw();
        let tri = convex_gjk(&a, &b);
        assert!(!tri.is_none(), "triangle and square overlap should yield a triangle");
        assert_terminating_origin_inside(&a, &b, &tri.unwrap());
    }
}
