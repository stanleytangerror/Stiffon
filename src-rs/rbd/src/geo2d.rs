#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]
#![allow(non_snake_case)]

use std::cmp::Ordering;
use std::f64;
use std::fmt;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use crate::math::*;

const EPS: f64 = 1e-12;


// --- Geometry ---

#[derive(Clone, Debug)]
pub enum Geometry2d {
    Rectangle { half_extents: Vec2 },
    Circle { radius: f64 },
    Convex { shape: Convex2d },
}

impl Geometry2d {
    pub fn rectangle(half_extents: Vec2) -> Self {
        Geometry2d::Rectangle { half_extents }
    }
    pub fn circle(radius: f64) -> Self {
        Geometry2d::Circle { radius }
    }
    pub fn convex(points: &[Vec2]) -> Self {
        Geometry2d::Convex { shape: Convex2d::build(points) }
    }
}

#[derive(Clone, Debug)]
pub struct Convex2d {
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

        // get the most -x point as the pivot
        let pivot_idx = points
            .iter().enumerate()
            .min_by(|(i, p), (j, q)| p.x().total_cmp(&q.x()))
            .unwrap().0;
        let pivot = points[pivot_idx];
        let remaining = points
            .iter().enumerate()
            .filter(|(i, _)| *i != pivot_idx)
            .map(|(i, p)| *p)
            .collect::<Vec<Vec2>>();

        // sort by sin(theta) from +x to pivot->point direction
        // sin(theta) is incremental from -pi/2 to pi/2, so pivot should be the most -x point
        let mut remaining_idx: Vec<usize> = (0..remaining.len()).collect();
        let dirs = remaining_idx
            .iter()
            .map(|&i| Vec2::unit_x().cross((remaining[i] - pivot).normalize())) // sort by sin(angle)
            .collect::<Vec<f64>>();

        remaining_idx.sort_by(|&i, &j| { dirs[i].total_cmp(&dirs[j]) });

        // create ccw order points
        let mut ccw_points = remaining_idx.iter().map(|&i| remaining[i]).collect::<Vec<Vec2>>();
        ccw_points.insert(0, pivot);

        for i in 0..ccw_points.len() {
            let pre = ccw_points[(i + ccw_points.len() - 1) % ccw_points.len()];
            let cur = ccw_points[i];
            let next = ccw_points[(i + 1) % ccw_points.len()];
            if almost_same_line(pre, cur, next, EPS) || !to_the_left_of_ray(next, pre, cur) {
                panic!("Convex2d is not convex");
            }
        }

        Self {
            ccw_points: ccw_points,
        }
    }

    pub fn n_points(&self) -> usize {
        self.ccw_points.len()
    }

    pub fn point(&self, index: usize) -> Vec2 {
        self.ccw_points[(index % self.n_points() + self.n_points()) % self.n_points()]
    }

    fn max_support_point_idx(&self, axis: Vec2, partial: &PartialConvex2d) -> usize {

        let mut found = false;
        let mut max_dot = f64::NEG_INFINITY;
        let mut max_index = 0;
        self.for_each_point(partial, &mut |i, point| {
            let dot = point.dot(axis);
            if dot > max_dot {
                max_dot = dot;
                max_index = i % self.n_points();
                found = true;
            }
        });

        max_index
    }

    fn for_each_point(&self, partial: &PartialConvex2d, f: &mut dyn FnMut(usize, Vec2)) {
        match partial {
            PartialConvex2d::Full => {
                for i in 0..self.n_points() {
                    f(i, self.point(i));
                }
            }
            PartialConvex2d::LeftSide { from, to } => {
                let mut i = *from;
                f(i, self.point(i));
                loop {
                    i = (i + self.n_points() - 1) % self.n_points();
                    f(i, self.point(i));
                    if i == *to {
                        break;
                    }
                }
            }
            PartialConvex2d::RightSide { from, to } => {
                let mut i = *from;
                f(i, self.point(i));
                loop {
                    i = (i + 1) % self.n_points();
                    f(i, self.point(i));
                    if i == *to {
                        break;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
enum PartialConvex2d {
    Full,
    LeftSide {
        from: usize,
        to: usize,
    },
    RightSide {
        from: usize,
        to: usize,
    },
}

impl PartialConvex2d {
    fn full() -> Self {
        Self::Full
    }

    fn left_side_of(from: usize, to: usize) -> Self {
        Self::LeftSide { from, to }
    }

    fn right_side_of(from: usize, to: usize) -> Self {
        Self::RightSide { from, to }
    }
}

#[derive(Clone, Copy)]
struct MinDiffPoint {
    index_a: usize,
    index_b: usize,
    pos: Vec2,
}

impl PartialEq for MinDiffPoint {
    fn eq(&self, other: &Self) -> bool {
        self.index_a == other.index_a && self.index_b == other.index_b
    }
}

impl Eq for MinDiffPoint {}

impl Hash for MinDiffPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index_a.hash(state);
        self.index_b.hash(state);
    }
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

struct MinDiffTriangle {
    ccw_points: [MinDiffPoint; 3],
}

// find the triangle of Minkowski difference covering the origin
fn convex_gjk(convex_a: &Convex2d, convex_b: &Convex2d) -> Option<MinDiffTriangle> {
    // first point
    let mut axis = Vec2::unit_x();

    let sp0 = max_support_point_of_minkowski_diff(convex_a, convex_b, 
        PartialConvex2d::full(), PartialConvex2d::full(), 
        -axis);

    let sp1 = max_support_point_of_minkowski_diff(convex_a, convex_b, 
        PartialConvex2d::full(), PartialConvex2d::full(), 
        axis);

    println!("sp0: {}, sp1: {}", sp0, sp1);

    let to_the_left = to_the_left_of_ray(Vec2::ZEROS, sp0.pos, sp1.pos);
    if to_the_left {
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp0, sp1);
    } else {
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp1, sp0);
    }
}

#[derive(Debug)]
struct MinDiffEdgeAndDist {
    p0: MinDiffPoint,
    p1: MinDiffPoint,
    dist: f64,
}

impl fmt::Display for MinDiffEdgeAndDist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge(\n\t p0={},\n\t p1={},\n\t dist={:.6})", self.p0, self.p1, self.dist)
    }
}

impl MinDiffEdgeAndDist {
    fn new(p0: &MinDiffPoint, p1: &MinDiffPoint) -> Self {
        Self { p0: *p0, p1: *p1, dist: dist_point_to_line(Vec2::ZEROS, p0.pos, p1.pos) }
    }
}

impl PartialEq for MinDiffEdgeAndDist {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == Ordering::Equal
    }
}

impl Eq for MinDiffEdgeAndDist {}

impl PartialOrd for MinDiffEdgeAndDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinDiffEdgeAndDist {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.total_cmp(&self.dist)
    }
}

// find the closest edge to the origin from the Minkowski difference convex
fn convex_epa(convex_a: &Convex2d, convex_b: &Convex2d, tri: &MinDiffTriangle) -> MinDiffEdgeAndDist {
    let mut heap = BinaryHeap::<MinDiffEdgeAndDist>::from(vec![
        MinDiffEdgeAndDist::new(&tri.ccw_points[0], &tri.ccw_points[1]),
        MinDiffEdgeAndDist::new(&tri.ccw_points[1], &tri.ccw_points[2]),
        MinDiffEdgeAndDist::new(&tri.ccw_points[2], &tri.ccw_points[0]),
    ]);

    let mut visited = HashSet::<MinDiffPoint>::new();
    visited.insert(tri.ccw_points[0]);
    visited.insert(tri.ccw_points[1]);
    visited.insert(tri.ccw_points[2]);

    println!("heap: {:?}", heap);

    loop {
        let closest_edge = heap.pop().unwrap();
        println!("pop: {:?}", closest_edge);

        let p0 = closest_edge.p0;
        let p1 = closest_edge.p1;
        // origin is to the left of all the edges, so to expand new points need to expand to the right of the edge
        let p2 = max_support_point_of_minkowski_diff(convex_a, convex_b, 
            PartialConvex2d::right_side_of(p0.index_a, p1.index_a),
            PartialConvex2d::right_side_of(p0.index_b, p1.index_b),
            (p1.pos - p0.pos).normalize().rotate(-std::f64::consts::PI / 2.0));

        if visited.contains(&p2) {
            // no new point found, no intersection
            return closest_edge;
        }
        
        visited.insert(p2);
        println!("new point: {:?}", p2);

        // origin is to the left of the edge p0->p1, p2 is to the right of the edge p1->p2
        // so we need to add the edge p1->p2 to the heap p0->p2 and p2->p1
        let new_edge = MinDiffEdgeAndDist::new(&p0, &p2);
        println!("new_edge: {:?}", new_edge);
        heap.push(new_edge);
        let new_edge = MinDiffEdgeAndDist::new(&p2, &p1);
        println!("new_edge: {:?}", new_edge);
        heap.push(new_edge);
    }

    panic!("should not reach here");
}

struct Convex2dContact {
    p_a: Vec2,
    p_b: Vec2,
    sp_vec: Vec2,
    index_p0_a: usize,
    index_p1_a: usize,    
    index_p0_b: usize,
    index_p1_b: usize,
    t: f64,
}

fn get_convex_contact_from_minkowski_diff_edge(convex_a: &Convex2d, convex_b: &Convex2d, edge: &MinDiffEdgeAndDist) -> Convex2dContact {
    let d = (-edge.p0.pos).dot(edge.p1.pos - edge.p0.pos);
    let edge_len = (edge.p1.pos - edge.p0.pos).norm();

    if edge_len < EPS {
        panic!("edge is too short");
    }
    if d < -EPS || d > edge_len + EPS {
        panic!("origin is not projected onto the edge");
    }

    let t = d / edge_len;

    let p_a = convex_a.point(edge.p0.index_a) * (1.0 - t) + convex_a.point(edge.p1.index_a) * t;
    let p_b = convex_b.point(edge.p0.index_b) * (1.0 - t) + convex_b.point(edge.p1.index_b) * t;
    let sp_vec = p_a - p_b;

    Convex2dContact {
        p_a: p_a,
        p_b: p_b,
        sp_vec: sp_vec,
        index_p0_a: edge.p0.index_a,
        index_p1_a: edge.p1.index_a,
        index_p0_b: edge.p0.index_b,
        index_p1_b: edge.p1.index_b,
        t: t,
    }
}


fn convex_gjk_expand_triangle_to_the_left(convex_a: &Convex2d, convex_b: &Convex2d, sp0: MinDiffPoint, sp1: MinDiffPoint) -> Option<MinDiffTriangle> {

    // origin is to the left of the edge sp0-sp1, i.e., to the left of ray sp0->sp1
    // so the axis is perpendicular to the edge sp0-sp1
    let axis = (sp1.pos - sp0.pos).normalize().rotate(std::f64::consts::PI / 2.0);

    let sp2 = max_support_point_of_minkowski_diff(convex_a, convex_b, 
        // convex points ccw order, so the left part of convex_a is opposite of sp0..sp1
        PartialConvex2d::left_side_of(sp0.index_a, sp1.index_a),
        // convex points ccw order, so the right part of convex_b is opposite of sp0..sp1 
        PartialConvex2d::left_side_of(sp0.index_b, sp1.index_b),
        axis);

    if sp2 == sp0 || sp2 == sp1 {
        // no new point found, no intersection
        return None;
    }

    println!("expand sp2: {}", sp2);

    if almost_same_line(sp0.pos, sp1.pos, sp2.pos, EPS) {
        // same line, no intersection
        println!("same line, no intersection");
        return None;
    }

    if !to_the_left_of_ray(Vec2::ZEROS, sp1.pos, sp2.pos) {
        // origin is outside of edge sp1-sp2
        // i.e., to the right of ray sp1->sp2
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp2, sp1);
    } else if !to_the_left_of_ray(Vec2::ZEROS, sp2.pos, sp0.pos) {
        // origin is outside of edge sp2-sp0
        // i.e., to the right of ray sp2->sp0
        return convex_gjk_expand_triangle_to_the_left(convex_a, convex_b, sp0, sp2);
    } else {
        // origin is inside of all three edges, intersection found
        // i.e., to the left of all rays sp0->sp1, sp1->sp2, sp2->sp0
        return Some(MinDiffTriangle { ccw_points: [sp0, sp1, sp2] });
    }
}

// find the maximum support point of the Minkowski difference between part of the two convex shapes
fn max_support_point_of_minkowski_diff(convex_a: &Convex2d, convex_b: &Convex2d, index_part_a: PartialConvex2d, index_part_b: PartialConvex2d, axis: Vec2) -> MinDiffPoint {
    let sp_idx_a = convex_a.max_support_point_idx(axis, &index_part_a);
    let sp_idx_b = convex_b.max_support_point_idx(-axis, &index_part_b);

    MinDiffPoint {
        index_a: sp_idx_a,
        index_b: sp_idx_b,
        pos: convex_a.point(sp_idx_a) - convex_b.point(sp_idx_b),
    }
}

fn pos_of_minkowski_diff(convex_a: &Convex2d, convex_b: &Convex2d, index_a: usize, index_b: usize) -> Vec2 {
    convex_a.point(index_a) - convex_b.point(index_b)
}

fn dist_point_to_line(p: Vec2, v0: Vec2, v1: Vec2) -> f64 {
    (p - v0).cross((v1 - v0).normalize()).abs()
}

fn almost_same_line(p0: Vec2, p1: Vec2, p2: Vec2, eps: f64) -> bool {
    (p1 - p0).normalize().cross((p2 - p0).normalize()).abs() < eps
}

fn to_the_left_of_ray(p: Vec2, v0: Vec2, v1: Vec2) -> bool {
    almost_to_the_left_of_ray(p, v0, v1, 0.0)
}

fn almost_to_the_left_of_ray(p: Vec2, v0: Vec2, v1: Vec2, eps: f64) -> bool {
    (v1 - v0).cross(p - v0) > -eps
}


fn contact_convex_convex(convex_a: &Convex2d, convex_b: &Convex2d) -> Option<Convex2dContact> {
    let tri = convex_gjk(convex_a, convex_b);
    if tri.is_none() {
        return None;
    }

    let edge_and_dist = convex_epa(convex_a, convex_b, &tri.unwrap());

    let contact = get_convex_contact_from_minkowski_diff_edge(convex_a, convex_b, &edge_and_dist);

    Some(contact)
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
        let sp0 = pos_of_minkowski_diff(convex_a, convex_b, tri[0].index_a, tri[0].index_b);
        let sp1 = pos_of_minkowski_diff(convex_a, convex_b, tri[1].index_a, tri[1].index_b);
        let sp2 = pos_of_minkowski_diff(convex_a, convex_b, tri[2].index_a, tri[2].index_b);
        assert!(almost_to_the_left_of_ray(Vec2::ZEROS, sp0, sp1, EPS));
        assert!(almost_to_the_left_of_ray(Vec2::ZEROS, sp1, sp2, EPS));
        assert!(almost_to_the_left_of_ray(Vec2::ZEROS, sp2, sp0, EPS));
    }
    
    #[test]
    fn convex_gjk_identical_squares_returns_triangle() {
        let a = unit_square_ccw();
        let b = unit_square_ccw();
        let tri = convex_gjk(&a, &b);
        assert!(!tri.is_none(), "overlapping shapes should yield a triangle");
        assert_terminating_origin_inside(&a, &b, &tri.unwrap().ccw_points);
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
        assert_terminating_origin_inside(&a, &b, &tri.unwrap().ccw_points);
    }

    #[test]
    fn convex_gjk_triangle_vs_square_overlap() {
        let a = triangle_ccw();
        let b = unit_square_ccw();
        let tri = convex_gjk(&a, &b);
        assert!(!tri.is_none(), "triangle and square overlap should yield a triangle");
        assert_terminating_origin_inside(&a, &b, &tri.unwrap().ccw_points);
    }

    #[test]
    fn convex_epa_returns_correct_distance() {
        let a = unit_square_ccw();
        let b = Convex2d::build(&[
            mvec!(0.0, 0.0),
            mvec!(0.0, 2.0),
            mvec!(2.0, 2.0),
            mvec!(2.0, 0.0),
        ]);
        let tri = convex_gjk(&a, &b);
        assert!(!tri.is_none(), "overlapping shapes should yield a triangle");
        let edge_and_dist = convex_epa(&a, &b, &tri.unwrap());
        assert!((edge_and_dist.dist - 1.0).abs() < 1e-9);
    }
}

