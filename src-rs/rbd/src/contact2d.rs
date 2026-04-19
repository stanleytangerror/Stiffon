#![allow(uncommon_codepoints)]
#![allow(mixed_script_confusables)]

use crate::geo2d::*;
use crate::math::*;
use crate::rbd2d::*;
use crate::solver2d::*;

/// Per-body geometry in a form used for contact detection: rectangle → convex polygon,
/// convex → same convex, circle → circle.
#[derive(Clone, Debug)]
pub struct GeoCache {
    pub body_id: usize,
    pub shape: GeoCacheShape,
}

#[derive(Clone, Debug)]
pub enum GeoCacheShape {
    Convex(Convex2d),
    Circle{ center: Vec2, circle: Circle2d },
}

impl GeoCache {
    pub fn from_geometry(body_id: usize, geom: &Geometry2d) -> Self {
        let shape = match geom {
            Geometry2d::Rectangle(r) => {
                GeoCacheShape::Convex(Convex2d::build_from_aabb(r.half_extents))
            }
            Geometry2d::Circle(c) => GeoCacheShape::Circle{ center: Vec2::ZEROS, circle: *c },
            Geometry2d::Convex { shape } => GeoCacheShape::Convex(shape.clone()),
        };
        GeoCache { body_id, shape }
    }

    pub fn transform(&self, transform: &Transform2d) -> Self {
        let shape = match &self.shape {
            GeoCacheShape::Convex(c) => GeoCacheShape::Convex(c.transform(transform)),
            GeoCacheShape::Circle{ center, circle } => GeoCacheShape::Circle{ center: transform.transform_position(*center), circle: *circle },
        };
        GeoCache { body_id: self.body_id, shape }
    }
}

#[derive(Clone, Debug)]
pub struct ContactDetect {
    pub geo_caches: Vec<GeoCache>,
}

impl ContactDetect {
    pub fn new() -> Self {
        ContactDetect {
            geo_caches: Vec::new(),
        }
    }

    pub fn on_add_body(&mut self, body_id: usize, geom: &Geometry2d) {
        self.geo_caches.push(GeoCache::from_geometry(body_id, geom));
    }

    pub fn step(&mut self, bodies: &Vec<Body2d>) -> Vec<Box<dyn ConstraintBasic>> {
        let mut geo_caches_world = Vec::new();
        for (i, body) in bodies.iter().enumerate() {
            geo_caches_world.push(self.geo_caches[i].transform(&body.pose));
        }

        let mut contact_constraints = Vec::<Box<dyn ConstraintBasic>>::new();

        for i in 0..self.geo_caches.len() {
            for j in i + 1..self.geo_caches.len() {
                let shape0 = &geo_caches_world[i].shape;
                let shape1 = &geo_caches_world[j].shape;

                match (shape0, shape1) {
                    (GeoCacheShape::Convex(convex_a), GeoCacheShape::Convex(convex_b)) => {
                        let contact = contact_convex_convex(convex_a, convex_b);
                        if let Some(contact) = contact {
                            contact_constraints.push(Box::new(ContactConstraint2d::new(bodies, i, j, contact.p_a, contact.p_b, contact.sp_vec)));
                        }
                    }
                    (GeoCacheShape::Convex(convex), GeoCacheShape::Circle{ center, circle }) => {
                        let contact = contact_circle_convex(&circle, *center, convex);
                        if let Some(contact) = contact {
                            contact_constraints.push(Box::new(ContactConstraint2d::new(bodies, j, i, contact.p_circle, contact.p_convex, -contact.sp_vec)));
                        }
                    }
                    (GeoCacheShape::Circle{ center, circle }, GeoCacheShape::Convex(convex)) => {
                        let contact = contact_circle_convex(&circle, *center, convex);
                        if let Some(contact) = contact {
                            contact_constraints.push(Box::new(ContactConstraint2d::new(bodies, i, j, contact.p_circle, contact.p_convex, contact.sp_vec)));
                        }
                    }
                    (GeoCacheShape::Circle{ center: center0, circle: circle0 }, GeoCacheShape::Circle{ center: center1, circle: circle1 }) => {
                        let contact = contact_circle_circle(&circle0, &circle1, *center0, *center1);
                        if let Some(contact) = contact {
                            contact_constraints.push(Box::new(ContactConstraint2d::new(bodies, i, j, contact.p_a, contact.p_b, contact.sp_vec)));
                        }
                    }
                }
            }
        }

        contact_constraints
    }
}

impl Default for ContactDetect {
    fn default() -> Self {
        Self::new()
    }
}
