import numpy as np
from math_utils import Vec3, Transform, normalized

class Box:
    def __init__(self, half_extents: Vec3):
        self.half_extents = half_extents

class Sphere:
    def __init__(self, radius: float):
        self.radius = radius

class Plane:
    def __init__(self, normal: Vec3, distance: float):
        self.normal = normalized(normal)
        self.distance = distance

class Shape:
    def __init__(self, geometry: Box | Sphere | Plane, transform: Transform):
        self.geometry = geometry
        self.transform = transform

class IntersectionResult:
    def __init__(self, intersects: bool, point_A: Vec3, point_B: Vec3, normal: Vec3):
        self.intersects = intersects
        self.point_A = point_A
        self.point_B = point_B
        self.normal = normal

def intersect_sphere_sphere(sphere1: Sphere, transform1: Transform, sphere2: Sphere, transform2: Transform):
    """
    Sphere-sphere intersection
    Returns intersection result with contact points and normals
    """
    # Calculate centers in world space
    center1 = transform1.origin
    center2 = transform2.origin
    
    # Calculate distance between centers
    distance = np.linalg.norm(center2 - center1)
    min_distance = sphere1.radius + sphere2.radius
    
    if distance > min_distance:
        return IntersectionResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    if distance < 1e-6:
        # Spheres are coincident, use arbitrary normal
        normal = Vec3(0, 0, 1)
        # Two points on the axis (same point when coincident)
        point_A = center1
        point_B = center1
    else:
        # Calculate normal (direction from center1 to center2)
        normal_vec = (center2 - center1) / distance
        normal = Vec3(normal_vec[0], normal_vec[1], normal_vec[2])
        
        # Calculate two points on the axis line
        # point_A is on sphere1's surface in the direction of sphere2
        point_A_vec = center1 + sphere1.radius * normal
        point_A = Vec3(point_A_vec[0], point_A_vec[1], point_A_vec[2])
        
        # point_B is on sphere2's surface in the direction of sphere1
        point_B_vec = center2 - sphere2.radius * normal
        point_B = Vec3(point_B_vec[0], point_B_vec[1], point_B_vec[2])
    
    return IntersectionResult(True, point_A, point_B, normal)

def intersect_sphere_plane(sphere: Sphere, transform: Transform, plane: Plane):
    """
    Sphere-plane intersection
    Returns intersection result with contact points and normals
    """
    # Calculate sphere center in world space
    sphere_center = transform.origin
    
    # Calculate signed distance from sphere center to plane
    # plane.normal should be unit vector, plane.distance is distance from origin
    signed_distance = np.dot(sphere_center, plane.normal) - plane.distance
    
    # Check if sphere intersects with plane
    if abs(signed_distance) > sphere.radius:
        # No intersection
        return IntersectionResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Calculate intersection point (closest point on sphere to plane)
    if signed_distance > 0:
        # Sphere is on positive side of plane
        intersection_point_vec = sphere_center - sphere.radius * plane.normal
        normal = Vec3(-plane.normal[0], -plane.normal[1], -plane.normal[2])  # Normal pointing from plane to sphere
    else:
        # Sphere is on negative side of plane
        intersection_point_vec = sphere_center + sphere.radius * plane.normal
        normal = Vec3(plane.normal[0], plane.normal[1], plane.normal[2])  # Normal pointing from plane to sphere
    
    intersection_point = Vec3(intersection_point_vec[0], intersection_point_vec[1], intersection_point_vec[2])
    
    # For sphere-plane intersection, both contact points are the same (the intersection point)
    point_A = intersection_point
    point_B = intersection_point
    
    return IntersectionResult(True, point_A, point_B, normal)

def intersect(shape1: Shape, shape2: Shape):
    if isinstance(shape1.geometry, Sphere) and isinstance(shape2.geometry, Sphere):
        return intersect_sphere_sphere(shape1.geometry, shape1.transform, shape2.geometry, shape2.transform)
    elif isinstance(shape1.geometry, Sphere) and isinstance(shape2.geometry, Plane):
        return intersect_sphere_plane(shape1.geometry, shape1.transform, shape2.geometry)
    elif isinstance(shape1.geometry, Plane) and isinstance(shape2.geometry, Sphere):
        return intersect_sphere_plane(shape2.geometry, shape2.transform, shape1.geometry)
    elif isinstance(shape1.geometry, Plane) and isinstance(shape2.geometry, Plane):
        return IntersectionResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0))
    else:
        raise NotImplementedError(f"Intersection between {type(shape1.geometry)} and {type(shape2.geometry)} not implemented")