import numpy as np
from math_utils import Vec3, Transform, normalized

class Box:
    def __init__(self, half_extents: Vec3):
        self.half_extents = half_extents

class Sphere:
    def __init__(self, radius: float):
        self.radius = radius

class Plane:
    def __init__(self, point: Vec3, normal: Vec3):
        self.point = point
        self.normal = normalized(normal)

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

class RaycastResult:
    def __init__(self, hits: bool, point: Vec3, normal: Vec3):
        self.hits = hits
        self.point = point
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

def intersect_sphere_plane(sphere: Sphere, sphere_transform: Transform, plane: Plane, plane_transform: Transform):
    """
    Sphere-plane intersection
    Returns intersection result with contact points and normals
    """
    # Calculate sphere center in world space
    sphere_center = sphere_transform.origin
    plane_normal = plane_transform.transformDirection(plane.normal)
    plane_point = plane_transform.transformPosition(plane.point)
    
    # Calculate signed distance from sphere center to plane
    # plane.normal should be unit vector, plane.point is a point on the plane
    signed_distance = np.dot(sphere_center - plane_point, plane_normal)
    
    # Check if sphere intersects with plane
    if abs(signed_distance) > sphere.radius:
        # No intersection
        return IntersectionResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Calculate intersection point (closest point on sphere to plane)
    if signed_distance > 0:
        # Sphere is on positive side of plane
        intersection_point_sphere = sphere_center - sphere.radius * plane_normal
        intersection_point_plane = sphere_center - signed_distance * plane_normal
        normal = Vec3(-plane_normal[0], -plane_normal[1], -plane_normal[2])  # Normal pointing from plane to sphere
    else:
        # Sphere is on negative side of plane
        intersection_point_sphere = sphere_center + sphere.radius * plane_normal
        intersection_point_plane = sphere_center - signed_distance * plane_normal
        normal = Vec3(plane_normal[0], plane_normal[1], plane_normal[2])  # Normal pointing from plane to sphere
    
    # For sphere-plane intersection, both contact points are the same (the intersection point)
    point_A = intersection_point_sphere
    point_B = intersection_point_plane
    
    return IntersectionResult(True, point_A, point_B, normal)

def intersect(shape1: Shape, shape2: Shape):
    if isinstance(shape1.geometry, Sphere) and isinstance(shape2.geometry, Sphere):
        return intersect_sphere_sphere(shape1.geometry, shape1.transform, shape2.geometry, shape2.transform)
    elif isinstance(shape1.geometry, Sphere) and isinstance(shape2.geometry, Plane):
        return intersect_sphere_plane(shape1.geometry, shape1.transform, shape2.geometry, shape2.transform)
    elif isinstance(shape1.geometry, Plane) and isinstance(shape2.geometry, Sphere):
        result = intersect_sphere_plane(shape2.geometry, shape2.transform, shape1.geometry, shape1.transform)
        return IntersectionResult(result.intersects, result.point_B, result.point_A, -result.normal)
    elif isinstance(shape1.geometry, Plane) and isinstance(shape2.geometry, Plane):
        return IntersectionResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0))
    else:
        return IntersectionResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0))

def raycast_box(box: Box, local_origin: Vec3, local_dir: Vec3):
    """
    Ray-Box intersection using slab method
    Returns raycast result with hit point and normal in local space
    """
    half_extents = box.half_extents
    
    # Calculate inverse direction for faster computation
    inv_dir = Vec3(1.0 / local_dir[0] if abs(local_dir[0]) > 1e-10 else float('inf'),
                   1.0 / local_dir[1] if abs(local_dir[1]) > 1e-10 else float('inf'),
                   1.0 / local_dir[2] if abs(local_dir[2]) > 1e-10 else float('inf'))
    
    # Calculate intersection with each slab
    t_min = (Vec3(-half_extents[0], -half_extents[1], -half_extents[2]) - local_origin) * inv_dir
    t_max = (Vec3(half_extents[0], half_extents[1], half_extents[2]) - local_origin) * inv_dir
    
    # Swap if needed
    t_min_vals = np.minimum(t_min, t_max)
    t_max_vals = np.maximum(t_min, t_max)
    
    # Find the largest t_min and smallest t_max
    t_near = max(t_min_vals[0], t_min_vals[1], t_min_vals[2])
    t_far = min(t_max_vals[0], t_max_vals[1], t_max_vals[2])
    
    if t_near > t_far or t_far < 0:
        return RaycastResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Use the closest intersection point
    t = t_near if t_near >= 0 else t_far
    
    if t < 0:
        return RaycastResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Calculate intersection point in local space
    local_point = local_origin + local_dir * t
    
    # Calculate normal in local space based on which face was hit
    # The face with the largest t_min value is the one that was hit
    normal_local = Vec3(0, 0, 0)
    epsilon = 1e-6
    
    # Check which axis has the largest t_min (with tolerance for floating point errors)
    if abs(t_near - t_min_vals[0]) < epsilon:
        # Hit the x-face
        normal_local = Vec3(-1 if local_dir[0] > 0 else 1, 0, 0)
    elif abs(t_near - t_min_vals[1]) < epsilon:
        # Hit the y-face
        normal_local = Vec3(0, -1 if local_dir[1] > 0 else 1, 0)
    elif abs(t_near - t_min_vals[2]) < epsilon:
        # Hit the z-face
        normal_local = Vec3(0, 0, -1 if local_dir[2] > 0 else 1)
    else:
        # Fallback: determine normal from intersection point position
        # This handles edge/corner cases
        diff = local_point
        abs_diff = np.abs(diff)
        if abs_diff[0] >= abs_diff[1] and abs_diff[0] >= abs_diff[2]:
            normal_local = Vec3(-1 if diff[0] > 0 else 1, 0, 0)
        elif abs_diff[1] >= abs_diff[2]:
            normal_local = Vec3(0, -1 if diff[1] > 0 else 1, 0)
        else:
            normal_local = Vec3(0, 0, -1 if diff[2] > 0 else 1)
    
    return RaycastResult(True, local_point, normal_local)

def raycast_sphere(sphere: Sphere, local_origin: Vec3, local_dir: Vec3):
    """
    Ray-Sphere intersection
    Returns raycast result with hit point and normal in local space
    """
    center = Vec3(0, 0, 0)  # Sphere center in local space
    
    # Vector from ray origin to sphere center
    oc = center - local_origin
    
    # Calculate discriminant
    a = np.dot(local_dir, local_dir)
    b = 2.0 * np.dot(oc, local_dir)
    c = np.dot(oc, oc) - sphere.radius * sphere.radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return RaycastResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    # Find the closest positive intersection
    t = None
    if t1 >= 0 and t2 >= 0:
        t = min(t1, t2)
    elif t1 >= 0:
        t = t1
    elif t2 >= 0:
        t = t2
    else:
        return RaycastResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Calculate intersection point in local space
    local_point = local_origin + local_dir * t
    
    # Calculate normal (from sphere center to intersection point)
    normal_local = normalized(local_point - center)
    
    return RaycastResult(True, local_point, normal_local)

def raycast_plane(plane: Plane, local_origin: Vec3, local_dir: Vec3):
    """
    Ray-Plane intersection
    Returns raycast result with hit point and normal in local space
    """
    plane_point = plane.point  # Point on plane in local space
    plane_normal = plane.normal  # Normal in local space (already normalized)
    
    # Calculate denominator
    denom = np.dot(plane_normal, local_dir)
    
    # Ray is parallel to plane
    if abs(denom) < 1e-10:
        return RaycastResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Calculate intersection parameter
    t = np.dot(plane_point - local_origin, plane_normal) / denom
    
    # Ray doesn't hit plane (intersection is behind the ray origin)
    if t < 0:
        return RaycastResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Calculate intersection point in local space
    local_point = local_origin + local_dir * t
    
    # Normal is the plane normal (already in local space)
    normal_local = plane_normal
    
    return RaycastResult(True, local_point, normal_local)

def raycast(origin: Vec3, dir: Vec3, shape: Shape) -> RaycastResult | None:
    # Transform ray to local space of the shape
    inv_transform = shape.transform.inverse()
    local_origin = inv_transform.transformPosition(origin)
    local_dir = inv_transform.transformDirection(dir)
    local_dir = normalized(local_dir)
    
    # Perform raycast in local space
    result = None
    if isinstance(shape.geometry, Box):
        result = raycast_box(shape.geometry, local_origin, local_dir)
    elif isinstance(shape.geometry, Sphere):
        result = raycast_sphere(shape.geometry, local_origin, local_dir)
    elif isinstance(shape.geometry, Plane):
        result = raycast_plane(shape.geometry, local_origin, local_dir)
    else:
        return RaycastResult(False, Vec3(0, 0, 0), Vec3(0, 0, 0))
    
    # Transform result back to world space
    if result.hits:
        world_point = shape.transform.transformPosition(result.point)
        world_normal = shape.transform.transformDirection(result.normal)
        world_normal = normalized(world_normal)
        return RaycastResult(True, world_point, world_normal)
    else:
        return result