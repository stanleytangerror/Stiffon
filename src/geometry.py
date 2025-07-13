import numpy as np
from math_utils import Vec3, Transform

class Box:
    def __init__(self, half_extents: Vec3):
        self.half_extents = half_extents

class Sphere:
    def __init__(self, radius: float):
        self.radius = radius

class Shape:
    def __init__(self, geometry: Box | Sphere, transform: Transform):
        self.geometry = geometry
        self.transform = transform

class IntersectionResult:
    def __init__(self, intersects: bool, points: list | None = None):
        self.intersects = intersects
        self.points = points if points is not None else []  # List of (position, normal) tuples

def intersect_box_box(box1: Box, transform1: Transform, box2: Box, transform2: Transform):
    """
    Separating Axis Theorem (SAT) for box-box intersection
    Returns intersection result with contact points and normals
    """
    # Transform box2 to box1's local coordinate system
    inv_transform1 = transform1.inverse()
    local_transform = inv_transform1 @ transform2
    
    # Get box2's corners in box1's local space
    box2_corners = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                corner = Vec3(x * box2.half_extents[0], y * box2.half_extents[1], z * box2.half_extents[2])
                transformed_corner = local_transform.origin + local_transform.basis @ corner
                box2_corners.append(transformed_corner)
    
    # Check if any corner of box2 is inside box1
    inside_corners = []
    for corner in box2_corners:
        if (abs(corner[0]) <= box1.half_extents[0] and 
            abs(corner[1]) <= box1.half_extents[1] and 
            abs(corner[2]) <= box1.half_extents[2]):
            inside_corners.append(corner)
    
    if not inside_corners:
        return IntersectionResult(False)
    
    # Find the closest point on box1's surface to each inside corner
    contact_points = []
    for corner in inside_corners:
        # Find the closest point on box1's surface
        closest_point = Vec3(
            np.clip(corner[0], -box1.half_extents[0], box1.half_extents[0]),
            np.clip(corner[1], -box1.half_extents[1], box1.half_extents[1]),
            np.clip(corner[2], -box1.half_extents[2], box1.half_extents[2])
        )
        
        # Calculate normal (direction from closest point to corner)
        normal = corner - closest_point
        if np.linalg.norm(normal) > 1e-6:
            normal = normal / np.linalg.norm(normal)
        else:
            # If corner is exactly on surface, use surface normal
            if abs(corner.x - box1.half_extents.x) < 1e-6:
                normal = Vec3(1, 0, 0)
            elif abs(corner.x + box1.half_extents.x) < 1e-6:
                normal = Vec3(-1, 0, 0)
            elif abs(corner.y - box1.half_extents.y) < 1e-6:
                normal = Vec3(0, 1, 0)
            elif abs(corner.y + box1.half_extents.y) < 1e-6:
                normal = Vec3(0, -1, 0)
            elif abs(corner.z - box1.half_extents.z) < 1e-6:
                normal = Vec3(0, 0, 1)
            else:
                normal = Vec3(0, 0, -1)
        
        # Transform back to world space
        world_point = transform1.origin + transform1.basis @ closest_point
        world_normal = transform1.basis @ normal
        
        contact_points.append((world_point, world_normal))
    
    return IntersectionResult(True, contact_points)

def intersect_box_sphere(box: Box, box_transform: Transform, sphere: Sphere, sphere_transform: Transform):
    """
    Box-sphere intersection using closest point on box to sphere center
    Returns intersection result with contact points and normals
    """
    # Transform sphere center to box's local coordinate system
    inv_box_transform = box_transform.inverse()
    local_sphere_center = inv_box_transform.origin + inv_box_transform.basis @ sphere_transform.origin
    
    # Find closest point on box to sphere center
    closest_point = Vec3(
        np.clip(local_sphere_center.x, -box.half_extents.x, box.half_extents.x),
        np.clip(local_sphere_center.y, -box.half_extents.y, box.half_extents.y),
        np.clip(local_sphere_center.z, -box.half_extents.z, box.half_extents.z)
    )
    
    # Calculate distance from sphere center to closest point
    distance = np.linalg.norm(local_sphere_center - closest_point)
    
    if distance > sphere.radius:
        return IntersectionResult(False)
    
    # Calculate normal (direction from closest point to sphere center)
    normal = local_sphere_center - closest_point
    if np.linalg.norm(normal) > 1e-6:
        normal = normal / np.linalg.norm(normal)
    else:
        # Sphere center is inside box, find the closest face normal
        min_dist = float('inf')
        best_normal = Vec3(0, 0, 0)
        
        for axis, half_extent in enumerate([box.half_extents[0], box.half_extents[1], box.half_extents[2]]):
            for direction in [-1, 1]:
                face_dist = abs(abs(local_sphere_center[axis]) - half_extent)
                if face_dist < min_dist:
                    min_dist = face_dist
                    best_normal = Vec3(0, 0, 0)
                    best_normal[axis] = direction
        
        normal = best_normal
    
    # Transform back to world space
    world_point = box_transform.origin + box_transform.basis @ closest_point
    world_normal = box_transform.basis @ normal
    
    return IntersectionResult(True, [(world_point, world_normal)])

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
        return IntersectionResult(False)
    
    if distance < 1e-6:
        # Spheres are coincident, use arbitrary normal
        normal = Vec3(0, 0, 1)
        contact_point = center1
    else:
        # Calculate normal and contact point
        normal = (center2 - center1) / distance
        
        # Contact point is on the line between centers
        # Weighted by sphere radii
        t = sphere1.radius / min_distance
        contact_point = center1 + t * (center2 - center1)
    
    return IntersectionResult(True, [(contact_point, normal)])

def intersect(shape1: Shape, shape2: Shape):
    if isinstance(shape1.geometry, Box) and isinstance(shape2.geometry, Box):
        return intersect_box_box(shape1.geometry, shape1.transform, shape2.geometry, shape2.transform)
    elif isinstance(shape1.geometry, Box) and isinstance(shape2.geometry, Sphere):
        return intersect_box_sphere(shape1.geometry, shape1.transform, shape2.geometry, shape2.transform)
    elif isinstance(shape1.geometry, Sphere) and isinstance(shape2.geometry, Box):
        return intersect_box_sphere(shape2.geometry, shape2.transform, shape1.geometry, shape1.transform)
    elif isinstance(shape1.geometry, Sphere) and isinstance(shape2.geometry, Sphere):
        return intersect_sphere_sphere(shape1.geometry, shape1.transform, shape2.geometry, shape2.transform)