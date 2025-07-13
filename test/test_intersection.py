#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from geometry import Box, Sphere, Shape, Transform, intersect
from math_utils import Vec3, Mat33

def test_sphere_sphere():
    """Test sphere-sphere intersection"""
    print("Testing sphere-sphere intersection...")
    
    # Create two spheres
    sphere1 = Sphere(2.0)
    sphere2 = Sphere(1.5)
    
    # Transform them
    transform1 = Transform(Vec3(0, 0, 0), Mat33.identity())
    transform2 = Transform(Vec3(3, 0, 0), Mat33.identity())  # Distance 3, should intersect
    
    shape1 = Shape(sphere1, transform1)
    shape2 = Shape(sphere2, transform2)
    
    result = intersect(shape1, shape2)
    print(f"Sphere-sphere intersection: {result.intersects}")
    if result.intersects:
        for i, (point, normal) in enumerate(result.points):
            print(f"  Contact point {i}: {point}, Normal: {normal}")
    
    # Test non-intersecting spheres
    transform2_no_intersect = Transform(Vec3(5, 0, 0), Mat33.identity())  # Distance 5, should not intersect
    shape2_no_intersect = Shape(sphere2, transform2_no_intersect)
    
    result_no_intersect = intersect(shape1, shape2_no_intersect)
    print(f"Sphere-sphere no intersection: {result_no_intersect.intersects}")

def test_box_sphere():
    """Test box-sphere intersection"""
    print("\nTesting box-sphere intersection...")
    
    # Create box and sphere
    box = Box(Vec3(2, 1, 1))  # 4x2x2 box
    sphere = Sphere(1.0)
    
    # Transform them
    box_transform = Transform(Vec3(0, 0, 0), Mat33.identity())
    sphere_transform = Transform(Vec3(2.5, 0, 0), Mat33.identity())  # Should intersect
    
    box_shape = Shape(box, box_transform)
    sphere_shape = Shape(sphere, sphere_transform)
    
    result = intersect(box_shape, sphere_shape)
    print(f"Box-sphere intersection: {result.intersects}")
    if result.intersects:
        for i, (point, normal) in enumerate(result.points):
            print(f"  Contact point {i}: {point}, Normal: {normal}")

def test_box_box():
    """Test box-box intersection"""
    print("\nTesting box-box intersection...")
    
    # Create two boxes
    box1 = Box(Vec3(1, 1, 1))  # 2x2x2 box
    box2 = Box(Vec3(0.5, 0.5, 0.5))  # 1x1x1 box
    
    # Transform them
    transform1 = Transform(Vec3(0, 0, 0), Mat33.identity())
    transform2 = Transform(Vec3(1.5, 0, 0), Mat33.identity())  # Should intersect
    
    shape1 = Shape(box1, transform1)
    shape2 = Shape(box2, transform2)
    
    result = intersect(shape1, shape2)
    print(f"Box-box intersection: {result.intersects}")
    if result.intersects:
        for i, (point, normal) in enumerate(result.points):
            print(f"  Contact point {i}: {point}, Normal: {normal}")

if __name__ == "__main__":
    test_sphere_sphere()
    test_box_sphere()
    test_box_box() 