import numpy as np
from math_utils import Vec2, Transform2d

class Rectangle:
    def __init__(self, half_extents: Vec2):
        self.half_extents = half_extents

class Circle:
    def __init__(self, radius: float):
        self.radius = radius

class Shape2d:
    def __init__(self, geometry: Rectangle | Circle, transform: Transform2d):
        self.geometry = geometry
        self.transform = transform
