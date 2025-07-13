from math_utils import Vec3, Mat33, Transform, integrate_transform
from geometry import Box, Sphere, Shape
from renderer import Renderer

class Body:
    def __init__(self, mass: float, inertia: Mat33 = Mat33.identity(), 
            linear_velocity: Vec3 = Vec3(0, 0, 0), angular_velocity: Vec3 = Vec3(0, 0, 0), 
            pose: Transform = Transform(Vec3(0, 0, 0), Mat33.identity()), 
            geometry: Box | Sphere = Box(Vec3(0.5, 0.5, 0.5))):
        self.mass = mass
        self.inertia = inertia
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.pose = pose
        self.geometry = geometry

class Scene:
    def __init__(self):
        self.bodies = []

    def add_body(self, body: Body):
        self.bodies.append(body)

    def step_simulation(self, dt: float):
        for body in self.bodies:
            body.pose = integrate_transform(body.pose, body.linear_velocity, body.angular_velocity, dt)

if __name__ == "__main__":
    scene = Scene()
    
    body1 = Body(mass=1.0, angular_velocity=Vec3(2, 0, 0))
    scene.add_body(body1)

    body2 = Body(mass=1.0, linear_velocity=Vec3(0.1, 0.0, 0.0))
    scene.add_body(body2)

    renderer = Renderer(width=800, height=600)

    while renderer.is_running():
        renderer.begin_frame()
        scene.step_simulation(0.01)

        renderer.draw_box(body1.pose.to_matrix())
        renderer.draw_box(body2.pose.to_matrix())
        renderer.end_frame()