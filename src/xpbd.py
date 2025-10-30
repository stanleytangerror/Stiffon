import numpy as np
np.seterr(all='raise')

from numpy.linalg import norm
from math_utils import Vec3, Mat33, Transform, integrate_transform, skew_symmetric_matrix, normalized, solve_jacobian, solve_gauss_seidel, create_world_matrix
from geometry import Box, Sphere, Plane, Shape, intersect   
from renderer import Renderer
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

class MassPoint:
    def __init__(self, inv_mass: float, x: Vec3, v: Vec3):
        self.inv_mass = inv_mass
        self.x = x
        self.x_prev = x
        self.x_prev_prev = x
        self.v = v
        self.f_ext = Vec3(0, 0, 0)
        self.x_tilde = Vec3(0, 0, 0)

class DistanceConstraint:
    def __init__(self, p1: MassPoint, p2: MassPoint, stiffness: float, distance: float):
        self.p1 = p1
        self.p2 = p2
        self.alpha = 1.0 / stiffness
        self.distance = distance
        self.inv_mass = np.zeros((6, 6))
        self.inv_mass[0:3, 0:3] = np.eye(3) * p1.inv_mass
        self.inv_mass[3:6, 3:6] = np.eye(3) * p2.inv_mass
        self.lambda_ = 0

    def solve(self, dt: float):
        n = normalized(self.p1.x_tilde - self.p2.x_tilde)
        c = np.linalg.norm(self.p1.x_tilde - self.p2.x_tilde) - self.distance
        j = np.zeros((1, 6))
        j[:, 0:3] = n
        j[:, 3:6] = -n

        alpha_tilde = self.alpha / dt / dt
        delta_lambda = -(alpha_tilde * self.lambda_ + c) / (j @ self.inv_mass @ j.T + alpha_tilde)[0, 0]
        self.lambda_ += delta_lambda

        delta_x = self.inv_mass @ j.T * delta_lambda
        self.p1.x_tilde += delta_x[0:3, :].reshape(3)
        self.p2.x_tilde += delta_x[3:6, :].reshape(3)

class Scene:
    def __init__(self):
        self.mass_points = []
        self.constraints = []
        self.gravity = Vec3(0.0, 0.0, -10.0)
        self.constraint_iterations = 5

    def add_mass_point(self, p: MassPoint):
        self.mass_points.append(p)
    
    def add_constraint(self, c: DistanceConstraint):
        self.constraints.append(c)

    def step_simulation(self, dt: float):
        for p in self.mass_points:
            if p.inv_mass != 0.0:
                p.f_ext += self.gravity / p.inv_mass

        for p in self.mass_points:
            p.x_tilde = p.x + p.v * dt + dt**2 * p.inv_mass * p.f_ext
            
        for _ in range(self.constraint_iterations):
            for c in self.constraints:
                c.solve(dt)
        
        for p in self.mass_points:
            p.v = (p.x_tilde - p.x) / dt - dt * p.inv_mass * p.f_ext
            p.x = p.x_tilde


class SceneDebugRenderer:
    def __init__(self, scene: Scene, width: int, height: int):
        self.scene = scene
        self.renderer = Renderer(width=width, height=height)

    def is_running(self):
        return self.renderer.is_running()

    def render(self):
        self.renderer.begin_frame()
        for p in self.scene.mass_points:
            self.draw_mass_point(p, np.array([1.0, 0.0, 0.0]))
        self.renderer.end_frame()

    def draw_mass_point(self, p: MassPoint, color: np.ndarray):
        world_matrix = create_world_matrix(translate=p.x)
        self.renderer.draw_sphere(world_matrix, color)
    

if __name__ == "__main__":
    scene = Scene()

    p1 = MassPoint(inv_mass=0.0, x=Vec3(0.0, 0.0, 0.0), v=Vec3(0.0, 0.0, 0.0))
    p2 = MassPoint(inv_mass=1.0, x=Vec3(1.0, 0.0, 0.0), v=Vec3(0.0, 0.0, 0.0))
    p3 = MassPoint(inv_mass=1.0, x=Vec3(2.0, 0.0, 0.0), v=Vec3(0.0, 0.0, 0.0))
    p4 = MassPoint(inv_mass=1.0, x=Vec3(3.0, 0.0, 0.0), v=Vec3(0.0, 0.0, 0.0))
    p5 = MassPoint(inv_mass=1.0, x=Vec3(4.0, 0.0, 0.0), v=Vec3(0.0, 0.0, 0.0))

    scene.add_mass_point(p1)
    scene.add_mass_point(p2)
    scene.add_mass_point(p3)
    scene.add_mass_point(p4)
    scene.add_mass_point(p5)

    c1 = DistanceConstraint(p1, p2, stiffness=float('inf'), distance=1.0)
    c2 = DistanceConstraint(p2, p3, stiffness=float('inf'), distance=1.0)
    c3 = DistanceConstraint(p3, p4, stiffness=1000.0, distance=1.0)
    c4 = DistanceConstraint(p4, p5, stiffness=1000.0, distance=1.0)

    scene.add_constraint(c1)
    scene.add_constraint(c2)
    scene.add_constraint(c3)
    scene.add_constraint(c4)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        scene.step_simulation(0.01)
        if frame_count % 10 == 0:
            renderer.render()
        frame_count += 1

    # while True:
    #     scene.step_simulation(0.01)
    #     print(scene.mass_points[2].x[2])
