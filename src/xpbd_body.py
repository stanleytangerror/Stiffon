import numpy as np
np.seterr(all='raise')

from numpy.linalg import norm
from math_utils import Vec3, Mat33, Transform, integrate_transform, skew_symmetric_matrix, normalized, solve_jacobian, solve_gauss_seidel, create_world_matrix
from geometry import Box, Sphere, Plane, Shape, intersect   
from renderer import Renderer
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

class Body:
    def __init__(self, 
        mass: float, inertia: Vec3 = Vec3(1.0, 1.0, 1.0), 
        x: Vec3 = Vec3(0.0, 0.0, 0.0), o: R = R.identity(), 
        v: Vec3 = Vec3(0, 0, 0), w: Vec3 = Vec3(0, 0, 0)):
        self.inv_mass = 1.0 / mass
        self.inv_inertia = np.diag(np.array([1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z]))
        self.x = x
        self.q = o.as_matrix()
        self.x_predict = x
        self.q_predict = o.as_matrix()
        self.x_prev = x
        self.q_prev = o.as_matrix()
        self.x_prev = x
        self.q_prev = o.as_matrix()
        self.v = v
        self.v_predict = v
        self.w = w
        self.w_predict = w
        self.force_ext = Vec3(0, 0, 0)
        self.torque_ext = Vec3(0, 0, 0)
        self.inv_inertia_world = self.inv_inertia
    
class DistanceConstraint:
    def __init__(self, 
        b1: Body, b2: Body, 
        p1: Vec3, p2: Vec3, 
        stiffness: float, damping: float, distance: float):
        
        self.b1 = b1
        self.b2 = b2
        self.anchor1 = p1 - b1.x
        self.anchor2 = p2 - b2.x

        self.alpha = 1.0 / stiffness
        self.beta = damping
        self.distance = distance
        self.inv_mass = np.zeros((6, 6))
        self.inv_mass[0:3, 0:3] = np.eye(3) * self.b1.inv_mass
        self.inv_mass[3:6, 3:6] = np.eye(3) * self.b2.inv_mass
        self.lambda_ = 0

    def solve(self, dt: float):

        r1 = self.b1.q_predict @ self.anchor1
        r2 = self.b2.q_predict @ self.anchor2
        
        disp = self.b1.x_predict + r1 - self.b2.x_predict - r2
        n = normalized(disp)
        c = np.linalg.norm(disp) - self.distance

        alpha_tilde = self.alpha / dt / dt

        w1 = self.b1.inv_mass + np.cross(r1, n).T @ self.b1.inv_inertia_world @ np.cross(r1, n)
        w2 = self.b2.inv_mass + np.cross(r2, n).T @ self.b2.inv_inertia_world @ np.cross(r2, n)

        delta_lambda = -(c + alpha_tilde * self.lambda_) / (w1 + w2 + alpha_tilde)
        self.lambda_ += delta_lambda

        p = delta_lambda * n

        self.b1.x_predict += p * self.b1.inv_mass
        self.b2.x_predict += -p * self.b2.inv_mass
        self.b1.q_predict = R.from_rotvec(self.b1.inv_inertia_world @ np.cross(r1, p)).as_matrix() @ self.b1.q_predict
        self.b2.q_predict = R.from_rotvec(self.b2.inv_inertia_world @ np.cross(r2, -p)).as_matrix() @ self.b2.q_predict


class Scene:
    def __init__(self):
        self.bodies = []
        self.constraints = []
        self.gravity = Vec3(0.0, 0.0, -10.0)
        self.constraint_iterations = 20

    def add_body(self, b: Body):
        self.bodies.append(b)
    
    def add_constraint(self, c: DistanceConstraint):
        self.constraints.append(c)

    def step_simulation(self, dt: float):
        for b in self.bodies:
            if b.inv_mass != 0.0:
                b.force_ext += self.gravity / b.inv_mass
        for b in self.bodies:
            b.v_predict = b.v + b.inv_mass * b.force_ext * dt
            b.x_predict = b.x + b.v_predict * dt

            b.w_predict = b.w + b.inv_inertia_world @ (b.torque_ext - np.cross(b.w, b.inv_inertia_world @ b.w)) * dt
            b.q_predict = R.from_rotvec(b.w_predict * dt).as_matrix() @ b.q
            
        for c in self.constraints:
            c.lambda_ = 0
        for _ in range(self.constraint_iterations):
            for c in self.constraints:
                c.solve(dt)
        
        for b in self.bodies:
            b.x_prev = b.x
            b.q_prev = b.q
            
            b.v = (b.x_predict - b.x) / dt
            delta_q = b.q_predict @ b.q.T
            b.w = R.from_matrix(delta_q).as_rotvec() * (1.0 / dt)

            b.inv_inertia_world = b.q @ b.inv_inertia @ b.q.T
            b.x = b.x_predict
            b.q = b.q_predict
            b.force_ext = Vec3(0.0, 0.0, 0.0)
            b.torque_ext = Vec3(0.0, 0.0, 0.0)


class SceneDebugRenderer:
    def __init__(self, scene: Scene, width: int, height: int):
        self.scene = scene
        self.renderer = Renderer(width=width, height=height)

    def is_running(self):
        return self.renderer.is_running()

    def render(self):
        self.renderer.begin_frame()
        for b in self.scene.bodies:
            self.draw_body(b, np.array([1.0, 0.0, 0.0]))
        self.renderer.end_frame()

    def draw_body(self, b: Body, color: np.ndarray):
        world_matrix = create_world_matrix(translate=b.x, rotate=R.from_matrix(b.q))
        self.renderer.draw_box(world_matrix, color)
    

if __name__ == "__main__":
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))
    # b3 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(4.0, 0.0, 0.0))
    # b4 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(6.0, 0.0, 0.0))
    # b5 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(8.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)
    # scene.add_body(b3)
    # scene.add_body(b4)
    # scene.add_body(b5)

    c1 = DistanceConstraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    # c2 = DistanceConstraint(b2, b3, Vec3(2.5, 0.0, 0.0), Vec3(3.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    # c3 = DistanceConstraint(b3, b4, Vec3(4.5, 0.0, 0.0), Vec3(5.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    # c4 = DistanceConstraint(b4, b5, Vec3(6.5, 0.0, 0.0), Vec3(7.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)

    scene.add_constraint(c1)
    # scene.add_constraint(c2)
    # scene.add_constraint(c3)
    # scene.add_constraint(c4)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        scene.step_simulation(0.01)
        if frame_count % 10 == 0:
            renderer.render()
        frame_count += 1
