import numpy as np
np.seterr(all='raise')

from numpy.linalg import norm
from math_utils import Vec3, generate_orthogonal_basis, normalized, create_world_matrix
from geometry import Box, Sphere, Plane, Shape, intersect   
from renderer import Renderer
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

class Body:
    def __init__(self, 
        mass: float, inertia: Vec3 = Vec3(1.0, 1.0, 1.0), 
        x: Vec3 = Vec3(0.0, 0.0, 0.0), o: R = R.identity(), 
        v: Vec3 = Vec3(0, 0, 0), w: Vec3 = Vec3(0, 0, 0),
        shape: Box | Sphere | Plane = Box(Vec3(0.5, 0.5, 0.5))):
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
        self.shape = shape
    
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
        self.lambda_ = 0

    def pre_solve(self, dt: float):
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


class RotationalConstraint_AlignAxis:
    def __init__(self, 
        body_A: Body, body_B: Body, 
        axis_A: Vec3, axis_B: Vec3,
        stiffness: float):
        
        self.body_A = body_A
        self.axis_A = axis_A
        self.body_B = body_B
        self.axis_B = axis_B

        self.alpha = 1.0 / stiffness
        self.lambda_ = 0

    def pre_solve(self, dt: float):
        self.lambda_ = 0

    def solve(self, dt: float):

        a_A = self.body_A.q_predict @ self.axis_A
        a_B = self.body_B.q_predict @ self.axis_B
        delta_q = -np.cross(a_A, a_B) # a rotation will rotate a_B to a_A

        n = normalized(delta_q)
        theta = np.linalg.norm(delta_q)

        if np.linalg.norm(delta_q) < 1e-8:
            # almost aligned, no need to solve
            return

        alpha_tilde = self.alpha / dt / dt

        w1 = n.T @ self.body_A.inv_inertia_world @ n
        w2 = n.T @ self.body_B.inv_inertia_world @ n

        delta_lambda = -(theta + alpha_tilde * self.lambda_) / (w1 + w2 + alpha_tilde)
        self.lambda_ += delta_lambda

        p = delta_lambda * n

        self.body_A.q_predict = R.from_rotvec(self.body_A.inv_inertia_world @ p).as_matrix() @ self.body_A.q_predict
        self.body_B.q_predict = R.from_rotvec(self.body_B.inv_inertia_world @ -p).as_matrix() @ self.body_B.q_predict

class RotationalConstraint_TargetAngle:
    def __init__(self, 
        body_A: Body, body_B: Body, 
        axis_A: Vec3, tangent_A: Vec3,
        axis_B: Vec3, tangent_B: Vec3,
        stiffness: float, tangent_angle: float):
        
        self.body_A = body_A
        self.axis_A = axis_A
        self.tangent_A = tangent_A
        self.body_B = body_B
        self.axis_B = axis_B
        self.tangent_B = tangent_B
        self.tangent_angle = tangent_angle

        self.alpha = 1.0 / stiffness
        self.lambda_ = 0

    def pre_solve(self, dt: float):
        self.lambda_ = 0
    
    def solve(self, dt: float):

        b_A = self.body_A.q_predict @ self.tangent_A
        b_B = R.from_rotvec(self.axis_B * self.tangent_angle).as_matrix() @ self.body_B.q_predict @ self.tangent_B
        delta_q = -np.cross(b_A, b_B) # a rotation will rotate b_B to b_A

        n = normalized(delta_q)
        theta = np.linalg.norm(delta_q)

        if np.linalg.norm(delta_q) < 1e-8:
            # almost aligned, no need to solve
            return

        alpha_tilde = self.alpha / dt / dt

        w1 = n.T @ self.body_A.inv_inertia_world @ n
        w2 = n.T @ self.body_B.inv_inertia_world @ n

        delta_lambda = -(theta + alpha_tilde * self.lambda_) / (w1 + w2 + alpha_tilde)
        self.lambda_ += delta_lambda

        p = delta_lambda * n

        self.body_A.q_predict = R.from_rotvec(self.body_A.inv_inertia_world @ p).as_matrix() @ self.body_A.q_predict
        self.body_B.q_predict = R.from_rotvec(self.body_B.inv_inertia_world @ -p).as_matrix() @ self.body_B.q_predict

class RotationalConstraint_Motor:
    def __init__(self, 
        body_A: Body, body_B: Body, 
        axis_A: Vec3, tangent_A: Vec3,
        axis_B: Vec3, tangent_B: Vec3,
        stiffness: float, angular_speed: float):
        
        self.body_A = body_A
        self.axis_A = axis_A
        self.tangent_A = tangent_A
        self.body_B = body_B
        self.axis_B = axis_B
        self.tangent_B = tangent_B
        self.tangent_angle = 0
        self.angular_speed = angular_speed

        self.alpha = 1.0 / stiffness
        self.lambda_ = 0

    def pre_solve(self, dt: float):
        self.lambda_ = 0
        self.tangent_angle += self.angular_speed * dt

    def solve(self, dt: float):

        b_A = self.body_A.q_predict @ self.tangent_A
        b_B = R.from_rotvec(self.axis_B * self.tangent_angle).as_matrix() @ self.body_B.q_predict @ self.tangent_B
        delta_q = -np.cross(b_A, b_B) # a rotation will rotate b_B to b_A

        n = normalized(delta_q)
        theta = np.linalg.norm(delta_q)

        if np.linalg.norm(delta_q) < 1e-8:
            # almost aligned, no need to solve
            return

        alpha_tilde = self.alpha / dt / dt

        w1 = n.T @ self.body_A.inv_inertia_world @ n
        w2 = n.T @ self.body_B.inv_inertia_world @ n

        delta_lambda = -(theta + alpha_tilde * self.lambda_) / (w1 + w2 + alpha_tilde)
        self.lambda_ += delta_lambda

        p = delta_lambda * n

        self.body_A.q_predict = R.from_rotvec(self.body_A.inv_inertia_world @ p).as_matrix() @ self.body_A.q_predict
        self.body_B.q_predict = R.from_rotvec(self.body_B.inv_inertia_world @ -p).as_matrix() @ self.body_B.q_predict


class Scene:
    def __init__(self):
        self.bodies = []
        self.constraints = []
        self.gravity = Vec3(0.0, 0.0, -10.0)
        self.constraint_iterations = 20

    def add_body(self, b: Body):
        self.bodies.append(b)
    
    def add_constraint(self, c):
        self.constraints.append(c)
    
    def add_constraints(self, cs):
        self.constraints.extend(cs)

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
            c.pre_solve(dt)

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
        if isinstance(b.shape, Box):
            scale = b.shape.half_extents * 2
            scale_matrix = np.diag(np.array([scale[0], scale[1], scale[2], 1.0]))
            self.renderer.draw_box(world_matrix @ scale_matrix, color)
        elif isinstance(b.shape, Sphere):
            scale_matrix = np.diag(np.array([b.shape.radius * 2, b.shape.radius * 2, b.shape.radius * 2, 1.0]))
            self.renderer.draw_sphere(world_matrix @ scale_matrix, color)
        elif isinstance(b.shape, Plane):
            self.renderer.draw_plane(world_matrix, color)
        else:
            raise ValueError(f"Unsupported shape: {type(b.shape)}")

def create_rotational_motor(b1: Body, b2: Body, p: Vec3, axis: Vec3, angular_speed: float, stiffness: float=float('inf')):
    c1 = DistanceConstraint(b1, b2, p, p, stiffness=float('inf'), damping=0.0, distance=0.0)

    v0, v1, v2 = generate_orthogonal_basis(axis)
    axis_A = b1.q.T @ v0
    axis_B = b2.q.T @ v0
    tangent_A = b1.q.T @ v1
    tangent_B = b2.q.T @ v1

    c2 = RotationalConstraint_AlignAxis(b1, b2, axis_A, axis_B, stiffness=float('inf'))
    c3 = RotationalConstraint_Motor(b1, b2, axis_A, tangent_A, axis_B, tangent_B, stiffness=stiffness, angular_speed=angular_speed)
    return [c1, c2, c3]

if __name__ == "__main__":
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)

    c1 = DistanceConstraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)

    scene.add_constraint(c1)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        scene.step_simulation(0.01)
        if frame_count % 10 == 0:
            renderer.render()
        frame_count += 1
