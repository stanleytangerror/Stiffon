import numpy as np
np.seterr(all='raise')

from numpy.linalg import norm
from math_utils import Vec3, generate_orthogonal_basis, normalized, create_world_matrix, Transform, decompose_to_n_and_t
from geometry import Box, Sphere, Plane, Shape, intersect, raycast, RaycastResult   
from renderer import Renderer
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

class Body:
    def __init__(self, 
        mass: float, inertia: Vec3 = Vec3(1.0, 1.0, 1.0), 
        x: Vec3 = Vec3(0.0, 0.0, 0.0), o: R = R.identity(), 
        v: Vec3 = Vec3(0, 0, 0), w: Vec3 = Vec3(0, 0, 0),
        shape: Box | Sphere | Plane = Box(Vec3(0.5, 0.5, 0.5)),
        static_friction_coefficient: float = 0.9,
        dynamic_friction_coefficient: float = 0.9):
        self.inv_mass = 1.0 / mass
        self.inv_inertia = np.diag(np.array([1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z]))
        self.x = x
        self.q = o.as_matrix()
        self.x_predict = x
        self.q_predict = o.as_matrix()
        self.v = v
        self.v_predict = v
        self.w = w
        self.w_predict = w
        self.force_ext = Vec3(0, 0, 0)
        self.torque_ext = Vec3(0, 0, 0)
        self.inv_inertia_world = self.inv_inertia
        self.shape = shape
        self.static_friction_coefficient = static_friction_coefficient
        self.dynamic_friction_coefficient = dynamic_friction_coefficient
    
class PositionalConstraint:
    def __init__(self, 
        body_A: Body, body_B: Body, 
        anchor_A: Vec3, anchor_B: Vec3, 
        stiffness: float, damping: float, distance: float):
        
        self.body_A = body_A
        self.body_B = body_B
        self.anchor_A = anchor_A
        self.anchor_B = anchor_B

        self.alpha = 1.0 / stiffness
        self.beta = damping
        self.distance = distance
        self.lambda_ = 0

    def pre_solve(self, dt: float):
        self.lambda_ = 0

    def solve(self, dt: float):

        r1 = self.body_A.q_predict @ self.anchor_A
        r2 = self.body_B.q_predict @ self.anchor_B
        
        disp = self.body_A.x_predict + r1 - self.body_B.x_predict - r2
        n = normalized(disp)
        c = np.linalg.norm(disp) - self.distance

        alpha_tilde = self.alpha / dt / dt

        w1 = self.body_A.inv_mass + np.cross(r1, n).T @ self.body_A.inv_inertia_world @ np.cross(r1, n)
        w2 = self.body_B.inv_mass + np.cross(r2, n).T @ self.body_B.inv_inertia_world @ np.cross(r2, n)

        delta_lambda = -(c + alpha_tilde * self.lambda_) / (w1 + w2 + alpha_tilde)
        self.lambda_ += delta_lambda

        p = delta_lambda * n

        self.body_A.x_predict += p * self.body_A.inv_mass
        self.body_B.x_predict += -p * self.body_B.inv_mass
        self.body_A.q_predict = R.from_rotvec(self.body_A.inv_inertia_world @ np.cross(r1, p)).as_matrix() @ self.body_A.q_predict
        self.body_B.q_predict = R.from_rotvec(self.body_B.inv_inertia_world @ np.cross(r2, -p)).as_matrix() @ self.body_B.q_predict

    def solve_velocity(self, dt: float):
        pass

class PositionalConstraint_Motor:
    def __init__(self, 
        body_A: Body, body_B: Body, 
        anchor_A: Vec3, anchor_B: Vec3, 
        stiffness: float, damping: float, speed: float):
        
        self.body_A = body_A
        self.body_B = body_B
        self.anchor_A = anchor_A
        self.anchor_B = anchor_B
        self.speed = speed

        self.alpha = 1.0 / stiffness
        self.beta = damping
        self.distance = 0
        self.lambda_ = 0

    def pre_solve(self, dt: float):
        self.lambda_ = 0
        self.distance += self.speed * dt

    def solve(self, dt: float):

        r1 = self.body_A.q_predict @ self.anchor_A
        r2 = self.body_B.q_predict @ self.anchor_B
        
        disp = self.body_A.x_predict + r1 - self.body_B.x_predict - r2
        n = normalized(disp)
        c = np.linalg.norm(disp) - self.distance

        alpha_tilde = self.alpha / dt / dt

        w1 = self.body_A.inv_mass + np.cross(r1, n).T @ self.body_A.inv_inertia_world @ np.cross(r1, n)
        w2 = self.body_B.inv_mass + np.cross(r2, n).T @ self.body_B.inv_inertia_world @ np.cross(r2, n)

        delta_lambda = -(c + alpha_tilde * self.lambda_) / (w1 + w2 + alpha_tilde)
        self.lambda_ += delta_lambda

        p = delta_lambda * n

        self.body_A.x_predict += p * self.body_A.inv_mass
        self.body_B.x_predict += -p * self.body_B.inv_mass
        self.body_A.q_predict = R.from_rotvec(self.body_A.inv_inertia_world @ np.cross(r1, p)).as_matrix() @ self.body_A.q_predict
        self.body_B.q_predict = R.from_rotvec(self.body_B.inv_inertia_world @ np.cross(r2, -p)).as_matrix() @ self.body_B.q_predict

    def solve_velocity(self, dt: float):
        pass


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

    def solve_velocity(self, dt: float):
        pass

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

    def solve_velocity(self, dt: float):
        pass

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

    def solve_velocity(self, dt: float):
        pass

class ContactConstraint:
    def __init__(self, 
        body_A: Body, body_B: Body, 
        anchor_A: Vec3, anchor_B: Vec3, normal_A: Vec3):

        self.body_A = body_A
        self.body_B = body_B
        self.anchor_A = anchor_A
        self.anchor_B = anchor_B
        self.normal_A = normal_A
        self.lambda_normal = 0
        self.lambda_tangent = 0
        self.static_friction_coefficient = 0.5 * (self.body_A.static_friction_coefficient + self.body_B.static_friction_coefficient)
        self.dynamic_friction_coefficient = 0.5 * (self.body_A.dynamic_friction_coefficient + self.body_B.dynamic_friction_coefficient)
        self.resolve_dynamic_friction = False
        self.resolve_friction = False
    
    def pre_solve(self, dt: float):
        self.lambda_normal = 0
        self.lambda_tangent = 0
        self.resolve_dynamic_friction = False
        self.resolve_friction = False
    
    def solve(self, dt: float):
        self.solve_penetration()
        if self.resolve_friction:
            self.solve_static_friction()

    def solve_velocity(self, dt: float):
        if self.resolve_dynamic_friction:
            self.solve_dynamic_friction(dt)

    def solve_penetration(self):
        # C = n_A^T * (x_A + r_A - x_B - r_B) <= 0
        n = self.body_A.q_predict @ self.normal_A
        r1 = self.body_A.q_predict @ self.anchor_A
        r2 = self.body_B.q_predict @ self.anchor_B
        p1 = self.body_A.x_predict + r1
        p2 = self.body_B.x_predict + r2

        c = (p1 - p2).T @ n
        if c <= 0:
            return False

        self.resolve_friction = True

        # handle peneration
        w1 = self.body_A.inv_mass + np.cross(r1, n).T @ self.body_A.inv_inertia_world @ np.cross(r1, n)
        w2 = self.body_B.inv_mass + np.cross(r2, n).T @ self.body_B.inv_inertia_world @ np.cross(r2, n)

        delta_lambda_normal = -c / (w1 + w2)
        self.lambda_normal += delta_lambda_normal

        p = delta_lambda_normal * n

        self.body_A.x_predict += p * self.body_A.inv_mass
        self.body_B.x_predict += -p * self.body_B.inv_mass
        self.body_A.q_predict = R.from_rotvec(self.body_A.inv_inertia_world @ np.cross(r1, p)).as_matrix() @ self.body_A.q_predict
        self.body_B.q_predict = R.from_rotvec(self.body_B.inv_inertia_world @ np.cross(r2, -p)).as_matrix() @ self.body_B.q_predict

    def solve_static_friction(self):
        
        n = self.body_A.q_predict @ self.normal_A
        r1 = self.body_A.q_predict @ self.anchor_A
        r2 = self.body_B.q_predict @ self.anchor_B
        p1 = self.body_A.x_predict + r1
        p2 = self.body_B.x_predict + r2
        p1_prev = self.body_A.x + self.body_A.q @ self.anchor_A
        p2_prev = self.body_B.x + self.body_B.q @ self.anchor_B

        delta_p = (p1 - p1_prev) - (p2 - p2_prev)
        _, delta_p_tangent = decompose_to_n_and_t(delta_p, n)

        c = np.linalg.norm(delta_p_tangent)

        if c < 1e-8:
            return

        t = normalized(delta_p_tangent)

        w1 = self.body_A.inv_mass + np.cross(r1, t).T @ self.body_A.inv_inertia_world @ np.cross(r1, t)
        w2 = self.body_B.inv_mass + np.cross(r2, t).T @ self.body_B.inv_inertia_world @ np.cross(r2, t)

        delta_lambda = -c / (w1 + w2)

        if abs(self.lambda_tangent + delta_lambda) >= self.static_friction_coefficient * abs(self.lambda_normal):
            # will slide, no need to solve
            self.resolve_dynamic_friction = True
            return

        self.lambda_tangent += delta_lambda

        p = delta_lambda * t

        self.body_A.x_predict += p * self.body_A.inv_mass
        self.body_B.x_predict += -p * self.body_B.inv_mass
        self.body_A.q_predict = R.from_rotvec(self.body_A.inv_inertia_world @ np.cross(r1, p)).as_matrix() @ self.body_A.q_predict
        self.body_B.q_predict = R.from_rotvec(self.body_B.inv_inertia_world @ np.cross(r2, -p)).as_matrix() @ self.body_B.q_predict


    def solve_dynamic_friction(self, dt: float):
        
        n = self.body_A.q_predict @ self.normal_A
        r1 = self.body_A.q_predict @ self.anchor_A
        r2 = self.body_B.q_predict @ self.anchor_B
        v1 = self.body_A.v_predict + np.cross(self.body_A.w_predict, r1)
        v2 = self.body_B.v_predict + np.cross(self.body_B.w_predict, r2)
        v = v1 - v2
        _, v_t = decompose_to_n_and_t(v, n)
        t = normalized(v_t)

        w1 = self.body_A.inv_mass + np.cross(r1, t).T @ self.body_A.inv_inertia_world @ np.cross(r1, t)
        w2 = self.body_B.inv_mass + np.cross(r2, t).T @ self.body_B.inv_inertia_world @ np.cross(r2, t)

        f_n = self.lambda_normal / dt**2
        delta_v = -t * min(self.dynamic_friction_coefficient * abs(f_n) * dt * (w1 + w2), np.linalg.norm(v_t))
        # delta_v = v_t

        p = delta_v / (w1 + w2)

        self.body_A.v_predict += p * self.body_A.inv_mass
        self.body_B.v_predict -= p * self.body_B.inv_mass
        self.body_A.w_predict += self.body_A.inv_inertia_world @ np.cross(r1, p)
        self.body_B.w_predict -= self.body_B.inv_inertia_world @ np.cross(r2, p)

class Scene:
    def __init__(self):
        self.bodies = []
        self.constraints = []
        self.temporary_constraints = []
        self.gravity = Vec3(0.0, 0.0, -10.0)
        self.position_iterations = 10

    def add_body(self, b: Body):
        self.bodies.append(b)
    
    def add_constraint(self, c):
        self.constraints.append(c)
    
    def add_constraints(self, cs):
        self.constraints.extend(cs)

    def remove_body(self, b: Body):
        self.bodies.remove(b)
    
    def remove_constraint(self, c):
        self.constraints.remove(c)

    def step_simulation(self, dt: float):
        for b in self.bodies:
            if b.inv_mass != 0.0:
                b.force_ext += self.gravity / b.inv_mass

        for b in self.bodies:
            b.v_predict = b.v + b.inv_mass * b.force_ext * dt
            b.x_predict = b.x + b.v_predict * dt

            b.w_predict = b.w + b.inv_inertia_world @ (b.torque_ext - np.cross(b.w, b.inv_inertia_world @ b.w)) * dt
            b.q_predict = R.from_rotvec(b.w_predict * dt).as_matrix() @ b.q

        self.contact_detection()

        for c in self.constraints:
            c.pre_solve(dt)
        for c in self.temporary_constraints:
            c.pre_solve(dt)

        for _ in range(self.position_iterations):
            for c in self.constraints:
                c.solve(dt)
            for c in self.temporary_constraints:
                c.solve(dt)
        
        for b in self.bodies:
            b.v_predict = (b.x_predict - b.x) / dt
            delta_q = b.q_predict @ b.q.T
            b.w_predict = R.from_matrix(delta_q).as_rotvec() * (1.0 / dt)
        
        for c in self.constraints:
            c.solve_velocity(dt)
        for c in self.temporary_constraints:
            c.solve_velocity(dt)

        for b in self.bodies:
            b.v = b.v_predict
            b.w = b.w_predict

            b.x = b.x + b.v * dt
            b.q = R.from_rotvec(b.w * dt).as_matrix() @ b.q
            b.inv_inertia_world = b.q @ b.inv_inertia @ b.q.T

            b.force_ext = Vec3(0.0, 0.0, 0.0)
            b.torque_ext = Vec3(0.0, 0.0, 0.0)
    
        self.temporary_constraints = []
    
    def contact_detection(self):
        for i, body in enumerate(self.bodies):
            for j, other_body in enumerate(self.bodies):
                if i >= j:
                    continue
                pose_A = Transform(body.x_predict, body.q_predict)
                pose_B = Transform(other_body.x_predict, other_body.q_predict)
                contact_result = intersect(Shape(body.shape, pose_A), Shape(other_body.shape, pose_B))
                if contact_result.intersects:
                    anchor_A = pose_A.inverse().transformPosition(contact_result.point_A)
                    anchor_B = pose_B.inverse().transformPosition(contact_result.point_B)
                    normal_A = pose_A.basis.T @ contact_result.normal
                    self.temporary_constraints.append(ContactConstraint(body, other_body, anchor_A, anchor_B, normal_A))

    def raycast(self, origin: Vec3, dir: Vec3) -> RaycastResult:
        hit_result = RaycastResult(hits=False, point=Vec3(0, 0, 0), normal=Vec3(0, 0, 1.0))
        hit_distance = float('inf')
        for body in self.bodies:
            pose = Transform(body.x_predict, body.q_predict)
            result = raycast(origin, dir, Shape(body.shape, pose))
            if result.hits:
                dist = np.linalg.norm(result.point - origin)
                if dist < hit_distance:
                    hit_result = result
        return hit_result


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

def create_positional_constraint(b1: Body, b2: Body, p1: Vec3, p2: Vec3, stiffness: float=float('inf'), damping: float=0.0, distance: float=0.0):
    c = PositionalConstraint(b1, b2, p1 - b1.x, p2 - b2.x, stiffness=stiffness, damping=0.0, distance=0.0)
    return c

def create_positional_constraint_motor(b1: Body, b2: Body, p: Vec3, speed: float, stiffness: float=float('inf'), damping: float=0.0):
    c = PositionalConstraint_Motor(b1, b2, p - b1.x, p - b2.x, stiffness=stiffness, damping=damping, speed=speed)
    return c

def create_rotational_motor(b1: Body, b2: Body, p: Vec3, axis: Vec3, angular_speed: float, stiffness: float=float('inf')):
    c1 = PositionalConstraint(b1, b2, p - b1.x, p - b2.x, stiffness=float('inf'), damping=0.0, distance=0.0)

    v0, v1, _ = generate_orthogonal_basis(axis)
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

    c1 = PositionalConstraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(-0.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)

    scene.add_constraint(c1)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()
