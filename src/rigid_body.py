import numpy as np
np.seterr(all='raise')

from numpy.linalg import norm
from math_utils import Vec3, Mat33, Transform, integrate_transform, skew_symmetric_matrix, normalized, solve_jacobian, solve_gauss_seidel
from geometry import Box, Sphere, Plane, Shape, intersect   
from renderer import Renderer
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

class Body:
    def __init__(self, mass: float, inertia: Vec3 = Vec3(1, 1, 1), 
            linear_velocity: Vec3 = Vec3(0, 0, 0), angular_velocity: Vec3 = Vec3(0, 0, 0), 
            pose: Transform = Transform(Vec3(0, 0, 0), Mat33.identity()), 
            geometry: Box | Sphere | Plane = Box(Vec3(0.5, 0.5, 0.5))):

        safe_inv_mass = lambda m: 1.0 / m if m != float('inf') else 0.0
        
        self.mass = mass
        self.inv_mass = np.eye(3) * safe_inv_mass(mass)
        self.inertia = inertia
        self.inv_inertia = np.diag(np.array([safe_inv_mass(inertia.x), safe_inv_mass(inertia.y), safe_inv_mass(inertia.z)]))  
        self.inv_inertia_world = pose.basis @ self.inv_inertia @ pose.basis.transpose()
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.pose = pose
        self.geometry = geometry
        self.total_force = Vec3(0, 0, 0)
        self.total_torque = Vec3(0, 0, 0)
        self.delta_linear_velocity = Vec3(0, 0, 0)
        self.delta_angular_velocity = Vec3(0, 0, 0)

    def apply_force(self, force: Vec3, point: Vec3):
        self.total_force += force
        self.total_torque += np.cross(point - self.pose.origin, force)

class ContactConstraint:
    # Constraint function: C = -normal_A * (X_A + R_A * r_A - X_B - R_B * r_B) >= 0
    # Jacobian: J = [ -normal_A, normal_A * [R_A*r_A]x, normal_A, -normal_A * [R_B*r_B]x ]
    def __init__(self, body_A: Body, body_B: Body, point_A: Vec3, point_B: Vec3, normal_A: Vec3):
        self.body_A = body_A
        self.body_B = body_B
        self.r_A = body_A.pose.basis.transpose() @ (point_A - body_A.pose.origin)
        self.r_B = body_B.pose.basis.transpose() @ (point_B - body_B.pose.origin)
        self.normal_A = normal_A
        self.C = np.dot(-normal_A, point_A - point_B)
        print(f"Constraint function: {self.C}")
        self.max_penetration = 0.01

    def setup(self, dt: float):
        self.jacobian = np.zeros((1, 12))
        self.jacobian[:, 0:3] = -self.normal_A
        self.jacobian[:, 3:6] = -np.cross(self.body_A.pose.basis @ self.r_A, self.normal_A)
        self.jacobian[:, 6:9] = self.normal_A
        self.jacobian[:, 9:12] = np.cross(self.body_B.pose.basis @ self.r_B, self.normal_A)

        self.generic_inv_mass = np.zeros((12, 12))
        self.generic_inv_mass[0:3, 0:3] = self.body_A.inv_mass
        self.generic_inv_mass[3:6, 3:6] = self.body_A.inv_inertia
        self.generic_inv_mass[6:9, 6:9] = self.body_B.inv_mass
        self.generic_inv_mass[9:12, 9:12] = self.body_B.inv_inertia

        self.generic_velocity = np.zeros((12, 1))
        self.generic_velocity[0:3, 0] = self.body_A.linear_velocity
        self.generic_velocity[3:6, 0] = self.body_A.angular_velocity
        self.generic_velocity[6:9, 0] = self.body_B.linear_velocity
        self.generic_velocity[9:12, 0] = self.body_B.angular_velocity

        self.generic_external_impulse = np.zeros((12, 1))
        self.generic_external_impulse[0:3, 0] = self.body_A.inv_mass @ self.body_A.total_force * dt
        self.generic_external_impulse[3:6, 0] = self.body_A.inv_inertia_world @ self.body_A.total_torque * dt
        self.generic_external_impulse[6:9, 0] = self.body_B.inv_mass @ self.body_B.total_force * dt
        self.generic_external_impulse[9:12, 0] = self.body_B.inv_inertia @ self.body_B.total_torque * dt

        erp = 0.2
        self.bias = erp * max(-self.max_penetration, self.C) / dt

    def iteration(self, with_baumgarte_stabilization: bool):
        generic_delta_velocity = np.zeros((12, 1))
        generic_delta_velocity[0:3, 0] = self.body_A.delta_linear_velocity
        generic_delta_velocity[3:6, 0] = self.body_A.delta_angular_velocity
        generic_delta_velocity[6:9, 0] = self.body_B.delta_linear_velocity
        generic_delta_velocity[9:12, 0] = self.body_B.delta_angular_velocity

        if with_baumgarte_stabilization:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse) - self.bias
        else:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse)

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()

        lamdba_ = np.linalg.inv(effective_mass) @ rhs

        impulse = self.jacobian.transpose() @ lamdba_
        generic_delta_velocity += self.generic_inv_mass @ impulse

        self.body_A.delta_linear_velocity = generic_delta_velocity[0:3, 0].reshape(3)
        self.body_A.delta_angular_velocity = generic_delta_velocity[3:6, 0].reshape(3)
        self.body_B.delta_linear_velocity = generic_delta_velocity[6:9, 0].reshape(3)
        self.body_B.delta_angular_velocity = generic_delta_velocity[9:12, 0].reshape(3)

class DistanceConstraint:
    def __init__(self, body_A: Body, body_B: Body, point_A: Vec3, point_B: Vec3, distance: float):
        self.body_A = body_A
        self.body_B = body_B
        self.r_A = body_A.pose.basis.transpose() @ (point_A - body_A.pose.origin)
        self.r_B = body_B.pose.basis.transpose() @ (point_B - body_B.pose.origin)
        self.distance = distance
    
    def setup(self, dt: float):
        a = self.body_A.pose.basis @ self.r_A + self.body_A.pose.origin
        b = self.body_B.pose.basis @ self.r_B + self.body_B.pose.origin
        n = normalized(a - b)
        c_init = norm(a - b) - self.distance

        self.jacobian = np.zeros((1, 12))
        self.jacobian[0, 0:3] = n.transpose()
        self.jacobian[0, 3:6] = -n.transpose() @ skew_symmetric_matrix(self.body_A.pose.basis @ self.r_A)
        self.jacobian[0, 6:9] = -n.transpose()
        self.jacobian[0, 9:12] = n.transpose() @ skew_symmetric_matrix(self.body_B.pose.basis @ self.r_B)

        self.generic_inv_mass = np.zeros((12, 12))
        self.generic_inv_mass[0:3, 0:3] = self.body_A.inv_mass
        self.generic_inv_mass[3:6, 3:6] = self.body_A.inv_inertia_world
        self.generic_inv_mass[6:9, 6:9] = self.body_B.inv_mass
        self.generic_inv_mass[9:12, 9:12] = self.body_B.inv_inertia_world

        self.generic_velocity = np.zeros((12, 1))
        self.generic_velocity[0:3, 0] = self.body_A.linear_velocity
        self.generic_velocity[3:6, 0] = self.body_A.angular_velocity
        self.generic_velocity[6:9, 0] = self.body_B.linear_velocity
        self.generic_velocity[9:12, 0] = self.body_B.angular_velocity

        self.generic_external_impulse = np.zeros((12, 1))
        self.generic_external_impulse[0:3, 0] = self.body_A.inv_mass @ self.body_A.total_force * dt
        self.generic_external_impulse[3:6, 0] = self.body_A.inv_inertia_world @ self.body_A.total_torque * dt
        self.generic_external_impulse[6:9, 0] = self.body_B.inv_mass @ self.body_B.total_force * dt
        self.generic_external_impulse[9:12, 0] = self.body_B.inv_inertia_world @ self.body_B.total_torque * dt

        erp = 0.0
        self.bias = erp / dt * c_init

    def iteration(self, with_baumgarte_stabilization: bool):
        generic_delta_velocity = np.zeros((12, 1))
        generic_delta_velocity[0:3, 0] = self.body_A.delta_linear_velocity
        generic_delta_velocity[3:6, 0] = self.body_A.delta_angular_velocity
        generic_delta_velocity[6:9, 0] = self.body_B.delta_linear_velocity
        generic_delta_velocity[9:12, 0] = self.body_B.delta_angular_velocity

        if with_baumgarte_stabilization:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse) - self.bias
        else:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse)

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = np.linalg.inv(effective_mass) @ rhs
        impulse = self.jacobian.transpose() @ lamdba_
        generic_delta_velocity += self.generic_inv_mass @ impulse

        self.body_A.delta_linear_velocity = generic_delta_velocity[0:3, 0].reshape(3)
        self.body_A.delta_angular_velocity = generic_delta_velocity[3:6, 0].reshape(3)
        self.body_B.delta_linear_velocity = generic_delta_velocity[6:9, 0].reshape(3)
        self.body_B.delta_angular_velocity = generic_delta_velocity[9:12, 0].reshape(3)


class PinConstraint:
    def __init__(self, body_A: Body, body_B: Body, anchor_A: Vec3, anchor_B: Vec3):
        self.body_A = body_A
        self.body_B = body_B
        self.r_A = body_A.pose.basis.transpose() @ (anchor_A - body_A.pose.origin)
        self.r_B = body_B.pose.basis.transpose() @ (anchor_B - body_B.pose.origin)
    
    def setup(self, dt: float):
        c_init = self.body_A.pose.origin + self.body_A.pose.basis @ self.r_A - \
                 (self.body_B.pose.origin + self.body_B.pose.basis @ self.r_B)

        self.jacobian = np.zeros((3, 12))
        self.jacobian[:, 0:3] = np.eye(3)
        self.jacobian[:, 3:6] = -skew_symmetric_matrix(self.body_A.pose.basis @ self.r_A)
        self.jacobian[:, 6:9] = -np.eye(3)
        self.jacobian[:, 9:12] = skew_symmetric_matrix(self.body_B.pose.basis @ self.r_B)

        self.generic_inv_mass = np.zeros((12, 12))
        self.generic_inv_mass[0:3, 0:3] = self.body_A.inv_mass
        self.generic_inv_mass[3:6, 3:6] = self.body_A.inv_inertia_world
        self.generic_inv_mass[6:9, 6:9] = self.body_B.inv_mass
        self.generic_inv_mass[9:12, 9:12] = self.body_B.inv_inertia_world

        self.generic_velocity = np.zeros((12, 1))
        self.generic_velocity[0:3, 0] = self.body_A.linear_velocity
        self.generic_velocity[3:6, 0] = self.body_A.angular_velocity
        self.generic_velocity[6:9, 0] = self.body_B.linear_velocity
        self.generic_velocity[9:12, 0] = self.body_B.angular_velocity

        self.generic_external_impulse = np.zeros((12, 1))
        self.generic_external_impulse[0:3, 0] = self.body_A.inv_mass @ self.body_A.total_force * dt
        self.generic_external_impulse[3:6, 0] = self.body_A.inv_inertia_world @ self.body_A.total_torque * dt
        self.generic_external_impulse[6:9, 0] = self.body_B.inv_mass @ self.body_B.total_force * dt
        self.generic_external_impulse[9:12, 0] = self.body_B.inv_inertia_world @ self.body_B.total_torque * dt

        erp = 0.2
        self.bias = erp / dt * c_init.reshape(3, 1)

    def iteration(self, with_baumgarte_stabilization: bool):
        generic_delta_velocity = np.zeros((12, 1))
        generic_delta_velocity[0:3, 0] = self.body_A.delta_linear_velocity
        generic_delta_velocity[3:6, 0] = self.body_A.delta_angular_velocity
        generic_delta_velocity[6:9, 0] = self.body_B.delta_linear_velocity
        generic_delta_velocity[9:12, 0] = self.body_B.delta_angular_velocity

        if with_baumgarte_stabilization:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse) - self.bias
        else:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse)

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = solve_gauss_seidel(effective_mass, rhs)
        impulse = self.jacobian.transpose() @ lamdba_
        generic_delta_velocity += self.generic_inv_mass @ impulse

        self.body_A.delta_linear_velocity = generic_delta_velocity[0:3, 0].reshape(3)
        self.body_A.delta_angular_velocity = generic_delta_velocity[3:6, 0].reshape(3)
        self.body_B.delta_linear_velocity = generic_delta_velocity[6:9, 0].reshape(3)
        self.body_B.delta_angular_velocity = generic_delta_velocity[9:12, 0].reshape(3)

class HingeRotationConstraintPart:
    def __init__(self, body_A: Body, body_B: Body, axis: Vec3):
        assert norm(axis) >= 1e-10
        axis_normalized = axis / norm(axis)
        self.body_A = body_A
        self.body_B = body_B
        self.axis_A = body_A.pose.basis.transpose() @ axis_normalized
        self.axis_B = body_B.pose.basis.transpose() @ axis_normalized
        if abs(np.dot(self.axis_A, Vec3(1, 0, 0))) <= 1e-10:
            self.normal_B = np.cross(self.axis_B, Vec3(0, 1, 0))
        else:
            self.normal_B = np.cross(self.axis_B, Vec3(1, 0, 0))
        self.normal_B = self.normal_B / norm(self.normal_B)
        self.tangent_B = np.cross(self.axis_B, self.normal_B)
    
    def setup(self, dt: float):
        # C = (a_A * n_B) = (0)
        #     (a_A * t_B)   (0)
        a_A = self.body_A.pose.basis.transpose() @ self.axis_A
        n_B = self.body_B.pose.basis.transpose() @ self.normal_B
        t_B = self.body_B.pose.basis.transpose() @ self.tangent_B

        c_init = np.array([
            [np.dot(a_A, n_B)], 
            [np.dot(a_A, t_B)]
        ])

        self.jacobian = np.zeros((2, 12))
        self.jacobian[0, 0:3] = np.zeros(3)
        self.jacobian[0, 3:6] = np.cross(a_A, n_B)
        self.jacobian[0, 6:9] = np.zeros(3)
        self.jacobian[0, 9:12] = -np.cross(a_A, n_B)
        self.jacobian[1, 0:3] = np.zeros(3)
        self.jacobian[1, 3:6] = np.cross(a_A, t_B)
        self.jacobian[1, 6:9] = np.zeros(3)
        self.jacobian[1, 9:12] = -np.cross(a_A, t_B)

        self.generic_inv_mass = np.zeros((12, 12))
        self.generic_inv_mass[0:3, 0:3] = self.body_A.inv_mass
        self.generic_inv_mass[3:6, 3:6] = self.body_A.inv_inertia_world
        self.generic_inv_mass[6:9, 6:9] = self.body_B.inv_mass
        self.generic_inv_mass[9:12, 9:12] = self.body_B.inv_inertia_world

        self.generic_velocity = np.zeros((12, 1))
        self.generic_velocity[0:3, 0] = self.body_A.linear_velocity
        self.generic_velocity[3:6, 0] = self.body_A.angular_velocity
        self.generic_velocity[6:9, 0] = self.body_B.linear_velocity
        self.generic_velocity[9:12, 0] = self.body_B.angular_velocity

        self.generic_external_impulse = np.zeros((12, 1))
        self.generic_external_impulse[0:3, 0] = self.body_A.inv_mass @ self.body_A.total_force * dt
        self.generic_external_impulse[3:6, 0] = self.body_A.inv_inertia_world @ self.body_A.total_torque * dt
        self.generic_external_impulse[6:9, 0] = self.body_B.inv_mass @ self.body_B.total_force * dt
        self.generic_external_impulse[9:12, 0] = self.body_B.inv_inertia_world @ self.body_B.total_torque * dt

        erp = 0.2
        self.bias = erp / dt * c_init

    def iteration(self, with_baumgarte_stabilization: bool):
        generic_delta_velocity = np.zeros((12, 1))
        generic_delta_velocity[0:3, 0] = self.body_A.delta_linear_velocity
        generic_delta_velocity[3:6, 0] = self.body_A.delta_angular_velocity
        generic_delta_velocity[6:9, 0] = self.body_B.delta_linear_velocity
        generic_delta_velocity[9:12, 0] = self.body_B.delta_angular_velocity

        if with_baumgarte_stabilization:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse) - self.bias
        else:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse)

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = solve_jacobian(effective_mass, rhs)
        impulse = self.jacobian.transpose() @ lamdba_
        generic_delta_velocity += self.generic_inv_mass @ impulse

        self.body_A.delta_linear_velocity = generic_delta_velocity[0:3, 0].reshape(3)
        self.body_A.delta_angular_velocity = generic_delta_velocity[3:6, 0].reshape(3)
        self.body_B.delta_linear_velocity = generic_delta_velocity[6:9, 0].reshape(3)
        self.body_B.delta_angular_velocity = generic_delta_velocity[9:12, 0].reshape(3)

class HingeConstraint:
    def __init__(self, body_A: Body, body_B: Body, anchor: Vec3, axis: Vec3):
        self.translation_constraint = PinConstraint(body_A, body_B, anchor, anchor)
        self.rotation_constraint = HingeRotationConstraintPart(body_A, body_B, axis)
    
    def setup(self, dt: float):
        self.translation_constraint.setup(dt)
        self.rotation_constraint.setup(dt)
    
    def iteration(self, with_baumgarte_stabilization: bool):
        self.translation_constraint.iteration(with_baumgarte_stabilization)
        self.rotation_constraint.iteration(with_baumgarte_stabilization)

class Scene:
    def __init__(self):
        self.gravity = Vec3(0.0, 0.0, -10.0)
        self.bodies = []
        self.temporary_constraints = []
        self.persistent_constraints = []
        self.position_iterations = 2
        self.velocity_iterations = 5

    def add_body(self, body: Body):
        self.bodies.append(body)

    def step_simulation(self, dt: float):
        self.apply_gravity()

        self.contact_detection()
        
        for constraint in self.temporary_constraints:
            constraint.setup(dt)
        for constraint in self.persistent_constraints:
            constraint.setup(dt)
        for _ in range(self.position_iterations):
            for constraint in self.temporary_constraints:
                constraint.iteration(with_baumgarte_stabilization=True)
        self.post_position_iteration(dt)
        for _ in range(self.velocity_iterations):
            for constraint in self.persistent_constraints:
                constraint.iteration(with_baumgarte_stabilization=False)
        self.post_velocity_iteration(dt)
        self.temporary_constraints.clear()

    def apply_gravity(self):
        for body in self.bodies:
            if body.mass != float('inf'):
                body.apply_force(self.gravity * body.mass, body.pose.origin)
    
    def add_distance_constraint(self, body_A: Body, body_B: Body, point_A: Vec3, point_B: Vec3, distance: float):
        self.persistent_constraints.append(DistanceConstraint(body_A, body_B, point_A, point_B, distance))
    
    def add_pin_constraint(self, body_A: Body, body_B: Body, anchor_A: Vec3, anchor_B: Vec3):
        self.persistent_constraints.append(PinConstraint(body_A, body_B, anchor_A, anchor_B))
    
    def add_hinge_constraint(self, body_A: Body, body_B: Body, anchor: Vec3, axis: Vec3):
        self.persistent_constraints.append(HingeConstraint(body_A, body_B, anchor, axis))
    
    def contact_detection(self):
        for i, body in enumerate(self.bodies):
            for j, other_body in enumerate(self.bodies):
                if i >= j:
                    continue
                contact_result = intersect(Shape(body.geometry, body.pose), Shape(other_body.geometry, other_body.pose))
                if contact_result.intersects:
                    print(f"Contact detected between {body} and {other_body}")
                    self.temporary_constraints.append(ContactConstraint(body, other_body, contact_result.point_A, contact_result.point_B, contact_result.normal))

    def post_position_iteration(self, dt: float):
        for body in self.bodies:
            new_linear_velocity = body.linear_velocity + body.delta_linear_velocity + body.inv_mass @ body.total_force * dt
            new_angular_velocity = body.angular_velocity + body.delta_angular_velocity + body.inv_inertia_world @ body.total_torque * dt
            
            body.pose = integrate_transform(body.pose, new_linear_velocity, new_angular_velocity, dt)
            body.inv_inertia_world = body.pose.basis @ body.inv_inertia @ body.pose.basis.transpose()

            body.delta_linear_velocity = Vec3(0, 0, 0)
            body.delta_angular_velocity = Vec3(0, 0, 0)

    def post_velocity_iteration(self, dt: float):
        for body in self.bodies:
            body.linear_velocity = body.linear_velocity + body.delta_linear_velocity + body.inv_mass @ body.total_force * dt
            body.angular_velocity = body.angular_velocity + body.delta_angular_velocity + body.inv_inertia_world @ body.total_torque * dt
            
            body.total_force = Vec3(0, 0, 0)
            body.total_torque = Vec3(0, 0, 0)
            body.delta_linear_velocity = Vec3(0, 0, 0)
            body.delta_angular_velocity = Vec3(0, 0, 0)

class SceneDebugRenderer:
    def __init__(self, scene: Scene, width: int, height: int):
        self.scene = scene
        self.renderer = Renderer(width=width, height=height)

    def is_running(self):
        return self.renderer.is_running()

    def render(self):
        self.renderer.begin_frame()
        for body in self.scene.bodies:
            self.draw_body(body, np.array([1.0, 0.0, 0.0]))
        for constraint in self.scene.persistent_constraints:
            self.draw_constraint(constraint, np.array([0.0, 1.0, 1.0]))
        self.renderer.end_frame()

    def draw_body(self, body: 'Body', color: np.ndarray):
        world_matrix = body.pose.to_matrix()

        if isinstance(body.geometry, Box):
            scale = body.geometry.half_extents * 2
            scale_matrix = np.diag(np.array([scale[0], scale[1], scale[2], 1.0]))
            self.renderer.draw_box(world_matrix @ scale_matrix, color)
        
        elif isinstance(body.geometry, Sphere):
            scale_matrix = np.diag(np.array([body.geometry.radius * 2, body.geometry.radius * 2, body.geometry.radius * 2, 1.0]))
            self.renderer.draw_sphere(world_matrix @ scale_matrix, color)
    
    def draw_constraint(self, constraint, color: np.ndarray):
        scale = 0.4
        scale_matrix = np.diag(np.array([scale, scale, scale, 1.0]))
        if isinstance(constraint, DistanceConstraint):
            mat_A = constraint.body_A.pose.to_matrix() @ Transform(constraint.r_A, Mat33.identity()).to_matrix() @ scale_matrix
            mat_B = constraint.body_B.pose.to_matrix() @ Transform(constraint.r_B, Mat33.identity()).to_matrix() @ scale_matrix
            self.renderer.draw_sphere(mat_A, color)
            self.renderer.draw_sphere(mat_B, color)


if __name__ == "__main__":
    scene = Scene()
    
    sphere1 = Body(mass=1.0, linear_velocity=Vec3(1.0, 0.0, 10.0), pose=Transform(Vec3(-1.0, 0.0, 3.0), Mat33.identity()), geometry=Sphere(0.5))
    scene.add_body(sphere1)

    ground = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(1.0, 0.0, 1.0), 0.0))
    scene.add_body(ground)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()