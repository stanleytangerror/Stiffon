from math_utils import Vec3, Mat33, Transform, integrate_transform, skew_symmetric_matrix
from geometry import Box, Sphere, Shape, intersect
from renderer import Renderer
import numpy as np

class Body:
    def __init__(self, mass: float, inertia: Vec3 = Vec3(1, 1, 1), 
            linear_velocity: Vec3 = Vec3(0, 0, 0), angular_velocity: Vec3 = Vec3(0, 0, 0), 
            pose: Transform = Transform(Vec3(0, 0, 0), Mat33.identity()), 
            geometry: Box | Sphere = Box(Vec3(0.5, 0.5, 0.5))):
        self.mass = mass
        self.inv_mass = np.eye(3) * (1.0 / mass)
        self.inertia = inertia
        self.inv_inertia = np.diag(np.array([1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z]))
        self.inv_inertia_world = pose.basis @ self.inv_inertia @ pose.basis.transpose()
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.pose = pose
        self.predicted_pose = pose
        self.geometry = geometry
        self.total_force = Vec3(0, 0, 0)
        self.total_torque = Vec3(0, 0, 0)
        self.delta_linear_velocity = Vec3(0, 0, 0)
        self.delta_angular_velocity = Vec3(0, 0, 0)

    def apply_force(self, force: Vec3, point: Vec3):
        self.total_force += force
        self.total_torque += np.cross(point - self.pose.origin, force)

class ContactConstraint:
    def __init__(self, body_A: Body, body_B: Body, point_A: Vec3, point_B: Vec3, contact_normal: Vec3):
        self.body_A = body_A
        self.body_B = body_B
        self.r_A = body_A.pose.basis.transpose() @ (point_A - body_A.pose.origin)
        self.r_B = body_B.pose.basis.transpose() @ (point_B - body_B.pose.origin)
        self.contact_normal = contact_normal

    def setup(self, dt: float):
        self.jacobian = np.zeros((3, 12))
        self.jacobian[0:3, 0:3] = np.diag(self.contact_normal)
        self.jacobian[0:3, 3:6] = self.contact_normal.transpose() @ skew_symmetric_matrix(-self.r_A)
        self.jacobian[0:3, 6:9] = -np.diag(self.contact_normal)
        self.jacobian[0:3, 9:12] = -self.contact_normal.transpose() @ skew_symmetric_matrix(-self.r_B)

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
        self.generic_external_impulse[0:3, 0] = self.body_A.total_force @ self.body_A.inv_mass * dt
        self.generic_external_impulse[3:6, 0] = self.body_A.total_torque @ self.body_A.inv_inertia * dt
        self.generic_external_impulse[6:9, 0] = self.body_B.total_force @ self.body_B.inv_mass * dt
        self.generic_external_impulse[9:12, 0] = self.body_B.total_torque @ self.body_B.inv_inertia * dt

    def iteration(self):
        generic_delta_velocity = np.zeros((12, 1))
        generic_delta_velocity[0:3, 0] = self.body_A.delta_linear_velocity
        generic_delta_velocity[3:6, 0] = self.body_A.delta_angular_velocity
        generic_delta_velocity[6:9, 0] = self.body_B.delta_linear_velocity
        generic_delta_velocity[9:12, 0] = self.body_B.delta_angular_velocity

        rhs = -self.jacobian @ (self.generic_velocity + self.generic_external_impulse + generic_delta_velocity)
        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = np.linalg.pinv(effective_mass) @ rhs
        impulse = self.jacobian.transpose() @ lamdba_
        generic_delta_velocity += self.generic_inv_mass @ impulse

        self.body_A.delta_linear_velocity = generic_delta_velocity[0:3, 0].reshape(3)
        self.body_A.delta_angular_velocity = generic_delta_velocity[3:6, 0].reshape(3)
        self.body_B.delta_linear_velocity = generic_delta_velocity[6:9, 0].reshape(3)
        self.body_B.delta_angular_velocity = generic_delta_velocity[9:12, 0].reshape(3)

class Scene:
    def __init__(self):
        self.bodies = []
        self.constraints = []

    def add_body(self, body: Body):
        self.bodies.append(body)

    def step_simulation(self, dt: float):
        self.predict_pose(dt)
        self.contact_detection()
        for constraint in self.constraints:
            constraint.setup(dt)
        for constraint in self.constraints:
            constraint.iteration()
        self.post_constraint(dt)

    def predict_pose(self, dt: float):
        for body in self.bodies:
            pred_linear_velocity = body.linear_velocity + body.inv_mass @ body.total_force * dt
            pred_angular_velocity = body.angular_velocity + body.inv_inertia_world @ body.total_torque * dt
            body.predicted_pose = integrate_transform(body.pose, pred_linear_velocity, pred_angular_velocity, dt)
    
    def contact_detection(self):
        for body in self.bodies:
            for other_body in self.bodies:
                if body == other_body:
                    continue
                contact_result = intersect(Shape(body.geometry, body.predicted_pose), Shape(other_body.geometry, other_body.predicted_pose))
                if contact_result.intersects:
                    print(f"Contact detected between {body} and {other_body}")
                    self.constraints.append(ContactConstraint(body, other_body, contact_result.point_A, contact_result.point_B, contact_result.normal))

    def post_constraint(self, dt: float):
        for body in self.bodies:
            body.linear_velocity = body.linear_velocity + body.delta_linear_velocity + body.inv_mass @ body.total_force * dt
            body.angular_velocity = body.angular_velocity + body.delta_angular_velocity + body.inv_inertia_world @ body.total_torque * dt
            
            body.pose = integrate_transform(body.pose, body.linear_velocity, body.angular_velocity, dt)
            body.inv_inertia_world = body.pose.basis @ body.inv_inertia @ body.pose.basis.transpose()

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
        self.renderer.end_frame()

    def draw_body(self, body: 'Body', color: np.ndarray):
        world_matrix = body.pose.to_matrix()

        if isinstance(body.geometry, Box):
            scale = body.geometry.half_extents * 2
            scale_matrix = np.diag(np.array([scale[0], scale[1], scale[2], 1.0]))
            self.renderer.draw_box(world_matrix @ scale_matrix, color)
        
        elif isinstance(body.geometry, Sphere):
            scale_matrix = np.diag(np.array([body.geometry.radius * 2, body.geometry.radius * 2, body.geometry.radius * 2, 1.0]))
            self.renderer.draw_sphere(world_matrix @ scale_matrix, color, 10)


if __name__ == "__main__":
    scene = Scene()
    
    sphere1 = Body(mass=1.0, linear_velocity=Vec3(0.3, 0.0, 1.0), pose=Transform(Vec3(-1.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(0.5))
    scene.add_body(sphere1)

    sphere2 = Body(mass=1.0, linear_velocity=Vec3(-0.3, 0.0, 1.0), pose=Transform(Vec3(1.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(0.5))
    scene.add_body(sphere2)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -40.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        sphere1.apply_force(Vec3(0.0, 0.0, -0.98 * sphere1.mass), sphere1.pose.origin)
        sphere2.apply_force(Vec3(0.0, 0.0, -0.98 * sphere2.mass), sphere2.pose.origin)
        scene.step_simulation(0.01)
        renderer.render()