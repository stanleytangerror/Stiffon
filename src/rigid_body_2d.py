import numpy as np
np.seterr(all='raise')

from math_utils import Vec2, Transform2d, integrate_transform2d, solve_gauss_seidel, cross21_2d, cross22_2d, rotational_inertia_around_offset_2d
from geometry_2d import Rectangle, Circle
from renderer import Renderer

class Body2d:
    def __init__(self, mass: float, inertia: float = 1.0, 
            linear_velocity: Vec2 = Vec2(0, 0), angular_velocity: float = 0.0, 
            pose: Transform2d = Transform2d(Vec2(0, 0), 0.0), 
            geometry: Rectangle | Circle = Rectangle(Vec2(0.5, 0.5))):  

        safe_inv_mass = lambda m: 1.0 / m if m != float('inf') else 0.0
        safe_inv_inertia = lambda i: 1.0 / i if i != float('inf') else 0.0
        
        self.mass = mass
        self.inv_mass = safe_inv_mass(mass)
        self.inertia = inertia
        self.inv_inertia = safe_inv_inertia(inertia)
        self.inv_inertia_world = self.inv_inertia
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.pose = pose
        self.geometry = geometry
        self.total_force = Vec2(0, 0)
        self.total_torque = 0.0
        self.delta_linear_velocity = Vec2(0, 0)
        self.delta_angular_velocity = 0.0

    def apply_force(self, force: Vec2, point: Vec2):
        """Apply a force at a point. In 2D, torque is a scalar (cross product z-component)"""
        self.total_force += force
        # In 2D, torque = cross2d(r, force) where r = point - origin
        r = point - self.pose.origin
        self.total_torque += cross22_2d(r, force)

def get_generic_inverse_mass(body_A: Body2d, body_B: Body2d):
    generic_inv_mass = np.zeros((6, 6))
    generic_inv_mass[0:2, 0:2] = np.eye(2) * body_A.inv_mass
    generic_inv_mass[2, 2] = body_A.inv_inertia_world
    generic_inv_mass[3:5, 3:5] = np.eye(2) * body_B.inv_mass
    generic_inv_mass[5, 5] = body_B.inv_inertia_world
    return generic_inv_mass

def get_generic_external_impulse(body_A: Body2d, body_B: Body2d, dt: float):
    generic_external_impulse = np.zeros((6, 1))
    generic_external_impulse[0:2, 0] = body_A.inv_mass * body_A.total_force * dt
    generic_external_impulse[2, 0] = body_A.inv_inertia_world * body_A.total_torque * dt
    generic_external_impulse[3:5, 0] = body_B.inv_mass * body_B.total_force * dt
    generic_external_impulse[5, 0] = body_B.inv_inertia_world * body_B.total_torque * dt
    return generic_external_impulse

def get_generic_velocity(body_A: Body2d, body_B: Body2d):
    generic_velocity = np.zeros((6, 1))
    generic_velocity[0:2, 0] = body_A.linear_velocity
    generic_velocity[2, 0] = body_A.angular_velocity
    generic_velocity[3:5, 0] = body_B.linear_velocity
    generic_velocity[5, 0] = body_B.angular_velocity
    return generic_velocity

def get_generic_delta_velocity(body_A: Body2d, body_B: Body2d):
    generic_delta_velocity = np.zeros((6, 1))
    generic_delta_velocity[0:2, 0] = body_A.delta_linear_velocity
    generic_delta_velocity[2, 0] = body_A.delta_angular_velocity
    generic_delta_velocity[3:5, 0] = body_B.delta_linear_velocity
    generic_delta_velocity[5, 0] = body_B.delta_angular_velocity
    return generic_delta_velocity

def add_back_generic_delta_velocity(body_A: Body2d, body_B: Body2d, generic_delta_velocity: np.ndarray):
    body_A.delta_linear_velocity += Vec2(generic_delta_velocity[0, 0], generic_delta_velocity[1, 0])
    body_A.delta_angular_velocity += generic_delta_velocity[2, 0]
    body_B.delta_linear_velocity += Vec2(generic_delta_velocity[3, 0], generic_delta_velocity[4, 0])
    body_B.delta_angular_velocity += generic_delta_velocity[5, 0]

class PinConstraint2d:
    def __init__(self, body_A: Body2d, body_B: Body2d, point_A: Vec2, point_B: Vec2):
        self.body_A = body_A
        self.body_B = body_B
        self.anchor_A = body_A.pose.inverse().transformPosition(point_A)
        self.anchor_B = body_B.pose.inverse().transformPosition(point_B)
        self.applied_impulse_magnitude = np.zeros((2, 1))
        self.generic_inv_mass = get_generic_inverse_mass(body_A, body_B)
    
    def setup(self, dt: float):
        # Rotate anchor points from local to world space
        r_A = self.body_A.pose.transformDirection(self.anchor_A)
        r_B = self.body_B.pose.transformDirection(self.anchor_B)

        c_init = self.body_A.pose.origin + r_A - (self.body_B.pose.origin + r_B)

        # C = v_A + ω_A × r_A - v_B - ω_B × r_B in R^2
        # J = [ I_2, -[r_A]x, -I_2, [r_B]x ] in R^2x6
        self.jacobian = np.zeros((2, 6))
        self.jacobian[:, 0:2] = np.eye(2)
        self.jacobian[:, 2] = -cross21_2d(r_A, 1.0)
        self.jacobian[:, 3:5] = -np.eye(2)
        self.jacobian[:, 5] = cross21_2d(r_B, 1.0)

        self.generic_velocity = get_generic_velocity(self.body_A, self.body_B)
        self.generic_external_impulse = get_generic_external_impulse(self.body_A, self.body_B, dt)

        erp = 0.2
        self.bias = erp / dt * c_init.reshape(2, 1)

    def warm_up(self):
        warm_start_delta_velocity = self.generic_inv_mass @ self.jacobian.transpose() @ self.applied_impulse_magnitude

        add_back_generic_delta_velocity(self.body_A, self.body_B, warm_start_delta_velocity)

    def iteration(self, is_positional_iteration: bool):
        generic_delta_velocity = get_generic_delta_velocity(self.body_A, self.body_B)

        if is_positional_iteration:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse) - self.bias
        else:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse)

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = solve_gauss_seidel(effective_mass, rhs)
        impulse = self.jacobian.transpose() @ lamdba_

        print('is_positional_iteration', is_positional_iteration, 'lambda: ', lamdba_, 'impulse: ', impulse, 'effective_mass: ', effective_mass, 'rhs: ', rhs)

        if not is_positional_iteration:
            self.applied_impulse_magnitude += lamdba_

        generic_delta_velocity_addition = self.generic_inv_mass @ impulse

        add_back_generic_delta_velocity(self.body_A, self.body_B, generic_delta_velocity_addition)

class SpringConstraint2d:
    def __init__(self, body_A: Body2d, body_B: Body2d, point_A: Vec2, point_B: Vec2, stiffness: float, damping: float):
        self.body_A = body_A
        self.body_B = body_B
        self.anchor_A = body_A.pose.inverse().transformPosition(point_A)
        self.anchor_B = body_B.pose.inverse().transformPosition(point_B)
        self.stiffness = stiffness
        self.damping = damping
        self.generic_inv_mass = get_generic_inverse_mass(body_A, body_B)
        self.reduced_mass = (body_A.mass if body_B.mass == float('inf') else
                             body_B.mass if body_A.mass == float('inf') else
                             (body_A.mass * body_B.mass) / (body_A.mass + body_B.mass))
    
    def warm_up(self):
        pass

    def setup(self, dt: float):
        
        r_A = self.body_A.pose.transformDirection(self.anchor_A)
        r_B = self.body_B.pose.transformDirection(self.anchor_B)

        # C = v_A + ω_A × r_A - v_B - ω_B × r_B in R^2
        # J = [ I_2, -[r_A]x, -I_2, [r_B]x ] in R^2x6
        self.jacobian = np.zeros((2, 6))
        self.jacobian[:, 0:2] = np.eye(2)
        self.jacobian[:, 2] = -cross21_2d(r_A, 1.0)
        self.jacobian[:, 3:5] = -np.eye(2)
        self.jacobian[:, 5] = cross21_2d(r_B, 1.0)

        # In 2D, angular velocity is scalar, so cross product: ω × r = ω * (-r.y, r.x)
        x_error = self.body_A.pose.origin + r_A - self.body_B.pose.origin - r_B
        v_angular_A = Vec2(-r_A.y * self.body_A.angular_velocity, r_A.x * self.body_A.angular_velocity)
        v_angular_B = Vec2(-r_B.y * self.body_B.angular_velocity, r_B.x * self.body_B.angular_velocity)
        v_error = self.body_A.linear_velocity + v_angular_A - self.body_B.linear_velocity - v_angular_B
        fs = -self.stiffness * x_error
        fd = -self.damping * v_error
        f = fs + fd
        self.delta_relative_velocity = (f * dt / self.reduced_mass).reshape(2, 1)
        self.impulse_lower_limit = np.minimum(fd, np.minimum(f, np.zeros(2))).reshape(2, 1) * dt
        self.impulse_upper_limit = np.maximum(fd, np.maximum(f, np.zeros(2))).reshape(2, 1) * dt

    def iteration(self, is_positional_iteration: bool):
        if is_positional_iteration:
            pass

        generic_delta_velocity = get_generic_delta_velocity(self.body_A, self.body_B)

        rhs = -self.jacobian @ generic_delta_velocity + self.delta_relative_velocity

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = solve_gauss_seidel(effective_mass, rhs)

        # the solved `lambda` is the impulse that will make the bodies behave exactly like being affected by this spring
        # however, as a soft constraint, there can be other hard constraints or more stiff constrains break the target delta velocity
        # so we need to clamp the impulse i.e. `lambda` to valid range
        lamdba_ = np.minimum(np.maximum(lamdba_, self.impulse_lower_limit), self.impulse_upper_limit)
        
        impulse = self.jacobian.transpose() @ lamdba_
        generic_delta_velocity_addition = self.generic_inv_mass @ impulse

        add_back_generic_delta_velocity(self.body_A, self.body_B, generic_delta_velocity_addition)

class AngularSpringConstraint2d:
    def __init__(self, body_A: Body2d, body_B: Body2d, point_A: Vec2, point_B: Vec2, stiffness: float, damping: float):
        self.body_A = body_A
        self.body_B = body_B
        self.anchor_A = body_A.pose.inverse().transformPosition(point_A)
        self.anchor_B = body_B.pose.inverse().transformPosition(point_B)
        self.stiffness = stiffness
        self.damping = damping

        self.generic_inv_mass = get_generic_inverse_mass(body_A, body_B)

        rotational_inertia_A = rotational_inertia_around_offset_2d(self.body_A.inertia, self.body_A.mass, self.anchor_A)
        rotational_inertia_B = rotational_inertia_around_offset_2d(self.body_B.inertia, self.body_B.mass, self.anchor_B)
        self.reduced_inertia = (rotational_inertia_A if rotational_inertia_B == float('inf') else \
                             rotational_inertia_B if rotational_inertia_A == float('inf') else \
                             (rotational_inertia_A * rotational_inertia_B) / (rotational_inertia_A + rotational_inertia_B))
    
    def warm_up(self):
        pass

    def setup(self, dt: float):
        
        # C = ω_A - ω_B in R
        # J = [ 0_2, I, 0_2, -I ] in R^6
        self.jacobian = np.zeros((1, 6))
        self.jacobian[0, 2] = 1.0
        self.jacobian[0, 5] = -1.0

        # In 2D, angular velocity is scalar, so cross product: ω × r = ω * (-r.y, r.x)
        x_error = self.body_A.pose.angle - self.body_B.pose.angle
        v_error = self.body_A.angular_velocity - self.body_B.angular_velocity
        fs = -self.stiffness * x_error
        fd = -self.damping * v_error
        f = fs + fd

        self.delta_relative_velocity = f * dt / self.reduced_inertia
        
        self.impulse_lower_limit = min(f, fd, 0)* dt
        self.impulse_upper_limit = max(f, fd, 0) * dt

    def iteration(self, is_positional_iteration: bool):
        if is_positional_iteration:
            pass

        generic_delta_velocity = get_generic_delta_velocity(self.body_A, self.body_B)

        rhs = -self.jacobian @ generic_delta_velocity + self.delta_relative_velocity

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = rhs / effective_mass

        # the solved `lambda` is the impulse that will make the bodies behave exactly like being affected by this spring
        # however, as a soft constraint, there can be other hard constraints or more stiff constrains break the target delta velocity
        # so we need to clamp the impulse i.e. `lambda` to valid range
        lamdba_ = np.minimum(np.maximum(lamdba_, self.impulse_lower_limit), self.impulse_upper_limit)
        
        impulse = self.jacobian.transpose() @ lamdba_
        generic_delta_velocity_addition = self.generic_inv_mass @ impulse

        add_back_generic_delta_velocity(self.body_A, self.body_B, generic_delta_velocity_addition)


class PrismaticConstraint2d_LinearPart:
    def __init__(self, body_A: Body2d, body_B: Body2d, 
        localPoint_A: Vec2, localAxis_A: Vec2, 
        localPoint_B: Vec2):

        self.body_A = body_A
        self.body_B = body_B
        self.localPoint_A = localPoint_A
        self.localAxis_A = localAxis_A
        self.localPoint_B = localPoint_B
        self.applied_impulse_magnitude = np.zeros((1, 1))
        self.generic_inv_mass = get_generic_inverse_mass(body_A, body_B)
    
    def setup(self, dt: float):
        # Rotate anchor points from local to world space
        r_A = self.body_A.pose.transformDirection(self.localPoint_A)
        r_B = self.body_B.pose.transformDirection(self.localPoint_B)

        axis_A = self.body_A.pose.transformDirection(self.localAxis_A)
        normal_A = Vec2(-axis_A.y, axis_A.x)
        c_init = normal_A.T @ (self.body_A.pose.origin + r_A - (self.body_B.pose.origin + r_B))

        # C = normal_A^T (v_A + ω_A × r_A - v_B - ω_B × r_B) in R
        # J = [ normal_A^T, (r_A x normal_A)^T, -normal_A^T, -(r_B x normal_A)^T ]
        self.jacobian = np.zeros((1, 6))
        self.jacobian[0, 0:2] = normal_A
        self.jacobian[0, 2] = cross22_2d(r_A, normal_A)
        self.jacobian[0, 3:5] = -normal_A
        self.jacobian[0, 5] = -cross22_2d(r_B, normal_A)

        self.generic_velocity = get_generic_velocity(self.body_A, self.body_B)
        self.generic_external_impulse = get_generic_external_impulse(self.body_A, self.body_B, dt)

        erp = 0.2
        self.bias = erp / dt * c_init

    def warm_up(self):
        warm_start_delta_velocity = self.generic_inv_mass @ self.jacobian.transpose() @ self.applied_impulse_magnitude

        add_back_generic_delta_velocity(self.body_A, self.body_B, warm_start_delta_velocity)

    def iteration(self, is_positional_iteration: bool):
        generic_delta_velocity = get_generic_delta_velocity(self.body_A, self.body_B)

        if is_positional_iteration:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse) - self.bias
        else:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse)

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = np.linalg.inv(effective_mass) @ rhs
        impulse = self.jacobian.transpose() @ lamdba_

        if not is_positional_iteration:
            self.applied_impulse_magnitude += lamdba_

        generic_delta_velocity_addition = self.generic_inv_mass @ impulse

        add_back_generic_delta_velocity(self.body_A, self.body_B, generic_delta_velocity_addition)


class PrismaticConstraint2d_AngularPart:
    def __init__(self, body_A: Body2d, body_B: Body2d, 
        localAxis_A: Vec2, localAxis_B: Vec2):

        self.body_A = body_A
        self.body_B = body_B
        self.localAxis_A = localAxis_A
        self.localAxis_B = localAxis_B
        self.applied_impulse_magnitude = np.zeros((1, 1))
        self.generic_inv_mass = get_generic_inverse_mass(body_A, body_B)
    
    def warm_up(self):
        warm_start_delta_velocity = self.generic_inv_mass @ self.jacobian.transpose() @ self.applied_impulse_magnitude

        add_back_generic_delta_velocity(self.body_A, self.body_B, warm_start_delta_velocity)

    def setup(self, dt: float):
        
        c_init = self.body_A.pose.transformDirection(self.localAxis_A) - self.body_B.pose.transformDirection(self.localAxis_B)

        # C = ω_A - ω_B in R
        # J = [ 0_2, I, 0_2, -I ] in R^6
        self.jacobian = np.zeros((1, 6))
        self.jacobian[0, 2] = 1.0
        self.jacobian[0, 5] = -1.0

        self.generic_velocity = get_generic_velocity(self.body_A, self.body_B)
        self.generic_external_impulse = get_generic_external_impulse(self.body_A, self.body_B, dt)

        erp = 0.2
        self.bias = erp / dt * c_init

    def iteration(self, is_positional_iteration: bool):
        generic_delta_velocity = get_generic_delta_velocity(self.body_A, self.body_B)

        if is_positional_iteration:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse) - self.bias
        else:
            rhs = -self.jacobian @ (self.generic_velocity + generic_delta_velocity + self.generic_external_impulse)

        effective_mass = self.jacobian @ self.generic_inv_mass @ self.jacobian.transpose()
        lamdba_ = solve_gauss_seidel(effective_mass, rhs)
        impulse = self.jacobian.transpose() @ lamdba_

        if not is_positional_iteration:
            self.applied_impulse_magnitude += lamdba_

        generic_delta_velocity_addition = self.generic_inv_mass @ impulse

        add_back_generic_delta_velocity(self.body_A, self.body_B, generic_delta_velocity_addition)


class Scene2d:
    def __init__(self):
        self.gravity = Vec2(0.0, -10.0)
        self.bodies = []
        self.last_delta_time = None
        self.persistent_constraints = []
        self.position_iterations = 1
        self.velocity_iterations = 1

    def set_gravity(self, f: Vec2):
        self.gravity = f

    def add_body(self, body: Body2d):
        self.bodies.append(body)

    def step_simulation(self, dt: float):
        print("======================")

        if self.last_delta_time is None:
            self.last_delta_time = dt

        self.apply_gravity()

        for constraint in self.persistent_constraints:
            constraint.setup(dt)

        for _ in range(self.position_iterations):
            for constraint in self.persistent_constraints:
                constraint.iteration(is_positional_iteration=True)
        self.post_position_iteration(dt)

        # for constraint in self.persistent_constraints:
        #     constraint.warm_up()
        for _ in range(self.position_iterations):
            for constraint in self.persistent_constraints:
                constraint.iteration(is_positional_iteration=False)
        self.post_velocity_iteration(dt)

        self.last_delta_time = dt

    def apply_gravity(self):
        for body in self.bodies:
            if body.mass != float('inf'):
                body.apply_force(self.gravity * body.mass, body.pose.origin)
    
    def add_pin_constraint(self, body_A: Body2d, body_B: Body2d, anchorInWorld_A: Vec2, anchorInWorld_B: Vec2):
        self.persistent_constraints.append(PinConstraint2d(body_A, body_B, anchorInWorld_A, anchorInWorld_B))
    
    def add_spring_constraint(self, body_A: Body2d, body_B: Body2d, point_A: Vec2, point_B: Vec2, stiffness: float, damping: float):
        self.persistent_constraints.append(SpringConstraint2d(body_A, body_B, point_A, point_B, stiffness, damping))
    
    def add_angular_spring_constraint(self, body_A: Body2d, body_B: Body2d, point_A: Vec2, point_B: Vec2, stiffness: float, damping: float):
        self.persistent_constraints.append(AngularSpringConstraint2d(body_A, body_B, point_A, point_B, stiffness, damping))
    
    def add_prismatic_constraint(self, body_A: Body2d, body_B: Body2d, pointInWorld_A: Vec2, axisInWorld_A: Vec2, pointInWorld_B: Vec2, axisInWorld_B: Vec2):
        localPoint_A = body_A.pose.inverse().transformPosition(pointInWorld_A)
        localPoint_B = body_B.pose.inverse().transformPosition(pointInWorld_B)
        localAxis_A = body_A.pose.inverse().transformDirection(axisInWorld_A)
        localAxis_B = body_B.pose.inverse().transformDirection(axisInWorld_B)
        self.persistent_constraints.append(PrismaticConstraint2d_LinearPart(body_A, body_B, localPoint_A, localAxis_A, localPoint_B))
        self.persistent_constraints.append(PrismaticConstraint2d_AngularPart(body_A, body_B, localAxis_A, localAxis_B))

    def post_position_iteration(self, dt: float):
        for body in self.bodies:
            new_linear_velocity = body.linear_velocity + body.delta_linear_velocity + body.inv_mass * body.total_force * dt
            new_angular_velocity = body.angular_velocity + body.delta_angular_velocity + body.inv_inertia_world * body.total_torque * dt
            
            body.pose = integrate_transform2d(body.pose, new_linear_velocity, new_angular_velocity, dt)
            body.inv_inertia_world = body.inv_inertia

            body.delta_linear_velocity = Vec2(0, 0)
            body.delta_angular_velocity = 0.0

            print("-----------------")
            print('post_position_iteration', body.pose.origin, body.pose.angle, body.linear_velocity, body.angular_velocity)

    def post_velocity_iteration(self, dt: float):
        for body in self.bodies:
            body.linear_velocity = body.linear_velocity + body.delta_linear_velocity + body.inv_mass * body.total_force * dt
            body.angular_velocity = body.angular_velocity + body.delta_angular_velocity + body.inv_inertia_world * body.total_torque * dt
            
            body.total_force = Vec2(0, 0)
            body.total_torque = 0.0
            body.delta_linear_velocity = Vec2(0, 0)
            body.delta_angular_velocity = 0.0

            print("-----------------")
            print('post_velocity_iteration', body.pose.origin, body.pose.angle, body.linear_velocity, body.angular_velocity)

class Scene2dDebugRenderer:
    def __init__(self, scene: Scene2d, width: int, height: int):
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

    def draw_body(self, body: Body2d, color: np.ndarray):
        world_matrix = body.pose.to_matrix()

        if isinstance(body.geometry, Rectangle):
            # Convert 2D rectangle to 3D box for rendering
            scale = body.geometry.half_extents * 2
            # Extend to 3D: use z=1.0 for depth
            scale_matrix = np.diag(np.array([scale[0], scale[1], 1.0, 1.0]))
            # Convert 3x3 matrix to 4x4
            world_4x4 = np.eye(4)
            world_4x4[:2, :2] = world_matrix[:2, :2]
            world_4x4[:2, 3] = world_matrix[:2, 2]
            self.renderer.draw_box(world_4x4 @ scale_matrix, color)
        
        elif isinstance(body.geometry, Circle):
            # Convert 2D circle to 3D sphere for rendering
            scale_matrix = np.diag(np.array([body.geometry.radius * 2, body.geometry.radius * 2, 1.0, 1.0]))
            world_4x4 = np.eye(4)
            world_4x4[:2, :2] = world_matrix[:2, :2]
            world_4x4[:2, 3] = world_matrix[:2, 2]
            self.renderer.draw_sphere(world_4x4 @ scale_matrix, color)
    
    def draw_constraint(self, constraint, color: np.ndarray):
        scale = 0.4
        scale_matrix = np.diag(np.array([scale, scale, scale, 1.0]))
        if isinstance(constraint, SpringConstraint2d) or isinstance(constraint, PinConstraint2d):
            # Transform anchor from local to world space
            anchor_A_world = constraint.body_A.pose.transformPosition(constraint.anchor_A)
            anchor_B_world = constraint.body_B.pose.transformPosition(constraint.anchor_B)
            # Create 4x4 matrices for rendering
            mat_A = np.eye(4)
            mat_A[0, 3] = anchor_A_world.x
            mat_A[1, 3] = anchor_A_world.y
            mat_B = np.eye(4)
            mat_B[0, 3] = anchor_B_world.x
            mat_B[1, 3] = anchor_B_world.y
            self.renderer.draw_sphere(mat_A @ scale_matrix, color)
            self.renderer.draw_sphere(mat_B @ scale_matrix, color)


if __name__ == "__main__":
    scene = Scene2d()
    scene.set_gravity(Vec2(0.0, -9.8))
    
    pivot = Body2d(mass=float('inf'), inertia=float('inf'), pose=Transform2d(Vec2(0.0, 0.0), 0.0), geometry=Rectangle(Vec2(1.0, 1.0)))
    scene.add_body(pivot)

    body = Body2d(mass=1.0, pose=Transform2d(Vec2(3.0, 0.0), 0.0), geometry=Rectangle(Vec2(1.0, 1.0)))
    scene.add_body(body)

    scene.add_pin_constraint(pivot, body, Vec2(1.5, 0.0), Vec2(1.5, 0.0))

    renderer = Scene2dDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, 0.0, -10.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()