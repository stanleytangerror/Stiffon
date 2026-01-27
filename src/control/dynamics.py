import jax.numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rigid_body_2d import Scene2d, Body2d, Scene2dDebugRenderer
from math_utils import Vec2, Transform2d, cross12_2d
from geometry_2d import Rectangle, Circle

class Pendulum:
    params = {
        'm': 1.0,
        'l': 0.5,
        'g': 9.81,
    }

    @staticmethod
    def dynamics(params, x, u) -> np.array:
        m = params['m']
        l = params['l']
        g = params['g']

        theta = x[0]
        theta_dot = x[1]
        theta_ddot = - g / l * np.sin(theta) + u[0] / (m * l**2)
        return np.array([theta_dot, theta_ddot]).flatten()

    @staticmethod
    def energy(params, x) -> float:
        m = params['m']
        l = params['l']
        g = params['g']

        theta = x[0]
        theta_dot = x[1]

        T = 0.5 * m * l**2 * theta_dot**2
        U = -m * g * l * np.cos(theta)

        return T + U


class CartPole:
    params = {
        'mc': 1.0,
        'mp': 0.2,
        'l': 0.5,
        'g': 9.81,
    }

    @staticmethod
    def dynamics(params, x, u) -> np.array:
        mc = params['mc']
        mp = params['mp']
        l = params['l']
        g = params['g']

        theta = x[1]
        q_dot = x[2:4].reshape(-1, 1)
        theta_dot = x[3]
        M = np.array([[mc + mp, mp * l * np.cos(theta)],
                    [mp * l * np.cos(theta), mp * l**2]])
        C = np.array([[0, -mp * l * theta_dot * np.sin(theta)],
                    [0, 0]])
        G = np.array([[0], 
                    [mp * g * l * np.sin(theta)]])
        B = np.array([[1], 
                    [0]])
        q_ddot = -np.linalg.inv(M) @ (C @ q_dot + G - B * u[0])
        return np.concatenate([q_dot, q_ddot]).flatten()
    
    @staticmethod
    def energy(params, x) -> float:
        mc = params['mc']
        mp = params['mp']
        l = params['l']
        g = params['g']

        theta = x[1]
        q_dot = x[2:4].reshape(-1, 1)
        theta_dot = x[3]

        T = 0.5 * (mc + mp) * q_dot[0]**2
        T += 0.5 * (mp * l**2) * theta_dot**2
        T += mp * l * q_dot[0] * theta_dot * np.cos(theta)

        U = -mp * g * l * np.cos(theta)

        return T + U

def rk4(params, dynamics, x, u, dt) -> np.array:
    # vanilla RK4
    k1 = dt*dynamics(params, x, u)
    k2 = dt*dynamics(params, x + k1/2, u)
    k3 = dt*dynamics(params, x + k2/2, u)
    k4 = dt*dynamics(params, x + k3, u)
    return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)


class CartPoleScene2d:
    def __init__(self, x):
        self.scene = Scene2d()
        self.scene.set_gravity(Vec2(0.0, -CartPole.params['g']))

        cart_pos = Vec2(x[0], 0.0)
        pole_angle = x[1]
        cart_linear_velocity = Vec2(x[2], 0)
        pole_angular_velocity = x[3]

        r_angle = pole_angle - np.pi / 2
        r = CartPole.params['l'] * Vec2(np.cos(r_angle), np.sin(r_angle))
        pole_linear_velocity = cross12_2d(pole_angular_velocity, r)
        pole_inertia = (1.0 / 3.0) * CartPole.params['mp'] * CartPole.params['l']**2
        
        self.cart = Body2d(
            mass=CartPole.params['mc'],
            inertia=1.0,
            linear_velocity=cart_linear_velocity,
            pose=Transform2d(cart_pos, 0.0),
            geometry=Rectangle(Vec2(0.1, 0.08))
        )
        self.scene.add_body(self.cart)
        
        self.pole = Body2d(
            mass=CartPole.params['mp'],
            inertia=1.0,
            linear_velocity=pole_linear_velocity,
            angular_velocity=pole_angular_velocity,
            pose=Transform2d(cart_pos + r, pole_angle),
            geometry=Circle(0.05)
        )
        self.scene.add_body(self.pole)

        self.pivot = Body2d(
            mass=float('inf'),
            inertia=float('inf'),
            pose=Transform2d(cart_pos, 0.0),
            geometry=Circle(0.02)
        )
        self.scene.add_body(self.pivot)
        
        self.scene.add_pin_constraint(self.cart, self.pole, cart_pos, cart_pos)
        self.scene.add_prismatic_constraint(self.pivot, self.cart, cart_pos, Vec2(1.0, 0.0), cart_pos, Vec2(1.0, 0.0))

    def apply_input(self, u):
        self.cart.apply_force(Vec2(u[0], 0), self.cart.pose.origin)

    def step_simulation(self, dt):
        self.scene.step_simulation(dt)

    def get_state(self):
        theta = self.pole.pose.angle
        return np.array([self.cart.pose.origin.x, theta, self.cart.linear_velocity.x, self.pole.angular_velocity])


if __name__ == "__main__":

    import numpy as np

    scene = CartPoleScene2d(np.array([0.0, np.pi/2, 0.0, 0.0]))
    renderer = Scene2dDebugRenderer(scene.scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, 0.0, -2.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

    while True:
        scene.apply_input(np.array([0.0]))
        scene.step_simulation(0.1)

        renderer.render()
