import jax.numpy as np

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

