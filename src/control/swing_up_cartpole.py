import jax.numpy as np
import jax
from dynamics import CartPole, rk4
from visualize import draw_cartpole
import math
from lqr import ihlqr

def collocated_pfl_energy(params, x):
    mc = params['mc']
    mp = params['mp']
    l = params['l']
    g = params['g']

    theta = x[1]
    q_dot = x[2:4].reshape(-1, 1)
    theta_dot = x[3]
    return 0.5 * mp * l**2 * theta_dot**2 - mp * g * l * np.cos(theta)

def test_swing_up():
    # 目标状态：小车在0位置，摆倒立（角度为π），速度为0
    x_goal = np.array([0.0, np.pi, 0.0, 0.0])
    u_goal = np.array([0.0])
    E_goal = collocated_pfl_energy(CartPole.params, x_goal)

    mc = CartPole.params['mc']
    mp = CartPole.params['mp']
    l = CartPole.params['l']
    g = CartPole.params['g']

    dt = 0.01
    t_total = 3.0
    k_e = 1.0
    k_d = 0.5
    k_p = 0.5

    N = int(t_total / dt)

    begin_lqr = False
    A = jax.jacfwd(lambda _x: rk4(CartPole.params, CartPole.dynamics, _x, u_goal, dt))(x_goal)
    B = jax.jacfwd(lambda _u: rk4(CartPole.params, CartPole.dynamics, x_goal, _u, dt))(u_goal)
    Q = np.diag(np.array([1.0, 1.0, 0.01, 0.01]))  # 增加角度权重
    R = 1.0 * np.eye(1)  # 增加控制权重，避免控制过大
    P, K = ihlqr(A, B, Q, R)

    x_init = np.array([0.0, 0.0, 0.0, 0.1])  # 小车位置从0开始，摆接近倒立
    x = x_init
    states = [x_init]
    for i in range(N-1):
        E_error = collocated_pfl_energy(CartPole.params, x) - E_goal
        theta_dot = x[3]
        theta = x[1]
        q = x[0]
        q_dot = x[2]
        s = np.sin(theta)
        c = np.cos(theta)
        
        # print("Energy Error", E_error, "Angle Error", abs(x[1] - x_goal[1]))

        if abs(x[1] - x_goal[1]) < math.radians(1):
            # begin_lqr = True
            print("Begin LQR", x)

        if begin_lqr:
            u = -K @ (x - x_goal) + u_goal
        else:
            x_ddot_desired = k_e * theta_dot * c * (E_error - 0.1) / (mp * l)
            f = (mc + mp - mp * c**2) * x_ddot_desired - mp * g * s * c + mp * l * theta_dot**2 * s
            print("Swing Up", E_error)

            # f += -k_p * (x[0] - x_goal[0]) - k_d * (x[2] - x_goal[2])
            u = np.array([f])

        x = rk4(CartPole.params, CartPole.dynamics, x, u, dt)
        states.append(x)
    
    states = np.array(states)
    draw_cartpole(states, dt)

if __name__ == "__main__":
    test_swing_up()