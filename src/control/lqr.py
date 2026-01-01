import jax.numpy as np
import jax
from dynamics import CartPole, Pendulum, rk4
from visualize import draw_cartpole
import math

def fhlqr(A, B, Q, R, Qf, N) -> (list[np.array], list[np.array]):
    nx = A.shape[0]
    nu = B.shape[1]

    P = [np.zeros((nx, nx)) for _ in range(N)]
    K = [np.zeros((nu, nx)) for _ in range(N-1)]

    P[N-1] = Qf

    for i in range(N-2, -1, -1):
        K[i] = np.linalg.inv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
        P[i] = Q + K[i].T @ R @ K[i] + (A - B @ K[i]).T @ P[i+1] @ (A - B @ K[i])
    return P, K

def ihlqr(A, B, Q, R, max_iter = 1000, tol = 1e-5) -> (np.array, np.array):
    P = Q
    for i in range(max_iter):
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P_new = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
        if np.linalg.norm(P_new - P) <= tol:
            break
        P = P_new
    return P, K

def test_ihlqr():
    # 目标状态：小车在0位置，摆倒立（角度为π），速度为0
    x_goal = np.array([0.0, np.pi, 0.0, 0.0])
    u_goal = np.array([0.0])

    dt = 0.01
    t_total = 4.0

    N = int(t_total / dt)

    A = jax.jacfwd(lambda _x: rk4(CartPole.params, CartPole.dynamics, _x, u_goal, dt))(x_goal)
    B = jax.jacfwd(lambda _u: rk4(CartPole.params, CartPole.dynamics, x_goal, _u, dt))(u_goal)
    Q = np.diag(np.array([1.0, 1.0, 0.05, 1.0]))  # 增加角度权重
    R = 0.1 * np.eye(1)  # 增加控制权重，避免控制过大
    P, K = ihlqr(A, B, Q, R)

    x_init = np.array([-2.280083 ,  3.0665107, -2.9715052,  2.0975614])
    x = x_init
    states = [x_init]
    for i in range(N):
        u = -K @ (x - x_goal) + u_goal
        x = rk4(CartPole.params, CartPole.dynamics, x, u, dt)
        states.append(x)
    
    states = np.array(states)
    draw_cartpole(states, dt)

def test_fhlqr():
    # 目标状态：小车在0位置，摆倒立（角度为π），速度为0
    x_goal = np.array([0.0, np.pi, 0.0, 0.0])
    u_goal = np.array([0.0])

    dt = 0.1
    t_total = 3.0

    N = int(t_total / dt)

    A = jax.jacfwd(lambda _x: rk4(CartPole.params, CartPole.dynamics, _x, u_goal, dt))(x_goal)
    B = jax.jacfwd(lambda _u: rk4(CartPole.params, CartPole.dynamics, x_goal, _u, dt))(u_goal)

    Q = np.diag(np.array([1.0, 1.0, 0.05, 1.0]))  # 增加角度权重
    R = 0.1 * np.eye(1)  # 增加控制权重，避免控制过大
    Qf = 5 * Q

    P, K = fhlqr(A, B, Q, R, Qf, N)

    x_init = np.array([0.0, np.pi + math.radians(-20), 0.3, 0.0])  # 小车位置从0开始，摆接近倒立
    x = x_init
    states = [x_init]
    for i in range(N-1):
        u = -K[i] @ (x - x_goal)
        x = rk4(CartPole.params, CartPole.dynamics, x, u, dt)
        states.append(x)
    
    states = np.array(states)
    draw_cartpole(states, dt)

if __name__ == "__main__":
    test_ihlqr()