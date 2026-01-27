import jax.numpy as np
import jax
from dynamics import Pendulum, rk4
from visualize import draw_pendulum
from lqr import ihlqr

def test_swing_up():
    x_goal = np.array([np.pi, 0.0])
    u_goal = np.array([0.0])
    E_goal = Pendulum.energy(Pendulum.params, x_goal)

    m = Pendulum.params['m']
    l = Pendulum.params['l']
    g = Pendulum.params['g']

    dt = 0.1
    t_total = 20.0
    k = 0.1

    N = int(t_total / dt)

    A = jax.jacfwd(lambda _x: rk4(Pendulum.params, Pendulum.dynamics, _x, u_goal, dt))(x_goal)
    B = jax.jacfwd(lambda _u: rk4(Pendulum.params, Pendulum.dynamics, x_goal, _u, dt))(u_goal)
    Q = np.diag(np.array([1.0, 1.0])) 
    R = 0.1 * np.eye(1) 
    P, K = ihlqr(A, B, Q, R)

    x_init = np.array([0.0, 0.1])
    x = x_init
    states = [x_init]
    for i in range(N-1):
        E_error = Pendulum.energy(Pendulum.params, x) - E_goal
        theta_dot = x[1]
        theta = x[0]
        print(E_error, theta_dot)
        
        u = np.array([-k * theta_dot * E_error])
        x = rk4(Pendulum.params, Pendulum.dynamics, x, u, dt)
        states.append(x)
    
    states = np.array(states)
    draw_pendulum(states, dt)

if __name__ == "__main__":
    test_swing_up()