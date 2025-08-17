import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from math_utils import solve_jacobian, solve_gauss_seidel

def test_solve_jacobian():
    A = np.array([
        [10.0, -1.0, -2.0], 
        [-1.0, 10.0, -2.0], 
        [-1.0, -1.0, 5.0]])
    b = np.array([7.2, 8.3, 4.2])

    x = solve_jacobian(A, b)
    print(x)

def test_solve_gauss_seidel():
    A = np.array([
        [10.0, -1.0, -2.0], 
        [-1.0, 10.0, -2.0], 
        [-1.0, -1.0, 5.0]])
    b = np.array([7.2, 8.3, 4.2])

    x = solve_gauss_seidel(A, b)
    print(x)

if __name__ == "__main__":
    test_solve_gauss_seidel()