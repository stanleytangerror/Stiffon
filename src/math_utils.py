import numpy as np
from scipy.spatial.transform import Rotation as R

def normalized(v):
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        return v / norm
    else:
        return v

class Vec3(np.ndarray):
    """3D vector class as an alias to numpy array with shape (3,)"""
    def __new__(cls, x=0.0, y=0.0, z=0.0):
        if isinstance(x, (list, tuple, np.ndarray)):
            # If first argument is already an array-like object
            arr = np.array(x, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(f"Vec3 requires shape (3,), got {arr.shape}")
        else:
            # If separate x, y, z arguments
            arr = np.array([x, y, z], dtype=np.float64)
        
        return arr.view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        if self.shape == ():
            return float(self)
        if self.shape != (3,) and self.shape != (3, 1) and self.shape != (1, 3):
            raise ValueError(f"Vec3 requires shape (3,), got {self.shape}")
    
    def __array_wrap__(self, out_arr, context=None):
        """Handle slicing and other operations that change the shape"""
        if out_arr.shape == (3,):
            return out_arr.view(Vec3)
        else:
            # Return as regular numpy array for non-3D shapes
            return out_arr.view(np.ndarray)
    
    @property
    def x(self):
        return self[0]
    
    @property
    def y(self):
        return self[1]
    
    @property
    def z(self):
        return self[2]
    
    @x.setter
    def x(self, value):
        self[0] = value
    
    @y.setter
    def y(self, value):
        self[1] = value
    
    @z.setter
    def z(self, value):
        self[2] = value
    
    def __add__(self, other):
        if isinstance(other, Vec3):
            result = super().__add__(other)
            return Vec3(result[0], result[1], result[2])
        else:
            return super().__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Vec3):
            result = super().__sub__(other)
            return Vec3(result[0], result[1], result[2])
        else:
            return super().__sub__(other)

class Mat33(np.ndarray):
    """3x3 matrix class as an alias to numpy array with shape (3, 3)"""
    def __new__(cls, data=None):
        if data is None:
            arr = np.zeros((3, 3), dtype=np.float64)
        elif isinstance(data, (list, tuple, np.ndarray)):
            arr = np.array(data, dtype=np.float64)
            if arr.shape != (3, 3):
                raise ValueError(f"Mat33 requires shape (3, 3), got {arr.shape}")
        else:
            raise ValueError("Mat33 requires array-like data or None for identity")
        
        return arr.view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        if self.shape == (3,):
            return Vec3(self[0], self[1], self[2])
        if self.shape != (3, 3):
            raise ValueError(f"Mat33 requires shape (3, 3), got {self.shape}")

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.shape == (3,):
            return Vec3(out_arr[0], out_arr[1], out_arr[2])
        elif out_arr.shape == (3, 3):
            return Mat33(out_arr)
        else:
            return out_arr.view(np.ndarray)

    @staticmethod
    def identity():
        return Mat33(np.eye(3, dtype=np.float64))

    @staticmethod
    def zero():
        return Mat33(np.zeros((3, 3), dtype=np.float64))

class Mat44(np.ndarray):
    """4x4 matrix class as an alias to numpy array with shape (4, 4)"""
    def __new__(cls, data=None):
        if data is None:
            arr = np.zeros((4, 4), dtype=np.float64)
        elif isinstance(data, (list, tuple, np.ndarray)):
            arr = np.array(data, dtype=np.float64)
            if arr.shape != (4, 4):
                raise ValueError(f"Mat44 requires shape (4, 4), got {arr.shape}")
        else:
            raise ValueError("Mat44 requires array-like data or None for identity")
        
        return arr.view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        if self.shape != (4, 4):
            raise ValueError(f"Mat44 requires shape (4, 4), got {self.shape}")

class Transform:
    def __init__(self, origin: Vec3, basis: Mat33):
        self.origin = origin
        self.basis = basis
    
    def __matmul__(self, other):
        if isinstance(other, Transform):
            # Matrix-vector multiplication returns Vec3, vector addition returns Vec3
            new_origin = self.origin + (self.basis @ other.origin)
            new_basis = self.basis @ other.basis
            return Transform(new_origin, new_basis)
        else:
            raise ValueError(f"Cannot multiply Transform with {type(other)}")

    def inverse(self):
        inv_t = Transform(-self.origin, Mat33.identity())
        inv_r = Transform(Vec3(0, 0, 0), self.basis.transpose())
        return inv_r @ inv_t
    
    def to_matrix(self):
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.basis
        result[:3, 3] = self.origin
        return Mat44(result)

def create_world_matrix(translate=None, rotate=None, scale=None):
    matrix = np.eye(4)
    
    if scale is not None:
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale[0]
        scale_matrix[1, 1] = scale[1]
        scale_matrix[2, 2] = scale[2]
        matrix = scale_matrix @ matrix
    
    if rotate is not None:
        if hasattr(rotate, 'as_matrix'):
            rot_matrix = rotate.as_matrix()
        else:
            rot_matrix = np.eye(3)
        full_rot_matrix = np.eye(4)
        full_rot_matrix[:3, :3] = rot_matrix
        matrix = full_rot_matrix @ matrix
    
    if translate is not None:
        translate_matrix = np.eye(4)
        translate_matrix[:3, 3] = translate
        matrix = translate_matrix @ matrix
    
    return matrix

def integrate_transform(transform: Transform, linear_velocity: Vec3, angular_velocity: Vec3, dt: float):
    new_origin = transform.origin + linear_velocity * dt
    new_basis = R.from_rotvec(angular_velocity * dt).as_matrix() @ R.from_matrix(transform.basis).as_matrix()
    return Transform(new_origin, new_basis)

def skew_symmetric_matrix(v: Vec3):
    return Mat33(np.array([[0, -v.z, v.y], [v.z, 0, -v.x], [-v.y, v.x, 0]]))

def solve_jacobian(A: np.ndarray, b: np.ndarray, max_iterations: int = 1000, tolerance: float = 1e-6):
    # A = D + L + U
    # x_next = -D^{-1} (L + U) @ x_prev + D^{-1} b

    inv_diag = 1.0 / np.diag(A)
    D_inv = np.zeros_like(A)
    np.fill_diagonal(D_inv, inv_diag)
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    G = -D_inv @ (L + U)
    C = D_inv @ b

    x = np.zeros_like(b)
    iterations = 0
    while iterations < max_iterations:
        x_new = G @ x + C
        err = np.linalg.norm(x_new - x)
        # print(f"Iteration {iterations}, error: {err}")
        if err < tolerance:
            break
        x = x_new
        iterations += 1

    return x

def solve_gauss_seidel(A: np.ndarray, b: np.ndarray, max_iterations: int = 1000, tolerance: float = 1e-6):
    # A = D + L + U
    # x_next = (D + L)^{-1} @ (-U @ x_prev + b)

    x = np.zeros_like(b)
    iterations = 0
    while iterations < max_iterations:
        x_new = np.zeros_like(x)
        for i in range(x_new.shape[0]):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        x = x_new
        err = np.linalg.norm(A @ x - b)
        # print(f"Iteration {iterations}, error: {err}")
        if err < tolerance:
            break
        iterations += 1

    return x