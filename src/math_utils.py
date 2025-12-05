import numpy as np
from scipy.spatial.transform import Rotation as R

def normalized(v):
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        return v / norm
    else:
        return v

def decompose_to_n_and_t(v, n):
    assert np.linalg.norm(n) - 1.0 < 1e-6
    
    v_d = np.dot(n, v) * n
    return v_d, v - v_d

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

class Vec2(np.ndarray):
    """2D vector class as an alias to numpy array with shape (2,)"""
    def __new__(cls, x=0.0, y=0.0):
        if isinstance(x, (list, tuple, np.ndarray)):
            # If first argument is already an array-like object
            arr = np.array(x, dtype=np.float64)
            if arr.shape != (2,):
                raise ValueError(f"Vec2 requires shape (2,), got {arr.shape}")
        else:
            # If separate x, y arguments
            arr = np.array([x, y], dtype=np.float64)
        
        return arr.view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        if self.shape == ():
            return float(self)
        if self.shape != (2,) and self.shape != (2, 1) and self.shape != (1, 2):
            raise ValueError(f"Vec2 requires shape (2,), got {self.shape}")
    
    def __array_wrap__(self, out_arr, context=None):
        """Handle slicing and other operations that change the shape"""
        if out_arr.shape == (2,):
            return out_arr.view(Vec2)
        else:
            # Return as regular numpy array for non-2D shapes
            return out_arr.view(np.ndarray)
    
    @property
    def x(self):
        return self[0]
    
    @property
    def y(self):
        return self[1]
    
    @x.setter
    def x(self, value):
        self[0] = value
    
    @y.setter
    def y(self, value):
        self[1] = value
    
    def __add__(self, other):
        if isinstance(other, Vec2):
            result = super().__add__(other)
            return Vec2(result[0], result[1])
        else:
            return super().__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Vec2):
            result = super().__sub__(other)
            return Vec2(result[0], result[1])
        else:
            return super().__sub__(other)

class Mat22(np.ndarray):
    """2x2 matrix class as an alias to numpy array with shape (2, 2)"""
    def __new__(cls, data=None):
        if data is None:
            arr = np.zeros((2, 2), dtype=np.float64)
        elif isinstance(data, (list, tuple, np.ndarray)):
            arr = np.array(data, dtype=np.float64)
            if arr.shape != (2, 2):
                raise ValueError(f"Mat22 requires shape (2, 2), got {arr.shape}")
        else:
            raise ValueError("Mat22 requires array-like data or None for identity")
        
        return arr.view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        if self.shape == (2,):
            return Vec2(self[0], self[1])
        if self.shape != (2, 2):
            raise ValueError(f"Mat22 requires shape (2, 2), got {self.shape}")

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.shape == (2,):
            return Vec2(out_arr[0], out_arr[1])
        elif out_arr.shape == (2, 2):
            return Mat22(out_arr)
        else:
            return out_arr.view(np.ndarray)

    @staticmethod
    def identity():
        return Mat22(np.eye(2, dtype=np.float64))

    @staticmethod
    def zero():
        return Mat22(np.zeros((2, 2), dtype=np.float64))
    
    @staticmethod
    def rotation(angle: float):
        """Create a 2D rotation matrix from an angle in radians"""
        c = np.cos(angle)
        s = np.sin(angle)
        return Mat22(np.array([[c, -s], [s, c]], dtype=np.float64))

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

    def transformDirection(self, d: Vec3):
        return self.basis @ d
    
    def transformPosition(self, p: Vec3):
        return self.basis @ p + self.origin

class Transform2d:
    def __init__(self, origin: Vec2, angle: float = 0.0):
        self.origin = origin
        self.angle = angle
    
    def __matmul__(self, other):
        if isinstance(other, Transform2d):
            # Rotate other's origin by self's angle, then add self's origin
            # Combine rotations by adding angles
            rotated_origin = self._rotate_vector(other.origin, self.angle)
            new_origin = self.origin + rotated_origin
            new_angle = self.angle + other.angle
            return Transform2d(new_origin, new_angle)
        else:
            raise ValueError(f"Cannot multiply Transform2d with {type(other)}")

    def inverse(self):
        # Inverse rotation: negative angle
        # Inverse translation: rotate origin by negative angle, then negate
        inv_origin = self._rotate_vector(-self.origin, -self.angle)
        return Transform2d(inv_origin, -self.angle)
    
    def _rotate_vector(self, v: Vec2, angle: float) -> Vec2:
        """Rotate a vector by an angle (in radians)"""
        c = np.cos(angle)
        s = np.sin(angle)
        x = v.x * c - v.y * s
        y = v.x * s + v.y * c
        return Vec2(x, y)
    
    def to_matrix(self):
        """Convert to 3x3 homogeneous transformation matrix"""
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        result = np.eye(3, dtype=np.float64)
        result[0, 0] = c
        result[0, 1] = -s
        result[1, 0] = s
        result[1, 1] = c
        result[0, 2] = self.origin.x
        result[1, 2] = self.origin.y
        return result

    def transformDirection(self, d: Vec2):
        """Transform a direction vector (rotation only, no translation)"""
        return self._rotate_vector(d, self.angle)
    
    def transformPosition(self, p: Vec2):
        """Transform a position vector (rotation + translation)"""
        rotated = self._rotate_vector(p, self.angle)
        return rotated + self.origin

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

def integrate_transform2d(transform: Transform2d, linear_velocity: Vec2, angular_velocity: float, dt: float):
    """Integrate a 2D transform with linear and angular velocity"""
    new_origin = transform.origin + linear_velocity * dt
    new_angle = transform.angle + angular_velocity * dt
    return Transform2d(new_origin, new_angle)

def cross22_2d(v1: Vec2, v2: Vec2) -> float:
    return np.cross(np.array([v1.x, v1.y, 0]), np.array([v2.x, v2.y, 0]))[2]

def cross12_2d(v1: float, v2: Vec2) -> Vec2:
    return Vec2(np.cross(np.array([0, 0, v1]), np.array([v2.x, v2.y, 0]))[2])

def cross21_2d(v1: Vec2, v2: float) -> Vec2:
    return Vec2(np.cross(np.array([v1.x, v1.y, 0]), np.array([0, 0, v2]))[0:2])

def skew_symmetric_matrix(v: Vec3):
    return Mat33(np.array([[0, -v.z, v.y], [v.z, 0, -v.x], [-v.y, v.x, 0]]))

def generate_orthogonal_basis(v: Vec3):
    v0 = normalized(v)
    v1 = np.cross(v0, Vec3(1, 0, 0))
    if np.linalg.norm(v1) < 1e-6:
        v1 = np.cross(v0, Vec3(0, 1, 0))
    v1 = normalized(v1)
    v2 = np.cross(v0, v1)
    return v0, v1, v2

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

def calculate_stiffness_damping(mass: float, omega: float, zeta: float) -> (float, float):
    return mass * omega ** 2, 2.0 * mass * omega * zeta