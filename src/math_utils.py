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
        """Handle slicing and other operations that change the shape"""
        return out_arr.view(np.ndarray)

    @staticmethod
    def identity():
        return Mat33(np.eye(3, dtype=np.float64))

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

def integrate_transform(transform: Transform, linear_velocity: Vec3, angular_velocity: Vec3, dt: float):
    new_origin = transform.origin + linear_velocity * dt
    new_basis = R.from_rotvec(angular_velocity * dt).as_matrix() @ R.from_matrix(transform.basis).as_matrix()
    return Transform(new_origin, new_basis)