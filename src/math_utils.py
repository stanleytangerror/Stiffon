import numpy as np

def normalized(v):
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        return v / norm
    else:
        return v
