import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import time
import math
from renderer import Renderer
from math_utils import normalized
from scipy.spatial.transform.rotation import Rotation as R

def render_boxs_and_spheres():
    program_start_time = time.time()

    renderer = Renderer(width=800, height=600)
    renderer.set_camera(eye=np.array([0.0, -5.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    renderer.set_fov(fov=90)

    while renderer.is_running():
        renderer.begin_frame()

        frame_start_time = time.time()
        relative_time = frame_start_time - program_start_time 
        # relative_time = 0

        # Create world matrices manually since we don't have the world_matrix function
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

        renderer.draw_box(create_world_matrix(
                                translate=np.array([np.sin(relative_time), 0.0, np.cos(relative_time)]) * 0.5,
                                rotate=R.from_rotvec(np.zeros(3)),
                                scale=np.array([0.3, 0.3, 0.3])),
                                color=np.array([1.0, 0.0, 0.0]))

        renderer.draw_box(create_world_matrix(
                                rotate=R.from_rotvec(rotvec=normalized(np.array([1.0, 1.0, 1.0])) * relative_time, degrees=False),
                                translate=np.array([1.0, 0.0, 0.0]),
                                scale=np.array([0.3, 0.3, 0.3])),
                                color=np.array([0.0, 1.0, 0.0]))

        renderer.draw_box(create_world_matrix(
                                rotate=R.from_rotvec(np.zeros(3)),
                                translate=np.array([-1.0, 0.0, 0.0]),
                                scale=np.array([np.sin(relative_time) * 0.5 + 0.5, 0.4, np.cos(relative_time) * 0.5 + 0.5])),
                                color=np.array([0.0, 0.0, 1.0]))

        renderer.draw_sphere(create_world_matrix(
                                translate=np.array([np.sin(-relative_time), 0.0, np.cos(-relative_time)]) * 0.5,
                                rotate=R.from_rotvec(np.zeros(3)),
                                scale=np.array([0.3, 0.3, 0.3])),
                                color=np.array([1.0, 1.0, 1.0]))

        renderer.draw_sphere(create_world_matrix(
                                rotate=R.from_rotvec(rotvec=normalized(np.array([1.0, 1.0, 1.0])) * -relative_time, degrees=False),
                                translate=np.array([1.0, 2.0, 0.0]),
                                scale=np.array([0.3, 0.3, 0.3])),
                                color=np.array([0.0, 1.0, 1.0]))

        renderer.end_frame()

if __name__ == "__main__":
    render_boxs_and_spheres()