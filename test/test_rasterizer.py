#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import time
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation as R

from rasterizer import Rasterizer, projection_matrix, view_matrix, world_matrix
from math_utils import normalized


if __name__ == "__main__":

    program_start_time = time.time()

    enable_kernel_profile = True
    ti.init(arch=ti.gpu, debug=False, default_fp=ti.f32, kernel_profiler=enable_kernel_profile)

    rasterizer = Rasterizer(width=800, height=600)
    gui = ti.GUI("Rasterizer", res=(rasterizer.window_size.x, rasterizer.window_size.y), fast_gui=True)

    # 构造 Box 的顶点缓冲和索引缓冲
    cube_vb = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
        [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ])
    cube_ib = np.array([
        [0,1,2], [2,3,0],   # 后面
        [4,5,6], [6,7,4],   # 前面
        [0,1,5], [5,4,0],   # 底面
        [2,3,7], [7,6,2],   # 顶面
        [1,2,6], [6,5,1],   # 右面
        [0,3,7], [7,4,0],   # 左面
    ])
    
    while gui.running:
        
        frame_start_time = time.time()
        relative_time = frame_start_time - program_start_time 

        ti.sync()

        if enable_kernel_profile:
            ti.profiler.clear_kernel_profiler_info()  #

        ti.sync()

        rasterizer.begin_frame()

        # print(f"Begin frame time: {(time.time() - frame_start_time) * 1000:.3f} ms")

        # 使用kernel来设置全局常量缓冲区
        proj_mat = projection_matrix(ti.math.pi * 0.7, rasterizer.window_size.x / rasterizer.window_size.y, 0.1, 100.0)
        view_mat = view_matrix(eye=np.array([-10.0, -20.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
        rasterizer.set_global_const_buffer(proj_mat, view_mat)

        # print(f"Global cb time: {(time.time() - frame_start_time) * 1000:.3f} ms")

        rasterizer.draw(world_matrix(
                                translate=np.array([np.sin(relative_time), 0.0, np.cos(relative_time)]) * 0.5,
                                rotate=R.from_rotvec(np.zeros(3)),
                                scale=np.array([0.3, 0.3, 0.3])),
                            np.array([1.0, 0.0, 0.0]),
                            cube_vb, cube_ib)

        rasterizer.draw(world_matrix(
                                rotate=R.from_rotvec(rotvec=normalized(np.array([1.0, 1.0, 1.0])) * relative_time, degrees=False),
                                translate=np.array([1.0, 0.0, 0.0]),
                                scale=np.array([0.3, 0.3, 0.3])), 
                            np.array([0.0, 1.0, 0.0]),
                            cube_vb, cube_ib)

        rasterizer.draw(world_matrix(
                                rotate=R.from_rotvec(np.zeros(3)),
                                translate=np.array([-1.0, 0.0, 0.0]),
                                scale=np.array([np.sin(relative_time) * 0.5 + 0.5, 0.4, np.cos(relative_time) * 0.5 + 0.5])), 
                            np.array([0.0, 0.0, 1.0]),
                            cube_vb, cube_ib)
        
        # print(f"Draw time: {(time.time() - frame_start_time) * 1000:.3f} ms")

        rasterizer.stage_input_assembly()
        rasterizer.stage_vertex_shader()
        rasterizer.stage_geometry_process()
        rasterizer.stage_rasterization_and_pixel_shader()
        rasterizer.stage_output_merge()

        # print(f"Execute time: {(time.time() - frame_start_time) * 1000:.3f} ms")

        if enable_kernel_profile:
            ti.profiler.print_kernel_profiler_info('trace')

        ti.sync()

        gui.set_image(rasterizer.back_buffer)
        gui.show()

        # print(f"Frame time: {(time.time() - frame_start_time) * 1000:.3f} ms")
