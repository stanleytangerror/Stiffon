import taichi as ti
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import time
from rasterizer import Rasterizer, projection_matrix, view_matrix, world_matrix
from math_utils import normalized

class Renderer:
    def __init__(self, width, height):
        # taichi initialization
        self.enable_kernel_profile = True
        ti.init(arch=ti.gpu, debug=False, default_fp=ti.f32, kernel_profiler=self.enable_kernel_profile)

        # rasterizer initialization
        self.renderer = Rasterizer(width=width, height=height)
        self.gui = ti.GUI("Rasterizer", res=(self.renderer.window_size.x, self.renderer.window_size.y), fast_gui=True)

        # camera initialization
        self.camera_eye = np.array([-10.0, -20.0, 0.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])

        # lens initialization
        self.lens_fov = ti.math.pi * 0.5
        self.lens_aspect = width / height
        self.lens_near = 0.1
        self.lens_far = 100.0

    def is_running(self):
        return self.gui.running

    def set_camera(self, eye, target, up):
        self.camera_eye = eye
        self.camera_target = target
        self.camera_up = up

    def set_lens(self, fov, near, far):
        self.lens_fov = fov
        self.lens_aspect = self.renderer.window_size.x / self.renderer.window_size.y
        self.lens_near = near
        self.lens_far = far

    def draw_box(self, transform):
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

        self.renderer.draw(transform, cube_vb, cube_ib)

    def begin_frame(self):
        self.renderer.begin_frame()

    def end_frame(self):

        if self.enable_kernel_profile:
            ti.profiler.clear_kernel_profiler_info()

        ti.sync()

        proj_mat = projection_matrix(self.lens_fov, self.lens_aspect, self.lens_near, self.lens_far)
        view_mat = view_matrix(eye=self.camera_eye, target=self.camera_target, up=self.camera_up)
        self.renderer.set_global_const_buffer(relative_time, proj_mat, view_mat)

        self.renderer.stage_input_assembly()
        self.renderer.stage_vertex_shader()
        self.renderer.stage_geometry_process()
        self.renderer.stage_rasterization_and_pixel_shader()
        self.renderer.stage_output_merge()

        if self.enable_kernel_profile:
            ti.profiler.print_kernel_profiler_info('trace')

        ti.sync()

        self.gui.set_image(self.renderer.back_buffer)
        self.gui.show()

if __name__ == "__main__":

    program_start_time = time.time()

    renderer = Renderer(width=800, height=600)

    while renderer.is_running():
        renderer.begin_frame()

        frame_start_time = time.time()
        relative_time = frame_start_time - program_start_time 

        renderer.draw_box(world_matrix(
                                translate=np.array([np.sin(relative_time), 0.0, np.cos(relative_time)]) * 0.5,
                                rotate=R.from_rotvec(np.zeros(3)),
                                scale=np.array([0.3, 0.3, 0.3])))

        renderer.draw_box(world_matrix(
                                rotate=R.from_rotvec(rotvec=normalized(np.array([1.0, 1.0, 1.0])) * relative_time, degrees=False),
                                translate=np.array([1.0, 0.0, 0.0]),
                                scale=np.array([0.3, 0.3, 0.3])))

        renderer.draw_box(world_matrix(
                                rotate=R.from_rotvec(np.zeros(3)),
                                translate=np.array([-1.0, 0.0, 0.0]),
                                scale=np.array([np.sin(relative_time) * 0.5 + 0.5, 0.4, np.cos(relative_time) * 0.5 + 0.5])))

        renderer.end_frame()