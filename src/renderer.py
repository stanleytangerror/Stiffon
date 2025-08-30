import taichi as ti
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import time
from math_utils import normalized

class Renderer:
    def __init__(self, width, height):
        # taichi initialization
        self.enable_kernel_profile = False
        ti.init(arch=ti.vulkan, debug=False, default_fp=ti.f32, kernel_profiler=self.enable_kernel_profile)

        # GGUI initialization
        self.window = ti.ui.Window("3D Renderer", (width, height), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()

        self.camera = ti.ui.Camera()

        # camera initialization
        self.camera_eye = np.array([0.0, -10.0, 0.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])
        self.lens_fov = ti.math.pi * 0.5

        # Store objects to render
        self.mesh_instances = MeshInstances()

    def is_running(self):
        return self.window.running

    def set_camera(self, eye, target, up):
        self.camera_eye = eye
        self.camera_target = target
        self.camera_up = up

    def set_fov(self, fov):
        self.lens_fov = fov

    def draw_box(self, world_mat, color):
        self.mesh_instances.add_box(world_mat, color)

    def draw_sphere(self, world_mat, color):
        self.mesh_instances.add_sphere(world_mat, color)
    
    def draw_plane(self, world_mat, color):
        self.mesh_instances.add_plane(world_mat, color)

    def begin_frame(self):
        # Clear objects from previous frame
        self.mesh_instances.clear()

    def end_frame(self):
        # self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        self.canvas.set_background_color((1.0, 1.0, 1.0))

        # Setup camera
        self.camera.position(self.camera_eye[0], self.camera_eye[1], self.camera_eye[2])
        self.camera.lookat(self.camera_target[0], self.camera_target[1], self.camera_target[2])
        self.camera.up(self.camera_up[0], self.camera_up[1], self.camera_up[2])
        self.camera.fov(self.lens_fov)
        self.scene.set_camera(self.camera)

        self.scene.ambient_light((1.0, 1.0, 1.0))
        self.scene.point_light(pos=(0, 5, 0), color=(1, 1, 1))

        # Render all objects
        self.mesh_instances.draw(self.scene)

        # Draw scene
        self.canvas.scene(self.scene)
        self.window.show()


@ti.data_oriented
class MeshInstances:
    max_mesh_instance_count = 100
    
    def __init__(self):
        # Box
        # Box vertices: 24 vertices (4 per face, 6 faces)
        # Each face has 4 vertices to allow proper normal interpolation
        vertices = np.array([
            # Front face (z = -0.5)
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            # Back face (z = 0.5)
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            # Left face (x = -0.5)
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
            # Right face (x = 0.5)
            [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
            # Bottom face (y = -0.5)
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5],
            # Top face (y = 0.5)
            [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ])
        
        # Box normals: 24 normals (1 per vertex, matching the face they belong to)
        normals = np.array([
            # Front face normals
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            # Back face normals
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            # Left face normals
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            # Right face normals
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            # Bottom face normals
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            # Top face normals
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]
        ])
        
        # Box indices: 12 triangles (2 per face, 6 faces)
        # Each face is made up of 2 triangles
        indices = np.array([
            # Front face: 2 triangles
            [0, 1, 2], [0, 2, 3],
            # Back face: 2 triangles
            [4, 6, 5], [4, 7, 6],
            # Left face: 2 triangles
            [8, 10, 9], [8, 11, 10],
            # Right face: 2 triangles
            [12, 13, 14], [12, 14, 15],
            # Bottom face: 2 triangles
            [16, 17, 18], [16, 18, 19],
            # Top face: 2 triangles
            [20, 22, 21], [20, 23, 22]
        ]).flatten().astype(np.int32)
        self.box_vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.box_vertices.from_numpy(vertices)
        self.box_normals = ti.Vector.field(3, dtype=ti.f32, shape=len(normals))
        self.box_normals.from_numpy(normals)
        self.box_indices = ti.field(dtype=ti.i32, shape=len(indices))
        self.box_indices.from_numpy(indices)
        self.box_transforms = ti.Matrix.field(4, 4, dtype = ti.f32, shape = self.max_mesh_instance_count)
        self.box_colors = []
        self.box_count = 0

        # Sphere
        vertices = []
        indices = []
        split_count = 10
        
        # Generate sphere vertices
        for lat in range(split_count + 1):
            theta = np.pi * lat / split_count
            for lon in range(split_count + 1):
                phi = 2 * np.pi * lon / split_count
                
                x = 0.5 * np.sin(theta) * np.cos(phi)
                y = 0.5 * np.sin(theta) * np.sin(phi)
                z = 0.5 * np.cos(theta)
                
                vertices.append([x, y, z])
        
        # Generate sphere indices for triangulation
        # Each grid cell (except poles) creates 2 triangles
        for lat in range(split_count):
            for lon in range(split_count):
                # Current grid cell vertices
                current = lat * (split_count + 1) + lon
                next_lat = current + (split_count + 1)
                next_lon = current + 1
                next_both = next_lat + 1
                
                # Skip triangles at the poles (first and last latitude)
                if lat == 0:
                    # First latitude: only one triangle per cell
                    indices.extend([current, next_lat, next_both])
                elif lat == split_count - 1:
                    # Last latitude: only one triangle per cell
                    indices.extend([current, next_lon, next_lat])
                else:
                    # Middle latitudes: two triangles per cell
                    indices.extend([current, next_lat, next_both])
                    indices.extend([current, next_both, next_lon])

        indices = np.array(indices).flatten().astype(np.int32)
        
        self.sphere_vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.sphere_vertices.from_numpy(np.array(vertices))
        self.sphere_normals = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.sphere_normals.from_numpy(np.array(vertices))
        self.sphere_indices = ti.field(dtype=ti.i32, shape=len(indices))
        self.sphere_indices.from_numpy(indices)
        self.sphere_transforms = ti.Matrix.field(4, 4, dtype = ti.f32, shape = self.max_mesh_instance_count)
        self.sphere_colors = []
        self.sphere_count = 0

        # Plane
        vertices = np.array([
            [-100.0, -100.0, 0.0],
            [100.0, -100.0, 0.0],
            [100.0, 100.0, 0.0],
            [-100.0, 100.0, 0.0],
        ])
        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],   
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        indices = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ]).flatten().astype(np.int32)
        self.plane_vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.plane_vertices.from_numpy(vertices)
        self.plane_normals = ti.Vector.field(3, dtype=ti.f32, shape=len(normals))
        self.plane_normals.from_numpy(normals)
        self.plane_indices = ti.field(dtype=ti.i32, shape=len(indices))
        self.plane_indices.from_numpy(indices)
        self.plane_transforms = ti.Matrix.field(4, 4, dtype = ti.f32, shape = self.max_mesh_instance_count)
        self.plane_colors = []
        self.plane_count = 0

    def add_box(self, world_mat, color):
        if self.box_count < self.max_mesh_instance_count:
            self.box_transforms[self.box_count] = world_mat.transpose()
            self.box_colors.append(color)
            self.box_count += 1

    def add_sphere(self, world_mat, color):
        if self.sphere_count < self.max_mesh_instance_count:
            self.sphere_transforms[self.sphere_count] = world_mat.transpose()
            self.sphere_colors.append(color)
            self.sphere_count += 1

    def add_plane(self, world_mat, color):
        if self.plane_count < self.max_mesh_instance_count:
            self.plane_transforms[self.plane_count] = world_mat.transpose()
            self.plane_colors.append(color)
            self.plane_count += 1

    def clear(self):
        self.box_count = 0
        self.box_colors = []
        self.sphere_count = 0
        self.sphere_colors = []
        self.plane_count = 0
        self.plane_colors = []

    def draw(self, scene):
        scene.mesh_instance(
            self.box_vertices, 
            indices=self.box_indices, 
            normals=self.box_normals,
            color=(1.0, 0.0, 0.0),
            transforms=self.box_transforms, 
            instance_offset=0,
            instance_count=self.box_count,
            )

        scene.mesh_instance(
            self.sphere_vertices, 
            indices=self.sphere_indices, 
            normals=self.sphere_normals,
            transforms=self.sphere_transforms, 
            color=(0.0, 1.0, 0.0),
            instance_offset=0,
            instance_count=self.sphere_count,
            )
            
        scene.mesh_instance(
            self.plane_vertices, 
            indices=self.plane_indices, 
            normals=self.plane_normals,
            transforms=self.plane_transforms, 
            color=(0.0, 0.0, 1.0),
            instance_offset=0,
            instance_count=self.plane_count,
            )


if __name__ == "__main__":
    program_start_time = time.time()

    renderer = Renderer(width=800, height=600)

    while renderer.is_running():
        renderer.begin_frame()

        frame_start_time = time.time()
        relative_time = frame_start_time - program_start_time 

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