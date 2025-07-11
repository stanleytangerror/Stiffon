import taichi as ti
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import time

Vec3is = ti.types.vector(3, ti.i16)
Vec2i = ti.types.vector(2, ti.i32)
Vec2f = ti.types.vector(2, ti.f32)
Vec3f = ti.types.vector(3, ti.f32)
Vec4f = ti.types.vector(4, ti.f32)
Mat33f = ti.types.matrix(3, 3, ti.f32)
Mat44f = ti.types.matrix(4, 4, ti.f32)

def normalized(v):
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        return v / norm
    else:
        return v

@ti.func
def edge(a, b, c): 
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)

@ti.func
def barycentric_coords(p, p0, p1, p2):
    result = Vec3f(1.0, 0.0, 0.0)
    area = edge(p0, p1, p2)
    # 非退化三角形正常计算
    if ti.abs(area) >= 1e-10:
        w0 = edge(p1, p2, p) / area
        w1 = edge(p2, p0, p) / area
        w2 = edge(p0, p1, p) / area
        result = Vec3f(w0, w1, w2)
    else:
        d01 = ti.math.dot(p1 - p0, p1 - p0)
        d12 = ti.math.dot(p2 - p1, p2 - p1)
        d20 = ti.math.dot(p0 - p2, p0 - p2)
        a, b, idx = p0, p1, 2
        if d01 >= d12 and d01 >= d20:
            a, b, idx = p0, p1, 2
        elif d12 >= d20:
            a, b, idx = p1, p2, 0
        else:
            a, b, idx = p2, p0, 1
        ba = b - a
        denom = ti.math.dot(ba, ba)
        t = 0.0
        if denom >= 1e-10:
            t = ti.math.dot(p - a, ba) / denom
            t = ti.min(ti.max(t, 0.0), 1.0)
        if idx == 2:
            result = Vec3f(1 - t, t, 0.0)
        elif idx == 0:
            result = Vec3f(0.0, 1 - t, t)
        else:
            result = Vec3f(t, 0.0, 1 - t)
    return result

def projection_matrix(fov, aspect, near, far):
    # https://github.com/g-truc/glm/blob/master/glm/ext/matrix_clip_space.inl
    # clip space (4D homogeneous):
    # +x: screen left, [-1, 1]
    # +y: screen up, [-1, 1]
    # +z: screen in
    # z/w: [0, 1], near: 0, far: 1
    inv_tan = 1.0 / ti.tan(fov * 0.5)
    mat = np.zeros(shape=(4, 4), dtype=np.float32)
    mat[0, 0] = inv_tan / aspect
    mat[1, 1] = inv_tan
    mat[2, 2] = far / (far - near)
    mat[2, 3] = -far * near / (far - near)
    mat[3, 2] = 1
    return mat
    

def view_matrix(eye, target, up):
    # view space:
    # +x: left, +y: up, +z: front
    front = normalized(target - eye)
    left = normalized(np.linalg.cross(up, front))

    inv_t = np.zeros(shape=(4, 4), dtype=np.float32)
    inv_t[0, 0] = 1.0
    inv_t[1, 1] = 1.0
    inv_t[2, 2] = 1.0
    inv_t[0:3, 3] = -eye
    inv_t[3, 3] = 1.0

    inv_r = np.zeros(shape=(4, 4), dtype=np.float32)
    inv_r[0, 0:3] = left
    inv_r[1, 0:3] = up
    inv_r[2, 0:3] = front
    inv_r[3, 3] = 1.0

    return inv_r @ inv_t

def world_matrix(t):
    # world space: right hand, z up
    mat = np.zeros(shape=(4, 4))
    mat[0:3, 0:3] = R.from_rotvec(rotvec=np.array([0.0, 0.0, 1.0]) * t, degrees=False).as_matrix()
    mat[3, 3] = 1.0
    return mat


@ti.func
def interp3(barycentric, v0, v1, v2):
    return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z

@ti.func
def interp2(t, v0, v1):
    return v0 * t.x + v1 * t.y

@ti.func
def is_inside(v, plane):
    result = ti.i8(0)
    # 0:left,1:right,2:bottom,3:top,4:near,5:far
    if plane == 0:
        result = ti.i8(v.pos.x + v.pos.w >= 0)
    if plane == 1:
        result = ti.i8(v.pos.w - v.pos.x >= 0)
    if plane == 2:
        result = ti.i8(v.pos.y + v.pos.w >= 0)
    if plane == 3:
        result = ti.i8(v.pos.w - v.pos.y >= 0)
    if plane == 4:
        result = ti.i8(v.pos.z >= 0)
    if plane == 5:
        result = ti.i8(v.pos.w - v.pos.z >= 0)
    return result


@ti.dataclass
class TConstBuffer:
    cur_time: ti.f32
    proj_mat: Mat44f
    view_mat: Mat44f
    world_mat: Mat44f

@ti.dataclass
class TVert:
    pos: Vec3f
    color: Vec3f

@ti.dataclass
class TInputAssem:
    v0: TVert
    v1: TVert
    v2: TVert

@ti.dataclass
class TVsOut:
    pos: Vec4f
    color: Vec4f

@ti.dataclass
class TClipInput:
    v0: TVsOut
    v1: TVsOut
    v2: TVsOut

@ti.dataclass
class TRasterInput:
    v0: TVsOut
    v1: TVsOut
    v2: TVsOut

@ti.dataclass
class TPsInput:
    prim: TVsOut
    clipped: ti.f16

@ti.func
def interp_vsout_2(t, v0, v1):
    pos = interp2(t, v0.pos, v1.pos)
    return TVsOut(pos = pos,
                color = interp2(t, v0.color, v1.color))

@ti.func
def interp_vsout_3(t, v0, v1, v2):
    pos = interp3(t, v0.pos, v1.pos, v2.pos)
    return TVsOut(pos = pos,
                color = interp3(t, v0.color, v1.color, v2.color))
    
@ti.func
def intersect_vert(v1: TVsOut, v2: TVsOut, plane):
    # 计算两个顶点在平面上的交点
    # 计算点到平面距离
    d1 = 0.0; d2 = 1.0
    if plane == 0:
        d1 = v1.pos.x + v1.pos.w; d2 = v2.pos.x + v2.pos.w
    elif plane == 1:
        d1 = v1.pos.w - v1.pos.x; d2 = v2.pos.w - v2.pos.x
    elif plane == 2:
        d1 = v1.pos.y + v1.pos.w; d2 = v2.pos.y + v2.pos.w
    elif plane == 3:
        d1 = v1.pos.w - v1.pos.y; d2 = v2.pos.w - v2.pos.y
    elif plane == 4:
        d1 = v1.pos.z; d2 = v2.pos.z
    elif plane == 5:
        d1 = v1.pos.w - v1.pos.z; d2 = v2.pos.w - v2.pos.z
    t = d1 / (d1 - d2)
    return interp_vsout_2(Vec2f(1.0 - t, t), v1, v2)


@ti.data_oriented
class Renderer:

    def __init__(self, width, height):
        self.window_size = Vec2i(width, height)
        
        self.tile_pixel_size = Vec2i(4, 4)
        self.tile_count = Vec2i(ti.ceil(self.window_size / self.tile_pixel_size))

        self.max_primitive_count = 1024
        
        # TConstBuffer = ti.types.struct(cur_time=ti.f32, proj_mat=Mat44f, view_mat=Mat44f, world_mat=Mat44f)
        self.constant_buffer = TConstBuffer.field(shape=())

        # TVert = ti.types.struct(pos=Vec3f, color=Vec3f)
        self.vertex_buffer = TVert.field(shape=8)
        self.index_buffer = ti.field(dtype=Vec3is, shape=12)

        # TInputAssem = ti.types.struct(v0=TVert, v1=TVert, v2=TVert)
        self.assembled_input = TInputAssem.field(shape=self.index_buffer.shape[0])

        # TVsOut = ti.types.struct(pos=Vec4f, color=Vec4f)
        # TClipInput = ti.types.struct(v0=TVsOut, v1=TVsOut, v2=TVsOut)
        self.clip_input = TClipInput.field(shape=self.max_primitive_count)
        self.clip_input_counter = ti.field(dtype=ti.i32, shape=())

        self.clip_buffer_0 = TVsOut.field(shape=(self.max_primitive_count, 10))
        self.clip_buffer_1 = TVsOut.field(shape=(self.max_primitive_count, 10))

        # TRasterInput = ti.types.struct(v0=TVsOut, v1=TVsOut, v2=TVsOut)
        self.rasterize_input = TRasterInput.field(shape=self.max_primitive_count)
        self.rasterize_input_counter = ti.field(dtype=ti.i32, shape=())

        # TPsInput = ti.types.struct(prim=TVsOut, clipped=ti.i8)
        self.output_merge_input = ti.field(dtype=Vec4f, shape=(self.window_size.x, self.window_size.y))

        self.screen_pixels = ti.Vector.field(3, ti.f32, shape=(self.window_size.x, self.window_size.y))
        self.depth_buffer = ti.field(dtype=ti.f32, shape=(self.window_size.x, self.window_size.y))

    @ti.func
    def vs(self, vertex: TVert) -> TVsOut:

        world_mat = self.constant_buffer[None].world_mat
        view_mat = self.constant_buffer[None].view_mat
        proj_mat = self.constant_buffer[None].proj_mat

        pos = Vec4f(vertex.pos, 1.0)
        pos = world_mat @ pos
        pos = view_mat @ pos
        pos = proj_mat @ pos

        return TVsOut(pos=pos,
                    color=Vec4f(vertex.color, 1.0))

    @ti.func
    def ps(self, vertex: TVsOut) -> Vec4f:
        return vertex.color

    @ti.kernel
    def stage_input_assembly(self):
        for i in self.index_buffer:
            indices = self.index_buffer[i]
            self.assembled_input[i].v0 = self.vertex_buffer[indices.x]
            self.assembled_input[i].v1 = self.vertex_buffer[indices.y]
            self.assembled_input[i].v2 = self.vertex_buffer[indices.z]

    @ti.kernel
    def stage_vertex_shader(self):
        self.clip_input_counter[None] = 0

        for prim_i in self.assembled_input:
            triangle_vertices = self.assembled_input[prim_i]
            idx = ti.atomic_add(self.clip_input_counter[None], 1)
            self.clip_input[idx].v0 = self.vs(triangle_vertices.v0)
            self.clip_input[idx].v1 = self.vs(triangle_vertices.v1)
            self.clip_input[idx].v2 = self.vs(triangle_vertices.v2)

    @ti.kernel
    def stage_geometry_process(self):
        self.rasterize_input_counter[None] = 0
        # 针对每个裁剪三角形执行 Sutherland-Hodgman 多边形裁剪
        for prim_i in range(self.clip_input_counter[None]):
            # 用固定数组承载临时顶点
            self.clip_buffer_0[prim_i, 0] = self.clip_input[prim_i].v0
            self.clip_buffer_0[prim_i, 1] = self.clip_input[prim_i].v1
            self.clip_buffer_0[prim_i, 2] = self.clip_input[prim_i].v2
            clip_buffer_0_count = 3

            # 6个裁剪平面
            for plane in range(6):

                clip_buffer_1_count = 0

                for j in range(clip_buffer_0_count):
                    curr = self.clip_buffer_0[prim_i, j]
                    next = self.clip_buffer_0[prim_i, (j + 1) % clip_buffer_0_count]
                    inside_curr = is_inside(curr, plane)
                    inside_next = is_inside(next, plane)

                    if inside_curr and inside_next:
                        # 都在内侧，保留 next
                        self.clip_buffer_1[prim_i, clip_buffer_1_count] = next
                        clip_buffer_1_count += 1
                    elif inside_curr and not inside_next:
                        # 边出内->外，添加交点
                        self.clip_buffer_1[prim_i, clip_buffer_1_count] = intersect_vert(curr, next, plane)
                        clip_buffer_1_count += 1
                    elif not inside_curr and inside_next:
                        # 边外->内，添加交点和 next
                        self.clip_buffer_1[prim_i, clip_buffer_1_count] = intersect_vert(curr, next, plane)
                        clip_buffer_1_count += 1
                        self.clip_buffer_1[prim_i, clip_buffer_1_count] = next
                        clip_buffer_1_count += 1

                # 更新多边形
                if clip_buffer_0_count == 0:
                    break
                
                for j in range(clip_buffer_1_count):
                    self.clip_buffer_0[prim_i, j] = self.clip_buffer_1[prim_i, j]
                clip_buffer_0_count = clip_buffer_1_count

            # 三角化并写入 rasterize_input
            if clip_buffer_0_count >= 3:
                for k in range(1, clip_buffer_0_count - 1):
                    idx = ti.atomic_add(self.rasterize_input_counter[None], 1)
                    self.rasterize_input[idx].v0 = self.clip_buffer_0[prim_i, 0]
                    self.rasterize_input[idx].v1 = self.clip_buffer_0[prim_i, k]
                    self.rasterize_input[idx].v2 = self.clip_buffer_0[prim_i, k + 1]

    @ti.kernel
    def stage_rasterization_and_pixel_shader(self):

        for tile_x, tile_y in ti.ndrange(self.tile_count.x, self.tile_count.y):
            for prim_i in range(self.rasterize_input_counter[None]):
                # 计算当前 tile 的像素范围
                tile_min = Vec2i(tile_x, tile_y) * self.tile_pixel_size
                tile_max = (Vec2i(tile_x, tile_y) + Vec2i(1, 1)) * self.tile_pixel_size - Vec2i(1, 1)

                # 遍历当前三角形的像素范围
                v0 = self.rasterize_input[prim_i].v0
                v1 = self.rasterize_input[prim_i].v1
                v2 = self.rasterize_input[prim_i].v2
                p0 = (v0.pos.xy * 0.5 + 0.5) * self.window_size
                p1 = (v1.pos.xy * 0.5 + 0.5) * self.window_size
                p2 = (v2.pos.xy * 0.5 + 0.5) * self.window_size

                min_pixel_uv = max(tile_min, int(ti.floor(min(p0, p1, p2))))
                max_pixel_uv = min(self.window_size - 1, tile_max, int(ti.ceil(max(p0, p1, p2))))

                for x in range(min_pixel_uv.x, max_pixel_uv.x + 1):
                    for y in range(min_pixel_uv.y, max_pixel_uv.y + 1):
                        p = Vec2f(x, y) + 0.5
                        w = barycentric_coords(p, p0.xy, p1.xy, p2.xy)
                        if w.x >= 0 and w.y >= 0 and w.z >= 0:
                            pos = interp3(w, v0.pos, v1.pos, v2.pos)
                            pos /= pos.w  
                            z = pos.z

                            old_z = ti.atomic_min(self.depth_buffer[x, y], z)
                            if z <= old_z:
                                ps_input = TPsInput()
                                ps_input.prim = interp_vsout_3(w, v0, v1, v2)
                                ps_input.prim.pos /= ps_input.prim.pos.w
                                
                                self.output_merge_input[x, y] = self.ps(ps_input.prim)
                                self.depth_buffer[x, y] = z

    @ti.kernel
    def stage_output_merge(self):
        for pixel_u, pixel_v in self.output_merge_input:
            self.screen_pixels[pixel_u, pixel_v] = self.output_merge_input[pixel_u, pixel_v].xyz

    @ti.kernel
    def clear_buffers(self):
        for I in ti.grouped(self.screen_pixels):
            self.screen_pixels[I] = ti.Vector([0.0, 0.0, 0.0])
        for I in ti.grouped(self.output_merge_input):
            self.output_merge_input[I] = Vec4f(0.0, 0.0, 0.0, 0.0)
        for I in ti.grouped(self.depth_buffer):
            self.depth_buffer[I] = 1

def update_constant_buffer(renderer, t):
    renderer.constant_buffer[None].cur_time = t
    renderer.constant_buffer[None].proj_mat = projection_matrix(ti.math.pi / 2, renderer.window_size.x / renderer.window_size.y, 0.1, 100.0)
    renderer.constant_buffer[None].view_mat = view_matrix(eye=np.array([0.0, -2.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    renderer.constant_buffer[None].world_mat = world_matrix(t)


def main():
    enable_kernel_profile = False
    ti.init(arch=ti.gpu, debug=False, default_fp=ti.f32, kernel_profiler=enable_kernel_profile)

    renderer = Renderer(width=800, height=600)

    # 构造 Box 的顶点缓冲和索引缓冲
    cube_vb = [
        ti.Vector([-0.5, -0.5, -0.5]), ti.Vector([0.5, -0.5, -0.5]),
        ti.Vector([0.5,  0.5, -0.5]), ti.Vector([-0.5,  0.5, -0.5]),
        ti.Vector([-0.5, -0.5,  0.5]), ti.Vector([0.5, -0.5,  0.5]),
        ti.Vector([0.5,  0.5,  0.5]), ti.Vector([-0.5,  0.5,  0.5]),
    ]
    cube_ib = [
        [0,1,2], [2,3,0],   # 后面
        [4,5,6], [6,7,4],   # 前面
        [0,1,5], [5,4,0],   # 底面
        [2,3,7], [7,6,2],   # 顶面
        [1,2,6], [6,5,1],   # 右面
        [0,3,7], [7,4,0],   # 左面
    ]
    # 将 Box 顶点写入顶点字段
    for i, v in enumerate(cube_vb):
        renderer.vertex_buffer[i] = TVert(pos=v, color=v + 0.5)
    
    for i, v in enumerate(cube_ib):
        renderer.index_buffer[i] = Vec3is(v[0], v[1], v[2])

    start_time = time.time()
    gui = ti.GUI("Renderer", (renderer.window_size.x, renderer.window_size.y))
    while gui.running:

        t = time.time() - start_time

        update_constant_buffer(renderer, t)

        if enable_kernel_profile:
            ti.profiler.clear_kernel_profiler_info()  #
        
        renderer.clear_buffers()

        renderer.stage_input_assembly()

        renderer.stage_vertex_shader()

        renderer.stage_geometry_process()

        renderer.stage_rasterization_and_pixel_shader()

        renderer.stage_output_merge()

        if enable_kernel_profile:
            ti.profiler.print_kernel_profiler_info('trace')

        gui.set_image(renderer.screen_pixels)
        gui.show()


if __name__ == "__main__":
    main()