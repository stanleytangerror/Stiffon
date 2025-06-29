import taichi as ti
import numpy as np
import time

Vec3is = ti.types.vector(3, ti.i16)
Vec2i = ti.types.vector(2, ti.i32)
Vec2f = ti.types.vector(2, ti.f32)
Vec3f = ti.types.vector(3, ti.f32)
Vec4f = ti.types.vector(4, ti.f32)
Mat33f = ti.math.mat3
Mat44f = ti.math.mat4

ti.init(arch=ti.gpu, debug=True, default_fp=ti.f32)

@ti.func
def edge(a, b, c): 
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)

@ti.func
def barycentric_coords(p, p0, p1, p2):
    result = Vec3f(0.0, 0.0, 0.0)
    area = edge(p0, p1, p2)
    if abs(area) >= 1e-10:
        w0 = edge(p1, p2, p) / area
        w1 = edge(p2, p0, p) / area
        w2 = edge(p0, p1, p) / area
        result = Vec3f(w0, w1, w2)
    return result

def projection_matrix(fov, aspect, near, far):
    # 构建标准透视投影矩阵
    inv_tan = 1.0 / ti.tan(fov * 0.5)
    # 行主序：
    return Mat44f([[inv_tan / aspect, 0.0, 0.0, 0.0],
                   [0.0, inv_tan, 0.0, 0.0],
                   [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
                   [0.0, 0.0, -1.0, 0.0]])

@ti.func
def view_matrix(eye, target, up):
    # 构建摄像机视图矩阵（LookAt）
    z = ti.math.normalize(eye - target)
    x = ti.math.normalize(ti.math.cross(up, z))
    y = ti.math.cross(z, x)
    # 行主序视图矩阵
    return Mat44f([[x.x, y.x, z.x, 0.0],
                   [x.y, y.y, z.y, 0.0],
                   [x.z, y.z, z.z, 0.0],
                   [-ti.math.dot(x, eye), -ti.math.dot(y, eye), -ti.math.dot(z, eye), 1.0]])

window_size = Vec2i(512, 128)

TConstBuffer = ti.types.struct(cur_time=ti.f32, proj_mat=Mat44f, view_mat=Mat44f, world_mat=Mat44f)
constant_buffer = TConstBuffer.field(shape=())

TVert = ti.types.struct(pos=Vec3f, color=Vec3f)
vertex_buffer = TVert.field(shape=8)
index_buffer = ti.field(dtype=Vec3is, shape=12)

TInputAssem = ti.types.struct(v0=TVert, v1=TVert, v2=TVert)
assembled_input = TInputAssem.field(shape=index_buffer.shape[0])

TVsOut = ti.types.struct(pos=Vec4f, color=Vec4f)
TRasterInput = ti.types.struct(v0=TVsOut, v1=TVsOut, v2=TVsOut)
rasterize_input = TRasterInput.field(shape=assembled_input.shape[0])

TPsInput = ti.types.struct(prim=TVsOut, clipped=ti.i8)
pixel_shading_input = TPsInput.field(shape=(window_size.x, window_size.y))
output_merge_input = ti.field(dtype=Vec4f, shape=(window_size.x, window_size.y))

screen_pixels = ti.Vector.field(3, ti.f32, shape=(window_size.x, window_size.y))
depth_buffer = ti.field(dtype=ti.f32, shape=(window_size.x, window_size.y))

@ti.func
def interp(barycentric, v0, v1, v2):
    return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z

@ti.func
def vs(vertex):

    world_mat = constant_buffer[None].world_mat
    view_mat = constant_buffer[None].view_mat
    proj_mat = constant_buffer[None].proj_mat

    pos = world_mat @ Vec4f(vertex.pos, 1.0)
    pos = view_mat @ pos
    pos = proj_mat @ pos

    return TVsOut(pos=pos,
                  color=Vec4f(vertex.color, 1.0))

@ti.func
def interp_vsout(barycentric, v0, v1, v2):
    pos = interp(barycentric, v0.pos, v1.pos, v2.pos)
    pos /= pos.w  # 透视除法
    return TVsOut(pos = pos,
                  color = interp(barycentric, v0.color, v1.color, v2.color))

@ti.func
def ps(vertex):
    return vertex.color

@ti.kernel
def stage_input_assembly():
    for i in index_buffer:
        indices = index_buffer[i]
        assembled_input[i].v0 = vertex_buffer[indices.x]
        assembled_input[i].v1 = vertex_buffer[indices.y]
        assembled_input[i].v2 = vertex_buffer[indices.z]

@ti.kernel
def stage_vertex_shader():
    for prim_i in assembled_input:
        triangle_vertices = assembled_input[prim_i]
        rasterize_input[prim_i].v0 = vs(triangle_vertices.v0)
        rasterize_input[prim_i].v1 = vs(triangle_vertices.v1)
        rasterize_input[prim_i].v2 = vs(triangle_vertices.v2)

@ti.kernel
def stage_rasterization():
    for u, v in pixel_shading_input:
        pixel_shading_input[u, v].clipped = 0

    for prim_i in rasterize_input:
        v0 = rasterize_input[prim_i].v0
        v1 = rasterize_input[prim_i].v1
        v2 = rasterize_input[prim_i].v2
        p0 = (v0.pos.xy * 0.5 + 0.5) * window_size
        p1 = (v1.pos.xy * 0.5 + 0.5) * window_size
        p2 = (v2.pos.xy * 0.5 + 0.5) * window_size

        min_pixel_uv = max(0, int(ti.floor(min(p0, p1, p2))))
        max_pixel_uv = min(window_size - 1, int(ti.ceil(max(p0, p1, p2))))

        for x in range(min_pixel_uv.x, max_pixel_uv.x + 1):
            for y in range(min_pixel_uv.y, max_pixel_uv.y + 1):
                p = Vec2f(x, y) + 0.5
                w = barycentric_coords(p, p0.xy, p1.xy, p2.xy)
                if w.x >= 0 and w.y >= 0 and w.z >= 0:
                    z = interp(w, v0.pos.z, v1.pos.z, v2.pos.z)
                    if z < depth_buffer[x, y]:  # 深度测试
                        pixel_shading_input[x, y].prim = interp_vsout(w, v0, v1, v2)
                        pixel_shading_input[x, y].clipped = 1
                        depth_buffer[x, y] = z

@ti.kernel
def stage_pixel_shader():
    for pixel_u, pixel_v in output_merge_input:
        ps_input = pixel_shading_input[pixel_u, pixel_v]
        if ps_input.clipped > 0.5:
            output_merge_input[pixel_u, pixel_v] = ps(ps_input.prim)

@ti.kernel
def stage_output_merge():
    for pixel_u, pixel_v in output_merge_input:
        screen_pixels[pixel_u, pixel_v] = output_merge_input[pixel_u, pixel_v].xyz

@ti.kernel
def clear_buffers():
    for I in ti.grouped(screen_pixels):
        screen_pixels[I] = ti.Vector([0.0, 0.0, 0.0])
    for I in ti.grouped(depth_buffer):
        depth_buffer[I] = 1

@ti.kernel
def update_constant_buffer(t: ti.f32):
    constant_buffer[None].cur_time = t
    constant_buffer[None].proj_mat = projection_matrix(ti.math.pi / 4, window_size.x / window_size.y, 0.1, 100.0)
    constant_buffer[None].view_mat = view_matrix(Vec3f(0.0, 0.0, 3.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0))
    constant_buffer[None].world_mat = ti.math.rot_by_axis(Vec3f(0.0, 1.0, 0.0), t)

def main():
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
        vertex_buffer[i] = TVert(pos=v, color=v + 0.5)
    
    for i, v in enumerate(cube_ib):
        index_buffer[i] = Vec3is(v[0], v[1], v[2])

    start_time = time.time()
    gui = ti.GUI("Renderer", (window_size.x, window_size.y))
    while gui.running:

        t = time.time() - start_time

        update_constant_buffer(t)
        
        clear_buffers()

        stage_input_assembly()

        stage_vertex_shader()

        stage_rasterization()

        stage_pixel_shader()

        stage_output_merge()

        gui.set_image(screen_pixels)
        gui.show()


if __name__ == "__main__":
    main()