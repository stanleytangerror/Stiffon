import taichi as ti
import numpy as np

Vec3is = ti.types.vector(3, ti.i16)
Vec2i = ti.types.vector(2, ti.i32)
Vec2f = ti.types.vector(2, ti.f32)
Vec3f = ti.types.vector(3, ti.f32)
Vec4f = ti.types.vector(4, ti.f32)
Mat33f = ti.types.matrix(3, 3, ti.f32)
Mat44f = ti.types.matrix(4, 4, ti.f32)

ti.init(arch=ti.gpu, debug=True, default_fp=ti.f32)

@ti.func
def edge(a, b, c): 
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)

@ti.func
def barycentric_coords(p0, p1, p2, p):
    result = Vec3f(0.0, 0.0, 0.0)
    area = edge(p0, p1, p2)
    if abs(area) >= 1e-10:
        w0 = edge(p1, p2, p) / area
        w1 = edge(p2, p0, p) / area
        w2 = edge(p0, p1, p) / area
        result = Vec3f(w0, w1, w2)
    return result

width = 512
height = 512

TVert = ti.types.struct(position=Vec3f)
vertex_buffer = TVert.field(shape=8)
index_buffer = ti.field(dtype=Vec3is, shape=12)

TInputAssem = ti.types.struct(v0=Vec4f, v1=Vec4f, v2=Vec4f)
assembled_input = TInputAssem.field(shape=index_buffer.shape[0])

TRasterInput = ti.types.struct(v0=Vec4f, v1=Vec4f, v2=Vec4f)
rasterize_input = TRasterInput.field(shape=assembled_input.shape[0])

TPsInput = ti.types.struct(prim=Vec4f, clipped=ti.i8)
pixel_shading_input = TPsInput.field(shape=(width, height))

output_merge_input = ti.field(dtype=Vec4f, shape=(width, height))

screen_pixels = ti.Vector.field(3, ti.f32, shape=(width, height))
depth_buffer = ti.field(dtype=ti.f32, shape=(width, height))

@ti.func
def interpoliate(barycentric, v0, v1, v2):
    return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z

@ti.func
def vs(vertex):
    return Vec4f([(vertex.x + 1) * 0.5 * width,
                      (vertex.y + 1) * 0.5 * height,
                      vertex.z], 1.0)

@ti.func
def ps(vertex):
    return Vec4f(0.5, 0.5, 0.5, 1.0)

@ti.kernel
def stage_input_assembly():
    for i in index_buffer:
        indices = index_buffer[i]
        v0 = vertex_buffer[indices.x].position
        v1 = vertex_buffer[indices.y].position
        v2 = vertex_buffer[indices.z].position
        assembled_input[i].v0 = ti.Vector([v0.x, v0.y, v0.z, 1.0])
        assembled_input[i].v1 = ti.Vector([v1.x, v1.y, v1.z, 1.0])
        assembled_input[i].v2 = ti.Vector([v2.x, v2.y, v2.z, 1.0])

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
        pixel_shading_input[u, v].prim = Vec4f(0.0, 0.0, 0.0, 1.0)
        pixel_shading_input[u, v].clipped = 0

    for prim_i in rasterize_input:
        v0 = rasterize_input[prim_i].v0
        v1 = rasterize_input[prim_i].v1
        v2 = rasterize_input[prim_i].v2

        min_x = int(min(v0.x, v1.x, v2.x))
        max_x = int(max(v0.x, v1.x, v2.x))
        min_y = int(min(v0.y, v1.y, v2.y))
        max_y = int(max(v0.y, v1.y, v2.y))

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                p = ti.Vector([x + 0.5, y + 0.5])
                w = barycentric_coords(v0.xy, v1.xy, v2.xy, p)
                if w.x >= 0 and w.y >= 0 and w.z >= 0:
                    depth_buffer[x, y] = 1e9  # Reset depth buffer for this pixel
                    pixel_shading_input[x, y].prim = interpoliate(w, v0, v1, v2)
                    pixel_shading_input[x, y].clipped = 1

@ti.kernel
def stage_pixel_shader():
    for pixel_u, pixel_v in output_merge_input:
        ps_input = pixel_shading_input[pixel_u, pixel_v]
        if ps_input.clipped > 0.5:
            output_merge_input[pixel_u, pixel_v] = ps(ps_input)

@ti.kernel
def stage_output_merge():
    for pixel_u, pixel_v in output_merge_input:
        screen_pixels[pixel_u, pixel_v] = output_merge_input[pixel_u, pixel_v].xyz

@ti.kernel
def clear_buffers():
    for I in ti.grouped(screen_pixels):
        screen_pixels[I] = ti.Vector([0.0, 0.0, 0.0])
    for I in ti.grouped(depth_buffer):
        depth_buffer[I] = 1e9


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
        vertex_buffer[i] = TVert(position=v)
    
    for i, v in enumerate(cube_ib):
        index_buffer[i] = Vec3is(v[0], v[1], v[2])

    gui = ti.GUI("Rasterizer", (width, height))
    while gui.running:
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