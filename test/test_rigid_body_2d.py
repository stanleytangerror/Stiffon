#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import math
import time
import numpy as np
from rigid_body_2d import Scene2d, Scene2dDebugRenderer, Body2d
from math_utils import Vec2, Transform2d
from geometry_2d import Circle, Rectangle

# def test_contact_constraint_0():
#     scene = Scene()

#     sphere1 = Body(mass=1.0, linear_velocity=Vec3(0.0, 0.0, -10.0), pose=Transform(Vec3(0.0, 0.0, 9.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(sphere1)
#     sphere2 = Body(mass=1.0, linear_velocity=Vec3(0.0, 0.0, -10.0), pose=Transform(Vec3(0.0, 0.0, 5.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(sphere2)
#     sphere3 = Body(mass=1.0, linear_velocity=Vec3(0.0, 0.0, -10.0), pose=Transform(Vec3(0.0, 0.0, 1.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(sphere3)
#     sphere4 = Body(mass=1.0, linear_velocity=Vec3(0.0, 0.0, -10.0), pose=Transform(Vec3(0.0, 0.0, -3.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(sphere4)
#     ground = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, -7.0), Mat33.identity()), geometry=Plane(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)))
#     scene.add_body(ground)

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -20.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     while renderer.is_running():
#         scene.step_simulation(0.01)
#         renderer.render()
        
# def test_contact_constraint_1():
#     scene = Scene()
    
#     sphere1 = Body(mass=1.0, linear_velocity=Vec3(1.0, 0.0, 10.0), pose=Transform(Vec3(-1.0, 0.0, 3.0), Mat33.identity()), geometry=Sphere(0.5))
#     scene.add_body(sphere1)

#     ground = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 1.0)))
#     scene.add_body(ground)

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     while renderer.is_running():
#         scene.step_simulation(0.01)
#         renderer.render()

# def test_contact_constraint_2():
#     last_spawn_time = time.time()
#     count = 10
#     scene = Scene()

#     scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 1.0))))
#     scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(0.0, 0.0, 0.0), Vec3(-1.0, 0.0, 1.0))))

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     frame_no = 0
#     while renderer.is_running():

#         print(f"Frame {frame_no}")

#         if time.time() - last_spawn_time > 1.0 and count > 0:
#             scene.add_body(Body(mass=1.0, linear_velocity=Vec3(np.random.uniform(-2.0, 2.0), 0.0, 10.0), pose=Transform(Vec3(0.0, 0.0, 3.0), Mat33.identity()), geometry=Sphere(0.5)))
#             last_spawn_time = time.time()
#             count -= 1

#         for _ in range(4):
#             scene.step_simulation(0.01)
#             renderer.render()

#             frame_no += 1

# def test_contact_constraint_3():
#     last_spawn_time = time.time()
#     count = 3
#     scene = Scene()

#     scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0))))
#     scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(3.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))))
#     scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(-3.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0))))

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     frame_no = 0
#     while renderer.is_running():

#         print(f"Frame {frame_no}")

#         if time.time() - last_spawn_time > 2.0 and count > 0:
#             scene.add_body(Body(mass=1.0, linear_velocity=Vec3(np.random.uniform(-2.0, 2.0), 0.0, 10.0), pose=Transform(Vec3(0.0, 0.0, 3.0), Mat33.identity()), geometry=Sphere(0.5)))
#             last_spawn_time = time.time()
#             count -= 1

#         scene.step_simulation(0.01)
#         renderer.render()

#         frame_no += 1

# def test_distance_constraint():
#     scene = Scene()
    
#     box1 = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
#     scene.add_body(box1)

#     box2 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(5.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
#     scene.add_body(box2)

#     scene.add_distance_constraint(box1, box2, Vec3(1.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0), 3.0)

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     while renderer.is_running():
#         for _ in range(4):
#             scene.step_simulation(1.0 / 240)
#         renderer.render()    

# def test_distance_constraint_chain_horizontal():
#     scene = Scene()
    
#     box1 = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box1)
#     box2 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(5.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box2)
#     box3 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(10.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box3)
#     box4 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(15.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box4)
#     box5 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(20.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box5)
#     box6 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(25.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box6)

#     scene.add_distance_constraint(box1, box2, Vec3(1.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0), 3.0)
#     scene.add_distance_constraint(box2, box3, Vec3(6.0, 0.0, 0.0), Vec3(9.0, 0.0, 0.0), 3.0)
#     scene.add_distance_constraint(box3, box4, Vec3(11.0, 0.0, 0.0), Vec3(14.0, 0.0, 0.0), 3.0)
#     scene.add_distance_constraint(box4, box5, Vec3(16.0, 0.0, 0.0), Vec3(19.0, 0.0, 0.0), 3.0)
#     scene.add_distance_constraint(box5, box6, Vec3(21.0, 0.0, 0.0), Vec3(24.0, 0.0, 0.0), 3.0)

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -30.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     while renderer.is_running():
#         for _ in range(8):
#             scene.step_simulation(1.0 / 60)
#         renderer.render()    


# def test_distance_constraint_chain_vertical():
#     scene = Scene()
    
#     box1 = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box1)
#     box2 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(0.0, 0.0, -5.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box2)
#     box3 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(0.0, 0.0, -10.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box3)
#     box4 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(0.0, 0.0, -15.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box4)
#     box5 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(0.0, 0.0, -20.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box5)
#     box6 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(0.0, 0.0, -25.0), Mat33.identity()), geometry=Sphere(1.0))
#     scene.add_body(box6)

#     scene.add_distance_constraint(box1, box2, Vec3(0.0, 0.0, -1.0), Vec3(0.0, 0.0, -4.0), 3.0)
#     scene.add_distance_constraint(box2, box3, Vec3(0.0, 0.0, -6.0), Vec3(0.0, 0.0, -9.0), 3.0)
#     scene.add_distance_constraint(box3, box4, Vec3(0.0, 0.0, -11.0), Vec3(0.0, 0.0, -14.0), 3.0)
#     scene.add_distance_constraint(box4, box5, Vec3(0.0, 0.0, -16.0), Vec3(0.0, 0.0, -19.0), 3.0)
#     scene.add_distance_constraint(box5, box6, Vec3(0.0, 0.0, -21.0), Vec3(0.0, 0.0, -24.0), 3.0)

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -30.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     while renderer.is_running():
#         for _ in range(8):
#             scene.step_simulation(1.0 / 60)
#         renderer.render()    


def test_pin_constraint_chain():
    scene = Scene2d()
    
    box1 = Body2d(mass=float('inf'), inertia=float('inf'), pose=Transform2d(Vec2(0.0, 0.0), 0.0), geometry=Circle(0.5))
    scene.add_body(box1)
    box2 = Body2d(mass=1.0, inertia=2.0 / 3, linear_velocity=Vec2(0.0, 0.0), pose=Transform2d(Vec2(5.0, 0.0), 0.0), geometry=Circle(0.5))
    scene.add_body(box2)
    box3 = Body2d(mass=1.0, inertia=2.0 / 3, linear_velocity=Vec2(0.0, 0.0), pose=Transform2d(Vec2(10.0, 0.0), 0.0), geometry=Circle(0.5))
    scene.add_body(box3)
    box4 = Body2d(mass=1.0, inertia=2.0 / 3, linear_velocity=Vec2(0.0, 0.0), pose=Transform2d(Vec2(15.0, 0.0), 0.0), geometry=Circle(0.5))
    scene.add_body(box4)
    box5 = Body2d(mass=1.0, inertia=2.0 / 3, linear_velocity=Vec2(0.0, 0.0), pose=Transform2d(Vec2(20.0, 0.0), 0.0), geometry=Circle(0.5))
    scene.add_body(box5)

    scene.add_pin_constraint(box1, box2, Vec2(2.5, 0.0), Vec2(2.5, 0.0))
    scene.add_pin_constraint(box2, box3, Vec2(7.5, 0.0), Vec2(7.5, 0.0))
    scene.add_pin_constraint(box3, box4, Vec2(12.5, 0.0), Vec2(12.5, 0.0))
    scene.add_pin_constraint(box4, box5, Vec2(17.5, 0.0), Vec2(17.5, 0.0))

    renderer = Scene2dDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, 0.0, -30.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

    while renderer.is_running():
        for _ in range(4):
            scene.step_simulation(1.0 / 60)
        renderer.render()    



# def test_hinge_constraint():
#     scene = Scene()
    
#     box1 = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
#     scene.add_body(box1)

#     box2 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(5.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
#     scene.add_body(box2)

#     scene.add_hinge_constraint(box1, box2, Vec3(2.5, 0.0, 0.0), Vec3(1.0, 0.0, 1.0))

#     renderer = SceneDebugRenderer(scene, width=800, height=600)
#     renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

#     while renderer.is_running():
#         for _ in range(4):
#             scene.step_simulation(1.0 / 60)
#         renderer.render()    


def test_spring_constraint_0():
    scene = Scene2d()
    
    box1 = Body2d(mass=float('inf'), inertia=float('inf'), pose=Transform2d(Vec2(0.0, 0.0), 0.0), geometry=Rectangle(Vec2(1.0, 1.0)))
    scene.add_body(box1)

    box2 = Body2d(mass=1.0, inertia=2.0 / 3, linear_velocity=Vec2(0.0, 0.0), pose=Transform2d(Vec2(5.0, 0.0), 0.0), geometry=Rectangle(Vec2(1.0, 1.0)))
    scene.add_body(box2)

    scene.add_spring_constraint(box1, box2, Vec2(1.0, 0.0), Vec2(4.0, 0.0), 100.0, 10.0)

    renderer = Scene2dDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, 0.0, -10.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

    while renderer.is_running():
        for _ in range(4):
            scene.step_simulation(1.0 / 60)
        renderer.render()  

def test_spring_constraint_1():
    scene = Scene2d()
    
    box1 = Body2d(mass=float('inf'), inertia=float('inf'), pose=Transform2d(Vec2(0.0, 0.0), 0.0), geometry=Circle(1.0))
    scene.add_body(box1)

    box2 = Body2d(mass=1.0, inertia=2.0 / 3, linear_velocity=Vec2(0.0, 0.0), pose=Transform2d(Vec2(0.0, 0.0), 0.0), geometry=Circle(1.0))
    scene.add_body(box2)

    box3 = Body2d(mass=float('inf'), inertia=float('inf'), pose=Transform2d(Vec2(0.0, 0.0), 0.0), geometry=Circle(1.0))
    scene.add_body(box3)

    scene.add_spring_constraint(box1, box2, Vec2(0.0, 0.0), Vec2(0.0, 0.0), 100.0, 10.0)
    scene.add_spring_constraint(box2, box3, Vec2(0.0, 0.0), Vec2(0.0, 0.0), 100.0, 10.0)

    renderer = Scene2dDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, 0.0, -100.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

    start_time = time.time()

    while renderer.is_running():
        t = time.time() - start_time
        box3.linear_velocity = Vec2(math.sin(t) * 10, 0.0)
        scene.step_simulation(1.0 / 60)
        renderer.render() 


def test_angular_spring_constraint_0():
    scene = Scene2d()
    
    box1 = Body2d(mass=float('inf'), inertia=float('inf'), pose=Transform2d(Vec2(-5.0, 0.0), 0.0), geometry=Rectangle(Vec2(2.0, 0.1)))
    scene.add_body(box1)

    box2 = Body2d(mass=1.0, inertia=2.0 / 3, pose=Transform2d(Vec2(-5.0, 0.0), 0.0), geometry=Rectangle(Vec2(2.0, 0.1)))
    scene.add_body(box2)

    box3 = Body2d(mass=float('inf'), inertia=float('inf'), pose=Transform2d(Vec2(5.0, 0.0), 0.0), geometry=Rectangle(Vec2(5.0, 0.1)))
    scene.add_body(box3)

    box4 = Body2d(mass=1.0, inertia=2.0 / 3, pose=Transform2d(Vec2(5.0, 0.0), 0.0), geometry=Rectangle(Vec2(5.0, 0.1)))
    scene.add_body(box4)


    scene.add_pin_constraint(box1, box2, Vec2(-7.0, 0.0), Vec2(-7.0, 0.0))
    scene.add_angular_spring_constraint(box1, box2, Vec2(-7.0, 0.0), Vec2(-7.0, 0.0), 100.0, 10.0)

    scene.add_pin_constraint(box3, box4, Vec2(10.0, 0.0), Vec2(10.0, 0.0))
    scene.add_angular_spring_constraint(box3, box4, Vec2(10.0, 0.0), Vec2(10.0, 0.0), 100.0, 10.0)

    renderer = Scene2dDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, 0.0, -10.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

    while renderer.is_running():
        scene.step_simulation(1.0 / 60)
        renderer.render()  

if __name__ == "__main__":
    # test_contact_constraint_2()
    # test_distance_constraint_chain_vertical()
    # test_distance_constraint_chain_horizontal()
    # test_pin_constraint_chain()
    # test_spring_constraint_0()
    # test_spring_constraint_1()
    test_angular_spring_constraint_0()