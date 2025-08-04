#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import time
import numpy as np
from rigid_body import Scene, SceneDebugRenderer, Body, Vec3, Transform, Mat33, Sphere, Plane, Box

def test_contact_constraint():
    scene = Scene()
    
    sphere1 = Body(mass=1.0, linear_velocity=Vec3(1.0, 0.0, 10.0), pose=Transform(Vec3(-1.0, 0.0, 3.0), Mat33.identity()), geometry=Sphere(0.5))
    scene.add_body(sphere1)

    ground = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(1.0, 0.0, 1.0), 0.0))
    scene.add_body(ground)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()

def test_contact_constraint_2():
    last_spawn_time = time.time()
    count = 3
    scene = Scene()

    scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(1.0, 0.0, 1.0), 0.0)))
    scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Plane(Vec3(-1.0, 0.0, 1.0), 0.0)))

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_no = 0
    while renderer.is_running():

        print(f"Frame {frame_no}")

        if time.time() - last_spawn_time > 2.0 and count > 0:
            scene.add_body(Body(mass=1.0, linear_velocity=Vec3(np.random.uniform(-2.0, 2.0), 0.0, 10.0), pose=Transform(Vec3(0.0, 0.0, 3.0), Mat33.identity()), geometry=Sphere(0.5)))
            last_spawn_time = time.time()
            count -= 1

        scene.step_simulation(0.01)
        renderer.render()

        frame_no += 1

def test_distance_constraint():
    scene = Scene()
    
    box1 = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
    scene.add_body(box1)

    box2 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(5.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
    scene.add_body(box2)

    scene.add_distance_constraint(box1, box2, Vec3(1.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0), 3.0)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        for _ in range(4):
            scene.step_simulation(1.0 / 240)
        renderer.render()    

def test_distance_constraint_chain():
    scene = Scene()
    
    box1 = Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), pose=Transform(Vec3(0.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
    scene.add_body(box1)

    box2 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(5.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
    scene.add_body(box2)

    box3 = Body(mass=1.0, inertia=Vec3(2.0 / 3, 2.0 / 3, 2.0 / 3), linear_velocity=Vec3(0.0, 0.0, 0.0), pose=Transform(Vec3(10.0, 0.0, 0.0), Mat33.identity()), geometry=Box(Vec3(1.0, 1.0, 1.0)))
    scene.add_body(box3)

    scene.add_distance_constraint(box1, box2, Vec3(1.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0), 3.0)
    scene.add_distance_constraint(box2, box3, Vec3(6.0, 0.0, 0.0), Vec3(9.0, 0.0, 0.0), 3.0)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        for _ in range(4):
            scene.step_simulation(1.0 / 240)
        renderer.render()    

if __name__ == "__main__":
    test_distance_constraint_chain()