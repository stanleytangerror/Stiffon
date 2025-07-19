#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import time
import numpy as np
from rigid_body import Scene, SceneDebugRenderer, Body, Vec3, Transform, Mat33, Sphere, Plane


if __name__ == "__main__":

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

