import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
np.seterr(all='raise')
from math_utils import Vec3
from xpbd_body import Scene, Body, DistanceConstraint, SceneDebugRenderer, HingeAxisConstraint

def single_distance_constraint():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)

    c1 = DistanceConstraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)

    scene.add_constraint(c1)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        scene.step_simulation(0.01)
        if frame_count % 10 == 0:
            renderer.render()
        frame_count += 1

def multiple_distance_constraint():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))
    b3 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(4.0, 0.0, 0.0))
    b4 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(6.0, 0.0, 0.0))
    b5 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(8.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)
    scene.add_body(b3)
    scene.add_body(b4)
    scene.add_body(b5)

    c1 = DistanceConstraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    c2 = DistanceConstraint(b2, b3, Vec3(2.5, 0.0, 0.0), Vec3(3.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    c3 = DistanceConstraint(b3, b4, Vec3(4.5, 0.0, 0.0), Vec3(5.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    c4 = DistanceConstraint(b4, b5, Vec3(6.5, 0.0, 0.0), Vec3(7.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)

    scene.add_constraint(c1)
    scene.add_constraint(c2)
    scene.add_constraint(c3)
    scene.add_constraint(c4)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        scene.step_simulation(0.01)
        if frame_count % 10 == 0:
            renderer.render()
        frame_count += 1

def hinge_axis_constraint():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)

    c1 = DistanceConstraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    c2 = HingeAxisConstraint(b1, b2, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))

    scene.add_constraint(c1)
    scene.add_constraint(c2)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        scene.step_simulation(0.01)
        if frame_count % 10 == 0:
            renderer.render()
        frame_count += 1

if __name__ == "__main__":
    hinge_axis_constraint()