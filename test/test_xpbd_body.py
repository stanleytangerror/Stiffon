import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import time
import numpy as np
np.seterr(all='raise')
from math_utils import Vec3
from xpbd_body import Scene, Body, create_positional_constraint, create_positional_constraint_motor, SceneDebugRenderer, RotationalConstraint_AlignAxis, RotationalConstraint_TargetAngle, RotationalConstraint_Motor, create_rotational_motor
import math
from geometry import Box, Plane, Sphere

def single_distance_constraint():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)

    c1 = create_positional_constraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)

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

    c1 = create_positional_constraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    c2 = create_positional_constraint(b2, b3, Vec3(2.5, 0.0, 0.0), Vec3(3.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    c3 = create_positional_constraint(b3, b4, Vec3(4.5, 0.0, 0.0), Vec3(5.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)
    c4 = create_positional_constraint(b4, b5, Vec3(6.5, 0.0, 0.0), Vec3(7.5, 0.0, 0.0), stiffness=16.0, damping=8.0, distance=1.0)

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

def rotational_constraint_align_axis():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)

    c1 = create_positional_constraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=float('inf'), damping=0.0, distance=1.0)
    c2 = RotationalConstraint_AlignAxis(b1, b2, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 100)

    scene.add_constraint(c1)
    scene.add_constraint(c2)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()


def rotational_constraint_target_angle():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)

    c1 = create_positional_constraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=float('inf'), damping=0.0, distance=1.0)
    c2 = RotationalConstraint_AlignAxis(b1, b2, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0), stiffness=float('inf'))
    c3 = RotationalConstraint_TargetAngle(b1, b2, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), stiffness=100, tangent_angle=math.pi / 4)

    scene.add_constraint(c1)
    scene.add_constraint(c2)
    scene.add_constraint(c3)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()


def rotational_constraint_motor():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0))

    scene.add_body(b1)
    scene.add_body(b2)

    c1 = create_positional_constraint(b1, b2, Vec3(0.5, 0.0, 0.0), Vec3(1.5, 0.0, 0.0), stiffness=float('inf'), damping=0.0, distance=1.0)
    c2 = RotationalConstraint_AlignAxis(b1, b2, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0), stiffness=float('inf'))
    c3 = RotationalConstraint_Motor(b1, b2, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), stiffness=float('inf'), angular_speed=math.pi)

    scene.add_constraint(c1)
    scene.add_constraint(c2)
    scene.add_constraint(c3)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()

def rotational_motor():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0), shape=Box(Vec3(1.0, 0.2, 0.2)))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0), shape=Box(Vec3(1.0, 0.2, 0.2)))
    b3 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(4.0, 0.0, 0.0), shape=Box(Vec3(1.0, 0.2, 0.2)))

    scene.add_body(b1)
    scene.add_body(b2)
    scene.add_body(b3)

    cs1 = create_rotational_motor(b1, b2, p=Vec3(1.0, 0.0, 0.0), axis=Vec3(0.0, 1.0, 0.0), angular_speed=math.pi * 10.0)
    scene.add_constraints(cs1)

    cs2 = create_rotational_motor(b2, b3, p=Vec3(3.0, 0.0, 0.0), axis=Vec3(0.0, 1.0, 0.0), angular_speed=-math.pi * 20.0)
    scene.add_constraints(cs2)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()

def positional_motor():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0), shape=Box(Vec3(1.0, 0.2, 0.2)))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0), shape=Box(Vec3(1.0, 0.2, 0.2)))
    b3 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(4.0, 0.0, 0.0), shape=Box(Vec3(1.0, 0.2, 0.2)))

    scene.add_body(b1)
    scene.add_body(b2)
    scene.add_body(b3)

    cs1 = create_positional_constraint_motor(b1, b2, p=Vec3(1.0, 0.0, 0.0), speed=0.0)
    scene.add_constraint(cs1)

    cs2 = create_positional_constraint_motor(b2, b3, p=Vec3(3.0, 0.0, 0.0), speed=0.0)
    scene.add_constraint(cs2)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        cs1.speed = math.sin(frame_count * 0.1) * 10.0
        cs2.speed = math.cos(frame_count * 0.1) * 10.0
        scene.step_simulation(0.01)
        renderer.render()
        frame_count += 1

def test_contact_constraint():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0), shape=Plane(Vec3(0.0, 0.0, -3.0), Vec3(0.1, 0.0, 1.0)))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(2.0, 0.0, 0.0), shape=Sphere(1.0))
    scene.add_body(b1)
    scene.add_body(b2)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_count = 0
    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()
        frame_count += 1

def test_contact_constraint_2():
    last_spawn_time = time.time()
    count = 10
    scene = Scene()

    scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), shape=Plane(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 1.0))))
    scene.add_body(Body(mass=float('inf'), inertia=Vec3(float('inf'), float('inf'), float('inf')), shape=Plane(Vec3(0.0, 0.0, 0.0), Vec3(-1.0, 0.0, 1.0))))

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    frame_no = 0
    while renderer.is_running():

        print(f"Frame {frame_no}")

        if time.time() - last_spawn_time > 1.0 and count > 0:
            scene.add_body(Body(mass=1.0, v=Vec3(np.random.uniform(-2.0, 2.0), 0.0, 10.0), x=Vec3(0.0, 0.0, 3.0), shape=Sphere(0.5)))
            last_spawn_time = time.time()
            count -= 1

        for _ in range(4):
            scene.step_simulation(0.01)
            renderer.render()

            frame_no += 1

def test_rolling():
    scene = Scene()
    b1 = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0), shape=Plane(Vec3(0.0, 0.0, -4.0), Vec3(0.0, 0.0, 1.0)))
    b2 = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(0.0, 0.0, 0.0), shape=Sphere(0.5))

    scene.add_body(b1)
    scene.add_body(b2)

    c = RotationalConstraint_Motor(b1, b2, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), stiffness=float('inf'), angular_speed=math.pi * 30.0)
    scene.add_constraint(c)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()


def test_2_wheels():
    scene = Scene()
    ground = Body(mass=float('inf'), inertia=Vec3(1.0, 1.0, 1.0) * float('inf'), x=Vec3(0.0, 0.0, 0.0), shape=Plane(Vec3(0.0, 0.0, -3.0), Vec3(0.0, 0.0, 1.0)))
    body = Body(mass=1.0, inertia=Vec3(1.0, 1.0, 1.0), x=Vec3(0.0, 0.0, 2.0), shape=Box(Vec3(4.0, 0.2, 0.2)))
    left_wheel = Body(mass=1.0, inertia=Vec3(0.01, 0.01, 0.01), x=Vec3(-2.0, 0.0, 0.0), shape=Sphere(0.5))
    right_wheel = Body(mass=1.0, inertia=Vec3(0.01, 0.01, 0.01), x=Vec3(2.0, 0.0, 0.0), shape=Sphere(0.5))

    scene.add_body(ground)
    scene.add_body(body)
    scene.add_body(left_wheel)
    scene.add_body(right_wheel)

    cs1 = create_rotational_motor(body, left_wheel, p=Vec3(-2.0, 0.0, 0.0), axis=Vec3(0.0, 1.0, 0.0), angular_speed=math.pi * 30.0)
    scene.add_constraints(cs1)

    cs2 = create_rotational_motor(body, right_wheel, p=Vec3(2.0, 0.0, 0.0), axis=Vec3(0.0, 1.0, 0.0), angular_speed=math.pi * 30.0)
    scene.add_constraints(cs2)

    renderer = SceneDebugRenderer(scene, width=800, height=600)
    renderer.renderer.set_camera(eye=np.array([0.0, -10.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

    while renderer.is_running():
        scene.step_simulation(0.01)
        renderer.render()


if __name__ == "__main__":
    test_rolling()