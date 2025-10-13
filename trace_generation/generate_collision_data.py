import pybullet as p
import pickle
import numpy as np
import math
import random
import time
import torch
import sys

import obb_calculator
from obb_forward_kinematics import OBBForwardKinematics
from robot_sphere_analyzer import RobotSphereAnalyzer


class PyBulletRobotSimulator:
    def __init__(self, use_gui=False):
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, 0)
        self.robot_id = None
        self.obstacle_ids = []

    def load_robot(self, robot_urdf):
        try:
            self.robot_id = p.loadURDF(robot_urdf, useFixedBase=True, physicsClientId=self.physics_client)
            return self.robot_id
        except Exception as e:
            print(f"Failed to load robot: {e}")
            return None

    def load_obstacles(self, obstacle_file):
        try:
            with open(obstacle_file, 'rb') as f:
                obstacles = pickle.load(f)
            for halfExtents, basePosition in obstacles:
                col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=self.physics_client)
                body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape_id, basePosition=basePosition, physicsClientId=self.physics_client)
                self.obstacle_ids.append(body_id)
        except Exception as e:
            print(f"Failed to load obstacles: {e}")

    def set_robot_config(self, joint_angles):
        if self.robot_id is None:
            return
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            p.resetJointState(self.robot_id, i, joint_angles[i], physicsClientId=self.physics_client)

    def check_collision_for_body(self, body_id):
        p.performCollisionDetection(physicsClientId=self.physics_client)
        for obstacle_id in self.obstacle_ids:
            contact_points = p.getContactPoints(bodyA=body_id, bodyB=obstacle_id, physicsClientId=self.physics_client)
            if len(contact_points) > 0:
                return True
        return False

    def disconnect(self):
        p.disconnect(physicsClientId=self.physics_client)


def read_configurations(filepath):
    configs = []
    with open(filepath, 'r') as f:
        for line in f:
            configs.append([float(x) for x in line.strip().split()])
    return configs


def initialize_obb_templates(robot_urdf_path):
    try:
        obb_data = obb_calculator.calculate_link_obbs(robot_urdf_path, verbose=False)
        return obb_data
    except Exception as e:
        print(f"Error: Failed to initialize OBB templates: {e}")
        return None


def main():
    if len(sys.argv) != 7:
        print("Usage: python generate_collision_data.py <config_file> <robot_urdf> <robot_name> <obstacle_file> <obb_output_file> <sphere_output_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    robot_urdf = sys.argv[2]
    robot_name = sys.argv[3]
    obstacle_file = sys.argv[4]
    obb_output_file = sys.argv[5]
    sphere_output_file = sys.argv[6]

    configs = read_configurations(config_file)

    sim = PyBulletRobotSimulator()
    sim.load_robot(robot_urdf)
    sim.load_obstacles(obstacle_file)

    obb_templates = initialize_obb_templates(robot_urdf)
    if not obb_templates:
        sim.disconnect()
        sys.exit(1)

    obb_fk = OBBForwardKinematics(sim.robot_id)
    sphere_analyzer = RobotSphereAnalyzer(robot_name, device="cpu")

    obb_results = []
    sphere_results = []

    for config in configs:
        sim.set_robot_config(config)

        # OBB collision detection
        obb_poses = obb_fk.compute_obb_poses(obb_templates)
        for obb_pose in obb_poses:
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=obb_pose['extents'] / 2.0)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=obb_pose['position'], baseOrientation=obb_pose['quaternion'])
            is_colliding = sim.check_collision_for_body(body)
            p.removeBody(body)

            obb_results.append({
                'center': obb_pose['position'].tolist(),
                'orientation': obb_pose['quaternion'].tolist(),
                'collision': is_colliding
            })

        # Sphere collision detection
        joint_config = torch.tensor(config, dtype=torch.float32).unsqueeze(0)
        world_spheres = sphere_analyzer.get_world_spheres(joint_config)
        for sphere in world_spheres:
            pos = sphere[:3].tolist()
            radius = sphere[3].item()
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=pos)
            is_colliding = sim.check_collision_for_body(body)
            p.removeBody(body)

            sphere_results.append({
                'center': pos,
                'radius': radius,
                'collision': is_colliding
            })

    with open(obb_output_file, 'wb') as f:
        pickle.dump(obb_results, f)

    with open(sphere_output_file, 'wb') as f:
        pickle.dump(sphere_results, f)

    sim.disconnect()

if __name__ == '__main__':
    main()
