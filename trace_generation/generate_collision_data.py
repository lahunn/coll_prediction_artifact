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


def read_obstacle_config_pair(filepath):
    """读取障碍物-配置对文件"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 新格式: {'obstacles': [...], 'configs': [...]}
    if isinstance(data, dict) and 'obstacles' in data and 'configs' in data:
        return data['obstacles'], data['configs']
    
    # 兼容旧格式: 只有configs列表
    elif isinstance(data, list):
        return None, data
    
    else:
        raise ValueError(f"无法识别的数据格式: {type(data)}")


def initialize_obb_templates(robot_urdf_path):
    try:
        obb_data = obb_calculator.calculate_link_obbs(robot_urdf_path, verbose=False)
        return obb_data
    except Exception as e:
        print(f"Error: Failed to initialize OBB templates: {e}")
        return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_collision_data.py <obstacle_config_pair_file> <robot_urdf> <robot_name> <obb_output_file> <sphere_output_file>")
        sys.exit(1)

    obstacle_config_file = sys.argv[1]
    robot_urdf = sys.argv[2]
    robot_name = sys.argv[3]
    obb_output_file = sys.argv[4]
    sphere_output_file = sys.argv[5]

    # 读取障碍物-配置对
    obstacles, configs = read_obstacle_config_pair(obstacle_config_file)
    
    if obstacles is None:
        print("Error: 无法读取障碍物信息")
        sys.exit(1)
    
    print(f"读取到 {len(obstacles)} 个障碍物, {len(configs)} 个配置")

    sim = PyBulletRobotSimulator()
    sim.load_robot(robot_urdf)
    
    if sim.robot_id is None:
        print("Error: Failed to load robot")
        sys.exit(1)
    
    # 直接使用从文件读取的obstacles
    for halfExtents, basePosition in obstacles:
        col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=sim.physics_client)
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape_id, basePosition=basePosition, physicsClientId=sim.physics_client)
        sim.obstacle_ids.append(body_id)

    obb_templates = initialize_obb_templates(robot_urdf)
    if not obb_templates:
        sim.disconnect()
        sys.exit(1)

    obb_fk = OBBForwardKinematics(sim.robot_id)
    sphere_analyzer = RobotSphereAnalyzer(robot_name)

    # --- Pre-create PyBullet bodies for OBBs ---
    obb_body_ids = []
    for obb_template in obb_templates:
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=obb_template['extents'] / 2.0)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape)
        obb_body_ids.append(body)

    # --- Pre-create PyBullet bodies for spheres ---
    # We need a sample configuration to know the number and radii of spheres
    # configs是边的列表,configs[0]是第一条边的配置数组,configs[0][0]是第一条边的第一个配置
    if configs and len(configs) > 0 and len(configs[0]) > 0:
        sample_config = configs[0][0]  # 第一条边的第一个配置
    else:
        sample_config = [0] * p.getNumJoints(sim.robot_id)
    
    joint_config_sample = torch.tensor(sample_config, dtype=torch.float32).unsqueeze(0).cuda()
    world_spheres_sample = sphere_analyzer.get_world_spheres(joint_config_sample)

    sphere_body_ids = []
    for sphere in world_spheres_sample:
        radius = sphere[3].item()
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape)
        sphere_body_ids.append(body)

    # configs是边的列表,每个元素是一条边上的配置数组
    obb_results = []  # 按edge组织: [[edge1_obbs], [edge2_obbs], ...]
    sphere_results = []  # 按edge组织: [[edge1_spheres], [edge2_spheres], ...]

    total_edges = len(configs)
    print(f"开始处理 {total_edges} 条边...")
    
    for edge_idx, edge_configs in enumerate(configs, 1):
        # 简洁的进度条
        if edge_idx % 10 == 0 or edge_idx == total_edges:
            progress = edge_idx / total_edges * 100
            print(f"进度: {edge_idx}/{total_edges} ({progress:.1f}%)", flush=True)
        
        # 为这条边创建结果列表
        edge_obb_results = []
        edge_sphere_results = []
        
        # 处理这条边上的每个配置
        for config in edge_configs:
            sim.set_robot_config(config)

            # --- OBB collision detection ---
            obb_poses = obb_fk.compute_obb_poses(obb_templates)
            config_obb_results = []
            for i, obb_pose in enumerate(obb_poses):
                p.resetBasePositionAndOrientation(obb_body_ids[i], obb_pose['position'], obb_pose['quaternion'])
                is_colliding = sim.check_collision_for_body(obb_body_ids[i])

                config_obb_results.append({
                    'center': obb_pose['position'],
                    'orientation': obb_pose['quaternion'],
                    'collision': is_colliding
                })
            edge_obb_results.append(config_obb_results)

            # --- Sphere collision detection ---
            joint_config = torch.tensor(config, dtype=torch.float32).unsqueeze(0).cuda()
            world_spheres = sphere_analyzer.get_world_spheres(joint_config)
            config_sphere_results = []
            for i, sphere in enumerate(world_spheres):
                pos = sphere[:3].tolist()
                p.resetBasePositionAndOrientation(sphere_body_ids[i], pos, [0,0,0,1])
                is_colliding = sim.check_collision_for_body(sphere_body_ids[i])

                config_sphere_results.append({
                    'center': pos,
                    'radius': sphere[3].item(),
                    'collision': is_colliding
                })
            edge_sphere_results.append(config_sphere_results)
        
        # 将这条边的结果添加到总结果中
        obb_results.append(edge_obb_results)
        sphere_results.append(edge_sphere_results)

    print("处理完成! 保存结果...")
    
    with open(obb_output_file, 'wb') as f:
        pickle.dump(obb_results, f)

    with open(sphere_output_file, 'wb') as f:
        pickle.dump(sphere_results, f)

    print(f"✓ OBB结果: {obb_output_file}")
    print(f"✓ Sphere结果: {sphere_output_file}")
    
    sim.disconnect()

if __name__ == '__main__':
    main()
