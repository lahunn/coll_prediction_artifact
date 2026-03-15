#!/usr/bin/env python3
"""
机器人球体建模集成可视化工具 (基于 RobotEnv)

功能：
1. 调用 RobotEnv 加载机器人 URDF。
2. 调用 RobotSphereAnalyzer 获取球体建模数据。
3. 创建无碰撞属性的视觉球体，并随机器人关节实时联动。
4. 提供滑块控制关节和相机视角。
5. 支持通过命令行参数设置机器人模型透明度。
"""

import sys
import time
import argparse
import numpy as np
import pybullet as p
import torch
from pathlib import Path

# 导入项目核心组件
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.robot.sphere_analyzer import RobotSphereAnalyzer

class RobotSphereVisualizer:
    def __init__(self, robot_name, robot_transparent=False):
        self.robot_name = robot_name
        self.robot_transparent = robot_transparent
        self.sphere_alpha = 0.6  # 固定球体透明度
        
        # 1. 直接调用 RobotEnv 初始化 PyBullet GUI 和加载机器人
        print(f"正在通过 RobotEnv 初始化机器人: {robot_name}")
        self.env = RobotEnv(robot_name, OBB_GUI=True)
        self.physics_client = self.env.physics_client
        self.robot_id = self.env.robotId
        
        # 2. 如果开启了机器人透明，遍历所有 link 设置透明度
        if self.robot_transparent:
            print("设置机器人模型为半透明状态...")
            # -1 代表 base link，0 到 numJoints-1 代表其他 link
            for link_idx in range(-1, p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
                p.changeVisualShape(self.robot_id, link_idx, rgbaColor=[1, 1, 1, 0.4], 
                                  physicsClientId=self.physics_client)
        
        # 3. 初始化球体分析器
        print(f"正在分析球体结构: {robot_name}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.analyzer = RobotSphereAnalyzer(robot_name, device=device)
        
        # 4. 存储球体视觉 ID
        self.sphere_body_ids = []
        
        # 5. 设置 GUI
        self._setup_gui()

    def _setup_gui(self):
        """设置滑块和相机"""
        self.sliders = {}
        
        # 关节控制
        for i, joint_idx in enumerate(self.env.valid_joints):
            info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.physics_client)
            name = info[1].decode("utf-8")
            low, high = self.env.pose_range[i]
            # 如果限位无效则给定默认值
            if low >= high: low, high = -3.14, 3.14
            self.sliders[f"j_{i}"] = p.addUserDebugParameter(f"关节 {name}", low, high, 0)
        
        # 相机初始位置
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
            physicsClientId=self.physics_client
        )

    def _update_visual_spheres(self, world_spheres):
        """创建或更新视觉球体（无碰撞属性）"""
        # 如果球体数量发生变化或首次运行，则重新创建
        if not self.sphere_body_ids or len(self.sphere_body_ids) != len(world_spheres):
            for b_id in self.sphere_body_ids:
                p.removeBody(b_id, physicsClientId=self.physics_client)
            self.sphere_body_ids = []
            
            colors = [[1, 0, 0, self.sphere_alpha], [0, 1, 0, self.sphere_alpha], 
                      [0, 0, 1, self.sphere_alpha], [1, 1, 0, self.sphere_alpha], 
                      [1, 0, 1, self.sphere_alpha], [0, 1, 1, self.sphere_alpha]]
            
            for i, (x, y, z, r) in enumerate(world_spheres):
                color = [1, 1, 0, 0.6]  # 黄色半透明
                # 关键：只创建 VisualShape，不创建 CollisionShape
                v_id = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color, 
                                          physicsClientId=self.physics_client)
                b_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, 
                                        basePosition=[x, y, z], 
                                        physicsClientId=self.physics_client)
                self.sphere_body_ids.append(b_id)
        else:
            # 仅仅移动位置
            for i, (x, y, z, r) in enumerate(world_spheres):
                p.resetBasePositionAndOrientation(self.sphere_body_ids[i], [x, y, z], [0, 0, 0, 1],
                                                 physicsClientId=self.physics_client)

    def run(self):
        """运行可视化主循环"""
        print("\n=== 可视化已启动 ===")
        print("- 拖动滑块：改变关节角度")
        print("- 球体跟随：实时更新建模位置")
        
        try:
            while True:
                # 1. 获取滑块值并设置机器人姿态
                current_q = []
                for i in range(self.env.config_dim):
                    val = p.readUserDebugParameter(self.sliders[f"j_{i}"])
                    current_q.append(val)
                
                # 使用 RobotEnv 的 set_config 方法
                self.env.set_config(current_q)
                
                # 2. 计算并更新球体
                q_tensor = torch.tensor([current_q], dtype=torch.float32).to(self.analyzer.device)
                
                # 如果 Analyzer 要求的 DOF 比 active 更多，补零
                expected_dof = self.analyzer.cuda_model.get_dof()
                if q_tensor.shape[1] < expected_dof:
                    padding = torch.zeros((1, expected_dof - q_tensor.shape[1])).to(self.analyzer.device)
                    q_tensor = torch.cat([q_tensor, padding], dim=1)
                
                world_spheres = self.analyzer.get_world_spheres(q_tensor)
                self._update_visual_spheres(world_spheres)
                
                p.stepSimulation(physicsClientId=self.physics_client)
                time.sleep(1./60.)
                
        except KeyboardInterrupt:
            print("\n退出可视化")
        finally:
            self.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="机器人球体建模集成可视化工具")
    parser.add_argument("--robot", default="jaco7", help="机器人名称 (franka, iiwa, jaco7等)")
    parser.add_argument("--robot-transparent", action="store_true", help="是否让机器人模型半透明")
    args = parser.parse_args()
    
    vis = RobotSphereVisualizer(args.robot, robot_transparent=args.robot_transparent)
    vis.run()
