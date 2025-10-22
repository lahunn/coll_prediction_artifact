"""
机器人碰撞检测数据生成脚本
用于生成微架构仿真所需的训练/测试数据集

主要功能：
1. 在给定障碍物环境中随机采样机器人姿态
2. 计算每个姿态下各个 link 的 OBB (Oriented Bounding Box) 几何信息
3. 执行精确的碰撞检测，生成 ground truth 标签
4. 输出 link 级和 pose 级的碰撞检测数据供后续仿真使用

输入参数：
- sys.argv[1]: numqueries (采样姿态数量)
- sys.argv[2]: foldername (环境文件夹)
- sys.argv[3]: filenumber (环境文件编号)

输出文件：
- obstacles_X_coord.pkl: link 级数据 (qarr, dirarr, yarr)
- obstacles_X_pose.pkl: pose 级数据 (qarr_pose, yarr_pose)
"""

import pybullet as p
import pickle
import numpy as np
import math
import random
import time
import obb_calculator
from obb_forward_kinematics import OBBForwardKinematics
from sphere_as.robot_sphere_analyzer import RobotSphereAnalyzer
import torch

# ========== PyBullet机器人仿真类 ==========


class PyBulletRobotSimulator:
    """PyBullet机器人仿真器，替换Klampt功能"""
    def __init__(self, use_gui=False):
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, 0)  # 无重力
        self.robot_id = None
        self.obstacle_ids = []
        self.joint_limits = []
        self.valid_joints = []

        # 用于球体碰撞检测的独立仿真器
        self.sphere_physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.sphere_physics_client)
        self.sphere_obstacle_ids = []

    def load_scene(self, scene_file):
        """加载MuJoCo格式的场景文件"""
        try:
            # 在主仿真器中加载场景
            scene_objects = self._load_scene_internal(scene_file, self.physics_client, "主仿真器")
            if scene_objects:
                self.obstacle_ids = scene_objects

            # 在球体仿真器中加载相同的场景
            sphere_scene_objects = self._load_scene_internal(scene_file, self.sphere_physics_client, "球体仿真器")
            if sphere_scene_objects:
                self.sphere_obstacle_ids = sphere_scene_objects

            return self.obstacle_ids
        except Exception as e:
            print(f"Failed to load scene: {e}")
            return []

    def _load_scene_internal(self, scene_file, physics_client_id, client_name):
        """内部场景加载函数，统一处理不同仿真器"""
        try:
            scene_objects = p.loadMJCF(scene_file, physicsClientId=physics_client_id)
            if scene_objects:
                obstacle_ids = []

                print(f"Loaded {len(scene_objects)} objects from scene in {client_name}:")
                for i, obj_id in enumerate(scene_objects):
                    # 获取对象信息来判断类型
                    info = p.getBodyInfo(obj_id, physicsClientId=physics_client_id)
                    body_name = info[0].decode("utf-8") if info[0] else f"Object_{i}"

                    # 跳过地面对象（通常名称包含ground、floor、plane等）
                    if any(keyword in body_name.lower() for keyword in ["ground", "floor", "plane", "terrain"]):
                        print(f"  Object {i}: {body_name} (Ground - skipped)")
                        continue

                    # 其他对象视为障碍物
                    obstacle_ids.append(obj_id)
                    print(f"  Object {i}: {body_name} (Obstacle - ID: {obj_id})")

                # 将所有场景物体设置为静态（质量为0）
                for body_id in scene_objects:
                    p.changeDynamics(body_id, -1, mass=0, physicsClientId=physics_client_id)

                print(f"Final obstacle count in {client_name}: {len(obstacle_ids)} static obstacles")
                return obstacle_ids

        except Exception as e:
            print(f"{client_name}场景加载失败: {e}")
            return []

    def load_robot(self, robot_urdf):
        """加载机器人URDF"""
        try:
            self.robot_id = p.loadURDF(robot_urdf, useFixedBase=True, physicsClientId=self.physics_client)  # 固定基座
            self._setup_joint_info()
            return self.robot_id
        except Exception as e:
            print(f"Failed to load robot: {e}")
            return None

    def _setup_joint_info(self):
        """设置关节信息"""
        if self.robot_id is None:
            return

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        self.joint_limits = []
        self.valid_joints = []

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            if joint_info[2] != p.JOINT_FIXED:  # 非固定关节
                self.valid_joints.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                # 处理无限制关节
                if lower_limit == 0 and upper_limit == -1:
                    lower_limit, upper_limit = -math.pi, math.pi
                self.joint_limits.append((lower_limit, upper_limit))

    def sample_feasible_config(self, max_attempts=1000):
        """采样可行的关节配置"""
        for _ in range(max_attempts):
            # 在关节限制内随机采样
            joint_config = []
            for lower, upper in self.joint_limits:
                angle = random.uniform(lower, upper)
                joint_config.append(angle)

            # 设置机器人配置
            self.set_robot_config(joint_config)

            # 检查自碰撞
            if not self.check_self_collision():
                return joint_config

        # 如果采样失败，返回默认配置
        return [0.0] * len(self.joint_limits)

    def set_robot_config(self, joint_angles):
        """设置机器人关节配置"""
        if self.robot_id is None:
            return

        for i, angle in enumerate(joint_angles):
            if i < len(self.valid_joints):
                p.resetJointState(
                    self.robot_id,
                    self.valid_joints[i],
                    angle,
                    physicsClientId=self.physics_client,
                )

    def get_robot_config(self):
        """获取当前机器人关节配置"""
        if self.robot_id is None:
            return []

        joint_states = p.getJointStates(self.robot_id, self.valid_joints, physicsClientId=self.physics_client)
        return [state[0] for state in joint_states]

    def check_self_collision(self):
        """检查机器人自碰撞"""
        if self.robot_id is None:
            return False

        # 检查机器人内部连杆间的碰撞
        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.robot_id,
            physicsClientId=self.physics_client,
        )
        return len(contacts) > 0

    def check_link_collision(self, link_id, ignore_links=None):
        """检查特定连杆与障碍物的碰撞"""
        if self.robot_id is None or not self.obstacle_ids:
            return False

        # 检查与所有障碍物的碰撞
        for obstacle_id in self.obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=obstacle_id,
                linkIndexA=link_id,
                physicsClientId=self.physics_client,
            )
            if len(contacts) > 0:
                return True
        return False

    def check_robot_collision(self):
        """检查整个机器人与障碍物的碰撞"""
        if self.robot_id is None or not self.obstacle_ids:
            return False

        for obstacle_id in self.obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=obstacle_id,
                physicsClientId=self.physics_client,
            )
            if len(contacts) > 0:
                return True
        return False

    def get_num_links(self):
        """获取连杆数量"""
        if self.robot_id is None:
            return 0
        return (p.getNumJoints(self.robot_id, physicsClientId=self.physics_client) + 1)  # 包括base link

    def find_valid_collision_links(self):
        """找到有碰撞几何体的连杆"""
        if self.robot_id is None:
            return []

        valid_links = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)

        # 检查base link
        try:
            collision_data = p.getCollisionShapeData(self.robot_id, -1, physicsClientId=self.physics_client)
            if collision_data:
                valid_links.append(-1)
        except Exception:
            pass

        # 检查其他连杆
        for i in range(num_joints):
            try:
                collision_data = p.getCollisionShapeData(self.robot_id, i, physicsClientId=self.physics_client)
                if collision_data:
                    valid_links.append(i)
            except Exception:
                pass

        return valid_links

    def get_link_state(self, link_id):
        """获取连杆状态"""
        if self.robot_id is None:
            return None

        if link_id == -1:  # base link
            pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
            return pos, orn
        else:
            link_state = p.getLinkState(self.robot_id, link_id, physicsClientId=self.physics_client)
            return link_state[0], link_state[1]  # position, orientation

    def disconnect(self):
        """断开PyBullet连接"""
        p.disconnect(physicsClientId=self.physics_client)
        if self.sphere_physics_client is not None:
            p.disconnect(physicsClientId=self.sphere_physics_client)


# ========== 可视化管理类 ==========


class VisualizationManager:
    """管理可视化界面和交互的类"""
    def __init__(self, sim, robot_urdf_path):
        self.sim = sim
        self.robot_urdf_path = robot_urdf_path
        self.obb_bodies = []
        self.obb_fk = None
        self.obb_templates = None
        self.valid_collision_links = []
        self.show_obbs = True
        self.last_distance_check = time.time()

        # 球体可视化相关属性
        self.sphere_bodies = []
        self.sphere_analyzer = None
        self.show_spheres = True
        self.last_sphere_update = time.time()

        # 相机控制参数
        self.camera_distance = 2.0
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_target = [0, 0, 0.5]

        # 创建相机控制滑块
        self.distance_slider = p.addUserDebugParameter("Camera Distance", 0.5, 10.0, self.camera_distance)
        self.yaw_slider = p.addUserDebugParameter("Camera Yaw", -180, 180, self.camera_yaw)
        self.pitch_slider = p.addUserDebugParameter("Camera Pitch", -89, 89, self.camera_pitch)

        # 设置初始相机位置
        self.update_camera()

    def update_camera(self):
        """根据滑块值更新相机位置"""
        # 读取滑块值
        new_distance = p.readUserDebugParameter(self.distance_slider)
        new_yaw = p.readUserDebugParameter(self.yaw_slider)
        new_pitch = p.readUserDebugParameter(self.pitch_slider)

        # 只有值发生变化时才更新相机
        if (
            abs(new_distance - self.camera_distance) > 0.01 or abs(new_yaw - self.camera_yaw) > 0.5
            or abs(new_pitch - self.camera_pitch) > 0.5
        ):
            self.camera_distance = new_distance
            self.camera_yaw = new_yaw
            self.camera_pitch = new_pitch

            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=self.camera_target,
            )

    def initialize_obb_system(self, obb_templates, valid_collision_links):
        """初始化OBB系统"""
        self.obb_templates = obb_templates
        self.valid_collision_links = valid_collision_links
        if self.sim.robot_id:
            self.obb_fk = OBBForwardKinematics(self.sim.robot_id)  # 传入整数ID

    def initialize_sphere_system(self):
        """初始化球体可视化系统"""
        try:
            # 创建球体分析器
            self.sphere_analyzer = RobotSphereAnalyzer("franka", device="cuda:0")
            print("✓ 可视化球体分析器初始化成功")

            # 创建球体实体用于可视化
            self.sphere_bodies = create_sphere_bodies(self.sim, self.sphere_analyzer)
            if self.sphere_bodies:
                print(f"✓ 可视化球体创建成功: {len(self.sphere_bodies)} 个球体")
            else:
                print("⚠️ 可视化球体创建失败")
                self.show_spheres = False

        except Exception as e:
            print(f"❌ 球体可视化系统初始化失败: {e}")
            self.show_spheres = False

    def clear_obbs(self):
        """清除所有OBB可视化"""
        for body_id in self.obb_bodies:
            try:
                p.removeBody(body_id)
            except Exception:
                pass
        self.obb_bodies.clear()

    def draw_obbs(self):
        """绘制OBB"""
        if not self.show_obbs or not self.obb_templates or not self.obb_fk:
            return

        self.clear_obbs()

        try:
            # 计算当前配置下的OBB位姿
            obb_poses = self.obb_fk.compute_obb_poses(self.obb_templates)

            # OBB颜色
            obb_colors = [
                [1.0, 0.0, 0.0, 0.3],  # 红色半透明
                [0.0, 1.0, 0.0, 0.3],  # 绿色半透明
                [0.0, 0.0, 1.0, 0.3],  # 蓝色半透明
                [1.0, 1.0, 0.0, 0.3],  # 黄色半透明
                [1.0, 0.0, 1.0, 0.3],  # 品红半透明
                [0.0, 1.0, 1.0, 0.3],  # 青色半透明
            ]

            for i, obb_pose in enumerate(obb_poses):
                if i >= len(self.valid_collision_links):
                    continue

                color = obb_colors[i % len(obb_colors)]

                # 创建OBB可视化
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=obb_pose["extents"] / 2.0,
                    rgbaColor=color,
                )

                body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=obb_pose["position"],
                    baseOrientation=obb_pose["quaternion"],
                )

                self.obb_bodies.append(body_id)

        except Exception as e:
            print(f"OBB绘制失败: {e}")

    def update_spheres(self):
        """更新球体位置"""
        if not self.show_spheres or not self.sphere_analyzer or not self.sphere_bodies:
            return

        try:
            # 获取当前关节配置
            current_config = self.sim.get_robot_config()
            if not current_config:
                return

            # 将关节配置转换为张量
            joint_config = torch.tensor(current_config, dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0)

            # 获取当前配置下的球体世界坐标
            world_spheres = self.sphere_analyzer.get_world_spheres(joint_config)

            # 更新球体位置
            update_sphere_positions(self.sim, self.sphere_bodies, world_spheres)

            # 执行球体碰撞检测并输出结果
            collision_results = check_spheres_collision(self.sim, self.sphere_bodies)

            # 统计碰撞结果
            collision_count = sum(collision_results)
            free_count = len(collision_results) - collision_count
            collision_rate = (collision_count / len(collision_results) * 100 if collision_results else 0)

            print("\n=== 球体碰撞检测结果 ===")
            print(f"总球体数: {len(collision_results)}")
            print(f"碰撞球体: {collision_count} ({collision_rate:.1f}%)")
            print(f"无碰撞球体: {free_count} ({(100 - collision_rate):.1f}%)")

            print("球体的详细碰撞状态:")
            for i in range(len(collision_results)):
                x, y, z, radius = world_spheres[i]
                collision_status = "碰撞" if collision_results[i] else "无碰撞"
                print(f"  球体{i + 1}: 位置[{x:.3f}, {y:.3f}, {z:.3f}] 半径{radius:.3f} → {collision_status}")

        except Exception as e:
            print(f"球体位置更新失败: {e}")

    def cleanup_spheres(self):
        """清理球体可视化"""
        if self.sphere_bodies:
            cleanup_sphere_bodies(self.sim, self.sphere_bodies)
            self.sphere_bodies.clear()

    def calculate_link_distances(self):
        """计算各个link到各个障碍物的距离"""
        if not self.sim.robot_id or not self.sim.obstacle_ids:
            return {}

        distances = {}

        for link_id in self.valid_collision_links:
            link_name = f"Link_{link_id}" if link_id >= 0 else "Base"
            distances[link_name] = {}

            for i, obstacle_id in enumerate(self.sim.obstacle_ids):
                # 使用PyBullet的getClosestPoints计算最短距离
                closest_points = p.getClosestPoints(
                    bodyA=self.sim.robot_id,
                    bodyB=obstacle_id,
                    linkIndexA=link_id,
                    distance=10.0,  # 最大查询距离
                    physicsClientId=self.sim.physics_client,
                )

                if closest_points:
                    # 取最近的点
                    min_distance = min([point[8] for point in closest_points])  # contactDistance
                    distances[link_name][f"Obstacle_{i}"] = min_distance
                else:
                    distances[link_name][f"Obstacle_{i}"] = float("inf")

        return distances

    def print_distances(self):
        """打印当前距离信息和碰撞检测结果"""
        distances = self.calculate_link_distances()

        print("\n=== Link-Obstacle Distances & Collision Status ===")
        for link_name, obstacle_distances in distances.items():
            # 提取link_id
            if link_name == "Base":
                link_id = -1
            else:
                link_id = int(link_name.split("_")[1])

            # 检查该link的碰撞状态
            collision_status = self.sim.check_link_collision(link_id)
            collision_indicator = "🔴 COLLISION" if collision_status else "🟢 FREE"

            print(f"{link_name} [{collision_indicator}]:")
            for obstacle_name, distance in obstacle_distances.items():
                if distance == float("inf"):
                    print(f"  {obstacle_name}: No collision geometry")
                else:
                    # 根据距离添加状态指示
                    status_icon = ("💥" if distance <= 0.0 else "⚠️" if distance < 0.05 else "✅")
                    print(f"  {obstacle_name}: {distance:.4f}m {status_icon}")

        # 整体机器人碰撞状态
        overall_collision = self.sim.check_robot_collision()
        overall_status = ("🔴 ROBOT IN COLLISION" if overall_collision else "🟢 ROBOT FREE")
        print(f"\nOverall Status: {overall_status}")
        print("=" * 50 + "\n")

    def update_visualization(self):
        """更新可视化"""
        try:
            # 更新相机位置
            self.update_camera()

            # 更新球体位置（每帧更新）
            if self.show_spheres:
                current_time = time.time()
                # 每隔0.1秒更新一次球体位置以提高性能
                if current_time - self.last_sphere_update >= 1.0:
                    self.update_spheres()
                    self.last_sphere_update = current_time

            # 每隔1秒打印距离信息
            current_time = time.time()
            if current_time - self.last_distance_check >= 1.0:
                self.print_distances()
                self.last_distance_check = current_time

        except Exception as e:
            print(f"可视化更新失败: {e}")

    def run_visualization_loop(self):
        """运行可视化循环"""
        print("\n=== 可视化模式 ===")
        print("显示机器人、障碍物、OBB和球体")
        print("关闭窗口或按Ctrl+C退出...")

        # 初始化球体系统
        if self.show_spheres:
            self.initialize_sphere_system()

        # 初始绘制
        if self.show_obbs:
            self.draw_obbs()

        try:
            while True:
                self.update_visualization()
                p.stepSimulation(physicsClientId=self.sim.physics_client)
                time.sleep(1.0 / 60.0)  # 60 FPS

        except KeyboardInterrupt:
            print("用户中断可视化")
        except Exception as e:
            print(f"可视化循环错误: {e}")
        finally:
            self.clear_obbs()
            self.cleanup_spheres()


# ========== 几何变换辅助函数 ==========


def transform_point(p, R, T):
    """
    应用旋转和平移变换到给定点

    Args:
        p: 输入点坐标 (3D 向量)
        R: 3x3 旋转矩阵
        T: 3x1 平移向量
    Returns:
        newT: 变换后的点坐标
    """
    new = np.zeros((3, 1))
    new[:, 0] = p
    newT = np.array(T)
    temp = np.matmul(R, new)
    newT[0] += temp[0, 0]
    newT[1] += temp[1, 0]
    newT[2] += temp[2, 0]
    return newT


def check_sphere_collision(sim, sphere_position, sphere_radius):
    """
    检查单个球体与环境障碍物的碰撞

    Args:
        sim: PyBullet仿真器实例
        sphere_position: 球体位置 [x, y, z]
        sphere_radius: 球体半径

    Returns:
        bool: True表示碰撞，False表示无碰撞
    """
    if not sim.obstacle_ids:
        return False

    try:
        # 在PyBullet中创建临时球体用于碰撞检测
        sphere_collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=sphere_radius, physicsClientId=sim.physics_client
        )

        sphere_body = p.createMultiBody(
            baseMass=0,  # 静态球体
            baseCollisionShapeIndex=sphere_collision_shape,
            basePosition=sphere_position,
            physicsClientId=sim.physics_client,
        )

        # 执行碰撞检测
        p.performCollisionDetection(physicsClientId=sim.physics_client)

        # 检查球体与所有障碍物的碰撞
        has_collision = False
        for obstacle_id in sim.obstacle_ids:
            contacts = p.getContactPoints(bodyA=sphere_body, bodyB=obstacle_id, physicsClientId=sim.physics_client)
            if len(contacts) > 0:
                has_collision = True
                break

        # 清理临时创建的球体
        p.removeBody(sphere_body, physicsClientId=sim.physics_client)

        return has_collision

    except Exception as e:
        print(f"球体碰撞检测失败: {e}")
        return False


def create_sphere_bodies(sim, sphere_analyzer):
    """
    创建用于碰撞检测的球体实体，使用默认关节配置下的实际球体数据

    Args:
        sim: PyBullet仿真器实例
        sphere_analyzer: 球体分析器实例

    Returns:
        list: 球体body ID列表
    """
    sphere_bodies = []

    # 确保球体仿真器已初始化
    if sim.sphere_physics_client is None:
        print("❌ 球体仿真器未初始化")
        return []

    try:
        # 获取默认关节配置下的球体数据
        world_spheres = sphere_analyzer.get_world_spheres()

        for i, (x, y, z, radius) in enumerate(world_spheres):
            # 在球体仿真器中创建球体碰撞形状
            sphere_collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=float(radius),
                physicsClientId=sim.sphere_physics_client,
            )

            # 创建球体可视化形状，使用半透明绿色
            sphere_visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=float(radius),
                rgbaColor=[0.0, 1.0, 0.0, 0.3],  # 半透明绿色
                physicsClientId=sim.sphere_physics_client,
            )

            # 在球体仿真器中创建球体body
            sphere_body = p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=sphere_collision_shape,
                baseVisualShapeIndex=sphere_visual_shape,
                basePosition=[float(x), float(y), float(z)],
                physicsClientId=sim.sphere_physics_client,
            )

            sphere_bodies.append(sphere_body)

        print(f"✓ 成功在球体仿真器中创建 {len(sphere_bodies)} 个球体用于碰撞检测")
        return sphere_bodies

    except Exception as e:
        print(f"❌ 球体创建失败: {e}")
        return []


def update_sphere_positions(sim, sphere_bodies, world_spheres):
    """
    更新球体位置和半径

    Args:
        sim: PyBullet仿真器实例
        sphere_bodies: 球体body ID列表
        world_spheres: 世界坐标下的球体信息 [N, 4] (x, y, z, radius)
    """
    try:
        for i, (sphere_body, (x, y, z, radius)) in enumerate(zip(sphere_bodies, world_spheres)):
            if i >= len(sphere_bodies):
                break

            # 在球体仿真器中更新球体位置
            p.resetBasePositionAndOrientation(
                sphere_body,
                [float(x), float(y), float(z)],
                [0, 0, 0, 1],  # 无旋转
                physicsClientId=sim.sphere_physics_client,
            )

            # 注意：PyBullet中动态更改半径比较复杂，这里暂时保持固定半径
            # 如果需要动态半径，需要重新创建collision shape

    except Exception as e:
        print(f"球体位置更新失败: {e}")


def check_spheres_collision(sim, sphere_bodies):
    """
    检查所有球体与障碍物的碰撞

    Args:
        sim: PyBullet仿真器实例
        sphere_bodies: 球体body ID列表

    Returns:
        list: 每个球体的碰撞结果列表 (True=碰撞, False=无碰撞)
    """
    collision_results = []

    if not sim.sphere_obstacle_ids:
        return [False] * len(sphere_bodies)

    try:
        # 在球体仿真器中执行碰撞检测
        p.performCollisionDetection(physicsClientId=sim.sphere_physics_client)

        for sphere_body in sphere_bodies:
            has_collision = False

            # 检查当前球体与所有障碍物的碰撞
            for obstacle_id in sim.sphere_obstacle_ids:
                contacts = p.getContactPoints(
                    bodyA=sphere_body,
                    bodyB=obstacle_id,
                    physicsClientId=sim.sphere_physics_client,
                )
                if len(contacts) > 0:
                    has_collision = True
                    break

            collision_results.append(has_collision)

        return collision_results

    except Exception as e:
        print(f"球体碰撞检测失败: {e}")
        return [False] * len(sphere_bodies)


def cleanup_sphere_bodies(sim, sphere_bodies):
    """
    清理球体实体

    Args:
        sim: PyBullet仿真器实例
        sphere_bodies: 球体body ID列表
    """
    try:
        for sphere_body in sphere_bodies:
            p.removeBody(sphere_body, physicsClientId=sim.sphere_physics_client)
        print(f"✓ 成功清理 {len(sphere_bodies)} 个球体")
    except Exception as e:
        print(f"球体清理失败: {e}")


# ========== 核心 OBB 计算函数 ==========


def calculate_direction_encoding(extents, rotation_matrix, center):
    """
    计算方向编码字符串

    Args:
        extents: OBB 尺寸 [x, y, z]
        rotation_matrix: 3x3 旋转矩阵
        center: OBB 中心位置

    Returns:
        dirstring: 2位方向编码字符串
    """
    try:
        # 使用 OBB 尺寸作为参考点
        newpoint = transform_point(extents, rotation_matrix, center)
        direction = np.sign(newpoint - center)

        if direction[0] < 0:
            direction = direction * -1  # 标准化 x 方向

        direction = (direction + 1) / 2  # 归一化到 [0, 1]
        dirstring = str(int(direction[1])) + str(int(direction[2]))  # y,z 编码

        return dirstring

    except Exception as e:
        print(f"    Warning: Direction encoding failed: {e}")
        return "00"  # 默认编码


def initialize_obb_templates(robot_urdf_path):
    """
    初始化 OBB 模板计算器

    Args:
        robot_urdf_path: 机器人 URDF 文件路径
        num_links: 链接数量

    Returns:
        obb_data: OBB 数据列表，用于 OBBForwardKinematics
    """
    try:
        # 使用 obb_calculator 生成初始 OBB 数据
        obb_data = obb_calculator.calculate_link_obbs(robot_urdf_path, verbose=False)
        print(f"    OBB templates initialized for {len(obb_data)} links")
        return obb_data
    except Exception as e:
        print(f"    Error: Failed to initialize OBB templates: {e}")
        return None


def parse_command_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(description="机器人碰撞检测数据生成脚本")
    parser.add_argument("numqueries", type=int, help="采样姿态数量")
    parser.add_argument("foldername", help="环境文件夹")
    parser.add_argument("filenumber", help="环境文件编号")
    parser.add_argument("--visualize", action="store_true", help="启用可视化模式")

    args = parser.parse_args()
    return args.numqueries, args.foldername, args.filenumber, args.visualize


def initialize_environment(foldername, filenumber, use_gui=False):
    """初始化环境和机器人模型"""
    robot_urdf_path = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf"

    # 创建PyBullet仿真器
    sim = PyBulletRobotSimulator(use_gui=use_gui)

    # 加载包含障碍物的环境
    scene_file = foldername + "/obstacles_" + filenumber + ".xml"
    print(scene_file)
    scene_objects = sim.load_scene(scene_file)  # 加载场景
    if not scene_objects:
        print("Warning: No objects loaded from scene")

    # 加载机器人模型
    robot_id = sim.load_robot(robot_urdf_path)

    if robot_id is None:
        raise RuntimeError("Failed to load robot")

    return sim, robot_urdf_path


def find_valid_collision_links(sim):
    """筛选有实际几何体的连杆"""
    valid_collision_links = sim.find_valid_collision_links()

    print(f"Found {len(valid_collision_links)} real collision links")
    print(f"Real collision links: {valid_collision_links}")

    # 输出连杆名称
    link_names = []
    for link_id in valid_collision_links:
        if link_id == -1:
            link_names.append("base_link")
        else:
            joint_info = p.getJointInfo(sim.robot_id, link_id, physicsClientId=sim.physics_client)
            link_name = joint_info[12].decode('utf-8')  # linkName is at index 12
            link_names.append(link_name)
    print(f"Link names: {link_names}")

    return valid_collision_links


# def setup_collision_detectors(sim, valid_collision_links):
#     """为每个实体link创建独立的碰撞检测器 (PyBullet版本中不需要单独创建)"""
#     # PyBullet中我们直接使用sim的方法进行碰撞检测
#     return sim  # 返回仿真器本身


def get_sphere_count():
    """获取机器人球体数量"""

    # 创建球体分析器（使用franka机器人，对应panda）
    analyzer = RobotSphereAnalyzer("franka")

    # 获取默认配置下的世界坐标球体
    world_spheres = analyzer.get_world_spheres()
    sphere_count = len(world_spheres)

    # print(f"✓ 通过球体分析器检测到机器人有 {sphere_count} 个球体")
    return sphere_count


def initialize_data_arrays(numqueries, num_real_links, num_dofs, sphere_count_per_query=0):
    """初始化数据存储数组"""
    # link级数据数组
    qarr = np.zeros((num_real_links * numqueries, 3))
    dirarr = []
    yarr = np.zeros((num_real_links * numqueries, 1))

    # pose级数据数组
    qarr_pose = np.zeros((numqueries, num_dofs))
    yarr_pose = np.zeros((numqueries, 1))

    # 球体数据数组
    total_spheres = sphere_count_per_query * numqueries
    qarr_sphere = np.zeros((total_spheres, 3))  # 球体位置
    rarr_sphere = np.zeros((total_spheres, 1))  # 球体半径
    yarr_sphere = np.zeros((total_spheres, 1))  # 球体对应link的碰撞结果

    return (
        qarr,
        dirarr,
        yarr,
        qarr_pose,
        yarr_pose,
        qarr_sphere,
        rarr_sphere,
        yarr_sphere,
    )


def sample_and_generate_data(
    sim,
    valid_collision_links,
    obb_templates,
    numqueries,
    qarr,
    dirarr,
    yarr,
    qarr_pose,
    yarr_pose,
    qarr_sphere=None,
    rarr_sphere=None,
    yarr_sphere=None,
    sphere_count_per_query=0,
):
    """主要的数据生成循环"""
    counter = 0
    coll_count = 0
    sphere_counter = 0  # 球体数据计数器
    sphere_collision_count = 0  # 球体碰撞总次数
    sphere_free_count = 0  # 球体无碰撞总次数
    num_real_links = len(valid_collision_links)

    # 初始化OBB正向运动学计算器
    obb_fk = OBBForwardKinematics(sim.robot_id)

    # 初始化球体分析器（如果需要）
    sphere_analyzer = None
    sphere_bodies = []  # 球体body ID列表
    if qarr_sphere is not None:
        sphere_analyzer = RobotSphereAnalyzer("franka", device="cuda:0")
        print("✓ 球体分析器初始化成功")

        # 创建用于碰撞检测的球体实体，使用默认配置下的球体数据
        sphere_bodies = create_sphere_bodies(sim, sphere_analyzer)
        if not sphere_bodies:
            print("⚠️ 球体创建失败，将跳过球体碰撞检测")
            qarr_sphere = None

    while counter < numqueries:
        # 采样可行的机器人配置
        q = sim.sample_feasible_config()
        sim.set_robot_config(q)
        # 获取当前关节配置
        current_config = sim.get_robot_config()
        p.performCollisionDetection(physicsClientId=sim.physics_client)
        # 使用OBB正向运动学直接计算当前配置下的OBB位姿
        if obb_templates is not None:
            obb_poses = obb_fk.compute_obb_poses(obb_templates)
        else:
            # 如果模板初始化失败，跳过OBB计算，使用默认值
            print(f"    Warning: OBB templates not available, using defaults for iteration {counter}")
            obb_poses = []

        # 准备关节配置张量（如果需要球体分析）
        joint_config = None
        if sphere_analyzer is not None:
            # 将当前关节配置转换为张量
            joint_config = torch.tensor(current_config, dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0)

        # 逐link碰撞检测
        real_link_idx = 0
        for lid in valid_collision_links:
            # 使用PyBullet检测碰撞
            collision = sim.check_link_collision(lid)
            ans = 0 if collision else 1

            # 存储当前实体link的数据
            if obb_poses and lid < len(obb_poses):
                # 使用计算出的OBB位姿
                qarr[counter * num_real_links + real_link_idx] = obb_poses[lid]["position"]
                # 计算方向编码 (保持与原有格式兼容)
                dirstring = calculate_direction_encoding(
                    obb_poses[lid]["extents"],
                    obb_poses[lid]["transform"][:3, :3],
                    obb_poses[lid]["position"],
                )
            else:
                # 使用默认值或链接位置
                link_state = sim.get_link_state(lid)
                if link_state:
                    qarr[counter * num_real_links + real_link_idx] = link_state[0]
                else:
                    qarr[counter * num_real_links + real_link_idx] = [0.0, 0.0, 0.0]
                dirstring = "00"  # 默认方向编码

            yarr[counter * num_real_links + real_link_idx] = ans
            dirarr.append(dirstring)
            real_link_idx += 1

        # 逐个球体碰撞检测
        if (
            sphere_analyzer is not None and qarr_sphere is not None and rarr_sphere is not None
            and yarr_sphere is not None and sphere_bodies
        ):
            try:
                # 获取当前配置下的球体世界坐标
                world_spheres = sphere_analyzer.get_world_spheres(joint_config)

                # 更新球体位置
                update_sphere_positions(sim, sphere_bodies, world_spheres)

                # 执行批量球体碰撞检测
                collision_results = check_spheres_collision(sim, sphere_bodies)

                # 遍历每个球体，保存数据
                for sphere_idx, ((x, y, z, radius),
                                 sphere_collision) in enumerate(zip(world_spheres, collision_results)):
                    if sphere_counter < len(qarr_sphere):
                        # 保存球体位置和半径
                        qarr_sphere[sphere_counter] = [x, y, z]  # 球体位置
                        rarr_sphere[sphere_counter] = [radius]  # 球体半径

                        # 保存球体碰撞结果 (0=碰撞, 1=无碰撞)
                        collision_result = 0 if sphere_collision else 1
                        yarr_sphere[sphere_counter] = [collision_result]

                        # 更新球体碰撞统计
                        if sphere_collision:
                            sphere_collision_count += 1
                        else:
                            sphere_free_count += 1

                        sphere_counter += 1

                # # 每100个查询打印一次球体碰撞统计
                # if counter % 100 == 0 and sphere_counter > 0:
                #     collision_rate = sphere_collision_count / sphere_counter * 100
                #     print(
                #         f"    已处理 {counter} 个查询，球体数据: {sphere_counter}/{len(qarr_sphere)}"
                #     )
                #     print(
                #         f"    球体碰撞统计: 碰撞{sphere_collision_count}, 无碰撞{sphere_free_count}, 碰撞率{collision_rate:.1f}%"
                #     )

            except Exception as e:
                if counter % 100 == 0:
                    print(f"    球体数据计算失败 (查询 {counter}): {e}")

        # 整体机器人碰撞检测
        overall_collision = sim.check_robot_collision()
        ans = 0 if overall_collision else 1
        if overall_collision:
            coll_count += 1

        # 存储姿态级数据
        current_config = sim.get_robot_config()
        # 确保配置长度匹配
        if len(current_config) == qarr_pose.shape[1]:
            qarr_pose[counter] = current_config
        else:
            # 如果长度不匹配，填充或截断
            config_padded = (current_config + [0.0] * qarr_pose.shape[1])[:qarr_pose.shape[1]]
            qarr_pose[counter] = config_padded

        yarr_pose[counter] = ans
        counter += 1

    # 打印最终球体碰撞统计
    if sphere_counter > 0:
        final_collision_rate = sphere_collision_count / sphere_counter * 100
        print("\n=== 球体碰撞检测最终统计 ===")
        print(f"总球体数: {sphere_counter}")
        print(f"碰撞球体: {sphere_collision_count} ({sphere_collision_count / sphere_counter * 100:.1f}%)")
        print(f"无碰撞球体: {sphere_free_count} ({sphere_free_count / sphere_counter * 100:.1f}%)")
        print(f"碰撞率: {final_collision_rate:.1f}%")

    # 清理球体实体
    if sphere_bodies:
        cleanup_sphere_bodies(sim, sphere_bodies)

    return coll_count


def save_results(
    foldername,
    filenumber,
    qarr,
    dirarr,
    yarr,
    qarr_pose,
    yarr_pose,
    qarr_sphere=None,
    rarr_sphere=None,
    yarr_sphere=None,
):
    """保存结果到文件"""
    import os

    # 创建新的输出文件夹
    output_folder = foldername + "_rs"
    os.makedirs(output_folder, exist_ok=True)

    # 保存link级数据
    with open(output_folder + "/obstacles_" + filenumber + "_coord.pkl", "wb") as f:
        pickle.dump((qarr, dirarr, yarr), f)

    # 保存pose级数据
    with open(output_folder + "/obstacles_" + filenumber + "_pose.pkl", "wb") as f:
        pickle.dump((qarr_pose, yarr_pose), f)

    # 保存球体数据（如果可用）
    with open(output_folder + "/obstacles_" + filenumber + "_sphere.pkl", "wb") as f:
        pickle.dump((qarr_sphere, rarr_sphere, yarr_sphere), f)
    print(f"Results saved to {output_folder}/")


def main():
    """主程序：数据生成流程"""
    # 解析命令行参数
    numqueries, foldername, filenumber, visualize_mode = parse_command_args()

    # 环境和机器人初始化
    sim, robot_urdf_path = initialize_environment(foldername, filenumber, use_gui=visualize_mode)

    # 预先筛选实体link
    valid_collision_links = find_valid_collision_links(sim)
    num_real_links = len(valid_collision_links)

    # 设置随机种子确保可重现性
    random.seed(2)

    # 一次性初始化OBB模板
    print("Initializing OBB templates...")
    obb_templates = initialize_obb_templates(robot_urdf_path)
    print(f"OBB templates initialization {'succeeded' if obb_templates else 'failed, will use fallback method'}")

    # 获取机器人结构信息
    num_links = sim.get_num_links()
    num_dofs = len(sim.joint_limits)
    print(f"Robot has {num_links} total links ({num_real_links} real) and {num_dofs} DOFs")

    # 获取球体数量
    sphere_count_per_query = get_sphere_count()

    # 如果是可视化模式，运行可视化
    if visualize_mode:
        print("\n启动可视化模式...")
        vis_manager = VisualizationManager(sim, robot_urdf_path)
        vis_manager.initialize_obb_system(obb_templates, valid_collision_links)

        # 设置初始配置
        q = sim.sample_feasible_config()
        sim.set_robot_config(q)

        try:
            vis_manager.run_visualization_loop()
        finally:
            sim.disconnect()
        return

    # 数据生成模式
    print("\n启动数据生成模式...")

    # 数据存储数组初始化
    qarr, dirarr, yarr, qarr_pose, yarr_pose, qarr_sphere, rarr_sphere, yarr_sphere = (
        initialize_data_arrays(numqueries, num_real_links, num_dofs, sphere_count_per_query)
    )

    # 初始化采样
    q = sim.sample_feasible_config()
    sim.set_robot_config(q)

    # 主要数据生成循环
    coll_count = sample_and_generate_data(
        sim,
        valid_collision_links,
        obb_templates,
        numqueries,
        qarr,
        dirarr,
        yarr,
        qarr_pose,
        yarr_pose,
        qarr_sphere,
        rarr_sphere,
        yarr_sphere,
        sphere_count_per_query,
    )

    # 结果输出和数据保存
    print("collision count", coll_count, "out of ", numqueries)
    save_results(
        foldername,
        filenumber,
        qarr,
        dirarr,
        yarr,
        qarr_pose,
        yarr_pose,
        qarr_sphere,
        rarr_sphere,
        yarr_sphere,
    )

    # 清理资源
    sim.disconnect()


if __name__ == "__main__":
    main()
