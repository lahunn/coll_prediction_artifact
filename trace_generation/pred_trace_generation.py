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

    def load_scene(self, scene_file):
        """加载MuJoCo格式的场景文件"""
        try:
            scene_objects = p.loadMJCF(scene_file)
            if scene_objects:
                # 分析场景对象并正确识别障碍物
                self.obstacle_ids = []

                print(f"Loaded {len(scene_objects)} objects from scene:")
                for i, obj_id in enumerate(scene_objects):
                    # 获取对象信息来判断类型
                    info = p.getBodyInfo(obj_id)
                    body_name = info[0].decode("utf-8") if info[0] else f"Object_{i}"

                    # 跳过地面对象（通常名称包含ground、floor、plane等）
                    if any(
                        keyword in body_name.lower()
                        for keyword in ["ground", "floor", "plane", "terrain"]
                    ):
                        print(f"  Object {i}: {body_name} (Ground - skipped)")
                        continue

                    # 其他对象视为障碍物
                    self.obstacle_ids.append(obj_id)
                    print(f"  Object {i}: {body_name} (Obstacle - ID: {obj_id})")

                # 将所有场景物体设置为静态（质量为0）并禁用碰撞响应
                for body_id in scene_objects:
                    p.changeDynamics(body_id, -1, mass=0)
                    # 原始代码中下面这行禁用了碰撞组，导致 getContactPoints 无法检测到碰撞。
                    # getClosestPoints 不受此影响，因此可以正确报告距离。
                    # p.setCollisionFilterGroupMask(body_id, -1, 0, 0)
                    # 通过注释掉此行，我们使用PyBullet的默认碰撞设置 (group=1, mask=1)，
                    # 从而使 getContactPoints 能够正常工作。

                print(
                    f"Final obstacle count: {len(self.obstacle_ids)} static obstacles"
                )
            return scene_objects
        except Exception as e:
            print(f"Failed to load scene: {e}")
            return []

    def load_robot(self, robot_urdf):
        """加载机器人URDF"""
        try:
            self.robot_id = p.loadURDF(robot_urdf, useFixedBase=True)  # 固定基座
            self._setup_joint_info()
            return self.robot_id
        except Exception as e:
            print(f"Failed to load robot: {e}")
            return None

    def _setup_joint_info(self):
        """设置关节信息"""
        if self.robot_id is None:
            return

        num_joints = p.getNumJoints(self.robot_id)
        self.joint_limits = []
        self.valid_joints = []

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
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
                p.resetJointState(self.robot_id, self.valid_joints[i], angle)

    def get_robot_config(self):
        """获取当前机器人关节配置"""
        if self.robot_id is None:
            return []

        joint_states = p.getJointStates(self.robot_id, self.valid_joints)
        return [state[0] for state in joint_states]

    def check_self_collision(self):
        """检查机器人自碰撞"""
        if self.robot_id is None:
            return False

        # 检查机器人内部连杆间的碰撞
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        return len(contacts) > 0

    def check_link_collision(self, link_id, ignore_links=None):
        """检查特定连杆与障碍物的碰撞"""
        if self.robot_id is None or not self.obstacle_ids:
            return False

        # 检查与所有障碍物的碰撞
        for obstacle_id in self.obstacle_ids:
            contacts = p.getContactPoints(
                bodyA=self.robot_id, bodyB=obstacle_id, linkIndexA=link_id
            )
            if len(contacts) > 0:
                return True
        return False

    def check_robot_collision(self):
        """检查整个机器人与障碍物的碰撞"""
        if self.robot_id is None or not self.obstacle_ids:
            return False

        for obstacle_id in self.obstacle_ids:
            contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obstacle_id)
            if len(contacts) > 0:
                return True
        return False

    def get_num_links(self):
        """获取连杆数量"""
        if self.robot_id is None:
            return 0
        return p.getNumJoints(self.robot_id) + 1  # 包括base link

    def find_valid_collision_links(self):
        """找到有碰撞几何体的连杆"""
        if self.robot_id is None:
            return []

        valid_links = []
        num_joints = p.getNumJoints(self.robot_id)

        # 检查base link
        try:
            collision_data = p.getCollisionShapeData(self.robot_id, -1)
            if collision_data:
                valid_links.append(-1)
        except Exception:
            pass

        # 检查其他连杆
        for i in range(num_joints):
            try:
                collision_data = p.getCollisionShapeData(self.robot_id, i)
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
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            return pos, orn
        else:
            link_state = p.getLinkState(self.robot_id, link_id)
            return link_state[0], link_state[1]  # position, orientation

    def disconnect(self):
        """断开PyBullet连接"""
        p.disconnect()


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

        # 相机控制参数
        self.camera_distance = 2.0
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_target = [0, 0, 0.5]

        # 创建相机控制滑块
        self.distance_slider = p.addUserDebugParameter(
            "Camera Distance", 0.5, 10.0, self.camera_distance
        )
        self.yaw_slider = p.addUserDebugParameter(
            "Camera Yaw", -180, 180, self.camera_yaw
        )
        self.pitch_slider = p.addUserDebugParameter(
            "Camera Pitch", -89, 89, self.camera_pitch
        )

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
            abs(new_distance - self.camera_distance) > 0.01
            or abs(new_yaw - self.camera_yaw) > 0.5
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
                )

                if closest_points:
                    # 取最近的点
                    min_distance = min(
                        [point[8] for point in closest_points]
                    )  # contactDistance
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
                    status_icon = (
                        "💥" if distance <= 0.0 else "⚠️" if distance < 0.05 else "✅"
                    )
                    print(f"  {obstacle_name}: {distance:.4f}m {status_icon}")

        # 整体机器人碰撞状态
        overall_collision = self.sim.check_robot_collision()
        overall_status = (
            "🔴 ROBOT IN COLLISION" if overall_collision else "🟢 ROBOT FREE"
        )
        print(f"\nOverall Status: {overall_status}")
        print("=" * 50 + "\n")

    def update_visualization(self):
        """更新可视化"""
        try:
            # 更新相机位置
            self.update_camera()

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
        print("显示机器人、障碍物和OBB")
        print("关闭窗口或按Ctrl+C退出...")

        # 初始绘制
        if self.show_obbs:
            self.draw_obbs()

        try:
            while True:
                self.update_visualization()
                p.stepSimulation()
                time.sleep(1.0 / 60.0)  # 60 FPS

        except KeyboardInterrupt:
            print("用户中断可视化")
        except Exception as e:
            print(f"可视化循环错误: {e}")
        finally:
            self.clear_obbs()


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
    robot_urdf_path = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/panda/panda.urdf"

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
    return valid_collision_links


# def setup_collision_detectors(sim, valid_collision_links):
#     """为每个实体link创建独立的碰撞检测器 (PyBullet版本中不需要单独创建)"""
#     # PyBullet中我们直接使用sim的方法进行碰撞检测
#     return sim  # 返回仿真器本身


def initialize_data_arrays(numqueries, num_real_links, num_dofs):
    """初始化数据存储数组"""
    # link级数据数组
    qarr = np.zeros((num_real_links * numqueries, 3))
    dirarr = []
    yarr = np.zeros((num_real_links * numqueries, 1))

    # pose级数据数组
    qarr_pose = np.zeros((numqueries, num_dofs))
    yarr_pose = np.zeros((numqueries, 1))

    return qarr, dirarr, yarr, qarr_pose, yarr_pose


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
):
    """主要的数据生成循环"""
    counter = 0
    coll_count = 0
    num_real_links = len(valid_collision_links)

    # 初始化OBB正向运动学计算器
    obb_fk = OBBForwardKinematics(sim.robot_id)

    while counter < numqueries:
        # 采样可行的机器人配置
        q = sim.sample_feasible_config()
        sim.set_robot_config(q)
        p.performCollisionDetection()
        # 使用OBB正向运动学直接计算当前配置下的OBB位姿
        if obb_templates is not None:
            obb_poses = obb_fk.compute_obb_poses(obb_templates)
        else:
            # 如果模板初始化失败，跳过OBB计算，使用默认值
            print(
                f"    Warning: OBB templates not available, using defaults for iteration {counter}"
            )
            obb_poses = []

        # 逐link碰撞检测
        real_link_idx = 0
        for lid in valid_collision_links:
            # 使用PyBullet检测碰撞
            collision = sim.check_link_collision(lid)
            ans = 0 if collision else 1

            # 存储当前实体link的数据
            if obb_poses and lid < len(obb_poses):
                # 使用计算出的OBB位姿
                qarr[counter * num_real_links + real_link_idx] = obb_poses[lid][
                    "position"
                ]
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
            config_padded = (current_config + [0.0] * qarr_pose.shape[1])[
                : qarr_pose.shape[1]
            ]
            qarr_pose[counter] = config_padded

        yarr_pose[counter] = ans
        counter += 1

    return coll_count


def save_results(foldername, filenumber, qarr, dirarr, yarr, qarr_pose, yarr_pose):
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

    print(f"Results saved to {output_folder}/")


def main():
    """主程序：数据生成流程"""
    # 解析命令行参数
    numqueries, foldername, filenumber, visualize_mode = parse_command_args()

    # 环境和机器人初始化
    sim, robot_urdf_path = initialize_environment(
        foldername, filenumber, use_gui=visualize_mode
    )

    # 预先筛选实体link
    valid_collision_links = find_valid_collision_links(sim)
    num_real_links = len(valid_collision_links)

    # 设置随机种子确保可重现性
    random.seed(2)

    # 一次性初始化OBB模板
    print("Initializing OBB templates...")
    obb_templates = initialize_obb_templates(robot_urdf_path)
    print(
        f"OBB templates initialization {'succeeded' if obb_templates else 'failed, will use fallback method'}"
    )

    # 获取机器人结构信息
    num_links = sim.get_num_links()
    num_dofs = len(sim.joint_limits)
    print(
        f"Robot has {num_links} total links ({num_real_links} real) and {num_dofs} DOFs"
    )

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
    qarr, dirarr, yarr, qarr_pose, yarr_pose = initialize_data_arrays(
        numqueries, num_real_links, num_dofs
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
    )

    # 结果输出和数据保存
    print("collision count", coll_count, "out of ", numqueries)
    save_results(foldername, filenumber, qarr, dirarr, yarr, qarr_pose, yarr_pose)

    # 清理资源
    sim.disconnect()


if __name__ == "__main__":
    main()
