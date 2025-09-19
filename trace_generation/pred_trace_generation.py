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
import sys
import pickle
import numpy as np
import math
import obb_calculator
import random

# ========== PyBullet机器人仿真类 ==========


class PyBulletRobotSimulator:
    """PyBullet机器人仿真器，替换Klampt功能"""

    def __init__(self):
        self.physics_client = p.connect(p.DIRECT)  # 无GUI模式
        p.setGravity(0, 0, -9.81)
        self.robot_id = None
        self.obstacle_ids = []
        self.joint_limits = []
        self.valid_joints = []

    def load_scene(self, scene_file):
        """加载MuJoCo格式的场景文件"""
        try:
            scene_objects = p.loadMJCF(scene_file)
            if scene_objects:
                # 假设第一个是地面，其余是障碍物
                self.obstacle_ids = scene_objects[2:] if len(scene_objects) > 2 else []
                print(f"Loaded scene with {len(self.obstacle_ids)} obstacles")
            return scene_objects
        except Exception as e:
            print(f"Failed to load scene: {e}")
            return []

    def load_robot(self, robot_urdf):
        """加载机器人URDF"""
        try:
            self.robot_id = p.loadURDF(robot_urdf)
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


# ========== 几何变换辅助函数 ==========


def give_RT(x):
    """
    从 4x4 变换矩阵中提取旋转矩阵 R 和平移向量 T

    Args:
        x: 4x4 齐次变换矩阵
    Returns:
        R: 展平的 3x3 旋转矩阵 (9个元素的列表)
        T: 3x1 平移向量 (3个元素的列表)
    """
    R = []
    T = []
    for j in range(0, 3):
        for i in range(0, 3):
            R.append(x[i][j])  # 按列优先顺序展平旋转矩阵
        T.append(x[j][3])  # 提取平移分量
    return (R, T)


def get_obbRT(R, T):
    """
    将展平的旋转矩阵和平移向量重构为标准的 numpy 数组格式

    Args:
        R: 展平的旋转矩阵 (9个元素)
        T: 平移向量 (3个元素)
    Returns:
        newR: 3x3 numpy 旋转矩阵
        newT: 3x1 numpy 平移向量
    """
    newR = np.eye(3)
    newT = np.array(T)
    pointer = 0
    for j in range(0, 3):
        for i in range(0, 3):
            newR[i][j] = R[pointer]  # 重构 3x3 旋转矩阵
            pointer += 1
    return newR, newT


def give_dh(d, r, th, al):
    """
    根据 DH (Denavit-Hartenberg) 参数构建齐次变换矩阵
    用于机器人运动学正向计算

    Args:
        d: 连杆偏移 (link offset)
        r: 连杆长度 (link length)
        th: 关节角 (joint angle)
        al: 连杆扭转角 (link twist)
    Returns:
        new: 4x4 DH 变换矩阵
    """
    new = np.eye(4)
    # DH 变换矩阵的标准公式
    new[0, 0] = math.cos(th)
    new[0, 1] = -1 * math.cos(al) * math.sin(th)
    new[0, 2] = math.sin(al) * math.sin(th)
    new[0, 3] = r * math.cos(th)

    new[1, 0] = math.sin(th)
    new[1, 1] = 1 * math.cos(al) * math.cos(th)
    new[1, 2] = -1 * math.sin(al) * math.cos(th)
    new[1, 3] = r * math.sin(th)

    new[2, 0] = 0
    new[2, 1] = math.sin(al)
    new[2, 2] = math.cos(al)
    new[2, 3] = d
    return new


def get_RT(x):
    """
    从旋转矩阵和平移向量重构 4x4 齐次变换矩阵

    Args:
        x: 包含 [R, T] 的元组，R为展平旋转矩阵，T为平移向量
    Returns:
        new: 4x4 齐次变换矩阵
    """
    R = x[0]
    T = x[1]
    new = np.eye(4)
    pointer = 0
    for j in range(0, 3):
        for i in range(0, 3):
            new[i][j] = R[pointer]
            pointer += 1
        new[j][3] = T[j]
    return new


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


def inverse(R):
    """
    从旋转矩阵提取欧拉角 (暂未在主程序中使用)
    """
    x = math.atan(R[1][2] / R[0][2])
    y = math.atan(math.sqrt(R[0][2] ** 2 + R[1][2] ** 2) / R[2][2])
    z = math.atan(-1 * R[2][1] / R[2][0])
    return [(x), (y), (z)]


# ========== 核心 OBB 计算函数 ==========


def initialize_obb_templates(robot_urdf_path, num_links):
    """
    一次性计算机器人所有link的OBB模板信息
    这些信息只依赖于几何结构，不依赖于关节配置

    Returns:
        obb_templates: list of OBB template info, or None if failed
    """
    # 1. 尝试使用精确的 OBB 计算
    if robot_urdf_path and obb_calculator.check_dependencies()[0]:
        try:
            print(f"Initializing precise OBB templates with {robot_urdf_path}")
            link_obbs = obb_calculator.calculate_link_obbs(
                robot_urdf_path, verbose=False
            )

            if link_obbs and len(link_obbs) == num_links:
                print(f"Successfully computed {len(link_obbs)} OBB templates")
                return link_obbs

        except Exception as e:
            print(f"Warning: Precise OBB template calculation failed: {e}")

    # 2. 如果精确计算失败，返回空列表
    return []


def get_obbs(world, qbase, obb_templates=None):
    """
    计算给定关节配置下所有 links 的 OBB 信息
    使用预计算的 OBB 模板或回退到几何包围盒方法

    Args:
        world: 世界模型 (在PyBullet版本中可能为None)
        qbase: 关节配置
        obb_templates: 预计算的OBB模板信息
    """
    # 1. 使用预计算的精确 OBB 模板
    if obb_templates is not None:
        dump_list = []
        for i, obb_info in enumerate(obb_templates):
            obbc = obb_info["position"]
            obbr = obb_info["rotation_matrix"]
            obbe = obb_info["extents"]
            dirstring = calculate_direction_encoding(obbe, obbr, obbc)
            dump_list.append([i, obbc, obbe, obbr, dirstring])
        return dump_list

    # 返回空列表
    return []


def calculate_direction_encoding(obbe, obbr, obbc):
    """
    计算方向编码字符串

    Args:
        obbe: OBB 尺寸 [x, y, z]
        obbr: 3x3 旋转矩阵
        obbc: OBB 中心位置

    Returns:
        dirstring: 2位方向编码字符串
    """
    try:
        # 使用 OBB 尺寸作为参考点
        newpoint = transform_point(obbe, obbr, obbc)
        direction = np.sign(newpoint - obbc)

        if direction[0] < 0:
            direction = direction * -1  # 标准化 x 方向

        direction = (direction + 1) / 2  # 归一化到 [0, 1]
        dirstring = str(int(direction[1])) + str(int(direction[2]))  # y,z 编码

        return dirstring

    except Exception as e:
        print(f"    Warning: Direction encoding failed: {e}")
        return "00"  # 默认编码


def parse_command_args():
    """解析命令行参数"""
    numqueries = int(sys.argv[1])
    foldername = sys.argv[2]
    filenumber = sys.argv[3]
    return numqueries, foldername, filenumber


def initialize_environment(foldername, filenumber):
    """初始化环境和机器人模型"""
    robot_urdf_path = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/jaco_7/jaco_7s.urdf"

    # 创建PyBullet仿真器
    sim = PyBulletRobotSimulator()

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

    while counter < numqueries:
        # 采样可行的机器人配置
        q = sim.sample_feasible_config()
        sim.set_robot_config(q)

        # 计算OBB信息 (使用虚拟的world参数)
        obbs = get_obbs(None, q, obb_templates=obb_templates)

        # 逐link碰撞检测
        real_link_idx = 0
        for lid in valid_collision_links:
            # 使用PyBullet检测碰撞
            collision = sim.check_link_collision(lid)
            ans = 0 if collision else 1

            # 存储当前实体link的数据
            if lid < len(obbs):
                qarr[counter * num_real_links + real_link_idx] = obbs[lid][1]
                yarr[counter * num_real_links + real_link_idx] = ans
                dirarr.append(obbs[lid][4])
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
    # 保存link级数据
    with open(foldername + "/obstacles_" + filenumber + "_coord.pkl", "wb") as f:
        pickle.dump((qarr, dirarr, yarr), f)

    # 保存pose级数据
    with open(foldername + "/obstacles_" + filenumber + "_pose.pkl", "wb") as f:
        pickle.dump((qarr_pose, yarr_pose), f)


def main():
    """主程序：数据生成流程"""
    # 解析命令行参数
    numqueries, foldername, filenumber = parse_command_args()

    # 环境和机器人初始化
    sim, robot_urdf_path = initialize_environment(foldername, filenumber)

    # 预先筛选实体link
    valid_collision_links = find_valid_collision_links(sim)
    num_real_links = len(valid_collision_links)

    # 设置随机种子确保可重现性
    random.seed(2)

    # 一次性初始化OBB模板
    print("Initializing OBB templates...")
    obb_templates = initialize_obb_templates(robot_urdf_path, sim.get_num_links())
    print(
        f"OBB templates initialization {'succeeded' if obb_templates else 'failed, will use fallback method'}"
    )

    # 获取机器人结构信息
    num_links = sim.get_num_links()
    num_dofs = len(sim.joint_limits)
    print(
        f"Robot has {num_links} total links ({num_real_links} real) and {num_dofs} DOFs"
    )

    # 数据存储数组初始化
    qarr, dirarr, yarr, qarr_pose, yarr_pose = initialize_data_arrays(
        numqueries, num_real_links, num_dofs
    )

    # 初始化采样
    q = sim.sample_feasible_config()
    sim.set_robot_config(q)

    # # 为每个实体link设置碰撞检测器（PyBullet版本中直接使用sim）
    # setup_collision_detectors(sim, valid_collision_links)

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
