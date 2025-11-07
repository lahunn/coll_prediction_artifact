import numpy as np
import pybullet as p
import sys
import os

# 添加 trace_generation 到 Python 路径
from trace_generation.utils.planning_utils import uniform_sample

robot_urdf_mapping = {
    "franka": "data/robots/franka_description/franka_panda.urdf",
    "iiwa": "data/robots/iiwa_allegro_description/iiwa.urdf",
    "iiwa_allegro": "data/robots/iiwa_allegro_description/iiwa_allegro.urdf",
    "jaco7": "data/robots/jaco_7/jaco_7s.urdf",
    "kinova_gen3": "data/robots/kinova/kinova_gen3_7dof.urdf",
    "quad_ur10e": "data/robots/ur_description/quad_ur10e.urdf",
    "simple_mimic_robot": "data/robots/simple/simple_mimic_robot.urdf",
    "tm12": "data/robots/techman/tm_description/urdf/tm12-nominal.urdf",
    "tri_ur10e": "data/robots/ur_description/tri_ur10e.urdf",
    "ur5e": "data/robots/ur_description/ur5e.urdf",
    "ur5e_robotiq_2f_140": "data/robots/ur_description/ur5e_robotiq_2f_140.urdf",
    "ur10e": "data/robots/ur_description/ur10e.urdf",
}

class RobotEnv:
    """机器人环境类，负责加载URDF、关节信息管理和机器人姿态控制

    构造函数参数说明:
      robot_name: 机器人名称 (例如: 'franka', 'ur5e')
      OBB_GUI: 是否启用GUI模式

    """

    def __init__(self, robot_name, OBB_GUI=None, enable_self_collision=False):
        """
        初始化机器人环境（通过 robot_name 查找 URDF）

        Args:
            robot_name: 机器人名称，用于在 `robot_urdf_mapping` 中查找相对URDF路径
            OBB_GUI: 是否启用GUI模式（可选，默认为False）
            enable_self_collision: 是否启用自碰撞检测（可选，默认为False）
        """

        # 将 robot_name 保存在实例中
        self.robot_name = robot_name
        self.enable_self_collision = enable_self_collision

        # 从映射表获取相对URDF路径（映射内可能以 / 开头）
        rel_path = robot_urdf_mapping.get(robot_name)
        if rel_path is None:
            print(f"错误: 未找到机器人名称 '{robot_name}' 对应的URDF路径")
            sys.exit(1)

        # 计算基准根目录：robot_method 所在路径的上两级目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # 最终URDF路径
        robot_file = os.path.join(base_dir, rel_path)
        self.robot_file = robot_file

        # 连接PyBullet
        if OBB_GUI:
            self.physics_client = p.connect(
                p.GUI,
                options="--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0",
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        # 加载机器人
        self.robotId = p.loadURDF(
            robot_file,
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=True,
            physicsClientId=self.physics_client,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        # 初始化关节信息
        self._setup_joint_info()

        # 初始化起始和目标状态
        self.init_state = [0.0] * self.config_dim
        self.goal_state = [0.0] * self.config_dim

    def _setup_joint_info(self):
        """设置关节信息，筛选有效关节"""
        if self.robotId is None:
            return

        num_joints = p.getNumJoints(self.robotId, physicsClientId=self.physics_client)
        self.valid_joints = []

        for i in range(num_joints):
            joint_info = p.getJointInfo(
                self.robotId, i, physicsClientId=self.physics_client
            )
            if joint_info[2] != p.JOINT_FIXED:  # 非固定关节
                self.valid_joints.append(i)

        self.config_dim = len(self.valid_joints)

        # 获取机器人关节上下限（基于有效关节）
        self.pose_range = [
            (
                p.getJointInfo(
                    self.robotId, jointId, physicsClientId=self.physics_client
                )[8],
                p.getJointInfo(
                    self.robotId, jointId, physicsClientId=self.physics_client
                )[9],
            )
            for jointId in self.valid_joints
        ]
        # 预计算正确的上下限（处理上下限可能颠倒的情况）
        self.lower_bounds = np.array([min(r[0], r[1]) for r in self.pose_range])
        self.upper_bounds = np.array([max(r[0], r[1]) for r in self.pose_range])
        self.bound = np.array(self.pose_range).T.reshape(-1)
        self.robotEndEffectorIndex = self.config_dim - 1

        p.setGravity(0, 0, -10, physicsClientId=self.physics_client)

        # 找到有碰撞几何体的link
        self.valid_collision_links = self._find_valid_collision_links()

        # 禁用相邻连杆之间的碰撞检测
        self._disable_adjacent_link_collisions()

    def _are_links_adjacent(self, link1, link2):
        """
        检查两个 link 是否相邻（通过关节连接）

        Args:
            link1: 第一个 link ID
            link2: 第二个 link ID

        Returns:
            bool: 是否相邻
        """

        if self.robot_name == "franka":
            if link1 == 9 and link2 == 7 or link1 == 7 and link2 == 9:
                return True
        # 遍历所有关节，检查父子链接关系
        num_joints = p.getNumJoints(self.robotId, physicsClientId=self.physics_client)
        for joint_id in range(num_joints):
            joint_info = p.getJointInfo(
                self.robotId, joint_id, physicsClientId=self.physics_client
            )
            parent_link = joint_info[16]  # 父链接ID
            child_link = joint_id  # 子链接ID

            # 检查是否直接通过关节相连
            if (parent_link == link1 and child_link == link2) or (
                parent_link == link2 and child_link == link1
            ):
                return True
        return False

    def _find_valid_collision_links(self):
        """找到有碰撞几何体的link"""
        if self.robotId is None:
            return []

        valid_links = []
        num_joints = p.getNumJoints(self.robotId, physicsClientId=self.physics_client)

        # 检查base link
        collision_data = p.getCollisionShapeData(
            self.robotId, -1, physicsClientId=self.physics_client
        )
        if collision_data:
            valid_links.append(-1)

        # 检查其他link
        for i in range(num_joints):
            collision_data = p.getCollisionShapeData(
                self.robotId, i, physicsClientId=self.physics_client
            )
            if collision_data:
                valid_links.append(i)

        return valid_links

    def _disable_adjacent_link_collisions(self):
        """
        禁用连杆之间的碰撞检测
        - 如果启用自碰撞检测，则只禁用相邻连杆之间的碰撞
        - 如果禁用自碰撞检测，则禁用所有连杆之间的碰撞
        """
        for i in range(len(self.valid_collision_links)):
            for j in range(i + 1, len(self.valid_collision_links)):
                link_a = self.valid_collision_links[i]
                link_b = self.valid_collision_links[j]

                # 如果不启用自碰撞检测，禁用所有连杆间的碰撞
                # 如果启用自碰撞检测，只禁用相邻连杆间的碰撞
                should_disable = (
                    not self.enable_self_collision
                ) or self._are_links_adjacent(link_a, link_b)

                if should_disable:
                    p.setCollisionFilterPair(
                        bodyUniqueIdA=self.robotId,
                        bodyUniqueIdB=self.robotId,
                        linkIndexA=link_a,
                        linkIndexB=link_b,
                        enableCollision=0,
                        physicsClientId=self.physics_client,
                    )

    def set_config(self, c, robotId=None):
        """
        设置机器人的关节配置

        Args:
            c: 关节配置数组
            robotId: 机器人ID（可选，默认使用self.robotId）
        """
        if robotId is None:
            robotId = self.robotId

        for i, angle in enumerate(c):
            if i < self.config_dim:
                p.resetJointState(
                    robotId,
                    self.valid_joints[i],
                    angle,
                    physicsClientId=self.physics_client,
                )

    def get_robot_config(self):
        """获取当前机器人关节配置"""
        if self.robotId is None:
            return []

        joint_states = p.getJointStates(
            self.robotId, self.valid_joints, physicsClientId=self.physics_client
        )
        return [state[0] for state in joint_states]

    def _valid_state(self, state):
        """检查配置是否在关节限位范围内"""
        return (state >= self.lower_bounds).all() and (state <= self.upper_bounds).all()

    def _get_link_pose(self, link_idx):
        """获取link的世界位姿"""
        if link_idx == -1:
            pos, orn = p.getBasePositionAndOrientation(
                self.robotId, physicsClientId=self.physics_client
            )
        else:
            link_state = p.getLinkState(
                self.robotId, link_idx, physicsClientId=self.physics_client
            )
            pos, orn = link_state[0], link_state[1]
        return list(pos) + list(orn)

    def close(self):
        """关闭PyBullet连接"""
        p.disconnect(physicsClientId=self.physics_client)

    def get_robot_points(self, config, end_point=True):
        """
        获取机器人在某一配置下的所有关节或末端执行器的空间位置

        Args:
            config: 关节配置
            end_point: 是否只返回末端执行器位置

        Returns:
            关节或末端执行器位置列表
        """
        points = []
        for i in range(len(self.valid_joints)):
            pose = self._get_link_pose(self.valid_joints[i])
            points.append(pose[:3])  # 只取位置
        if end_point:
            points = [points[self.robotEndEffectorIndex]]
        return points

    def sample_n_points(self, n, need_negative=False):
        """
        在配置空间采样 n 个可行点（正样本），可选采样不可行点（负样本）

        Args:
            n: 采样数量
            need_negative: 是否需要负样本

        Returns:
            采样点列表
        """

        if need_negative:
            samples = uniform_sample(
                self.lower_bounds, self.upper_bounds, self.config_dim, n * 2
            )
            # 这里需要碰撞检测来筛选正负样本，但由于没有碰撞环境，暂时返回所有样本
            # 实际使用时需要传入 collision_env
            return samples[:n], samples[n:]  # 正样本, 负样本
        else:
            samples = uniform_sample(
                self.lower_bounds, self.upper_bounds, self.config_dim, n
            )
            return samples

    def interpolate(self, from_state, to_state, ratio):
        """
        在两个配置之间插值

        Args:
            from_state: 起始配置
            to_state: 目标配置
            ratio: 插值比例

        Returns:
            插值后的配置
        """
        diff = to_state - from_state
        new_state = from_state + diff * ratio
        # 确保在关节限位范围内
        new_state = np.maximum(new_state, self.lower_bounds)
        new_state = np.minimum(new_state, self.upper_bounds)
        return new_state
