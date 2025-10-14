import numpy as np
import pybullet as p
import pybullet_data
import pickle

class CollisionEnv:
    """完整的碰撞检测环境类,包含PyBullet初始化、机器人加载和碰撞检测功能"""
    
    RRT_EPS = 0.25
    
    def __init__(self, GUI=False, robot_file="kuka_iiwa/model_0.urdf", z_offset=0.0, config_output_file=None):
        """
        初始化碰撞检测环境
        
        Args:
            GUI: 是否启用GUI模式
            robot_file: 机器人URDF文件路径
            z_offset: Z轴偏移量
            config_output_file: 配置输出文件路径（可选）
        """
        self.robot_file = robot_file
        self.z_offset = z_offset
        self.obstacles = []
        self.obstacle_body_ids = []
        
        # config_output_file 相关逻辑
        self.config_output_file = config_output_file
        self.config_list = []
        
        # 连接PyBullet
        if GUI:
            p.connect(p.GUI, options="--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0")
        else:
            p.connect(p.DIRECT)
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[0, 0, 0.1])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载机器人
        self.robotId = p.loadURDF(robot_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.performCollisionDetection()
        
        # 获取机器人配置信息
        self.config_dim = p.getNumJoints(self.robotId)
        self.pose_range = [
            (p.getJointInfo(self.robotId, jointId)[8], p.getJointInfo(self.robotId, jointId)[9])
            for jointId in range(self.config_dim)
        ]
        # 预计算正确的上下限（处理上下限可能颠倒的情况）
        self.lower_bounds = np.array([min(r[0], r[1]) for r in self.pose_range])
        self.upper_bounds = np.array([max(r[0], r[1]) for r in self.pose_range])
        self.bound = np.array(self.pose_range).T.reshape(-1)
        self.robotEndEffectorIndex = self.config_dim - 1
        
        p.setGravity(0, 0, -10)
        
        # 规划相关属性
        self.init_state = [0.0] * self.config_dim
        self.goal_state = [0.0] * self.config_dim
    
    def close(self):
        """关闭配置输出文件句柄和PyBullet连接"""
        if self.config_output_file is not None and len(self.config_list) > 0:
            with open(self.config_output_file, 'wb') as f:
                pickle.dump(self.config_list, f)
        p.disconnect()
    
    def uniform_sample(self, n=1):
        """
        在配置空间的关节限位范围内均匀随机采样
        
        Args:
            n: 采样数量
            
        Returns:
            采样的配置，n=1时返回一维数组，否则返回二维数组
        """
        sample = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(n, self.config_dim),
        )
        return sample.reshape(-1) if n == 1 else sample
    
    def distance(self, from_state, to_state):
        """
        计算两个配置之间的欧几里得距离
        
        Args:
            from_state: 起始配置
            to_state: 目标配置
            
        Returns:
            欧几里得距离（L2范数）
        """
        to_state = np.maximum(to_state, self.lower_bounds)
        to_state = np.minimum(to_state, self.upper_bounds)
        diff = np.abs(to_state - from_state)
        return np.sqrt(np.sum(diff**2, axis=-1))
    
    def set_config(self, c, robotId=None):
        """
        设置机器人的关节配置
        
        Args:
            c: 关节配置数组
            robotId: 机器人ID（可选，默认使用self.robotId）
        """
        if robotId is None:
            robotId = self.robotId
        for i in range(p.getNumJoints(robotId)):
            p.resetJointState(robotId, i, c[i])
        p.performCollisionDetection()

    def create_voxel(self, halfExtents, basePosition):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        groundVisID = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            rgbaColor=np.random.uniform(0, 1, size=3).tolist() + [0.8],
            specularColor=[0.4, 0.4, 0],
            halfExtents=halfExtents,
        )
        groundId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=groundColId,
            baseVisualShapeIndex=groundVisID,
            basePosition=basePosition,
        )
        return groundId

    def init_obstacle_bodies(self, num_obstacles, initial_obstacles=None):
        self.obstacles = initial_obstacles
        self.obstacle_body_ids = []
        for i in range(num_obstacles):
            if initial_obstacles is not None and i < len(initial_obstacles):
                halfExtents, basePosition = initial_obstacles[i]
            else:
                halfExtents = np.array([0.1, 0.1, 0.1])
                basePosition = np.array([0, 0, -10])
            body_id = self.create_voxel(halfExtents, basePosition)
            self.obstacle_body_ids.append(body_id)
        return self.obstacle_body_ids

    def update_obstacle_poses(self, new_obstacles):
        if not hasattr(self, 'obstacle_body_ids'):
            raise RuntimeError("请先调用 init_obstacle_bodies() 初始化障碍物")
        for i, (halfExtents, basePosition) in enumerate(new_obstacles):
            if i < len(self.obstacle_body_ids):
                p.resetBasePositionAndOrientation(
                    self.obstacle_body_ids[i],
                    basePosition,
                    [0, 0, 0, 1]
                )
        self.obstacles = new_obstacles

    def randomize_obstacle_poses(self, workspace_range=(-1.0, 1.0), safe_zone_center=(0.0, 0.0, 0.0), safe_zone_radius=0.3, max_attempts_per_obstacle=100):
        if not hasattr(self, 'obstacles') or self.obstacles is None:
            raise RuntimeError("请先设置 self.obstacles")
        
        w_min, w_max = workspace_range
        safe_center = np.array(safe_zone_center)
        new_obstacles = []
        for halfExtents, old_position in self.obstacles:
            for attempt in range(max_attempts_per_obstacle):
                new_position = np.random.uniform(w_min, w_max, size=3)
                distance_to_base = np.linalg.norm(new_position - safe_center)
                min_safe_distance = safe_zone_radius + np.max(halfExtents)
                if distance_to_base > min_safe_distance:
                    new_obstacles.append((halfExtents, new_position))
                    break
            else:
                new_obstacles.append((halfExtents, old_position))
        self.update_obstacle_poses(new_obstacles)
        return new_obstacles

    def cleanup_obstacles(self):
        if hasattr(self, 'obstacle_body_ids'):
            for body_id in self.obstacle_body_ids:
                try:
                    p.removeBody(body_id)
                except Exception:
                    pass
            self.obstacle_body_ids.clear()

    def _valid_state(self, state):
        """检查配置是否在关节限位范围内"""
        return (state >= self.lower_bounds).all() and (state <= self.upper_bounds).all()

    def _point_in_free_space(self, state):
        """检查单个配置是否无碰撞,如果设置了config_output_file则记录配置"""
        if self.config_output_file is not None:
            self.config_list.append(state.copy())
        
        if not self._valid_state(state):
            return False
        for i in range(p.getNumJoints(self.robotId)):
            p.resetJointState(self.robotId, i, state[i])
        p.performCollisionDetection()
        if len(p.getContactPoints(self.robotId)) == 0:
            return True
        else:
            return False

    def _state_fp(self, state):
        return self._point_in_free_space(state)

    def _edge_fp(self, state, new_state, RRT_EPS=0.25):
        assert state.size == new_state.size
        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._point_in_free_space(state) or not self._point_in_free_space(new_state):
            return False
        disp = new_state - state
        d = np.linalg.norm(new_state - state)
        K = int(d / RRT_EPS)
        for k in range(0, K):
            c = state + k * 1.0 / K * disp
            if not self._point_in_free_space(c):
                return False
        return True
