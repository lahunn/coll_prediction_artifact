#!/usr/bin/env python3
"""
基于几何计算的球体碰撞检测实现 - 不依赖PyBullet

使用geometric_collision_detection模块中的sphere_aabb函数进行碰撞检测
假设环境中只有AABB格式的障碍物
"""

import torch
import pickle
from typing import Optional, List, Tuple

# 使用新的模块结构导入
from trace_generation.core.robot.sphere_analyzer import RobotSphereAnalyzer
from trace_generation.core.robot.environment import RobotEnv
from trace_generation.core.collision.geometric_collision_detection import (
    Sphere,
    AABB,
    sphere_aabb,
    sphere_sphere,
)
from trace_generation.core.collision.cpp_collision import cpp_collision


class SphereEnvGeometric:
    """
    基于几何计算的球体碰撞检测环境类

    不依赖PyBullet，使用纯几何算法进行碰撞检测
    假设所有障碍物都是AABB格式
    """

    def __init__(
        self,
        robot_env: RobotEnv,
        robot_name: Optional[str] = None,
        SPH_GUI=None,  # 保留接口一致性，但不使用
        return_cycles: bool = False,  # 新增：是否返回周期数
    ):
        """
        初始化球体环境

        Args:
            robot_env: 复用的机器人环境实例
            robot_name: 机器人名称
            SPH_GUI: GUI标志（保留接口但不使用，因为无可视化）
            return_cycles: 是否返回周期数（默认False保持向后兼容）
        """
        # 初始化球体分析器
        resolved_name = robot_name or getattr(robot_env, "robot_name", None)
        if resolved_name is None:
            raise ValueError("SphereEnvGeometric requires a valid robot name.")

        self.sphere_analyzer = RobotSphereAnalyzer(resolved_name, device="cuda:0")

        # 复用传入的机器人环境（用于 link 邻接检查）
        self.robot_env = robot_env
        self.robot_name = resolved_name

        # 存储障碍物AABB列表
        self.obstacles_aabb: List[AABB] = []

        # 存储球体信息（位置、半径、link_id）
        self.sphere_link_ids: List[int] = []
        self.adjacent_sphere_pairs: set = set()  # 需要忽略碰撞的球体对

        # 数据收集
        self.link_data = []
        self.link_coll_data = []
        self.link_coll_cycles = []  # 新增：存储每次碰撞检测的周期数

        # 是否返回周期数
        self.return_cycles = return_cycles

        # ====================================================================
        # 尝试加载 C++ 加速模块
        # ====================================================================
        self.cpp_checker = cpp_collision.SphereCollisionChecker()
        self.use_cpp = True
        print("✓ [SphereEnvGeometric] 使用 C++ 加速碰撞检测")
        if self.return_cycles:
            print("✓ [SphereEnvGeometric] 周期计数已启用")

    def close(self):
        """关闭环境（保留接口一致性）"""
        self.obstacles_aabb.clear()
        self.sphere_link_ids.clear()
        self.adjacent_sphere_pairs.clear()

    def load_obstacles(self, obstacles: List[Tuple]) -> List[int]:
        """
        加载并初始化AABB障碍物

        Args:
            obstacles: 障碍物列表，每个元素为 (halfExtents, basePosition) 元组

        Returns:
            障碍物ID列表（简单的索引列表）
        """
        self.cleanup_obstacles()
        self.obstacles_aabb = []

        for halfExtents, basePosition in obstacles:
            # 将PyBullet格式的box转换为AABB
            # halfExtents = (hx, hy, hz), basePosition = (cx, cy, cz)
            hx, hy, hz = halfExtents
            cx, cy, cz = basePosition

            aabb = AABB(
                min_x=cx - hx,
                min_y=cy - hy,
                min_z=cz - hz,
                max_x=cx + hx,
                max_y=cy + hy,
                max_z=cz + hz,
            )
            self.obstacles_aabb.append(aabb)

        # 同步障碍物到 C++ 检测器
        if self.use_cpp:
            # 转换为C++的AABB类型
            cpp_aabbs = []
            for aabb in self.obstacles_aabb:
                cpp_aabb = cpp_collision.AABB(
                    aabb.min_x,
                    aabb.min_y,
                    aabb.min_z,
                    aabb.max_x,
                    aabb.max_y,
                    aabb.max_z,
                )
                cpp_aabbs.append(cpp_aabb)
            self.cpp_checker.set_obstacles(cpp_aabbs)

        # 返回障碍物索引列表
        return list(range(len(self.obstacles_aabb)))

    def update_obstacle_poses(self, new_obstacles: List[Tuple]):
        """
        更新障碍物位置

        Args:
            new_obstacles: 新的障碍物列表 [(halfExtents, basePosition), ...]
        """
        for i, (halfExtents, basePosition) in enumerate(new_obstacles):
            if i < len(self.obstacles_aabb):
                hx, hy, hz = halfExtents
                cx, cy, cz = basePosition

                # 更新AABB
                self.obstacles_aabb[i] = AABB(
                    min_x=cx - hx,
                    min_y=cy - hy,
                    min_z=cz - hz,
                    max_x=cx + hx,
                    max_y=cy + hy,
                    max_z=cz + hz,
                )

    def cleanup_obstacles(self):
        """清理障碍物"""
        self.obstacles_aabb.clear()

    def _initialize_sphere_metadata(self):
        """
        初始化球体元数据（link_id和需要忽略的碰撞对）
        只需要执行一次
        """
        if self.sphere_link_ids:
            return  # 已初始化

        # 使用默认配置获取球体和link信息
        joint_config = torch.tensor(
            [0.0] * 7, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        spheres, link_ids = self.sphere_analyzer.get_world_spheres_with_links(
            joint_config
        )

        self.sphere_link_ids = (
            link_ids.tolist() if hasattr(link_ids, "tolist") else list(link_ids)
        )

        # 预计算需要忽略碰撞的球体对（同一link或相邻link）
        self.adjacent_sphere_pairs = set()
        for i in range(len(link_ids)):
            for j in range(i + 1, len(link_ids)):
                if link_ids[i] == link_ids[j] or self.robot_env._are_links_adjacent(
                    link_ids[i], link_ids[j]
                ):
                    self.adjacent_sphere_pairs.add((i, j))

        # 同步邻接对到 C++ 检测器
        if self.use_cpp:
            pairs_list = list(self.adjacent_sphere_pairs)
            self.cpp_checker.set_adjacent_pairs(pairs_list)

    def _get_sphere_data(self, state) -> Tuple:
        """
        获取关节配置下的球体数据（位置、半径、link ID）

        Args:
            state: 关节配置

        Returns:
            tuple: (positions, radii, link_ids)
        """
        joint_config = torch.tensor(
            state, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        spheres, link_ids = self.sphere_analyzer.get_world_spheres_with_links(
            joint_config
        )
        positions = spheres[:, :3]  # numpy array
        radii = spheres[:, 3]  # numpy array
        return positions, radii, link_ids

    def _update_sphere_positions(self, state) -> List[List[float]]:
        """
        更新球体位置到当前关节配置并返回球体坐标

        Args:
            state: 关节配置

        Returns:
            list: 所有球体的中心坐标列表 [[x, y, z, r], ...]
        """
        # 确保元数据已初始化
        self._initialize_sphere_metadata()

        joint_config = torch.tensor(
            state, dtype=torch.float32, device=torch.device("cuda:0")
        ).unsqueeze(0)
        world_spheres = self.sphere_analyzer.get_world_spheres(joint_config)

        sphere_coords = []
        for x, y, z, radius in world_spheres:
            sphere_coords.append([float(x), float(y), float(z), float(radius)])

        return sphere_coords

    def _check_sphere_collision(self, state):
        """
        检查球体碰撞（包括障碍物碰撞和自碰撞）

        Args:
            state: 关节配置

        Returns:
            如果 self.return_cycles=False: (bool, List[int])
                - 是否有碰撞
                - 各球体碰撞状态列表[0/1] (0表示碰撞，1表示无碰撞)
            如果 self.return_cycles=True: (bool, List[int], List[int])
                - 是否有碰撞
                - 各球体碰撞状态列表[0/1]
                - 各球体的周期数列表
        """
        # 确保元数据已初始化
        self._initialize_sphere_metadata()

        # 获取当前状态下的球体信息
        sphere_coords = self._update_sphere_positions(state)

        # ====================================================================
        # 使用 C++ 加速（如果可用）
        # ====================================================================
        if self.use_cpp:
            any_collision, sphere_colls, sphere_cycles = (
                self.cpp_checker.check_collisions(sphere_coords)
            )
            if self.return_cycles:
                return any_collision, sphere_colls, sphere_cycles
            return any_collision, sphere_colls

        # ====================================================================
        # Python 实现（作为备用）
        # ====================================================================
        return self._check_sphere_collision_python(sphere_coords)

    def _check_sphere_collision_python(self, sphere_coords):
        """
        Python 版本的球体碰撞检测（作为 C++ 的备用实现）

        Args:
            sphere_coords: 球体坐标列表 [[x, y, z, r], ...]

        Returns:
            如果 self.return_cycles=False: (bool, List[int])
            如果 self.return_cycles=True: (bool, List[int], List[int])
        """
        # 初始化碰撞状态，默认无碰撞（1表示无碰撞）
        sphere_colls = [1] * len(sphere_coords)
        sphere_cycles = [0] * len(sphere_coords)  # 每个球体的周期数
        any_collision = False

        # 检查每个球体与障碍物的碰撞
        for i, (x, y, z, radius) in enumerate(sphere_coords):
            sphere = Sphere(x, y, z, radius)

            # 检查与所有AABB障碍物的碰撞
            for aabb in self.obstacles_aabb:
                collision_result, cycles = sphere_aabb(sphere, aabb)
                sphere_cycles[i] += cycles  # 累加该球体的周期
                if collision_result == 0:  # 0表示碰撞
                    sphere_colls[i] = 0
                    any_collision = True
                    break  # 该球体已碰撞，无需继续检查其他障碍物

        # 检查球体之间的自碰撞（使用sphere_sphere函数）
        for i in range(len(sphere_coords)):
            if sphere_colls[i] == 0:  # 已经与障碍物碰撞，跳过
                continue

            for j in range(i + 1, len(sphere_coords)):
                # 跳过同一link或相邻link的球体
                if (i, j) in self.adjacent_sphere_pairs:
                    continue

                # 使用sphere_sphere函数检测碰撞
                xi, yi, zi, ri = sphere_coords[i]
                xj, yj, zj, rj = sphere_coords[j]

                sphere_i = Sphere(xi, yi, zi, ri)
                sphere_j = Sphere(xj, yj, zj, rj)

                collision_result = sphere_sphere(sphere_i, sphere_j)
                # sphere_sphere 固定1周期，分配给两个球体
                sphere_cycles[i] += 1
                sphere_cycles[j] += 1
                if collision_result == 0:  # 0表示碰撞
                    sphere_colls[i] = 0
                    sphere_colls[j] = 0
                    any_collision = True

        if self.return_cycles:
            return any_collision, sphere_colls, sphere_cycles
        return any_collision, sphere_colls

    def get_sphere_collision_data(self, state):
        """
        获取球体碰撞数据

        Args:
            state: 关节配置状态

        Returns:
            如果 self.return_cycles=False: (collision, coords, colls)
            如果 self.return_cycles=True: (collision, coords, colls, cycles)
        """
        coords = self._update_sphere_positions(state)
        result = self._check_sphere_collision(state)

        if self.return_cycles:
            collision, colls, cycles = result  # pyright: ignore[reportAssignmentType]
            return collision, coords, colls, cycles
        else:
            collision, colls = result  # pyright: ignore[reportAssignmentType]
            return collision, coords, colls

    def store_sphere_data(self, coords, colls, cycles=None, is_edge=True):
        """
        存储球体数据

        Args:
            coords: 坐标数据
            colls: 碰撞标签
            cycles: 周期数据（可选，如果return_cycles=True时提供）
            is_edge: 是否为边数据
        """
        if not coords:
            return

        if is_edge:
            self.link_data.append(coords)
            self.link_coll_data.append(colls)
            if self.return_cycles and cycles is not None:
                self.link_coll_cycles.append(cycles)
        else:
            self.link_data.append([coords])
            self.link_coll_data.append([colls])
            if self.return_cycles and cycles is not None:
                self.link_coll_cycles.append([cycles])

    def save_collision_data(self, output_file: str):
        """
        保存球体碰撞数据到文件

        Args:
            output_file: 输出文件路径
        """
        with open(output_file, "wb") as f:
            if self.return_cycles:
                pickle.dump(
                    (self.link_data, self.link_coll_data, self.link_coll_cycles), f
                )
                print(f"保存球体碰撞数据（含周期数）到: {output_file}")
            else:
                pickle.dump((self.link_data, self.link_coll_data), f)
                print(f"保存球体碰撞数据到: {output_file}")

    def get_collision_stats(self) -> dict:
        """
        获取碰撞检测统计信息（新增功能）

        Returns:
            dict: 统计信息字典
        """
        return {
            "num_spheres": len(self.sphere_link_ids),
            "num_obstacles": len(self.obstacles_aabb),
            "num_adjacent_pairs": len(self.adjacent_sphere_pairs),
            "robot_name": self.robot_name,
        }

    def check_pose(self, state: Tuple) -> Tuple[bool, dict]:
        """
        检查姿态是否无碰撞（与LinkCollisionDetector接口一致）

        Args:
            state: 关节配置状态

        Returns:
            (is_free, collision_dict): 
                - is_free (bool): True表示无碰撞，False表示有碰撞
                - collision_dict (dict): 包含碰撞信息的字典
                  {
                      'sphere_coords': [...],  # 球体坐标列表
                      'sphere_colls': [...],   # 球体碰撞信息列表
                      'timestamp': ...         # 检测时间戳（可选）
                  }
        """
        import time
        timestamp = time.time()
        
        # 调用现有的碰撞检测方法
        result = self.get_sphere_collision_data(state)
        
        # 处理返回值（考虑return_cycles标志）
        # pyright: ignore[reportGeneralTypeIssues]
        coords = result[1]
        colls = result[2]
        collision = result[0]
        
        # 构建返回字典（与LinkCollisionDetector格式一致）
        collision_dict = {
            'sphere_coords': coords,
            'sphere_colls': colls,
            'timestamp': timestamp
        }
        
        # 返回无碰撞状态和字典（collision为True表示碰撞，需要取反）
        return (not collision, collision_dict)


# 为了保持向后兼容，提供别名
SphereEnv = SphereEnvGeometric


if __name__ == "__main__":
    """简单测试"""
    print("=" * 70)
    print("SphereEnvGeometric 测试")
    print("=" * 70)

    # 注意：这里需要实际的RobotEnv实例才能运行
    # 这只是展示接口用法
    print("\n接口说明:")
    print("1. load_obstacles(obstacles) - 加载AABB障碍物")
    print("2. update_obstacle_poses(new_obstacles) - 更新障碍物位置")
    print("3. get_sphere_collision_data(state) - 获取碰撞检测结果")
    print("4. store_sphere_data(coords, colls, is_edge) - 存储数据")
    print("5. save_collision_data(output_file) - 保存数据")
    print("6. get_collision_stats() - 获取统计信息")

    print("\n特性:")
    print("✓ 不依赖PyBullet，纯几何计算")
    print("✓ 使用sphere_aabb进行球-障碍物碰撞检测")
    print("✓ 自动处理同link和相邻link的球体碰撞过滤")
    print("✓ 保持与原SphereEnv接口一致")
    print("✓ 支持硬件周期成本跟踪")
