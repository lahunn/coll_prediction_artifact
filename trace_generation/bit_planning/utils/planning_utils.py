import numpy as np


def uniform_sample(lower_bounds, upper_bounds, config_dim, n=1):
    """
    在配置空间的关节限位范围内均匀随机采样

    Args:
        lower_bounds: 配置空间的下界
        upper_bounds: 配置空间的上界
        config_dim: 配置维度
        n: 采样数量

    Returns:
        采样的配置，n=1时返回一维数组，否则返回二维数组
    """
    sample = np.random.uniform(
        lower_bounds,
        upper_bounds,
        size=(n, config_dim),
    )
    return sample.reshape(-1) if n == 1 else sample


def distance(from_state, to_state):
    """
    计算两个配置之间的欧几里得距离

    Args:
        from_state: 起始配置
        to_state: 目标配置

    Returns:
        两个配置之间的欧几里得距离
    """
    diff = np.abs(to_state - from_state)
    return np.sqrt(np.sum(diff**2, axis=-1))


def plot(path, robot_env=None, obstacle_manager=None, make_gif=False):
    """
    可视化路径，支持生成 GIF

    Args:
        path: 路径（配置列表）
        robot_env: 机器人环境实例
        obstacle_manager: 障碍物管理器实例
        make_gif: 是否生成GIF

    Returns:
        GIF列表（如果make_gif=True）
    """
    import pybullet as p
    import pybullet_data

    if robot_env is None:
        raise ValueError("robot_env 不能为空")

    path = np.array(path)

    # 重置仿真
    p.resetSimulation(physicsClientId=robot_env.physics_client)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 加载障碍物
    if obstacle_manager and hasattr(obstacle_manager, 'obstacles'):
        for halfExtents, basePosition in obstacle_manager.obstacles:
            obstacle_manager.create_voxel(halfExtents, basePosition)

    # 加载机器人
    robot_id = p.loadURDF(
        robot_env.robot_file,
        [0, 0, 0],
        [0, 0, 0, 1],
        useFixedBase=True,
        flags=p.URDF_IGNORE_COLLISION_SHAPES,
        physicsClientId=robot_env.physics_client,
    )

    # 设置重力
    p.setGravity(0, 0, -10, physicsClientId=robot_env.physics_client)
    p.stepSimulation(physicsClientId=robot_env.physics_client)

    gifs = []

    # 简化版本：只显示起点和终点
    if len(path) >= 2:
        # 设置起点
        robot_env.set_config(path[0], robot_id)

        # 创建目标机器人
        target_robot_id = p.loadURDF(
            robot_env.robot_file,
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_IGNORE_COLLISION_SHAPES,
            physicsClientId=robot_env.physics_client,
        )
        robot_env.set_config(path[-1], target_robot_id)

        # 简单的动画（如果需要）
        if make_gif:
            # 这里可以实现GIF生成逻辑
            pass

    return gifs