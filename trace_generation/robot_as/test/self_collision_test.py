import sys
import os
import pickle

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from robot_as.modular_env import ModularEnv


def test_self_collision(robot_file, num_samples=100):
    """
    测试无障碍物情况下的自碰撞检查

    Args:
        robot_file: 机器人URDF文件路径
        num_samples: 采样配置数量
    """
    print(f"初始化机器人环境: {robot_file}")
    # 初始化ModularEnv，不加载障碍物
    env = ModularEnv(robot_file, GUI=False)

    print(f"采样 {num_samples} 个配置进行自碰撞测试...")

    # 采样配置
    samples = env.sample_n_points(num_samples)

    self_collision_count = 0
    free_count = 0

    for i, config in enumerate(samples):
        # 检查配置是否自由（无碰撞）
        is_free, link_coords, link_colls = env._state_fp_probe(config)

        if is_free:
            free_count += 1
        else:
            self_collision_count += 1
            print(f"配置 {i}: 自碰撞检测到")
            # 可以打印更多细节，如哪些链接碰撞
            # 但为了简洁，只计数

    print("\n测试结果:")
    print(f"总配置数: {num_samples}")
    print(f"无碰撞配置: {free_count}")
    print(f"自碰撞配置: {self_collision_count}")
    print(".2f")

    env.close()


def test_self_collision_from_inconsistent_edges(robot_file, inconsistent_edges_file):
    """
    从inconsistent_edges文件中加载配置，进行自碰撞检查

    Args:
        robot_file: 机器人URDF文件路径
        inconsistent_edges_file: 不一致边数据文件路径
    """
    print(f"初始化机器人环境: {robot_file}")
    env = ModularEnv(robot_file, GUI=False)

    print(f"加载不一致边数据: {inconsistent_edges_file}")
    if not os.path.exists(inconsistent_edges_file):
        print(f"错误: 文件不存在: {inconsistent_edges_file}")
        env.close()
        return

    # 加载数据
    with open(inconsistent_edges_file, "rb") as f:
        data = pickle.load(f)

    # 数据结构是字典，包含 "edge_configs" 等键
    if "edge_configs" not in data:
        print("错误: 数据文件中缺少 'edge_configs' 键")
        env.close()
        return

    edges = data["edge_configs"]  # edges 是边列表，每个边是配置列表

    print(f"处理 {len(edges)} 条边...")

    total_configs = 0
    self_collision_count = 0
    free_count = 0

    for edge_idx, edge in enumerate(edges):
        for pose_idx, pose in enumerate(edge):
            if isinstance(pose, list) and len(pose) > 0:
                # 如果pose是配置（numpy array）
                config = pose
                total_configs += 1
                is_free, _, _ = env._state_fp_probe(config)
                if is_free:
                    free_count += 1
                else:
                    self_collision_count += 1
                    print(f"边 {edge_idx}, 配置 {pose_idx}: 自碰撞检测到")

    print("\n测试结果:")
    print(f"总配置数: {total_configs}")
    print(f"无碰撞配置: {free_count}")
    print(f"自碰撞配置: {self_collision_count}")
    if total_configs > 0:
        print(".2f")

    env.close()


if __name__ == "__main__":
    # 示例URDF路径，需要根据实际路径调整
    robot_file = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/franka_description/franka_panda.urdf"  # 假设路径

    # 检查文件是否存在
    if not os.path.exists(robot_file):
        print(f"错误: URDF文件不存在: {robot_file}")
        print("请检查路径或提供正确的URDF文件路径")
        sys.exit(1)

    # 运行采样测试
    test_self_collision(robot_file, num_samples=500)

    # 示例inconsistent_edges文件路径
    inconsistent_edges_file = "/home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/sphere_as/inconsistent_edge/inconsistent_edges_10.pkl"

    # 运行不一致边测试
    test_self_collision_from_inconsistent_edges(robot_file, inconsistent_edges_file)
