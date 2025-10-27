import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from environment.kuka_env_old import KukaEnv


def test_kuka_collision():
    """
    单独测试 KukaEnv 的碰撞检测功能
    """
    print("Initializing KukaEnv...")

    # 初始化环境
    kuka_env = KukaEnv(
        GUI=False,  # 设置为 False 以避免 GUI 窗口
        kuka_file="kuka_iiwa/model_3.urdf",
        map_file="maze_files/kukas_13_3000.pkl",
    )

    # 初始化问题
    print("Initializing problem index 0...")
    kuka_env.init_new_problem(0)

    print("Testing _state_fp_probe...")

    # 测试 _state_fp_probe
    num_tests = 1000
    collision_count = 0
    for i in range(num_tests):
        # 生成随机 pose (使用 uniform_sample，直接生成，不过滤碰撞)
        random_pose = kuka_env.uniform_sample()
        print(f"Testing pose {i}: {random_pose}")

        # 调用 _state_fp_probe
        is_free, link_positions, link_feasibilities = kuka_env._state_fp_probe(random_pose)

        if not is_free:
            collision_count += 1
            print(f"  Collision detected: free={is_free}, link_feas={link_feasibilities}")
        else:
            print(f"  Free: free={is_free}")

        if i % 100 == 0:  # 每100次打印进度
            print(f"  Progress: {i + 1}/{num_tests}")

    print(f"Total collisions detected: {collision_count}")

    print("Testing _edge_fp_probe...")

    # 测试 _edge_fp_probe
    edge_collision_count = 0
    for i in range(num_tests):
        # 生成两个随机 pose 作为 edge (使用 uniform_sample)
        pose1 = kuka_env.uniform_sample()
        pose2 = kuka_env.uniform_sample()

        print(f"Testing edge {i}: pose1={pose1}, pose2={pose2}")

        # 调用 _edge_fp_probe
        edge_free, edge_positions, edge_feasibilities = kuka_env._edge_fp_probe(pose1, pose2)

        if not edge_free:
            edge_collision_count += 1
            print(f"  Edge collision detected: free={edge_free}")
        else:
            print(f"  Edge free: free={edge_free}")

        if i % 100 == 0:  # 每100次打印进度
            print(f"  Progress: {i + 1}/{num_tests}")

    print(f"Total edge collisions detected: {edge_collision_count}")

    print("KukaEnv collision test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_kuka_collision()
    if not success:
        sys.exit(1)