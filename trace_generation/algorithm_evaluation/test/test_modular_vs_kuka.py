import sys

# 添加路径

from trace_generation.robot_as.modular_env import ModularEnv
from environment.kuka_env_old import KukaEnv

def test_modular_vs_kuka():
    """
    测试 ModularEnv 和 KukaEnv 的碰撞检测方法是否一致
    """
    print("Initializing environments...")

    # 初始化两个环境

    modular_env = ModularEnv(
        GUI=False,
        robot_file="kuka_iiwa/model_0.urdf",
        map_file="maze_files/kukas_7_3000.pkl",
    )

    kuka_env = KukaEnv(
        GUI=False,
        kuka_file="kuka_iiwa/model_0.urdf",
        map_file="maze_files/kukas_7_3000.pkl",
    )

    # 初始化相同的问题（使用索引 0）
    print("Initializing problem index 0...")
    modular_env.init_new_problem(0)
    kuka_env.init_new_problem(0)
    print("Testing _state_fp_probe...")
    collision_count = 0
    # 测试 _state_fp_probe
    num_tests = 10000
    for i in range(num_tests):
        # 生成随机 pose (使用 KukaEnv 的采样方法)
        random_pose = kuka_env.uniform_sample()
        # 调用 ModularEnv 的 _state_fp_probe
        modular_result = modular_env._state_fp_probe(random_pose)
        # 调用 KukaEnv 的 _state_fp_probe
        kuka_result = kuka_env._state_fp_probe(random_pose)
        # 比较结果 (只比较碰撞检查结果：is_free 和 link_colls)
        modular_free, _, modular_colls = modular_result
        kuka_free, _, kuka_colls = kuka_result

        # 跳过 kuka_free 为 True 的情况
        if kuka_free:
            # print(f"  Test {i + 1}: KukaEnv free, skipping comparison.")
            continue
        else:
            collision_count += 1
        if modular_free != kuka_free or modular_colls != kuka_colls:
            print(f"MISMATCH in _state_fp_probe for pose {i}:")
            print(f"  ModularEnv: free={modular_free}, colls={modular_colls}")
            print(f"  KukaEnv: free={kuka_free}, colls={kuka_colls}")
            return False
        else:
            if i % 100 == 0:  # 每100次打印一次以减少输出
                print(f"  Test {i + 1}: OK")

    print(f"Total collisions tested: {collision_count}")

    print("Testing _edge_fp_probe...")

    # 测试 _edge_fp_probe
    for i in range(num_tests):
        # 生成两个随机 pose 作为 edge (使用 KukaEnv 的采样方法)
        pose1 = kuka_env.sample_n_points(1)[0]
        pose2 = kuka_env.sample_n_points(1)[0]

        # 调用 ModularEnv 的 _edge_fp_probe
        modular_result = modular_env._edge_fp_probe(pose1, pose2)

        # 调用 KukaEnv 的 _edge_fp_probe
        kuka_result = kuka_env._edge_fp_probe(pose1, pose2)

        # 比较结果 (只比较碰撞检查结果：edge_free 和 edge_link_colls)
        modular_free, _, modular_colls = modular_result
        kuka_free, _, kuka_colls = kuka_result

        if modular_free != kuka_free or modular_colls != kuka_colls:
            print(f"MISMATCH in _edge_fp_probe for edge {i}:")
            print(f"  Pose1: {pose1}")
            print(f"  Pose2: {pose2}")
            print(f"  ModularEnv: free={modular_free}, colls={modular_colls}")
            print(f"  KukaEnv: free={kuka_free}, colls={kuka_colls}")
            return False
        else:
            if i % 100 == 0:  # 每100次打印一次以减少输出
                print(f"  Test {i + 1}: OK")

    print("All tests passed!")
    return True

if __name__ == "__main__":
    success = test_modular_vs_kuka()
    if not success:
        sys.exit(1)
