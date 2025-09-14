"""
球体参数和 get_spheres 函数验证脚本
用于在数据生成前验证球体配置和计算的正确性

验证内容：
1. 球体配置文件加载是否正确
2. get_spheres 函数计算的球体位置是否合理
3. 球体是否正确跟随机器人运动
4. 球体大小和位置是否符合机器人几何形状
5. 可视化验证球体覆盖效果
"""

import klampt
from klampt import vis
import numpy as np
import math
import yaml
import sys
import time
from sphere_trace_generation import (
    give_dh,
    give_RT,
    get_obbRT,
    transform_point,
    load_sphere_config,
    get_spheres,
)


def create_test_configurations():
    """
    创建一系列测试关节配置用于验证
    包括零配置、极限配置和随机配置
    """
    test_configs = {
        "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "home": [0.0, 0.0, 0.0, -math.pi / 2, 0.0, math.pi / 2, 0.0],
        "extended": [0.0, math.pi / 4, 0.0, -math.pi / 2, 0.0, math.pi / 4, 0.0],
        "folded": [0.0, math.pi / 2, 0.0, -math.pi / 2, 0.0, math.pi / 2, 0.0],
        "random1": [0.5, -0.3, 0.8, -1.2, 0.4, 1.1, -0.6],
        "random2": [-0.4, 0.7, -0.5, -0.9, -0.3, 0.8, 0.5],
    }
    return test_configs


def validate_sphere_config():
    """
    验证球体配置文件的正确性
    """
    print("=" * 50)
    print("1. 验证球体配置文件")
    print("=" * 50)

    sphere_config = load_sphere_config()

    total_spheres = 0
    for link_name, spheres in sphere_config.items():
        print(f"Link: {link_name}")
        print(f"  球体数量: {len(spheres)}")

        for i, sphere in enumerate(spheres):
            center = sphere["center"]
            radius = sphere["radius"]
            print(f"    球体 {i}: 中心={center}, 半径={radius:.3f}")

            # 验证合理性
            if radius <= 0:
                print(f"    ⚠️  警告: 球体 {i} 半径非正值!")
            if radius > 0.5:
                print(f"    ⚠️  警告: 球体 {i} 半径过大!")
            if abs(center[0]) > 1.0 or abs(center[1]) > 1.0 or abs(center[2]) > 1.0:
                print(f"    ⚠️  警告: 球体 {i} 中心位置可能不合理!")

        total_spheres += len(spheres)
        print()

    print(f"总球体数量: {total_spheres}")
    return sphere_config


def validate_get_spheres_function(sphere_config):
    """
    验证 get_spheres 函数的计算正确性
    """
    print("=" * 50)
    print("2. 验证 get_spheres 函数")
    print("=" * 50)

    # 创建简单的测试环境
    world = klampt.WorldModel()
    world.readFile("jaco_collision.xml")
    robot = world.robot(0)

    test_configs = create_test_configurations()

    for config_name, q in test_configs.items():
        print(f"\n测试配置: {config_name}")
        print(f"关节角: {[f'{angle:.3f}' for angle in q]}")

        # 设置机器人配置
        robot.setConfig(q)

        # 计算球体
        spheres = get_spheres(world, q, sphere_config)

        print(f"计算得到 {len(spheres)} 个球体")

        # 分析球体分布
        if spheres:
            centers = np.array([sphere[2].flatten() for sphere in spheres])

            # 计算边界
            min_coords = np.min(centers, axis=0)
            max_coords = np.max(centers, axis=0)

            print(f"球体分布范围:")
            print(f"  X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
            print(f"  Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
            print(f"  Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")

            # 检查异常值
            if np.any(np.abs(centers) > 5.0):
                print("⚠️  警告: 发现球体位置异常值!")

            # 按link分组统计
            link_count = {}
            for sphere in spheres:
                link_id = sphere[0]
                if link_id not in link_count:
                    link_count[link_id] = 0
                link_count[link_id] += 1

            print(f"每个link的球体数量: {link_count}")


def validate_sphere_movement(sphere_config):
    """
    验证球体是否正确跟随机器人运动
    """
    print("=" * 50)
    print("3. 验证球体运动跟踪")
    print("=" * 50)

    world = klampt.WorldModel()
    world.readFile("jaco_collision.xml")
    robot = world.robot(0)

    # 测试关节运动对球体位置的影响
    base_config = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print("测试单个关节运动对球体位置的影响:")

    for joint_idx in range(7):
        print(f"\n关节 {joint_idx} 运动测试:")

        # 计算基准位置
        robot.setConfig(base_config)
        spheres_base = get_spheres(world, base_config, sphere_config)

        # 移动该关节
        test_config = base_config.copy()
        test_config[joint_idx] = math.pi / 4  # 45度
        robot.setConfig(test_config)
        spheres_moved = get_spheres(world, test_config, sphere_config)

        # 计算位移
        if len(spheres_base) == len(spheres_moved):
            max_displacement = 0
            for i in range(len(spheres_base)):
                base_pos = spheres_base[i][2].flatten()
                moved_pos = spheres_moved[i][2].flatten()
                displacement = np.linalg.norm(moved_pos - base_pos)
                max_displacement = max(max_displacement, displacement)

            print(f"  最大球体位移: {max_displacement:.3f} m")

            if max_displacement == 0:
                print("  ⚠️  警告: 球体位置未发生变化!")
            elif max_displacement > 2.0:
                print("  ⚠️  警告: 球体位移过大!")
        else:
            print("  ❌ 错误: 球体数量不一致!")


def visualize_spheres(sphere_config):
    """
    可视化验证球体覆盖效果
    """
    print("=" * 50)
    print("4. 可视化验证")
    print("=" * 50)

    world = klampt.WorldModel()
    world.readFile("jaco_collision.xml")
    robot = world.robot(0)

    # 使用一个典型配置
    test_config = [0.0, math.pi / 4, 0.0, -math.pi / 2, 0.0, math.pi / 2, 0.0]
    robot.setConfig(test_config)

    spheres = get_spheres(world, test_config, sphere_config)

    print(f"可视化 {len(spheres)} 个球体")
    print("启动可视化窗口...")

    # 初始化可视化
    vis.init()
    vis.add("world", world)

    # 添加球体到可视化
    for i, (link_id, sphere_id, center, radius) in enumerate(spheres):
        sphere_name = f"sphere_{link_id}_{sphere_id}"

        # 创建球体几何
        from klampt.model.create import primitives

        sphere_geom = primitives.sphere(radius, center=center.flatten())

        # 根据link_id设置颜色
        colors = [
            [1, 0, 0, 0.5],  # 红色 - link 0
            [0, 1, 0, 0.5],  # 绿色 - link 1
            [0, 0, 1, 0.5],  # 蓝色 - link 2
            [1, 1, 0, 0.5],  # 黄色 - link 3
            [1, 0, 1, 0.5],  # 洋红 - link 4
            [0, 1, 1, 0.5],  # 青色 - link 5
            [1, 0.5, 0, 0.5],  # 橙色 - link 6
        ]

        color = colors[link_id % len(colors)]
        vis.add(sphere_name, sphere_geom, color=color)

    print("\n可视化说明:")
    print("- 不同颜色代表不同的link")
    print("- 球体应该合理覆盖机器人各个部件")
    print("- 按 'q' 退出可视化")

    # 运行可视化
    vis.show()

    while vis.shown():
        time.sleep(0.1)

    vis.kill()


def run_comprehensive_validation():
    """
    运行完整的验证流程
    """
    print("🔍 开始球体参数和函数验证")
    print("=" * 60)

    try:
        # 1. 验证配置文件
        sphere_config = validate_sphere_config()

        # 2. 验证函数计算
        validate_get_spheres_function(sphere_config)

        # 3. 验证运动跟踪
        validate_sphere_movement(sphere_config)

        # 4. 可视化验证
        if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
            visualize_spheres(sphere_config)
        else:
            print("\n💡 提示: 运行时添加 --visualize 参数以启用可视化验证")

        print("\n" + "=" * 60)
        print("✅ 验证完成!")
        print("如果没有看到警告或错误信息，说明球体配置和函数基本正确")
        print("建议运行可视化验证以确认球体覆盖效果")

    except Exception as e:
        print(f"\n❌ 验证过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


def quick_sphere_test():
    """
    快速球体测试，用于调试
    """
    print("🔧 快速球体测试")

    sphere_config = load_sphere_config()

    world = klampt.WorldModel()
    world.readFile("jaco_collision.xml")
    robot = world.robot(0)

    q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    robot.setConfig(q)

    spheres = get_spheres(world, q, sphere_config)

    print(f"零配置下生成 {len(spheres)} 个球体:")
    for i, (link_id, sphere_id, center, radius) in enumerate(
        spheres[:5]
    ):  # 只显示前5个
        print(f"  球体 {i}: Link{link_id}, 中心{center.flatten()}, 半径{radius:.3f}")

    if len(spheres) > 5:
        print(f"  ... 还有 {len(spheres) - 5} 个球体")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_sphere_test()
    else:
        run_comprehensive_validation()
