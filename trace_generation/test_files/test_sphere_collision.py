#!/usr/bin/env python3
"""
球体碰撞检测测试脚本
创建简单的球体和长方体进行碰撞检测测试
"""

import pybullet as p
import time


class SimpleBulletSim:
    """简化的PyBullet仿真器"""

    def __init__(self, use_gui=True):
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, 0)
        self.obstacle_ids = []

    def create_obstacles(self):
        """创建测试用的障碍物（长方体）"""
        # 创建长方体障碍物
        box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        box_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.8, 0.2, 0.2, 1.0]
        )

        obstacle1 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_shape,
            baseVisualShapeIndex=box_visual,
            basePosition=[1.0, 0.0, 0.5],
        )

        obstacle2 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_shape,
            baseVisualShapeIndex=box_visual,
            basePosition=[-1.0, 0.0, 0.5],
        )

        self.obstacle_ids = [obstacle1, obstacle2]
        print(f"创建了 {len(self.obstacle_ids)} 个障碍物")

    def disconnect(self):
        """断开连接"""
        p.disconnect(physicsClientId=self.physics_client)


def check_spheres_collision_closest_points(sim, sphere_bodies):
    """
    使用getClosestPoints的球体碰撞检测函数
    """
    collision_results = []

    if not sim.obstacle_ids:
        return [False] * len(sphere_bodies)

    try:
        for i, sphere_body in enumerate(sphere_bodies):
            has_collision = False
            min_distance = float("inf")

            # 检查当前球体与所有障碍物的碰撞
            for j, obstacle_id in enumerate(sim.obstacle_ids):
                # 使用getClosestPoints计算最近距离
                closest_points = p.getClosestPoints(
                    bodyA=sphere_body,
                    bodyB=obstacle_id,
                    distance=2.0,  # 最大查询距离
                    physicsClientId=sim.physics_client,
                )

                if closest_points:
                    # 获取最近的距离
                    distance = closest_points[0][8]  # contactDistance字段
                    min_distance = min(min_distance, distance)

                    print(
                        f"[ClosestPoints] 球体{i + 1} vs 障碍物{j + 1}: 距离 = {distance:.4f}"
                    )

                    # 如果距离 <= 0，表示碰撞
                    if distance <= 0.0:
                        has_collision = True
                        print("  -> ClosestPoints检测到碰撞！")
                        break

            collision_results.append(has_collision)
            print(
                f"[ClosestPoints] 球体{i + 1} 最小距离: {min_distance:.4f}, 碰撞: {has_collision}"
            )

        return collision_results

    except Exception as e:
        print(f"ClosestPoints球体碰撞检测失败: {e}")
        return [False] * len(sphere_bodies)


def check_spheres_collision_contact_points(sim, sphere_bodies):
    """
    使用getContactPoints的球体碰撞检测函数
    根据PyBullet文档，getContactPoints用于获取实际的接触点信息
    """
    collision_results = []

    if not sim.obstacle_ids:
        return [False] * len(sphere_bodies)

    try:
        # 首先执行碰撞检测步骤（这对getContactPoints很重要）
        p.performCollisionDetection(physicsClientId=sim.physics_client)

        for i, sphere_body in enumerate(sphere_bodies):
            has_collision = False

            # 检查当前球体与所有障碍物的碰撞
            for j, obstacle_id in enumerate(sim.obstacle_ids):
                # 使用getContactPoints获取接触点
                contact_points = p.getContactPoints(
                    bodyA=sphere_body,
                    bodyB=obstacle_id,
                    physicsClientId=sim.physics_client,
                )

                # 如果有接触点，说明有碰撞
                if len(contact_points) > 0:
                    has_collision = True
                    print(
                        f"[ContactPoints] 球体{i + 1} vs 障碍物{j + 1}: 检测到 {len(contact_points)} 个接触点"
                    )

                    # 打印第一个接触点的详细信息
                    if contact_points:
                        contact = contact_points[0]
                        contact_distance = contact[8]  # contactDistance
                        print(f"  -> 接触距离: {contact_distance:.4f}")
                        print("  -> ContactPoints检测到碰撞！")
                    break
                else:
                    print(f"[ContactPoints] 球体{i + 1} vs 障碍物{j + 1}: 无接触点")

            collision_results.append(has_collision)
            print(f"[ContactPoints] 球体{i + 1} 碰撞: {has_collision}")

        return collision_results

    except Exception as e:
        print(f"ContactPoints球体碰撞检测失败: {e}")
        return [False] * len(sphere_bodies)


def check_spheres_collision_closest_points_silent(sim, sphere_bodies):
    """
    使用getClosestPoints的球体碰撞检测函数（静默版本，用于性能测试）
    """
    collision_results = []

    if not sim.obstacle_ids:
        return [False] * len(sphere_bodies)

    try:
        for sphere_body in sphere_bodies:
            has_collision = False

            # 检查当前球体与所有障碍物的碰撞
            for obstacle_id in sim.obstacle_ids:
                # 使用getClosestPoints计算最近距离
                closest_points = p.getClosestPoints(
                    bodyA=sphere_body,
                    bodyB=obstacle_id,
                    distance=2.0,  # 最大查询距离
                    physicsClientId=sim.physics_client,
                )

                if closest_points:
                    # 获取最近的距离
                    distance = closest_points[0][8]  # contactDistance字段

                    # 如果距离 <= 0，表示碰撞
                    if distance <= 0.0:
                        has_collision = True
                        break

            collision_results.append(has_collision)

        return collision_results

    except Exception:
        return [False] * len(sphere_bodies)


def check_spheres_collision_contact_points_silent(sim, sphere_bodies):
    """
    使用getContactPoints的球体碰撞检测函数（静默版本，用于性能测试）
    """
    collision_results = []

    if not sim.obstacle_ids:
        return [False] * len(sphere_bodies)

    try:
        # 首先执行碰撞检测步骤
        p.performCollisionDetection(physicsClientId=sim.physics_client)

        for sphere_body in sphere_bodies:
            has_collision = False

            # 检查当前球体与所有障碍物的碰撞
            for obstacle_id in sim.obstacle_ids:
                # 使用getContactPoints获取接触点
                contact_points = p.getContactPoints(
                    bodyA=sphere_body,
                    bodyB=obstacle_id,
                    physicsClientId=sim.physics_client,
                )

                # 如果有接触点，说明有碰撞
                if len(contact_points) > 0:
                    has_collision = True
                    break

            collision_results.append(has_collision)

        return collision_results

    except Exception:
        return [False] * len(sphere_bodies)


def performance_test(sim, sphere_bodies, iterations=1000):
    """
    性能测试：比较两种碰撞检测方法的耗时
    """
    print(f"\n=== 性能测试 ({iterations} 次迭代) ===")

    # 测试 ClosestPoints 方法
    start_time = time.time()
    for _ in range(iterations):
        check_spheres_collision_closest_points_silent(sim, sphere_bodies)
    closest_points_time = time.time() - start_time

    # 测试 ContactPoints 方法
    start_time = time.time()
    for _ in range(iterations):
        check_spheres_collision_contact_points_silent(sim, sphere_bodies)
    contact_points_time = time.time() - start_time

    print(
        f"ClosestPoints方法: {closest_points_time:.4f}秒 (平均: {closest_points_time / iterations * 1000:.4f}ms/次)"
    )
    print(
        f"ContactPoints方法: {contact_points_time:.4f}秒 (平均: {contact_points_time / iterations * 1000:.4f}ms/次)"
    )

    # 计算性能比率
    if contact_points_time > 0:
        ratio = closest_points_time / contact_points_time
        if ratio > 1:
            print(f"ContactPoints方法比ClosestPoints方法快 {ratio:.2f} 倍")
        else:
            print(f"ClosestPoints方法比ContactPoints方法快 {1 / ratio:.2f} 倍")

    return closest_points_time, contact_points_time


def main():
    print("=== 球体碰撞检测测试 ===")

    # 创建仿真器
    sim = SimpleBulletSim(use_gui=True)

    # 创建障碍物
    sim.create_obstacles()
    radius = 0.5  # 减小球体半径
    # 创建测试球体
    sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    sphere_visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=[0.0, 1.0, 0.0, 0.7]
    )

    # 球体位置：一个明显碰撞，一个不碰撞，一个边缘情况
    test_positions = [
        [1.0, 0.0, 0.5],  # 应该碰撞 (直接在第一个障碍物中心)
        [0.0, 0.0, 0.5],  # 应该不碰撞 (在中间，距离障碍物0.5m)
        [1.8, 0.0, 0.5],  # 应该不碰撞 (距离障碍物边缘0.3m)
    ]

    sphere_bodies = []
    for i, pos in enumerate(test_positions):
        # 创建可视化和碰撞检测球体
        sphere_body = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=sphere_shape,
            baseVisualShapeIndex=sphere_visual,
            basePosition=pos,
            physicsClientId=sim.physics_client,
        )
        sphere_bodies.append(sphere_body)
        print(f"球体{i + 1} 位置: {pos}")

    # 执行碰撞检测 - 测试两种方法
    print("\n=== 执行ClosestPoints碰撞检测 ===")
    collision_results_closest = check_spheres_collision_closest_points(
        sim, sphere_bodies
    )

    print("\n=== 执行ContactPoints碰撞检测 ===")
    collision_results_contact = check_spheres_collision_contact_points(
        sim, sphere_bodies
    )

    # 输出结果对比
    print("\n=== 碰撞检测结果对比 ===")
    for i, pos in enumerate(test_positions):
        closest_status = "碰撞" if collision_results_closest[i] else "无碰撞"
        contact_status = "碰撞" if collision_results_contact[i] else "无碰撞"
        print(f"球体{i + 1} 位置{pos}:")
        print(f"  ClosestPoints: {closest_status}")
        print(f"  ContactPoints: {contact_status}")

        # 检查两种方法是否一致
        if collision_results_closest[i] != collision_results_contact[i]:
            print("  ⚠️  两种方法结果不一致！")

    print(
        f"\nClosestPoints方法: {sum(collision_results_closest)}/{len(collision_results_closest)} 个球体发生碰撞"
    )
    print(
        f"ContactPoints方法: {sum(collision_results_contact)}/{len(collision_results_contact)} 个球体发生碰撞"
    )

    # 执行性能测试
    performance_test(
        sim, sphere_bodies, iterations=10000
    )  # 使用较少迭代次数避免过长等待

    # 等待用户查看结果
    print("\n按任意键退出...")
    try:
        input()
    except KeyboardInterrupt:
        pass

    # 清理
    sim.disconnect()


if __name__ == "__main__":
    main()
