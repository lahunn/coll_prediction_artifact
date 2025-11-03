#!/usr/bin/env python3
"""
基于几何计算的碰撞检测耗时评估程序

基于geometric_collision_detection.py重新实现碰撞检测性能评估
对比不同碰撞检测类型的计算成本和耗时
"""

import time
import numpy as np
import sys
import os

# 添加当前目录到Python路径，以便导入geometric_collision_detection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geometric_collision_detection_optimized import (
    Cuboid,
    Sphere,
    Capsule,
    Triangle,
    HeightField,
    cuboid_sphere,
    cuboid_capsule,
    cuboid_cuboid,
    cuboid_triangle,
    cuboid_heightfield,
    sphere_sphere,
    sphere_triangle,
    sphere_capsule,
    sphere_cuboid,
    sphere_heightfield,
)


class CollisionBenchmark:
    """碰撞检测基准测试类"""

    def __init__(self):
        self.collision_functions = {
            "cuboid_sphere": cuboid_sphere,
            "cuboid_capsule": cuboid_capsule,
            "cuboid_cuboid": cuboid_cuboid,
            "cuboid_triangle": cuboid_triangle,
            "cuboid_heightfield": cuboid_heightfield,
            "sphere_sphere": sphere_sphere,
            "sphere_triangle": sphere_triangle,
            "sphere_capsule": sphere_capsule,
            "sphere_cuboid": sphere_cuboid,
            "sphere_heightfield": sphere_heightfield,
        }

    def random_cuboid(self, position_range=(-10, 10), size_range=(0.1, 2.0)):
        """随机生成一个Cuboid对象

        Args:
            position_range: 位置坐标范围 (min, max)
            size_range: 尺寸范围 (min, max)

        Returns:
            Cuboid: 随机生成的Cuboid对象
        """
        # 随机位置
        x = np.random.uniform(position_range[0], position_range[1])
        y = np.random.uniform(position_range[0], position_range[1])
        z = np.random.uniform(position_range[0], position_range[1])

        # 随机轴方向（归一化）
        axis1_dir = np.random.normal(0, 1, 3)
        axis1_dir = axis1_dir / np.linalg.norm(axis1_dir)
        axis2_dir = np.random.normal(0, 1, 3)
        axis2_dir = axis2_dir / np.linalg.norm(axis2_dir)
        axis3_dir = np.random.normal(0, 1, 3)
        axis3_dir = axis3_dir / np.linalg.norm(axis3_dir)

        # 随机半长度
        axis1_r = np.random.uniform(size_range[0], size_range[1])
        axis2_r = np.random.uniform(size_range[0], size_range[1])
        axis3_r = np.random.uniform(size_range[0], size_range[1])

        return Cuboid(
            x,
            y,
            z,
            (axis1_dir[0], axis1_dir[1], axis1_dir[2], axis1_r),
            (axis2_dir[0], axis2_dir[1], axis2_dir[2], axis2_r),
            (axis3_dir[0], axis3_dir[1], axis3_dir[2], axis3_r),
        )

    def random_sphere(self, position_range=(-10, 10), radius_range=(0.1, 1.0)):
        """随机生成一个Sphere对象

        Args:
            position_range: 位置坐标范围 (min, max)
            radius_range: 半径范围 (min, max)

        Returns:
            Sphere: 随机生成的Sphere对象
        """
        x = np.random.uniform(position_range[0], position_range[1])
        y = np.random.uniform(position_range[0], position_range[1])
        z = np.random.uniform(position_range[0], position_range[1])
        r = np.random.uniform(radius_range[0], radius_range[1])

        return Sphere(x, y, z, r)

    def random_capsule(
        self,
        position_range=(-10, 10),
        length_range=(0.5, 3.0),
        radius_range=(0.05, 0.5),
    ):
        """随机生成一个Capsule对象

        Args:
            position_range: 位置坐标范围 (min, max)
            length_range: 长度范围 (min, max)
            radius_range: 半径范围 (min, max)

        Returns:
            Capsule: 随机生成的Capsule对象
        """
        # 随机起点
        x1 = np.random.uniform(position_range[0], position_range[1])
        y1 = np.random.uniform(position_range[0], position_range[1])
        z1 = np.random.uniform(position_range[0], position_range[1])

        # 随机方向向量（归一化）
        direction = np.random.normal(0, 1, 3)
        direction = direction / np.linalg.norm(direction)
        length = np.random.uniform(length_range[0], length_range[1])

        # 计算终点
        xv = direction[0] * length
        yv = direction[1] * length
        zv = direction[2] * length

        # 随机半径
        r = np.random.uniform(radius_range[0], radius_range[1])

        return Capsule(x1, y1, z1, xv, yv, zv, r)

    def random_heightfield(
        self, position_range=(-5, 5), size_range=(5, 20), height_range=(-1, 2)
    ):
        """随机生成一个HeightField对象

        Args:
            position_range: 基准位置范围 (min, max)
            size_range: 网格尺寸范围 (min, max)
            height_range: 高度范围 (min, max)

        Returns:
            HeightField: 随机生成的HeightField对象
        """
        # 随机基准位置
        x = np.random.uniform(position_range[0], position_range[1])
        y = np.random.uniform(position_range[0], position_range[1])
        z = np.random.uniform(position_range[0], position_range[1])

        # 随机尺寸
        xd = np.random.randint(size_range[0], size_range[1] + 1)
        yd = np.random.randint(size_range[0], size_range[1] + 1)

        # 随机缩放因子
        xs = np.random.uniform(0.5, 2.0)
        ys = np.random.uniform(0.5, 2.0)
        zs = np.random.uniform(0.5, 2.0)

        # 生成随机高度数据
        num_points = xd * yd
        height_data = np.random.uniform(
            height_range[0], height_range[1], num_points
        ).tolist()

        return HeightField(x, y, z, xs, ys, zs, xd, yd, height_data)

    def random_triangle(self, position_range=(-10, 10), size_range=(0.5, 3.0)):
        """随机生成一个Triangle对象

        Args:
            position_range: 顶点位置范围 (min, max)
            size_range: 三角形尺寸范围 (min, max)

        Returns:
            Triangle: 随机生成的Triangle对象
        """
        # 随机生成第一个顶点
        v0 = np.random.uniform(position_range[0], position_range[1], 3)

        # 随机生成另外两个顶点，确保形成有效的三角形
        size = np.random.uniform(size_range[0], size_range[1])

        # 生成两个随机向量
        vec1 = np.random.normal(0, size, 3)
        vec2 = np.random.normal(0, size, 3)

        # 确保vec1和vec2不平行（叉积不为零）
        while np.linalg.norm(np.cross(vec1, vec2)) < 1e-6:
            vec2 = np.random.normal(0, size, 3)

        v1 = v0 + vec1
        v2 = v0 + vec2

        return Triangle(tuple(v0), tuple(v1), tuple(v2))

    def generate_random_obstacles(
        self,
        num_cuboids=5,
        num_spheres=5,
        num_capsules=3,
        num_heightfields=2,
        num_triangles=5,
        position_range=(-15, 15),
    ):
        """生成随机数量的各种障碍物

        Args:
            num_cuboids: Cuboid数量
            num_spheres: Sphere数量
            num_capsules: Capsule数量
            num_heightfields: HeightField数量
            num_triangles: Triangle数量
            position_range: 位置范围

        Returns:
            dict: 包含所有障碍物的字典
        """
        obstacles = {
            "cuboids": [],
            "spheres": [],
            "capsules": [],
            "heightfields": [],
            "triangles": [],
        }

        # 生成Cuboid障碍物
        for i in range(num_cuboids):
            obstacles["cuboids"].append(self.random_cuboid(position_range))

        # 生成Sphere障碍物
        for i in range(num_spheres):
            obstacles["spheres"].append(self.random_sphere(position_range))

        # 生成Capsule障碍物
        for i in range(num_capsules):
            obstacles["capsules"].append(self.random_capsule(position_range))

        # 生成HeightField障碍物
        for i in range(num_heightfields):
            obstacles["heightfields"].append(self.random_heightfield(position_range))

        # 生成Triangle障碍物
        for i in range(num_triangles):
            obstacles["triangles"].append(self.random_triangle(position_range))

        print(
            f"生成了 {len(obstacles['cuboids'])} 个Cuboid, "
            f"{len(obstacles['spheres'])} 个Sphere, "
            f"{len(obstacles['capsules'])} 个Capsule, "
            f"{len(obstacles['heightfields'])} 个HeightField, "
            f"{len(obstacles['triangles'])} 个Triangle"
        )

        return obstacles

    def cuboid_vs_obstacles(self, cuboid, obstacles):
        """OBB与指定障碍物集合进行碰撞检测

        Args:
            cuboid: Cuboid对象
            obstacles: 障碍物字典

        Returns:
            dict: 碰撞检测结果
        """
        results = {"total_obstacles": 0, "collisions": 0, "collision_details": []}

        # 检查与Sphere的碰撞
        for i, sphere in enumerate(obstacles["spheres"]):
            collision = cuboid_sphere(cuboid, sphere)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Sphere {i} 碰撞")

        # 检查与Capsule的碰撞
        for i, capsule in enumerate(obstacles["capsules"]):
            collision = cuboid_capsule(cuboid, capsule)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Capsule {i} 碰撞")

        # 检查与Cuboid的碰撞
        for i, other_cuboid in enumerate(obstacles["cuboids"]):
            collision = cuboid_cuboid(cuboid, other_cuboid)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Cuboid {i} 碰撞")

        # 检查与Triangle的碰撞
        for i, triangle in enumerate(obstacles["triangles"]):
            collision = cuboid_triangle(cuboid, triangle)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Triangle {i} 碰撞")

        # 检查与HeightField的碰撞
        for i, heightfield in enumerate(obstacles["heightfields"]):
            collision = cuboid_heightfield(cuboid, heightfield)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与HeightField {i} 碰撞")

        return results

    def sphere_vs_obstacles(self, sphere, obstacles):
        """Sphere与指定障碍物集合进行碰撞检测

        Args:
            sphere: Sphere对象
            obstacles: 障碍物字典

        Returns:
            dict: 碰撞检测结果
        """
        results = {"total_obstacles": 0, "collisions": 0, "collision_details": []}

        # 检查与Sphere的碰撞
        for i, other_sphere in enumerate(obstacles["spheres"]):
            collision = sphere_sphere(sphere, other_sphere)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Sphere {i} 碰撞")

        # 检查与Capsule的碰撞
        for i, capsule in enumerate(obstacles["capsules"]):
            collision = sphere_capsule(capsule, sphere)  # 注意参数顺序
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Capsule {i} 碰撞")

        # 检查与Cuboid的碰撞
        for i, cuboid in enumerate(obstacles["cuboids"]):
            collision = sphere_cuboid(cuboid, sphere)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Cuboid {i} 碰撞")

        # 检查与Triangle的碰撞
        for i, triangle in enumerate(obstacles["triangles"]):
            collision = sphere_triangle(sphere, triangle)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与Triangle {i} 碰撞")

        # 检查与HeightField的碰撞
        for i, heightfield in enumerate(obstacles["heightfields"]):
            collision = sphere_heightfield(heightfield, sphere)
            results["total_obstacles"] += 1
            if collision == 0:  # 碰撞
                results["collisions"] += 1
                results["collision_details"].append(f"与HeightField {i} 碰撞")

        return results

    def create_test_objects(self):
        """创建测试用的几何对象"""
        objects = {}

        # 创建Cuboid对象
        objects["cuboid1"] = Cuboid(
            0, 0, 0, (1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)
        )
        objects["cuboid2"] = Cuboid(
            2, 0, 0, (1, 0, 0, 0.3), (0, 1, 0, 0.3), (0, 0, 1, 0.3)
        )

        # 创建Sphere对象
        objects["sphere1"] = Sphere(0, 0, 0, 0.5)
        objects["sphere2"] = Sphere(1.5, 0, 0, 0.3)

        # 创建Capsule对象
        objects["capsule"] = Capsule(0, 2, 0, 0, 0, 1, 0.2)

        # 创建Triangle对象
        objects["triangle"] = Triangle((1, 0, 0), (1.5, 1, 0), (2, 0, 0))

        # 创建HeightField对象
        height_data = [0.0] * 100  # 10x10 平面
        objects["heightfield"] = HeightField(
            0, 0, 0, 1.0, 1.0, 1.0, 10, 10, height_data
        )

        return objects

    def benchmark_single_collision(self, func_name, func, obj1, obj2, num_tests=10000):
        """测试单个碰撞检测函数的性能"""
        result = 1
        print(f"测试 {func_name} ({num_tests} 次)...")

        # 预热运行，避免首次调用的额外开销
        for _ in range(min(100, num_tests // 10)):
            func(obj1, obj2)

        # 正式测试
        start_time = time.perf_counter()
        for _ in range(num_tests):
            result = func(obj1, obj2)
        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        avg_time = total_time / num_tests

        return {
            "function": func_name,
            "total_time_ms": total_time,
            "avg_time_us": avg_time * 1000,  # 转换为微秒
            "avg_time_ms": avg_time,
            "num_tests": num_tests,
            "last_result": result,
        }

    def run_full_benchmark(self, num_tests=10000):
        """运行完整的碰撞检测基准测试"""
        print("几何碰撞检测性能基准测试")
        print("=" * 60)
        print(f"每次碰撞检测函数执行 {num_tests} 次测试")
        print()

        objects = self.create_test_objects()
        results = []

        # 定义测试用例：(函数名, 对象1, 对象2)
        test_cases = [
            ("cuboid_sphere", objects["cuboid1"], objects["sphere1"]),
            ("cuboid_capsule", objects["cuboid1"], objects["capsule"]),
            ("cuboid_cuboid", objects["cuboid1"], objects["cuboid2"]),
            ("cuboid_triangle", objects["cuboid1"], objects["triangle"]),
            ("cuboid_heightfield", objects["cuboid1"], objects["heightfield"]),
            ("sphere_sphere", objects["sphere1"], objects["sphere2"]),
            ("sphere_triangle", objects["sphere1"], objects["triangle"]),
            (
                "sphere_capsule",
                objects["capsule"],
                objects["sphere1"],
            ),  # capsule first, then sphere
            ("sphere_cuboid", objects["cuboid1"], objects["sphere1"]),
            ("sphere_heightfield", objects["heightfield"], objects["sphere1"]),
        ]

        for func_name, obj1, obj2 in test_cases:
            if func_name in self.collision_functions:
                result = self.benchmark_single_collision(
                    func_name,
                    self.collision_functions[func_name],
                    obj1,
                    obj2,
                    num_tests,
                )
                results.append(result)
                print(f"  平均耗时: {result['avg_time_ms']:.4f} ms")
        return results

    def analyze_results(self, results):
        """分析测试结果"""
        print("\n" + "=" * 60)
        print("性能分析结果")
        print("=" * 60)

        if not results:
            print("没有测试结果可分析")
            return

        # 按平均耗时排序
        sorted_results = sorted(results, key=lambda x: x["avg_time_us"])

        print(
            f"{'函数名':<18} {'平均耗时(μs)':<12} {'平均耗时(ms)':<12} {'测试次数':<8}"
        )
        print("-" * 60)

        for result in sorted_results:
            print(
                f"{result['function']:<18} {result['avg_time_us']:<12.4f} {result['avg_time_ms']:<12.4f} {result['num_tests']:<8}"
            )

        # 计算统计信息
        avg_times = [r["avg_time_us"] for r in results]
        min_time = min(avg_times)
        max_time = max(avg_times)
        mean_time = np.mean(avg_times)
        std_time = np.std(avg_times)

        print("\n统计信息:")
        print(f"最快函数耗时: {min_time:.2f} μs")
        print(f"最慢函数耗时: {max_time:.2f} μs")
        print(f"平均耗时: {mean_time:.2f} μs")
        print(f"标准差: {std_time:.2f} μs")

        # 性能分类
        print("\n性能分类:")
        very_fast = [r for r in results if r["avg_time_us"] < 10]
        fast = [r for r in results if 10 <= r["avg_time_us"] < 50]
        medium = [r for r in results if 50 <= r["avg_time_us"] < 200]
        slow = [r for r in results if r["avg_time_us"] >= 200]

        print(f"非常快 (< 10μs): {len(very_fast)} 个函数")
        for r in very_fast:
            print(f"  - {r['function']}")

        print(f"快 (10-50μs): {len(fast)} 个函数")
        for r in fast:
            print(f"  - {r['function']}")

        print(f"中速 (50-200μs): {len(medium)} 个函数")
        for r in medium:
            print(f"  - {r['function']}")

        print(f"慢 (>= 200μs): {len(slow)} 个函数")
        for r in slow:
            print(f"  - {r['function']}")

        return {
            "sorted_results": sorted_results,
            "statistics": {
                "min_time": min_time,
                "max_time": max_time,
                "mean_time": mean_time,
                "std_time": std_time,
            },
            "categories": {
                "very_fast": len(very_fast),
                "fast": len(fast),
                "medium": len(medium),
                "slow": len(slow),
            },
        }

    def compare_collision_types(self, results):
        """比较不同类型的碰撞检测"""
        print("\n" + "=" * 60)
        print("碰撞类型对比分析")
        print("=" * 60)

        # 按碰撞类型分组
        type_groups = {"球体相关": [], "Cuboid相关": [], "混合类型": []}

        for result in results:
            func_name = result["function"]
            if func_name.startswith("sphere"):
                type_groups["球体相关"].append(result)
            elif func_name.startswith("cuboid"):
                type_groups["Cuboid相关"].append(result)
            else:
                type_groups["混合类型"].append(result)

        for group_name, group_results in type_groups.items():
            if group_results:
                print(f"\n{group_name}:")
                group_avg = np.mean([r["avg_time_us"] for r in group_results])
                print(f"  平均耗时: {group_avg:.2f} μs")
                for result in sorted(group_results, key=lambda x: x["avg_time_us"]):
                    print(f"    {result['function']}: {result['avg_time_us']:.2f} μs")

    def run_stress_test(self, num_iterations=5, tests_per_iter=50000):
        """运行压力测试，验证性能稳定性"""
        print(f"\n运行压力测试: {num_iterations} 轮，每轮 {tests_per_iter} 次检测")
        print("-" * 60)

        objects = self.create_test_objects()
        cuboid = objects["cuboid1"]
        sphere = objects["sphere1"]

        all_times = []

        for i in range(num_iterations):
            start_time = time.perf_counter()
            for _ in range(tests_per_iter):
                cuboid_sphere(cuboid, sphere)
            end_time = time.perf_counter()

            iter_time = (end_time - start_time) * 1000  # 毫秒
            avg_time_per_test = (iter_time / tests_per_iter) * 1000  # 微秒
            all_times.append(avg_time_per_test)
            print(f"第 {i + 1} 轮: {avg_time_per_test:.4f} μs/次")

        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        cv = (std_time / mean_time) * 100 if mean_time > 0 else 0  # 变异系数

        print("\n压力测试结果:")
        print(f"平均耗时: {mean_time:.2f} μs")
        print(f"标准差: {std_time:.2f} μs")
        print(f"变异系数: {cv:.1f}%")
        return {
            "iterations": num_iterations,
            "tests_per_iter": tests_per_iter,
            "mean_time": mean_time,
            "std_time": std_time,
            "coefficient_of_variation": cv,
        }

    def run_random_objects_benchmark(
        self,
        num_test_objects=100,
        num_obstacles_cuboids=10,
        num_obstacles_spheres=10,
        num_obstacles_capsules=5,
        num_obstacles_heightfields=3,
        num_obstacles_triangles=10,
        position_range=(-20, 20),
        object_position_range=(-15, 15),
    ):
        """运行随机对象在预先生成障碍物集合中的碰撞检测性能评估

        Args:
            num_test_objects: 测试对象的数量
            num_obstacles_*: 各种障碍物的数量
            position_range: 障碍物位置范围
            object_position_range: 测试对象位置范围

        Returns:
            dict: 性能评估结果
        """
        print("\n" + "=" * 80)
        print("随机对象在预生成障碍物集合中的碰撞检测性能评估")
        print("=" * 80)
        print(f"测试对象数量: {num_test_objects}")
        print(f"障碍物位置范围: {position_range}")
        print(f"测试对象位置范围: {object_position_range}")
        print()

        # 预先生成障碍物集合
        print("预生成障碍物集合...")
        obstacles = self.generate_random_obstacles(
            num_cuboids=num_obstacles_cuboids,
            num_spheres=num_obstacles_spheres,
            num_capsules=num_obstacles_capsules,
            num_heightfields=num_obstacles_heightfields,
            num_triangles=num_obstacles_triangles,
            position_range=position_range,
        )

        total_obstacles = sum(len(v) for v in obstacles.values())
        print(f"总障碍物数量: {total_obstacles}")
        print()

        # 准备测试数据收集
        cuboid_results = []
        sphere_results = []
        cuboid_times = []
        sphere_times = []

        print(f"开始测试 {num_test_objects} 个随机对象...")
        print("-" * 80)

        for i in range(num_test_objects):
            if (i + 1) % 10 == 0:
                print(f"正在测试第 {i + 1}/{num_test_objects} 个对象...")

            # 生成随机测试对象
            test_cuboid = self.random_cuboid(
                position_range=object_position_range, size_range=(0.1, 1.0)
            )
            test_sphere = self.random_sphere(
                position_range=object_position_range, radius_range=(0.1, 0.8)
            )

            # 测试Cuboid与障碍物的碰撞检测性能
            start_time = time.perf_counter()
            cuboid_result = self.cuboid_vs_obstacles(test_cuboid, obstacles)
            end_time = time.perf_counter()
            cuboid_time = (end_time - start_time) * 1000  # 毫秒

            cuboid_results.append(cuboid_result)
            cuboid_times.append(cuboid_time)

            # 测试Sphere与障碍物的碰撞检测性能
            start_time = time.perf_counter()
            sphere_result = self.sphere_vs_obstacles(test_sphere, obstacles)
            end_time = time.perf_counter()
            sphere_time = (end_time - start_time) * 1000  # 毫秒

            sphere_results.append(sphere_result)
            sphere_times.append(sphere_time)

        print("测试完成!")
        print()

        # 分析结果
        print("性能分析结果:")
        print("-" * 80)

        # Cuboid性能统计
        cuboid_avg_time = np.mean(cuboid_times)
        cuboid_std_time = np.std(cuboid_times)
        cuboid_min_time = min(cuboid_times)
        cuboid_max_time = max(cuboid_times)

        # Sphere性能统计
        sphere_avg_time = np.mean(sphere_times)
        sphere_std_time = np.std(sphere_times)
        sphere_min_time = min(sphere_times)
        sphere_max_time = max(sphere_times)

        print("Cuboid性能统计:")
        print(f"  平均耗时: {cuboid_avg_time:.4f} ms")
        print(f"  标准差: {cuboid_std_time:.4f} ms")
        print(f"  最快: {cuboid_min_time:.4f} ms")
        print(f"  最慢: {cuboid_max_time:.4f} ms")
        print(f"  耗时范围: [{cuboid_min_time:.4f}, {cuboid_max_time:.4f}] ms")

        print("\nSphere性能统计:")
        print(f"  平均耗时: {sphere_avg_time:.4f} ms")
        print(f"  标准差: {sphere_std_time:.4f} ms")
        print(f"  最快: {sphere_min_time:.4f} ms")
        print(f"  最慢: {sphere_max_time:.4f} ms")
        print(f"  耗时范围: [{sphere_min_time:.4f}, {sphere_max_time:.4f}] ms")

        # 碰撞统计
        cuboid_collisions = [r["collisions"] for r in cuboid_results]
        sphere_collisions = [r["collisions"] for r in sphere_results]

        cuboid_collision_rate = np.mean(cuboid_collisions)
        sphere_collision_rate = np.mean(sphere_collisions)

        print("\n碰撞统计:")
        print(f"  Cuboid平均碰撞数: {cuboid_collision_rate:.2f} / {total_obstacles}")
        print(f"  Sphere平均碰撞数: {sphere_collision_rate:.2f} / {total_obstacles}")
        print(f"  Cuboid碰撞率: {cuboid_collision_rate / total_obstacles * 100:.1f}%")
        print(f"  Sphere碰撞率: {sphere_collision_rate / total_obstacles * 100:.1f}%")

        # 性能对比
        print("\n性能对比:")
        if cuboid_avg_time < sphere_avg_time:
            speedup = sphere_avg_time / cuboid_avg_time
            print(f"  Cuboid比Sphere快 {speedup:.2f} 倍")
        else:
            speedup = cuboid_avg_time / sphere_avg_time
            print(f"  Sphere比Cuboid快 {speedup:.2f} 倍")

        return {
            "configuration": {
                "num_test_objects": num_test_objects,
                "total_obstacles": total_obstacles,
                "obstacles_breakdown": {
                    "cuboids": len(obstacles["cuboids"]),
                    "spheres": len(obstacles["spheres"]),
                    "capsules": len(obstacles["capsules"]),
                    "heightfields": len(obstacles["heightfields"]),
                    "triangles": len(obstacles["triangles"]),
                },
                "position_ranges": {
                    "obstacles": position_range,
                    "test_objects": object_position_range,
                },
            },
            "cuboid_performance": {
                "avg_time_ms": cuboid_avg_time,
                "std_time_ms": cuboid_std_time,
                "min_time_ms": cuboid_min_time,
                "max_time_ms": cuboid_max_time,
                "times": cuboid_times,
            },
            "sphere_performance": {
                "avg_time_ms": sphere_avg_time,
                "std_time_ms": sphere_std_time,
                "min_time_ms": sphere_min_time,
                "max_time_ms": sphere_max_time,
                "times": sphere_times,
            },
            "collision_statistics": {
                "cuboid_avg_collisions": cuboid_collision_rate,
                "sphere_avg_collisions": sphere_collision_rate,
                "cuboid_collision_rate_percent": cuboid_collision_rate
                / total_obstacles
                * 100,
                "sphere_collision_rate_percent": sphere_collision_rate
                / total_obstacles
                * 100,
            },
            "raw_results": {
                "cuboid_results": cuboid_results,
                "sphere_results": sphere_results,
            },
        }


def main():
    """主函数"""
    print("基于几何计算的碰撞检测耗时评估")
    print("=" * 60)

    # 创建基准测试器
    benchmark = CollisionBenchmark()

    try:
        # 运行随机对象碰撞检测性能评估
        random_benchmark_results = benchmark.run_random_objects_benchmark(
            num_test_objects=5000,
            num_obstacles_cuboids=8,
            num_obstacles_spheres=8,
            num_obstacles_capsules=4,
            num_obstacles_heightfields=2,
            num_obstacles_triangles=100,
        )
        # 运行完整基准测试
        results = benchmark.run_full_benchmark(num_tests=10000)

        # 分析结果
        analysis = benchmark.analyze_results(results)

        # 碰撞类型对比
        benchmark.compare_collision_types(results)

        # 运行压力测试
        stress_results = benchmark.run_stress_test(
            num_iterations=5, tests_per_iter=50000
        )

        print("\n" + "=" * 60)
        print("评估完成!")
        print("=" * 60)
        print("基于geometric_collision_detection.py的碰撞检测性能评估已完成")
        print("所有测试均使用纯Python几何计算，无外部库依赖")
        print(f"总测试函数数量: {len(results)}")
        print(
            f"随机对象测试数量: {random_benchmark_results['configuration']['num_test_objects']}"
        )
        print(
            f"障碍物总数: {random_benchmark_results['configuration']['total_obstacles']}"
        )
        print(f"性能变异系数: {stress_results['coefficient_of_variation']:.1f}%")
        return {
            "benchmark_results": results,
            "analysis": analysis,
            "random_benchmark": random_benchmark_results,
            "stress_test": stress_results,
        }

    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
