"""
机器人球体近似碰撞检测数据生成脚本
用于生成基于球体几何近似的微架构仿真训练/测试数据集

主要功能：
1. 在给定障碍物环境中随机采样机器人姿态
2. 计算每个姿态下各个 link 的球体近似几何信息
3. 执行基于球体的精确碰撞检测，生成 ground truth 标签
4. 输出球体级和 pose 级的碰撞检测数据供后续仿真使用

输入参数：
- sys.argv[1]: numqueries (采样姿态数量)
- sys.argv[2]: foldername (环境文件夹)
- sys.argv[3]: filenumber (环境文件编号)

输出文件：
- obstacles_X_sphere.pkl: 球体级数据 (qarr_sphere, yarr_sphere, radius_arr)
- obstacles_X_pose.pkl: pose 级数据 (qarr_pose, yarr_pose)
"""

import klampt
from klampt.plan import robotplanning
from klampt.model import collide
import klampt.model
import sys
from klampt import vis
import pickle
import numpy as np
import math
from klampt.model.create import primitives
import yaml

# ========== 几何变换辅助函数 (与原程序相同) ==========


def give_RT(x):
    """
    从 4x4 变换矩阵中提取旋转矩阵 R 和平移向量 T

    Args:
        x: 4x4 齐次变换矩阵
    Returns:
        R: 展平的 3x3 旋转矩阵 (9个元素的列表)
        T: 3x1 平移向量 (3个元素的列表)
    """
    R = []
    T = []
    for j in range(0, 3):
        for i in range(0, 3):
            R.append(x[i][j])  # 按列优先顺序展平旋转矩阵
        T.append(x[j][3])  # 提取平移分量
    return (R, T)


def get_obbRT(R, T):
    """
    将展平的旋转矩阵和平移向量重构为标准的 numpy 数组格式

    Args:
        R: 展平的旋转矩阵 (9个元素)
        T: 平移向量 (3个元素)
    Returns:
        newR: 3x3 numpy 旋转矩阵
        newT: 3x1 numpy 平移向量
    """
    newR = np.eye(3)
    newT = np.array(T)
    pointer = 0
    for j in range(0, 3):
        for i in range(0, 3):
            newR[i][j] = R[pointer]  # 重构 3x3 旋转矩阵
            pointer += 1
    return newR, newT


def give_dh(d, r, th, al):
    """
    根据 DH (Denavit-Hartenberg) 参数构建齐次变换矩阵
    用于机器人运动学正向计算

    Args:
        d: 连杆偏移 (link offset)
        r: 连杆长度 (link length)
        th: 关节角 (joint angle)
        al: 连杆扭转角 (link twist)
    Returns:
        new: 4x4 DH 变换矩阵
    """
    new = np.eye(4)
    # DH 变换矩阵的标准公式
    new[0, 0] = math.cos(th)
    new[0, 1] = -1 * math.cos(al) * math.sin(th)
    new[0, 2] = math.sin(al) * math.sin(th)
    new[0, 3] = r * math.cos(th)

    new[1, 0] = math.sin(th)
    new[1, 1] = 1 * math.cos(al) * math.cos(th)
    new[1, 2] = -1 * math.sin(al) * math.cos(th)
    new[1, 3] = r * math.sin(th)

    new[2, 0] = 0
    new[2, 1] = math.sin(al)
    new[2, 2] = math.cos(al)
    new[2, 3] = d
    return new


def transform_point(p, R, T):
    """
    应用旋转和平移变换到给定点

    Args:
        p: 输入点坐标 (3D 向量)
        R: 3x3 旋转矩阵
        T: 3x1 平移向量
    Returns:
        newT: 变换后的点坐标
    """
    new = np.zeros((3, 1))
    new[:, 0] = p
    newT = np.array(T)
    temp = np.matmul(R, new)
    newT[0] += temp[0, 0]
    newT[1] += temp[1, 0]
    newT[2] += temp[2, 0]
    return newT


# ========== 球体配置加载函数 ==========


def load_sphere_config(config_path = "../configs/robot/spheres/iiwa.yml"):
    """
    从 YAML 配置文件加载球体近似信息

    Returns:
        sphere_config: 字典，包含每个 link 的球体定义
    """
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["collision_spheres"]
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default sphere config.")
        # 默认球体配置（基于 YAML 文件内容）
        return {
            "iiwa7_link_0": [{"center": [0.0, 0.0, 0.05], "radius": 0.10}],
            "iiwa7_link_1": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.08},
                {"center": [0.0, -0.05, 0.1], "radius": 0.07},
                {"center": [0.0, -0.05, 0.18], "radius": 0.08},
            ],
            "iiwa7_link_2": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.08},
                {"center": [0.0, 0.02, 0.06], "radius": 0.07},
                {"center": [0.0, 0.1, 0.03], "radius": 0.07},
                {"center": [0.0, 0.18, 0.0], "radius": 0.08},
            ],
            "iiwa7_link_3": [
                {"center": [0.0, 0.0, 0.08], "radius": 0.08},
                {"center": [0.0, 0.06, 0.16], "radius": 0.07},
                {"center": [0.0, 0.05, 0.22], "radius": 0.07},
            ],
            "iiwa7_link_4": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.08},
                {"center": [0.0, 0.0, 0.05], "radius": 0.07},
                {"center": [0.0, 0.07, 0.05], "radius": 0.06},
                {"center": [0.0, 0.11, 0.03], "radius": 0.06},
                {"center": [0.0, 0.15, 0.01], "radius": 0.07},
            ],
            "iiwa7_link_5": [
                {"center": [0.0, 0.0, 0.02], "radius": 0.08},
                {"center": [0.0, 0.03, 0.07], "radius": 0.06},
                {"center": [0.0, 0.08, 0.13], "radius": 0.05},
            ],
            "iiwa7_link_6": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.06},
                {"center": [0.0, 0.0, 0.05], "radius": 0.08},
                {"center": [0.0, -0.04, 0.075], "radius": 0.06},
                {"center": [0.0, 0.08, 0.06], "radius": 0.065},
                {"center": [0.0, 0.16, 0.06], "radius": 0.05},
            ],
        }


# ========== 核心球体计算函数 ==========


def get_spheres(world, qbase, sphere_config):
    """
    计算给定关节配置下所有 links 的球体近似信息
    这是整个球体数据生成的核心函数，为每个 link 的每个球体计算世界坐标位置

    Args:
        world: Klampt 世界模型
        qbase: 7-DOF 关节角配置 [q0, q1, q2, q3, q4, q5, q6]
        sphere_config: 球体配置字典

    Returns:
        sphere_list: 包含所有球体信息的列表
                    每个元素为 [link_id, sphere_id, 世界坐标中心, 半径]
    """

    # ========== 通过 DH 参数计算各 link 的变换矩阵 (与原程序相同) ==========
    # 基于 KUKA iiwa 机器人的 DH 参数
    dh01 = give_dh(0.1535, 0, 0, 0)  # base -> link1
    dh1e = give_dh(0.08, 0, qbase[1], 0)  # link1 末端调整

    dh12 = give_dh(0.1185, 0, qbase[1], -1 * math.pi / 2)  # link1 -> link2
    dh2e = give_dh(-0.029, 0.206, qbase[2], 0)  # link2 末端调整

    dh23 = give_dh(0.0, 0.41, qbase[2], 0)  # link2 -> link3
    dh3e = give_dh(
        0.014, -0.08, -1 * math.pi / 2 + qbase[3], -1 * math.pi / 2
    )  # link3 末端

    dh34 = give_dh(0.01125, 0.0, qbase[3], -1 * math.pi / 2)  # link3 -> link4
    # link4 的复杂变换链
    dh4e = give_dh(0.207, 0.0, qbase[4], 0)
    dh4ed = np.matmul(
        give_dh(0.05, 0.0, 0, 0.61), give_dh(0.0, 0.014, -1 * math.pi / 2, 0.0)
    )
    dh45 = give_dh(0.207 + 0.0658, 0.00, qbase[4], math.pi + 0.9599)  # link4 -> link5

    # link5 变换
    dh5e = give_dh(-0.028, 0.01968, (math.pi / 2), 0.0)
    dh5ed = give_dh(-0.050, 0.01, -qbase[5], 0.0)

    # link6 变换
    dh56 = np.matmul(
        give_dh(-0.028, 0.01968, (math.pi / 2), 0.0), give_dh(0, 0, -qbase[5], 0)
    )
    dh6e = np.matmul(
        give_dh(0, 0, -1 * math.pi / 2, 0.9599),
        give_dh(-0.0658, -0.0343, math.pi / 2, 0),
    )
    dh6ed = give_dh(-0.055, 0, -qbase[6], 0)

    # ========== 累积变换：从基座到各 link 的世界坐标变换 ==========
    fin1e = np.matmul(dh01, dh1e)
    fin2e = np.matmul(dh01, np.matmul(dh12, dh2e))
    fin3e = np.matmul(dh01, np.matmul(dh12, np.matmul(dh23, dh3e)))
    fin4e = np.matmul(
        dh01, np.matmul(dh12, np.matmul(dh23, np.matmul(dh34, np.matmul(dh4e, dh4ed))))
    )
    fin5e = np.matmul(
        dh01,
        np.matmul(
            dh12,
            np.matmul(dh23, np.matmul(dh34, np.matmul(dh45, np.matmul(dh5e, (dh5ed))))),
        ),
    )
    fin6e = np.matmul(
        dh01,
        np.matmul(
            dh12,
            np.matmul(
                dh23,
                np.matmul(
                    dh34, np.matmul(dh45, np.matmul(dh56, np.matmul(dh6e, dh6ed)))
                ),
            ),
        ),
    )

    # ========== link 变换矩阵列表 ==========
    # link0 (base) 使用单位变换
    finlist = [np.eye(4), fin1e, fin2e, fin3e, fin4e, fin5e, fin6e]
    link_names = [
        "iiwa7_link_0",
        "iiwa7_link_1",
        "iiwa7_link_2",
        "iiwa7_link_3",
        "iiwa7_link_4",
        "iiwa7_link_5",
        "iiwa7_link_6",
    ]

    sphere_list = []

    # ========== 处理每个 link 的球体 ==========
    for link_id, (link_name, transform_matrix) in enumerate(zip(link_names, finlist)):
        if link_name not in sphere_config:
            continue  # 跳过没有球体定义的 link

        # 从变换矩阵提取旋转和平移
        R, T = give_RT(transform_matrix)
        obbr, link_origin = get_obbRT(R, T)

        # 处理该 link 的所有球体
        spheres = sphere_config[link_name]
        for sphere_id, sphere_def in enumerate(spheres):
            # 球体在 link 坐标系中的中心位置
            local_center = np.array(sphere_def["center"])
            radius = sphere_def["radius"]

            # 将球体中心变换到世界坐标系
            world_center = transform_point(local_center, obbr, link_origin)

            # 存储球体信息: [link_id, sphere_id, 世界坐标中心, 半径]
            sphere_list.append([link_id, sphere_id, world_center, radius])

    return sphere_list


# ========== 球体碰撞检测函数 ==========


def check_sphere_collision(
    sphere_center, sphere_radius, world, robot_index, num_obstacles
):
    """
    检查单个球体与环境障碍物的碰撞

    Args:
        sphere_center: 球体中心在世界坐标系的位置
        sphere_radius: 球体半径
        world: Klampt 世界模型
        robot_index: 机器人索引
        num_obstacles: 障碍物数量

    Returns:
        collision: 是否发生碰撞 (True=碰撞, False=自由)
    """
    # 创建球体几何体
    sphere_geom = primitives.sphere(sphere_radius)

    # 设置球体变换矩阵 - Klampt 使用 SO3 + 平移向量格式
    # SO3 旋转矩阵 (单位矩阵，因为球体无方向)
    rotation = [
        1,
        0,
        0,  # 第一行
        0,
        1,
        0,  # 第二行
        0,
        0,
        1,
    ]  # 第三行

    # 平移向量
    translation = sphere_center.flatten().tolist()

    # 设置变换
    sphere_geom.setCurrentTransform(rotation, translation)

    # 检查与所有障碍物的碰撞
    for obstacle_id in range(num_obstacles):
        obstacle = world.terrain(obstacle_id)
        obstacle_geom = obstacle.geometry()

        # 执行几何体间的碰撞检测
        if sphere_geom.collides(obstacle_geom):
            return True  # 发现碰撞

    return False  # 无碰撞


# ========== 主程序：球体数据生成流程 ==========


def main():
    # 解析命令行参数
    if len(sys.argv) != 4:
        print(
            "Usage: python sphere_trace_generation.py <numqueries> <foldername> <filenumber>"
        )
        sys.exit(1)

    numqueries = int(sys.argv[1])  # 需要采样的姿态数量
    foldername = sys.argv[2]  # 环境文件夹路径
    filenumber = sys.argv[3]  # 环境文件编号

    # ========== 环境和机器人初始化 ==========
    # 加载包含障碍物的环境
    world = klampt.WorldModel()
    world.readFile(foldername + "/obstacles_" + filenumber + ".xml")
    print("Loaded environment:", foldername + "/obstacles_" + filenumber + ".xml")

    # 加载无障碍物的机器人模型 (用于可行性检查)
    world1 = klampt.WorldModel()
    world1.readFile("jaco_collision.xml")

    # 初始化碰撞检测器
    collider_w = collide.WorldCollider(world)
    num_ob = collider_w.world.numTerrains()  # 障碍物数量
    print(f"Number of obstacles: {num_ob}")

    # 获取机器人对象
    robot = world.robot(0)  # 带障碍物环境中的机器人
    robot1 = world1.robot(0)  # 无障碍物环境中的机器人 (用于采样)
    qbase = robot.getConfig()  # 初始关节配置

    # 设置随机种子确保可重现性
    klampt.plan.motionplanning.setRandomSeed(2)

    # 创建配置空间用于可行性采样
    space = robotplanning.makeSpace(world1, robot1, edgeCheckResolution=0.005)

    # ========== 加载球体配置 ==========
    sphere_config = load_sphere_config()
    print("Loaded sphere configuration")

    # 计算总球体数量
    total_spheres = 0
    for link_name, spheres in sphere_config.items():
        if "iiwa7_link_" in link_name:  # 只计算 iiwa 的 links
            total_spheres += len(spheres)
    print(f"Total spheres per pose: {total_spheres}")

    # ========== 数据存储数组初始化 ==========
    # 球体级数据数组 (总共 total_spheres*numqueries 行)
    qarr_sphere = []  # 球体中心坐标: [x, y, z] (动态列表)
    yarr_sphere = []  # 碰撞标签: 1=自由, 0=碰撞 (动态列表)
    radius_arr = []  # 球体半径 (动态列表)
    link_id_arr = []  # 所属 link ID (动态列表)
    sphere_id_arr = []  # 球体 ID (动态列表)

    # pose 级数据数组 (总共 numqueries 行)
    qarr_pose = np.zeros((numqueries, 7))  # 关节配置: [q0, q1, ..., q6]
    yarr_pose = np.zeros((numqueries, 1))  # 整体碰撞标签: 1=整个机器人自由, 0=有碰撞

    # ========== 主要数据生成循环 ==========
    counter = 0  # 成功采样的姿态计数器
    coll_count = 0  # 发生碰撞的姿态计数器

    print(f"Starting data generation for {numqueries} poses...")

    while counter < numqueries:
        if counter % 100 == 0:
            print(f"Progress: {counter}/{numqueries} poses processed")

        # ========== 采样可行的机器人配置 ==========
        attempts = 0
        spheres = []

        while attempts < 1000:  # 避免无限循环
            q = space.sample()  # 随机采样关节配置
            if space.isFeasible(q):  # 检查配置是否在关节限制内且无自碰撞
                robot.setConfig(q)  # 设置机器人到采样配置
                spheres = get_spheres(world, q, sphere_config)  # 计算所有球体信息
                break
            attempts += 1

        if attempts >= 1000:
            print("Warning: Could not find feasible configuration after 1000 attempts")
            continue

        # ========== 逐球体碰撞检测 ==========
        pose_has_collision = False

        for sphere_info in spheres:
            link_id, sphere_id, sphere_center, radius = sphere_info

            # 执行球体碰撞检测
            collision = check_sphere_collision(
                sphere_center, radius, world, robot.index, num_ob
            )

            # 存储球体数据
            qarr_sphere.append(sphere_center.flatten())  # 球体中心坐标
            yarr_sphere.append(0 if collision else 1)  # 碰撞标签 (0=碰撞, 1=自由)
            radius_arr.append(radius)  # 球体半径
            link_id_arr.append(link_id)  # 所属 link ID
            sphere_id_arr.append(sphere_id)  # 球体 ID

            if collision:
                pose_has_collision = True

        # ========== 存储姿态级数据 ==========
        # 对某些关节角进行偏移调整 (与原程序保持一致)
        q_adjusted = q.copy()
        q_adjusted[2] += 3.97935067
        q_adjusted[3] += 4.41568301

        qarr_pose[counter] = q_adjusted  # 存储调整后的关节配置
        yarr_pose[counter] = 0 if pose_has_collision else 1  # 存储整体碰撞标签

        if pose_has_collision:
            coll_count += 1

        counter += 1

    # ========== 转换为 numpy 数组 ==========
    qarr_sphere = np.array(qarr_sphere)
    yarr_sphere = np.array(yarr_sphere).reshape(-1, 1)
    radius_arr = np.array(radius_arr)
    link_id_arr = np.array(link_id_arr)
    sphere_id_arr = np.array(sphere_id_arr)

    # ========== 结果输出和数据保存 ==========
    print("Data generation completed!")
    print(f"Collision count: {coll_count} out of {numqueries} poses")
    print(f"Total spheres generated: {len(qarr_sphere)}")
    print(
        f"Sphere collision rate: {np.sum(yarr_sphere == 0) / len(yarr_sphere) * 100:.2f}%"
    )

    # 保存球体级数据到 sphere.pkl
    sphere_data = (qarr_sphere, yarr_sphere, radius_arr, link_id_arr, sphere_id_arr)
    sphere_filename = foldername + "/obstacles_" + filenumber + "_sphere.pkl"
    with open(sphere_filename, "wb") as f:
        pickle.dump(sphere_data, f)
    print(f"Saved sphere data to: {sphere_filename}")

    # 保存 pose 级数据到 pose.pkl
    pose_data = (qarr_pose, yarr_pose)
    pose_filename = foldername + "/obstacles_" + filenumber + "_pose.pkl"
    with open(pose_filename, "wb") as f:
        pickle.dump(pose_data, f)
    print(f"Saved pose data to: {pose_filename}")

    # 可选：可视化调试
    vis.debug(world)


if __name__ == "__main__":
    main()
