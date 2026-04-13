"""
碰撞检测分析参数统一管理模块

该模块统一管理OBB和Sphere碰撞检测的关键参数:
- obb_num: OBB包围盒的数量
- obb_cost: 单个OBB碰撞检测的计算成本
- sphere_num: 球体的数量
- sphere_cost: 单个球体碰撞检测的计算成本

这些参数在整个项目中被多个模块使用，统一管理便于维护和调整
"""

# 默认碰撞检测参数（用于向后兼容）
obb_num = 8  # 默认OBB包围盒数量
obb_cost = 45  # OBB碰撞检测成本
sphere_num = 22  # 默认球体数量
sphere_cost = 45  # 球体碰撞检测成本

# 各机器人的OBB数量（通常等于Link数量）
ROBOT_OBB_NUM = {
    "franka": 11,  # Franka Panda - 7自由度机械臂
    "ur5e": 11,  # UR5e - 6自由度机械臂
    "iiwa": 8,  # KUKA iiwa - 7自由度机械臂
    "kinova_gen3": 24,  # Kinova Gen3 - 7自由度机械臂（含手爪）
    "ur10e": 11,  # UR10e - 6自由度机械臂
    "jaco7": 15,  # Kinova Jaco - 7自由度机械臂
    "iiwa_allegro": 30,  # KUKA iiwa + Allegro手 - 7+16自由度
}

# 各机器人的Sphere数量（基于CuRobo球体模型）
ROBOT_SPHERE_NUM = {
    "franka": 61,  # Franka Panda球体近似
    "ur5e": 28,  # UR5e球体近似
    "iiwa": 22,  # KUKA iiwa球体近似
    "kinova_gen3": 24,  # Kinova Gen3球体近似
    "ur10e": 28,  # UR10e球体近似（与UR5e相似）
    "jaco7": 32,  # Kinova Jaco球体近似（估计）
}


def get_robot_params(robot_name: str):
    """
    获取指定机器人的碰撞检测参数

    Args:
        robot_name: 机器人名称

    Returns:
        dict: 包含 obb_num, sphere_num, obb_cost, sphere_cost 的字典
    """
    return {
        "obb_num": ROBOT_OBB_NUM.get(robot_name, obb_num),
        "sphere_num": ROBOT_SPHERE_NUM.get(robot_name, sphere_num),
        "obb_cost": obb_cost,
        "sphere_cost": sphere_cost,
    }
