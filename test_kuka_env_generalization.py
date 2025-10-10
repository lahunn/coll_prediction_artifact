#!/usr/bin/env python3
"""
测试 KukaEnv 泛化修改的脚本

测试要点:
1. 构造函数是否正确接受任意 robot_file 参数
2. z_offset 参数是否正常工作
3. 变量重命名是否完整(robotId, robot_file, robotEndEffectorIndex)
4. __str__ 方法是否正确提取机器人名称
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trace_generation/bit_planning'))

from environment.kuka_env import KukaEnv

def test_initialization():
    """测试初始化和参数传递"""
    print("=" * 60)
    print("测试 1: 初始化测试")
    print("=" * 60)
    
    # 测试默认参数
    try:
        env1 = KukaEnv(GUI=False)
        print(f"✓ 默认参数初始化成功")
        print(f"  - Robot file: {env1.robot_file}")
        print(f"  - Z offset: {env1.z_offset}")
        print(f"  - Config dim: {env1.config_dim}")
        print(f"  - End effector index: {env1.robotEndEffectorIndex}")
        print(f"  - Environment name: {env1}")
    except Exception as e:
        print(f"✗ 默认参数初始化失败: {e}")
        return False
    
    # 测试自定义 z_offset
    try:
        env2 = KukaEnv(
            GUI=False,
            robot_file="kuka_iiwa/model_0.urdf",
            z_offset=-0.4
        )
        print(f"\n✓ 自定义 z_offset 初始化成功")
        print(f"  - Z offset: {env2.z_offset}")
    except Exception as e:
        print(f"\n✗ 自定义 z_offset 初始化失败: {e}")
        return False
    
    return True

def test_str_method():
    """测试 __str__ 方法是否正确提取机器人名称"""
    print("\n" + "=" * 60)
    print("测试 2: __str__ 方法测试")
    print("=" * 60)
    
    try:
        env = KukaEnv(GUI=False, robot_file="kuka_iiwa/model_0.urdf")
        env_name = str(env)
        print(f"✓ __str__ 方法成功")
        print(f"  - Robot file: {env.robot_file}")
        print(f"  - Environment name: {env_name}")
        print(f"  - Expected format: 'model_0_<config_dim>dof'")
        
        # 验证格式
        if "_" in env_name and "dof" in env_name:
            print(f"✓ 格式验证通过")
            return True
        else:
            print(f"✗ 格式验证失败: {env_name}")
            return False
    except Exception as e:
        print(f"✗ __str__ 方法测试失败: {e}")
        return False

def test_attribute_access():
    """测试变量是否正确重命名"""
    print("\n" + "=" * 60)
    print("测试 3: 属性访问测试")
    print("=" * 60)
    
    try:
        env = KukaEnv(GUI=False)
        
        # 测试新属性
        assert hasattr(env, 'robotId'), "缺少 robotId 属性"
        assert hasattr(env, 'robot_file'), "缺少 robot_file 属性"
        assert hasattr(env, 'robotEndEffectorIndex'), "缺少 robotEndEffectorIndex 属性"
        assert hasattr(env, 'z_offset'), "缺少 z_offset 属性"
        
        print(f"✓ 所有新属性存在")
        print(f"  - robotId: {env.robotId}")
        print(f"  - robot_file: {env.robot_file}")
        print(f"  - robotEndEffectorIndex: {env.robotEndEffectorIndex}")
        print(f"  - z_offset: {env.z_offset}")
        
        # 确保旧属性不存在
        assert not hasattr(env, 'kukaId'), "旧属性 kukaId 仍然存在!"
        assert not hasattr(env, 'kuka_file'), "旧属性 kuka_file 仍然存在!"
        assert not hasattr(env, 'kukaEndEffectorIndex'), "旧属性 kukaEndEffectorIndex 仍然存在!"
        
        print(f"✓ 确认旧属性已移除")
        return True
        
    except AssertionError as e:
        print(f"✗ 属性检查失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 属性访问测试失败: {e}")
        return False

def test_functionality():
    """测试基本功能是否正常"""
    print("\n" + "=" * 60)
    print("测试 4: 基本功能测试")
    print("=" * 60)
    
    try:
        env = KukaEnv(GUI=False, z_offset=-0.3)
        
        # 初始化一个问题
        env.init_new_problem(index=0)
        print(f"✓ 初始化问题成功")
        
        # 获取初始状态
        init_state = env.get_init()
        print(f"✓ 获取初始状态成功: shape={init_state.shape}")
        
        # 获取目标状态
        goal_state = env.get_goal()
        print(f"✓ 获取目标状态成功: shape={goal_state.shape}")
        
        # 测试 get_robot_points
        points = env.get_robot_points(init_state, end_point=True)
        print(f"✓ get_robot_points 成功: {points}")
        print(f"  注意: Z坐标应该考虑 z_offset={env.z_offset}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("KukaEnv 泛化修改测试")
    print("=" * 60)
    
    results = []
    
    results.append(("初始化测试", test_initialization()))
    results.append(("__str__ 方法测试", test_str_method()))
    results.append(("属性访问测试", test_attribute_access()))
    results.append(("基本功能测试", test_functionality()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit(main())
