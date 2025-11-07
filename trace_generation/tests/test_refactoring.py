#!/usr/bin/env python3
"""
重构后的功能验证测试

测试新的目录结构和导入路径是否正常工作。
"""

import sys
import os

# 添加trace_generation到路径

def test_backward_compatibility():
    """测试向后兼容性 - 旧导入仍然可用"""
    print("=" * 60)
    print("测试 1: 向后兼容性")
    print("=" * 60)
    
    try:
        # 测试geo_collision兼容层
        from geo_collision.geometric_collision_detection import Sphere, AABB, sphere_aabb
        print("✓ geo_collision 兼容层导入成功")
        
        # 测试robot_as兼容层
        from robot_as import RobotEnv, robot_urdf_mapping
        print("✓ robot_as 兼容层导入成功")
        
        # 测试sphere_as兼容层
        from sphere_as import SphereEnvGeometric
        print("✓ sphere_as 兼容层导入成功")
        
        # 测试workspace_bound兼容层
        from workspace_bound import WorkspaceAnalyzer
        print("✓ workspace_bound 兼容层导入成功")
        
        print("\n✅ 所有向后兼容测试通过！\n")
        return True
    except Exception as e:
        print(f"\n❌ 向后兼容测试失败: {e}\n")
        return False

def test_new_imports():
    """测试新的导入路径"""
    print("=" * 60)
    print("测试 2: 新导入路径")
    print("=" * 60)
    
    try:
        # 测试core.collision
        from core.collision.geometric_collision_detection import Sphere, AABB, sphere_aabb
        print("✓ core.collision.geometric_collision_detection 导入成功")
        
        # 测试core.robot
        from core.robot.environment import RobotEnv, robot_urdf_mapping
        print("✓ core.robot.environment 导入成功")
        
        # 测试core.collision.sphere_detector  
        from core.collision.sphere_detector import SphereEnvGeometric
        print("✓ core.collision.sphere_detector 导入成功")
        
        # 测试data.workspace_bounds
        from data.workspace_bounds.workspace_analyzer import WorkspaceAnalyzer
        print("✓ data.workspace_bounds.workspace_analyzer 导入成功")
        
        print("\n✅ 所有新导入测试通过！\n")
        return True
    except Exception as e:
        print(f"\n❌ 新导入测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_collision_detection():
    """测试碰撞检测功能"""
    print("=" * 60)
    print("测试 3: 碰撞检测功能")
    print("=" * 60)
    
    try:
        from core.collision.geometric_collision_detection import Sphere, AABB, sphere_aabb
        
        # 创建测试对象
        sphere = Sphere(0, 0, 0, 1)
        aabb = AABB(-2, -2, -2, 2, 2, 2)
        
        # 测试碰撞检测
        result, cycles = sphere_aabb(sphere, aabb)
        print(f"  sphere_aabb 测试: result={result}, cycles={cycles}")
        
        if result == 0:  # 球在AABB内，应该碰撞
            print("✓ 碰撞检测逻辑正确")
        else:
            print("✗ 碰撞检测逻辑可能有误")
            return False
        
        print("\n✅ 碰撞检测功能测试通过！\n")
        return True
    except Exception as e:
        print(f"\n❌ 碰撞检测测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_cpp_extension():
    """测试C++扩展"""
    print("=" * 60)
    print("测试 4: C++扩展")
    print("=" * 60)
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core/collision/cpp_collision'))
        import cpp_collision
        
        # 创建C++对象
        cpp_sphere = cpp_collision.Sphere(0, 0, 0, 1)
        cpp_aabb = cpp_collision.AABB(-2, -2, -2, 2, 2, 2)
        
        # 测试C++碰撞检测
        result, cycles = cpp_collision.sphere_aabb(cpp_sphere, cpp_aabb)
        print(f"  C++ sphere_aabb 测试: result={result}, cycles={cycles}")
        
        if result == 0:
            print("✓ C++扩展工作正常")
        else:
            print("✗ C++扩展逻辑可能有误")
            return False
        
        print("\n✅ C++扩展测试通过！\n")
        return True
    except ImportError as e:
        print(f"⚠️  C++扩展未编译或路径不正确: {e}")
        print("   提示: 请先编译C++扩展 (cd core/collision/cpp_collision && bash build.sh)")
        print("\n")
        return None  # None表示跳过
    except Exception as e:
        print(f"\n❌ C++扩展测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "     trace_generation 重构验证测试".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    results = []
    
    # 运行测试
    results.append(("向后兼容性", test_backward_compatibility()))
    results.append(("新导入路径", test_new_imports()))
    results.append(("碰撞检测功能", test_collision_detection()))
    results.append(("C++扩展", test_cpp_extension()))
    
    # 汇总结果
    print("=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            print(f"✅ {name}: 通过")
            passed += 1
        elif result is False:
            print(f"❌ {name}: 失败")
            failed += 1
        else:
            print(f"⚠️  {name}: 跳过")
            skipped += 1
    
    print("=" * 60)
    print(f"总计: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("=" * 60)
    
    if failed == 0 and passed > 0:
        print("\n🎉 所有测试通过！重构成功！\n")
        return 0
    elif failed > 0:
        print("\n⚠️  部分测试失败，请检查相关模块\n")
        return 1
    else:
        print("\n⚠️  没有测试被执行\n")
        return 2

if __name__ == "__main__":
    exit(main())
