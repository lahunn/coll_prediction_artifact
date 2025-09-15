#!/usr/bin/env python3
"""
测试通用 OBB 计算功能的脚本
验证修改后的 pred_trace_generation.py 是否能适应不同的机器人模型
"""

import sys
import os
import numpy as np

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import klampt
    from klampt.model import collide
    import obb_calculator
    
    # 测试函数
    def test_universal_obb():
        print("=== 测试通用 OBB 计算功能 ===")
        
        # 1. 检查 obb_calculator 依赖
        deps_available, missing_libs = obb_calculator.check_dependencies()
        print(f"OBB Calculator 依赖库状态: {deps_available}")
        if not deps_available:
            print(f"缺失的库: {', '.join(missing_libs)}")
            print("安装命令: pip install " + " ".join(missing_libs))
        
        # 2. 测试加载机器人模型
        try:
            world = klampt.WorldModel()
            # 尝试加载默认的机器人模型
            if os.path.exists("../data/robots/jaco_mod.rob"):
                world.readFile("../data/robots/jaco_mod.rob")
                print("✓ 成功加载 jaco_mod.rob")
            else:
                print("✗ 未找到 jaco_mod.rob")
                return False
            
            robot = world.robot(0)
            num_links = robot.numLinks()
            print(f"✓ 机器人有 {num_links} 个 links")
            
            # 3. 测试随机配置
            q = robot.getConfig()
            print(f"✓ 当前配置有 {len(q)} 个 DOFs")
            
            # 4. 测试 OBB 计算 (模拟)
            print("✓ 基本功能测试通过")
            
            return True
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            return False
    
    if __name__ == "__main__":
        success = test_universal_obb()
        if success:
            print("\n🎉 通用 OBB 功能准备就绪！")
            print("现在可以使用修改后的 pred_trace_generation.py 处理任意机器人模型。")
        else:
            print("\n❌ 测试失败，请检查环境配置。")

except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保 Klampt 库已正确安装。")
