#!/usr/bin/env python3
"""
演示通用 OBB 计算功能
生成少量样本数据来验证新的通用版本
"""

import os
import sys
import subprocess

def demo_universal_obb():
    print("=== 通用 OBB 计算系统演示 ===\n")
    
    # 检查必要文件
    if not os.path.exists("pred_trace_generation.py"):
        print("❌ 找不到 pred_trace_generation.py")
        return False
    
    if not os.path.exists("../data/robots/jaco_mod.rob"):
        print("❌ 找不到机器人模型文件")
        return False
    
    # 创建测试环境目录
    test_dir = "demo_test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"✓ 创建测试目录: {test_dir}")
    
    # 创建简单的测试环境文件
    test_env_content = '''<?xml version="1.0"?>
<world>
    <robot name="jaco" file="../data/robots/jaco_mod.rob"/>
    <terrain name="obstacle1">
        <geometry type="box" extents="0.1 0.1 0.1"/>
        <translation>0.5 0.0 0.5</translation>
    </terrain>
    <terrain name="obstacle2">
        <geometry type="box" extents="0.1 0.1 0.1"/>
        <translation>-0.5 0.0 0.5</translation>
    </terrain>
</world>'''
    
    env_file = f"{test_dir}/obstacles_0.xml"
    with open(env_file, 'w') as f:
        f.write(test_env_content)
    print(f"✓ 创建测试环境: {env_file}")
    
    # 运行数据生成 (小样本)
    print("\n🚀 开始生成演示数据...")
    cmd = [
        "python", "pred_trace_generation.py",
        "10",  # 只生成 10 个样本
        test_dir,
        "0"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ 数据生成成功！")
            print("\n生成的文件:")
            
            # 检查输出文件
            coord_file = f"{test_dir}/obstacles_0_coord.pkl"
            pose_file = f"{test_dir}/obstacles_0_pose.pkl"
            
            if os.path.exists(coord_file):
                size = os.path.getsize(coord_file)
                print(f"  📁 {coord_file} ({size} bytes)")
            
            if os.path.exists(pose_file):
                size = os.path.getsize(pose_file)
                print(f"  📁 {pose_file} ({size} bytes)")
            
            # 显示部分输出
            if result.stdout:
                print(f"\n📋 输出信息:")
                for line in result.stdout.strip().split('\n')[-5:]:  # 最后5行
                    print(f"  {line}")
                    
            return True
            
        else:
            print("❌ 数据生成失败")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 数据生成超时")
        return False
    except Exception as e:
        print(f"❌ 执行错误: {e}")
        return False

def cleanup_demo():
    """清理演示文件"""
    import shutil
    test_dir = "demo_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"🧹 清理测试目录: {test_dir}")

if __name__ == "__main__":
    print("通用机器人 OBB 计算系统")
    print("支持任意 DOF 和 link 数量的机器人模型\n")
    
    success = demo_universal_obb()
    
    if success:
        print("\n🎉 演示完成！新的通用系统工作正常。")
        print("\n主要改进:")
        print("  ✓ 支持任意机器人模型")
        print("  ✓ 自动检测 link 和 DOF 数量") 
        print("  ✓ 智能选择 OBB 计算方法")
        print("  ✓ 保持与原版相同的接口")
        
        # 询问是否清理
        response = input("\n是否清理演示文件? (y/N): ")
        if response.lower() in ['y', 'yes']:
            cleanup_demo()
            
    else:
        print("\n❌ 演示失败，请检查配置。")
        cleanup_demo()
