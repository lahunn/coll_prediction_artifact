#!/usr/bin/env python3
"""
测试转换后的XML文件能否在Klampt中正常工作
"""

import sys
import os
from pathlib import Path

try:
    import klampt
    from klampt import WorldModel, RobotModel
    print("✅ Klampt导入成功")
except ImportError as e:
    print(f"❌ 导入Klampt失败: {e}")
    sys.exit(1)

def test_converted_xml():
    """测试转换后的XML文件"""
    print("\n🔄 测试转换后的XML文件...")
    
    # 测试转换后的XML文件
    test_files = [
        "/home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/scene_benchmarks_urdf/dens6/obstacles_0.xml",
        "/home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/scene_benchmarks_urdf/dens9/obstacles_0.xml",
        "/home/lanh/project/robot_sim/coll_prediction_artifact/trace_generation/scene_benchmarks_urdf/dens12/obstacles_0.xml"
    ]
    
    results = []
    
    for xml_file in test_files:
        if not Path(xml_file).exists():
            print(f"⚠️  文件不存在: {xml_file}")
            continue
            
        print(f"\n📄 测试文件: {Path(xml_file).name}")
        print(f"   路径: {xml_file}")
        
        try:
            # 先检查XML文件内容
            with open(xml_file, 'r') as f:
                content = f.read()
                if 'jaco_7/jaco_7s.urdf' in content:
                    print("   ✅ 包含正确的URDF引用")
                else:
                    print("   ⚠️  未找到预期的URDF引用")
            
            # 尝试加载
            world = WorldModel()
            # 切换到XML文件所在目录，这样相对路径才能正确解析
            original_cwd = os.getcwd()
            xml_dir = Path(xml_file).parent
            os.chdir(str(xml_dir))
            
            try:
                success = world.loadFile(xml_file)
                
                if success:
                    print("   ✅ XML世界文件加载成功!")
                    print(f"   🤖 机器人数量: {world.numRobots()}")
                    print(f"   🌍 地形数量: {world.numTerrains()}")
                    
                    if world.numRobots() > 0:
                        robot = world.robot(0)
                        print(f"   🤖 机器人名称: {robot.getName()}")
                        print(f"   🔗 链接数量: {robot.numLinks()}")
                        print(f"   ⚙️ 关节数量: {robot.numDrivers()}")
                        
                        # 测试一个简单的运动学计算
                        try:
                            config = robot.getConfig()
                            print(f"   📐 配置空间维度: {len(config)}")
                            
                            # 设置一个随机配置
                            import random
                            random_config = [random.uniform(-1, 1) for _ in range(len(config))]
                            robot.setConfig(random_config)
                            
                            # 获取末端执行器位置
                            end_link = robot.numLinks() - 1
                            transform = robot.link(end_link).getTransform()
                            pos = transform[1][:3]
                            print(f"   🎯 末端执行器位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                            
                            results.append({'file': Path(xml_file).name, 'status': 'success'})
                            
                        except Exception as kinematics_error:
                            print(f"   ⚠️  运动学计算错误: {kinematics_error}")
                            results.append({'file': Path(xml_file).name, 'status': 'partial_success'})
                    else:
                        print("   ⚠️  没有找到机器人")
                        results.append({'file': Path(xml_file).name, 'status': 'no_robot'})
                else:
                    print("   ❌ XML文件加载失败")
                    results.append({'file': Path(xml_file).name, 'status': 'load_failed'})
                    
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"   ❌ 测试出错: {e}")
            results.append({'file': Path(xml_file).name, 'status': 'error', 'error': str(e)})
    
    return results

def test_relative_path_fix():
    """测试相对路径修复"""
    print("\n🔄 测试相对路径修复...")
    
    # 创建一个包含正确相对路径的临时XML文件
    xml_content = '''<?xml version='1.0' encoding='UTF-8'?>
<world>
    <robot name="jaco" file="/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/jaco_7/jaco_7s.urdf" translation="0.0 0.0 0" scale="1 1 1" />
    <terrain file="/home/lanh/project/robot_sim/coll_prediction_artifact/data/terrains/cube.off" scale="0.263081 0.054247 0.377450" translation="0.230036 0.086652 0.995170">
        <display color="0.2 0.2 0.0" opacity="0.2" />
    </terrain>
</world>'''
    
    temp_xml = "/tmp/test_world_fixed.xml"
    
    try:
        with open(temp_xml, 'w') as f:
            f.write(xml_content)
        
        print(f"📄 创建修复的临时XML: {temp_xml}")
        
        world = WorldModel()
        success = world.loadFile(temp_xml)
        
        if success:
            print("✅ 修复的XML世界文件加载成功!")
            print(f"🤖 机器人数量: {world.numRobots()}")
            print(f"🌍 地形数量: {world.numTerrains()}")
            
            if world.numRobots() > 0:
                robot = world.robot(0)
                print(f"🤖 机器人名称: {robot.getName()}")
                print(f"🔗 链接数量: {robot.numLinks()}")
                print(f"⚙️ 关节数量: {robot.numDrivers()}")
                return True
        else:
            print("❌ 修复的XML世界文件加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 相对路径测试出错: {e}")
        return False
    finally:
        if os.path.exists(temp_xml):
            os.remove(temp_xml)

def main():
    print("=" * 80)
    print("🧪 转换后XML文件测试")
    print("=" * 80)
    
    # 测试1: 相对路径修复
    path_fix_ok = test_relative_path_fix()
    
    # 测试2: 转换后的XML文件
    results = test_converted_xml()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)
    
    if path_fix_ok:
        print("✅ 相对路径修复测试通过")
    else:
        print("❌ 相对路径修复测试失败")
    
    success_count = 0
    for result in results:
        status = result['status']
        file_name = result['file']
        
        if status == 'success':
            print(f"✅ {file_name}: 完全成功")
            success_count += 1
        elif status == 'partial_success':
            print(f"⚠️ {file_name}: 部分成功")
            success_count += 0.5
        elif status == 'no_robot':
            print(f"⚠️ {file_name}: 加载成功但无机器人")
        elif status == 'load_failed':
            print(f"❌ {file_name}: 加载失败")
        else:
            print(f"❌ {file_name}: 错误 - {result.get('error', '')}")
    
    total_files = len(results)
    if total_files > 0:
        success_rate = success_count / total_files * 100
        print(f"\n🎯 XML文件成功率: {success_count}/{total_files} ({success_rate:.1f}%)")
    
    # 提供建议
    if path_fix_ok and success_count == total_files:
        print("\n🎉 所有测试通过！转换后的XML文件在Klampt中工作正常。")
        print("💡 建议: 可以使用转换后的XML文件进行碰撞预测实验。")
    elif path_fix_ok:
        print("\n⚠️ 路径修复测试通过，但XML文件测试有问题。")
        print("💡 建议: 需要修复转换后XML文件中的相对路径问题。")
    else:
        print("\n❌ 基础路径测试失败。")
        print("💡 建议: 检查URDF文件和mesh文件的路径配置。")

if __name__ == "__main__":
    main()
