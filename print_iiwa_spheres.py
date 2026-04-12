import torch
from trace_generation.core.robot.sphere_analyzer import RobotSphereAnalyzer

def print_iiwa_info():
    # 1. 初始化分析器，指定机器人名称为 iiwa
    # 注意：确保 curobo 的 config 目录下有 iiwa.yml 文件
    robot_name = "iiwa"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"正在加载 {robot_name} 机器人模型 (设备: {device})...")
    
    try:
        analyzer = RobotSphereAnalyzer(robot_name, device)
        
        # 2. 获取连杆坐标系下的球体信息
        link_spheres = analyzer.get_link_spheres_info()
        
        print(f"\n--- {robot_name} 连杆球体信息 (Link Spheres Info) ---")
        print(f"包含碰撞球体的连杆总数: {len(link_spheres)}")
        
        for link_name, spheres in link_spheres.items():
            print(f"\n连杆: {link_name}")
            print(f"  球体数量: {len(spheres)}")
            # 打印每个球体的 [x, y, z, radius]
            for i, s in enumerate(spheres):
                print(f"  球体 {i+1}: POS({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}) R: {s[3]:.4f}")
                
    except Exception as e:
        print(f"获取信息失败: {e}")
        print("请检查是否已在 curobo 中配置 iiwa 机器人，或尝试使用 'franka' 测试脚本是否工作。")

if __name__ == "__main__":
    print_iiwa_info()