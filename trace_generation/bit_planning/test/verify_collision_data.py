#!/usr/bin/env python3
"""
验证碰撞数据格式的脚本
"""
import pickle
import sys

def verify_data_format(filepath, data_type):
    """验证数据格式是否正确"""
    print(f"\n{'='*60}")
    print(f"验证文件: {filepath}")
    print(f"数据类型: {data_type}")
    print('='*60)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if data_type == "config":
        # config_list: List[np.ndarray]
        print(f"边数量: {len(data)}")
        if len(data) > 0:
            print(f"第1条边形状: {data[0].shape}")
            print(f"第1条边第1个配置: {data[0][0]}")
    
    elif data_type in ["obb", "sphere"]:
        # (link_data, link_coll_data)
        link_data, link_coll_data = data
        print(f"边数量: {len(link_data)}")
        
        if len(link_data) > 0:
            edge0_data = link_data[0]
            edge0_coll = link_coll_data[0]
            print(f"第1条边姿态数: {len(edge0_data)}")
            
            if len(edge0_data) > 0:
                pose0_data = edge0_data[0]
                pose0_coll = edge0_coll[0]
                
                if data_type == "obb":
                    print(f"第1姿态OBB数: {len(pose0_data)}")
                    if len(pose0_data) > 0:
                        print(f"第1个OBB位姿: {pose0_data[0][:3]}... (前3维)")
                        print(f"第1个OBB碰撞标签: {pose0_coll[0]}")
                else:  # sphere
                    print(f"第1姿态球体数: {len(pose0_data)}")
                    if len(pose0_data) > 0:
                        print(f"第1个球体坐标: {pose0_data[0]}")
                        print(f"第1个球体碰撞标签: {pose0_coll[0]}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 使用测试数据
        files = [
            ("test_configs.pkl", "config"),
            ("test_obb_data.pkl", "obb"),
            ("test_sphere_data.pkl", "sphere"),
        ]
    else:
        files = [(sys.argv[1], sys.argv[2])]
    
    for filepath, data_type in files:
        try:
            verify_data_format(filepath, data_type)
        except FileNotFoundError:
            print(f"⚠️  文件不存在: {filepath}")
        except Exception as e:
            print(f"❌ 验证失败: {e}")
