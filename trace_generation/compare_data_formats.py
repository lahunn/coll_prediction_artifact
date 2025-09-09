"""
数据格式对比脚本
用于比较 OBB 方法和球体方法生成的数据格式和内容

功能：
1. 加载 OBB 数据 (coord.pkl) 和球体数据 (sphere.pkl)
2. 分析数据结构和维度差异
3. 比较碰撞检测结果的一致性
4. 生成对比报告
"""

import pickle
import numpy as np
import sys
import os

def load_data(folder, file_number):
    """加载 OBB 和球体数据文件"""
    
    # OBB 数据文件
    coord_file = f"{folder}/obstacles_{file_number}_coord.pkl"
    sphere_file = f"{folder}/obstacles_{file_number}_sphere.pkl"
    pose_file = f"{folder}/obstacles_{file_number}_pose.pkl"
    
    data = {}
    
    # 加载 OBB 数据
    if os.path.exists(coord_file):
        with open(coord_file, 'rb') as f:
            qarr_obb, dirarr_obb, yarr_obb = pickle.load(f)
        data['obb'] = {
            'coordinates': qarr_obb,
            'directions': dirarr_obb, 
            'labels': yarr_obb
        }
        print(f"✓ Loaded OBB data from {coord_file}")
    else:
        print(f"✗ OBB data file not found: {coord_file}")
        
    # 加载球体数据
    if os.path.exists(sphere_file):
        with open(sphere_file, 'rb') as f:
            qarr_sphere, yarr_sphere, radius_arr, link_id_arr, sphere_id_arr = pickle.load(f)
        data['sphere'] = {
            'coordinates': qarr_sphere,
            'labels': yarr_sphere,
            'radii': radius_arr,
            'link_ids': link_id_arr,
            'sphere_ids': sphere_id_arr
        }
        print(f"✓ Loaded sphere data from {sphere_file}")
    else:
        print(f"✗ Sphere data file not found: {sphere_file}")
        
    # 加载 pose 数据
    if os.path.exists(pose_file):
        with open(pose_file, 'rb') as f:
            qarr_pose, yarr_pose = pickle.load(f)
        data['pose'] = {
            'configurations': qarr_pose,
            'labels': yarr_pose
        }
        print(f"✓ Loaded pose data from {pose_file}")
    else:
        print(f"✗ Pose data file not found: {pose_file}")
        
    return data

def analyze_data_structure(data):
    """分析数据结构"""
    
    print("\n" + "="*60)
    print("DATA STRUCTURE ANALYSIS")
    print("="*60)
    
    if 'obb' in data:
        obb = data['obb']
        print(f"\n📦 OBB Data Structure:")
        print(f"  - Coordinates shape: {obb['coordinates'].shape}")
        print(f"  - Labels shape: {obb['labels'].shape}")
        print(f"  - Directions count: {len(obb['directions'])}")
        print(f"  - Total OBB elements: {len(obb['coordinates'])}")
        
        # 计算每个 pose 的 link 数量
        if len(obb['coordinates']) > 0:
            poses_count = len(obb['coordinates']) // 7  # 假设 7 个 links
            print(f"  - Estimated poses: {poses_count}")
            print(f"  - Links per pose: 7 (fixed)")
    
    if 'sphere' in data:
        sphere = data['sphere']
        print(f"\n🔵 Sphere Data Structure:")
        print(f"  - Coordinates shape: {sphere['coordinates'].shape}")
        print(f"  - Labels shape: {sphere['labels'].shape}")
        print(f"  - Radii shape: {sphere['radii'].shape}")
        print(f"  - Link IDs shape: {sphere['link_ids'].shape}")
        print(f"  - Sphere IDs shape: {sphere['sphere_ids'].shape}")
        print(f"  - Total sphere elements: {len(sphere['coordinates'])}")
        
        # 计算每个 link 的球体数量
        unique_links = np.unique(sphere['link_ids'])
        print(f"  - Unique links: {len(unique_links)} (IDs: {unique_links})")
        
        for link_id in unique_links:
            link_spheres = np.sum(sphere['link_ids'] == link_id)
            poses_count = link_spheres // np.max(sphere['sphere_ids'][sphere['link_ids'] == link_id] + 1)
            spheres_per_pose = np.max(sphere['sphere_ids'][sphere['link_ids'] == link_id]) + 1
            print(f"    - Link {link_id}: {spheres_per_pose} spheres per pose")
    
    if 'pose' in data:
        pose = data['pose']
        print(f"\n🤖 Pose Data Structure:")
        print(f"  - Configurations shape: {pose['configurations'].shape}")
        print(f"  - Labels shape: {pose['labels'].shape}")
        print(f"  - Total poses: {len(pose['configurations'])}")

def compare_collision_rates(data):
    """比较碰撞检测率"""
    
    print("\n" + "="*60)
    print("COLLISION RATE COMPARISON")
    print("="*60)
    
    if 'obb' in data:
        obb_labels = data['obb']['labels']
        obb_collision_rate = np.sum(obb_labels == 0) / len(obb_labels) * 100
        print(f"\n📦 OBB Method:")
        print(f"  - Total elements: {len(obb_labels)}")
        print(f"  - Collisions: {np.sum(obb_labels == 0)}")
        print(f"  - Free: {np.sum(obb_labels == 1)}")
        print(f"  - Collision rate: {obb_collision_rate:.2f}%")
    
    if 'sphere' in data:
        sphere_labels = data['sphere']['labels']
        sphere_collision_rate = np.sum(sphere_labels == 0) / len(sphere_labels) * 100
        print(f"\n🔵 Sphere Method:")
        print(f"  - Total elements: {len(sphere_labels)}")
        print(f"  - Collisions: {np.sum(sphere_labels == 0)}")
        print(f"  - Free: {np.sum(sphere_labels == 1)}")
        print(f"  - Collision rate: {sphere_collision_rate:.2f}%")
    
    if 'pose' in data:
        pose_labels = data['pose']['labels']
        pose_collision_rate = np.sum(pose_labels == 0) / len(pose_labels) * 100
        print(f"\n🤖 Pose-level (Overall):")
        print(f"  - Total poses: {len(pose_labels)}")
        print(f"  - Colliding poses: {np.sum(pose_labels == 0)}")
        print(f"  - Free poses: {np.sum(pose_labels == 1)}")
        print(f"  - Pose collision rate: {pose_collision_rate:.2f}%")

def analyze_geometric_distribution(data):
    """分析几何分布"""
    
    print("\n" + "="*60)
    print("GEOMETRIC DISTRIBUTION ANALYSIS")
    print("="*60)
    
    if 'obb' in data:
        obb_coords = data['obb']['coordinates']
        print(f"\n📦 OBB Coordinates Distribution:")
        print(f"  - X range: [{np.min(obb_coords[:, 0]):.3f}, {np.max(obb_coords[:, 0]):.3f}]")
        print(f"  - Y range: [{np.min(obb_coords[:, 1]):.3f}, {np.max(obb_coords[:, 1]):.3f}]")
        print(f"  - Z range: [{np.min(obb_coords[:, 2]):.3f}, {np.max(obb_coords[:, 2]):.3f}]")
        print(f"  - Mean position: [{np.mean(obb_coords[:, 0]):.3f}, {np.mean(obb_coords[:, 1]):.3f}, {np.mean(obb_coords[:, 2]):.3f}]")
    
    if 'sphere' in data:
        sphere_coords = data['sphere']['coordinates']
        sphere_radii = data['sphere']['radii']
        print(f"\n🔵 Sphere Coordinates Distribution:")
        print(f"  - X range: [{np.min(sphere_coords[:, 0]):.3f}, {np.max(sphere_coords[:, 0]):.3f}]")
        print(f"  - Y range: [{np.min(sphere_coords[:, 1]):.3f}, {np.max(sphere_coords[:, 1]):.3f}]")
        print(f"  - Z range: [{np.min(sphere_coords[:, 2]):.3f}, {np.max(sphere_coords[:, 2]):.3f}]")
        print(f"  - Mean position: [{np.mean(sphere_coords[:, 0]):.3f}, {np.mean(sphere_coords[:, 1]):.3f}, {np.mean(sphere_coords[:, 2]):.3f}]")
        print(f"\n🔵 Sphere Radii Distribution:")
        print(f"  - Radius range: [{np.min(sphere_radii):.3f}, {np.max(sphere_radii):.3f}]")
        print(f"  - Mean radius: {np.mean(sphere_radii):.3f}")
        print(f"  - Unique radii: {len(np.unique(sphere_radii))}")

def compare_per_link_collision_rates(data):
    """比较每个 link 的碰撞率"""
    
    print("\n" + "="*60)
    print("PER-LINK COLLISION RATE COMPARISON")
    print("="*60)
    
    if 'obb' in data and 'sphere' in data:
        obb_labels = data['obb']['labels']
        sphere_labels = data['sphere']['labels']
        sphere_link_ids = data['sphere']['link_ids']
        
        print(f"\nLink-wise collision rates:")
        
        # OBB: 假设每 7 个元素对应一个 pose 的 7 个 links
        obb_poses = len(obb_labels) // 7
        for link_id in range(7):
            link_start = link_id
            link_indices = np.arange(link_start, len(obb_labels), 7)
            if len(link_indices) > 0:
                link_labels = obb_labels[link_indices]
                obb_rate = np.sum(link_labels == 0) / len(link_labels) * 100
                
                # 对应的球体数据
                sphere_mask = sphere_link_ids == link_id
                if np.any(sphere_mask):
                    sphere_link_labels = sphere_labels[sphere_mask]
                    sphere_rate = np.sum(sphere_link_labels == 0) / len(sphere_link_labels) * 100
                    
                    print(f"  Link {link_id}: OBB {obb_rate:.1f}% vs Sphere {sphere_rate:.1f}%")
                else:
                    print(f"  Link {link_id}: OBB {obb_rate:.1f}% vs Sphere N/A")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_data_formats.py <foldername> <filenumber>")
        print("Example: python compare_data_formats.py maze_data 1")
        sys.exit(1)
    
    folder = sys.argv[1]
    file_number = sys.argv[2]
    
    print("Loading data files...")
    data = load_data(folder, file_number)
    
    if not data:
        print("No data files found!")
        return
    
    # 执行各种分析
    analyze_data_structure(data)
    compare_collision_rates(data)
    analyze_geometric_distribution(data)
    compare_per_link_collision_rates(data)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    
    # 总结建议
    print(f"\n💡 Key Differences:")
    print(f"  - OBB method: Fixed 1 element per link per pose")
    print(f"  - Sphere method: Variable elements per link (based on sphere count)")
    print(f"  - OBB includes direction encoding, sphere includes radius")
    print(f"  - Both methods should show similar overall collision trends")

if __name__ == "__main__":
    main()
