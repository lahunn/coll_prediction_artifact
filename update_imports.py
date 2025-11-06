#!/usr/bin/env python3
"""
自动更新 motion_planning_prediction 和 prediction_approaches 的导入路径
"""
import os
import re
from pathlib import Path

# 需要修改的文件列表
FILES_TO_UPDATE = [
    # motion_planning_prediction
    "motion_planning_prediction/prediction_simulation_nDOF.py",
    "motion_planning_prediction/prediction_simulation_nDOF_sphere.py",
    
    # prediction_approaches
    "prediction_approaches/coord_hashing.py",
    "prediction_approaches/coord_hashing_sphere.py",
    "prediction_approaches/encoord_hashing.py",
    "prediction_approaches/pose_hashing.py",
    "prediction_approaches/enpose_hashing.py",
    "prediction_approaches/enpose_hashing_cpu.py",
    "prediction_approaches/optimize_s_parameters.py",
    "prediction_approaches/optimize_s_parameters_sphere.py",
    "prediction_approaches/test_cht_inheritance_same_benchmark.py",
    "prediction_approaches/test_cht_inheritance_sphere.py",
    "prediction_approaches/analyze_training_progression.py",
    "prediction_approaches/test_strategies.py",
]

# 替换规则
REPLACEMENTS = [
    # 更新 ana_parameters 导入路径
    (
        r"from trace_generation\.robot_as\.ana_parameters import",
        "from trace_generation.config.ana_parameters import"
    ),
    # 备用匹配（防止有不同的写法）
    (
        r"import trace_generation\.robot_as\.ana_parameters",
        "import trace_generation.config.ana_parameters"
    ),
]

def update_imports(base_dir: str):
    """更新所有文件的导入路径"""
    base_path = Path(base_dir)
    updated_count = 0
    skipped_count = 0
    not_found_count = 0
    
    print("=" * 70)
    print("开始更新导入路径...")
    print("=" * 70)
    
    for file_rel_path in FILES_TO_UPDATE:
        file_path = base_path / file_rel_path
        
        if not file_path.exists():
            print(f"⚠️  文件不存在: {file_rel_path}")
            not_found_count += 1
            continue
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 应用替换规则
        for pattern, replacement in REPLACEMENTS:
            content = re.sub(pattern, replacement, content)
        
        # 检查是否有修改
        if content != original_content:
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已更新: {file_rel_path}")
            updated_count += 1
        else:
            print(f"ℹ️  无需修改: {file_rel_path}")
            skipped_count += 1
    
    print("\n" + "=" * 70)
    print(f"更新完成！")
    print(f"  已更新: {updated_count} 个文件")
    print(f"  无需修改: {skipped_count} 个文件")
    print(f"  未找到: {not_found_count} 个文件")
    print("=" * 70)
    
    return updated_count, skipped_count, not_found_count

if __name__ == "__main__":
    project_root = "/home/lanh/project/robot_sim/coll_prediction_artifact"
    update_imports(project_root)
