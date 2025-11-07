#!/usr/bin/env python3
"""
coll_prediction_artifact 项目安装配置

使用方法：
    开发模式（推荐）: pip install -e .
    正式安装: pip install .
"""
from setuptools import setup, find_packages
from pathlib import Path

# 读取项目根目录的 README
project_root = Path(__file__).parent
readme_file = project_root / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 核心依赖（使用灵活的版本约束，避免旧版本问题）
requirements = [
    "numpy>=1.19.0",
    "pybullet>=3.0.0",
    "tqdm>=4.60.0",
]

setup(
    name="coll_prediction_artifact",
    version="1.0.0",
    author="Robot Collision Prediction Team",
    description="机器人碰撞预测与轨迹生成工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # 自动发现所有包
    packages=find_packages(include=[
        "trace_generation",
        "trace_generation.*",
        "motion_planning_prediction",
        "motion_planning_prediction.*",
        "prediction_approaches",
        "prediction_approaches.*"
    ]),
    
    # 包含数据文件
    package_data={
        "trace_generation": [
            "data/**/*",
            "core/robot/robot_config/*.py",
        ],
    },
    include_package_data=True,
    
    # 依赖
    install_requires=requirements,
    
    # Python版本要求
    python_requires=">=3.7",
    
    # 可选依赖
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "obb": [
            "open3d>=0.13.0",
            "coacd>=1.0.0",
            "yourdfpy>=0.0.53",
            "trimesh>=3.9.0",
        ],
        "cuda": [
            "torch>=1.9.0",
            "curobo>=0.1.0",
        ],
    },
    
    # 命令行工具（如果有的话）
    entry_points={
        "console_scripts": [
            # 可以在这里添加命令行工具
            # "collision-check=trace_generation.scripts.collision_check:main",
        ],
    },
    
    # 项目分类
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
