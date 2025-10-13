#!/bin/bash
################################################################################
# 测试问题集生成脚本
# 用途: 测试 generate_problem_dataset.py 的基本功能
################################################################################

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "测试问题集生成功能"
echo "========================================================================"
echo "工作目录: $SCRIPT_DIR"
echo ""

# 确保 maze_files 目录存在
echo ">>> 步骤 1: 创建输出目录"
mkdir -p maze_files
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ maze_files 目录已创建${NC}"
else
    echo -e "${RED}✗ 创建目录失败${NC}"
    exit 1
fi
echo ""

# 测试1: 生成小型测试集 (5个问题，快速测试)
echo "========================================================================"
echo "测试 1: 快速测试 - 生成5个问题"
echo "========================================================================"
echo "参数:"
echo "  - 问题数量: 5"
echo "  - 障碍物数量: 5"
echo "  - 最大规划时间: 30秒/问题"
echo "  - 机器人: kuka_iiwa/model_0.urdf (7DOF)"
echo ""

python generate_problem_dataset.py \
    --robot-file kuka_iiwa/model_0.urdf \
    --num-problems 5 \
    --num-obstacles 5 \
    --max-time 30.0 \
    --output-file maze_files/test_quick_5.pkl

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 测试 1 成功完成${NC}"
    
    # 检查输出文件
    if [ -f "maze_files/test_quick_5.pkl" ]; then
        FILE_SIZE=$(du -h "maze_files/test_quick_5.pkl" | cut -f1)
        echo -e "${GREEN}✓ 输出文件已生成: maze_files/test_quick_5.pkl (大小: $FILE_SIZE)${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ 测试 1 失败${NC}"
    exit 1
fi
echo ""

# 测试2: 生成中型测试集 (10个问题，更多障碍物)
echo "========================================================================"
echo "测试 2: 中等难度测试 - 生成10个问题"
echo "========================================================================"
echo "参数:"
echo "  - 问题数量: 10"
echo "  - 障碍物数量: 10"
echo "  - 最大规划时间: 45秒/问题"
echo "  - 机器人: kuka_iiwa/model_0.urdf (7DOF)"
echo ""

python generate_problem_dataset.py \
    --robot-file kuka_iiwa/model_0.urdf \
    --num-problems 10 \
    --num-obstacles 100 \
    --max-time 60.0 \
    --output-file maze_files/test_medium_10.pkl

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 测试 2 成功完成${NC}"
    
    # 检查输出文件
    if [ -f "maze_files/test_medium_10.pkl" ]; then
        FILE_SIZE=$(du -h "maze_files/test_medium_10.pkl" | cut -f1)
        echo -e "${GREEN}✓ 输出文件已生成: maze_files/test_medium_10.pkl (大小: $FILE_SIZE)${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}⚠ 测试 2 失败或部分成功${NC}"
fi
echo ""

# 验证生成的数据集
echo "========================================================================"
echo "验证生成的数据集"
echo "========================================================================"

# 创建Python验证脚本
cat > verify_dataset.py << 'VERIFY_EOF'
import pickle
import sys

def verify_dataset(filename):
    print(f"正在验证: {filename}")
    try:
        with open(filename, 'rb') as f:
            problems = pickle.load(f)
        
        print(f"  ✓ 文件加载成功")
        print(f"  ✓ 问题数量: {len(problems)}")
        
        if len(problems) > 0:
            obstacles, start, goal, path = problems[0]
            print(f"  ✓ 第一个问题:")
            print(f"    - 障碍物数量: {len(obstacles)}")
            print(f"    - 起点维度: {len(start)}")
            print(f"    - 终点维度: {len(goal)}")
            print(f"    - 路径长度: {len(path)}")
            
            # 检查数据结构
            if len(obstacles) > 0:
                halfExtents, basePosition = obstacles[0]
                print(f"    - 体素格式正确: halfExtents={halfExtents.shape}, basePosition={basePosition.shape}")
            
            return True
    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        return False

if __name__ == "__main__":
    all_ok = True
    for filename in sys.argv[1:]:
        if not verify_dataset(filename):
            all_ok = False
        print()
    
    sys.exit(0 if all_ok else 1)
VERIFY_EOF

# 运行验证
python verify_dataset.py maze_files/test_quick_5.pkl maze_files/test_medium_10.pkl

VERIFY_STATUS=$?

# 清理验证脚本
rm verify_dataset.py

echo "========================================================================"
echo "测试总结"
echo "========================================================================"

if [ -f "maze_files/test_quick_5.pkl" ]; then
    echo -e "${GREEN}✓ 快速测试数据集: maze_files/test_quick_5.pkl${NC}"
else
    echo -e "${RED}✗ 快速测试数据集生成失败${NC}"
fi

if [ -f "maze_files/test_medium_10.pkl" ]; then
    echo -e "${GREEN}✓ 中等测试数据集: maze_files/test_medium_10.pkl${NC}"
else
    echo -e "${YELLOW}⚠ 中等测试数据集生成失败或不完整${NC}"
fi

if [ $VERIFY_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ 数据集验证通过${NC}"
else
    echo -e "${YELLOW}⚠ 数据集验证有问题${NC}"
fi

echo ""
echo "========================================================================"
echo "完整用法示例"
echo "========================================================================"
echo ""
echo "1. 生成小型测试集 (100个问题):"
echo "   python generate_problem_dataset.py --num-problems 100"
echo ""
echo "2. 生成标准数据集 (3000个问题):"
echo "   python generate_problem_dataset.py --num-problems 3000"
echo ""
echo "3. 自定义参数:"
echo "   python generate_problem_dataset.py \\"
echo "       --robot-file kuka_iiwa/model_0.urdf \\"
echo "       --num-problems 1000 \\"
echo "       --num-obstacles 15 \\"
echo "       --max-time 60.0 \\"
echo "       --workspace-min -0.8 \\"
echo "       --workspace-max 0.8 \\"
echo "       --voxel-size-min 0.05 \\"
echo "       --voxel-size-max 0.12 \\"
echo "       --output-file maze_files/custom_dataset.pkl"
echo ""
echo "4. 查看帮助:"
echo "   python generate_problem_dataset.py --help"
echo ""
echo "========================================================================"
echo "✓ 测试脚本执行完成"
echo "========================================================================"
