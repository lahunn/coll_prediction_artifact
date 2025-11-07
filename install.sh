#!/bin/bash
# 一键安装脚本 - 解决所有路径问题

set -e

echo "========================================"
echo "🚀 coll_prediction_artifact 一键安装"
echo "========================================"
echo ""

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 项目目录: $SCRIPT_DIR"
echo ""

# 检查Python版本
echo "🔍 检查Python版本..."
python3 --version

# 检查pip
echo "🔍 检查pip..."
pip3 --version
echo ""

# 询问是否创建虚拟环境
read -p "是否创建新的虚拟环境？(y/N): " create_venv
echo ""

if [[ "$create_venv" =~ ^[Yy]$ ]]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    
    echo "✓ 虚拟环境已创建"
    echo ""
    echo "请运行以下命令激活虚拟环境，然后重新运行此脚本："
    echo "  source venv/bin/activate"
    echo ""
    exit 0
fi

# 开发模式安装
echo "📦 开发模式安装项目..."
pip3 install -e .

echo ""
echo "========================================"
echo "✅ 安装完成！"
echo "========================================"
echo ""

# 验证安装
echo "🧪 验证安装..."
if python3 -c "import trace_generation; print('✓ trace_generation 导入成功')" 2>/dev/null; then
    echo "✅ 安装验证通过！"
else
    echo "❌ 安装验证失败"
    exit 1
fi

echo ""
echo "📚 使用指南："
echo "  1. 现在可以在任何位置导入模块："
echo "     from trace_generation.core.collision import sphere_detector"
echo ""
echo "  2. 运行测试："
echo "     python trace_generation/tests/test_refactoring.py"
echo ""
echo "  3. 可选：清理旧的路径配置代码："
echo "     python cleanup_imports.py --dry-run  # 预览"
echo "     python cleanup_imports.py            # 执行"
echo ""
echo "  4. 可选：安装额外依赖："
echo "     pip install -e \".[obb]\"    # OBB碰撞检测"
echo "     pip install -e \".[cuda]\"   # CUDA支持"
echo "     pip install -e \".[dev]\"    # 开发工具"
echo ""
echo "📖 详细文档: INSTALL.md"
echo ""
