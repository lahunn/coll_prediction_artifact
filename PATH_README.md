# 路径问题解决方案 - 快速指引 🚀

> **问题**: 经常遇到 `ImportError: No module named 'trace_generation'` 或需要手动配置 `sys.path`
>
> **解决**: 一行命令，彻底解决！

## ⚡ 快速解决（30秒）

```bash
cd /home/lanh/project/robot_sim/coll_prediction_artifact
pip install -e .
```

**就这么简单！** 现在可以在任何地方使用：

```python
from trace_generation.core.collision.obb_detector import OBBCollisionEnv
from trace_generation.core.robot.environment import RobotEnv
# 无需任何路径配置！
```

## 📖 详细文档

根据你的需求选择：

| 文档 | 适合 | 内容 |
|------|------|------|
| **[PATH_FIX_REPORT.md](PATH_FIX_REPORT.md)** | 📋 想了解完整解决方案 | 问题分析、解决方案、完成报告 |
| **[PATH_SOLUTION.md](PATH_SOLUTION.md)** | 🎯 快速参考 | 常见场景、故障排除、使用示例 |
| **[INSTALL.md](INSTALL.md)** | 📚 深入学习 | 详细安装指南、可选依赖、高级配置 |

## 🛠️ 可用工具

### 1. 清理旧的路径配置
```bash
python cleanup_imports.py --dry-run  # 预览
python cleanup_imports.py            # 执行
```

### 2. 修复相对导入
```bash
python fix_imports.py --dry-run  # 预览
python fix_imports.py            # 执行
```

### 3. 一键安装脚本
```bash
./install.sh
```

## ✅ 验证安装

```bash
python3 -c "import trace_generation; print('✅ 安装成功！')"
```

## ❓ 遇到问题？

查看 **[PATH_SOLUTION.md](PATH_SOLUTION.md)** 的"故障排除"部分

---

**核心理念**: 使用Python标准的包管理方式，而不是手动修改 `sys.path`

**一次配置，终身受益** 🎉
