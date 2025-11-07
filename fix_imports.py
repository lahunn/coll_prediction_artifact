#!/usr/bin/env python3
"""
修复相对导入为绝对导入

将 trace_generation 目录下所有文件中的相对导入转换为绝对导入。

转换规则:
    from core.xxx -> from trace_generation.core.xxx
    from utils.xxx -> from trace_generation.utils.xxx
    from config.xxx -> from trace_generation.config.xxx

使用方法:
    python fix_imports.py --dry-run  # 预览
    python fix_imports.py            # 执行
"""

import os
import re
import argparse
import shutil
from datetime import datetime

# 需要修复的导入模式
IMPORT_PATTERNS = [
    (r'^from core\.', 'from trace_generation.core.'),
    (r'^from utils\.', 'from trace_generation.utils.'),
    (r'^from config\.', 'from trace_generation.config.'),
]


def find_python_files(root_dir):
    """查找所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过特定目录
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.venv', 'build', 'dist', '*.egg-info']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files


def analyze_file(filepath):
    """分析文件，返回需要修改的导入行"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    changes = []
    for i, line in enumerate(lines):
        for old_pattern, new_prefix in IMPORT_PATTERNS:
            if re.match(old_pattern, line.strip()):
                # 构建新的导入语句
                old_line = line.strip()
                new_line = re.sub(old_pattern, new_prefix, line)
                changes.append((i, old_line, new_line.strip()))
                break
    
    return changes


def fix_file(filepath, dry_run=False):
    """修复文件中的导入语句"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 收集需要修改的行
    changes = {}
    for i, line in enumerate(lines):
        for old_pattern, new_prefix in IMPORT_PATTERNS:
            if re.match(old_pattern, line.strip()):
                new_line = re.sub(old_pattern, new_prefix, line)
                changes[i] = new_line
                break
    
    if not changes:
        return False, []
    
    # 应用修改
    modified_lines = []
    for i in changes.keys():
        modified_lines.append((lines[i].strip(), changes[i].strip()))
    
    if not dry_run:
        # 创建备份
        backup_path = filepath + '.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(filepath, backup_path)
        
        # 应用修改
        for i, new_line in changes.items():
            lines[i] = new_line
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return True, modified_lines


def main():
    parser = argparse.ArgumentParser(
        description='修复相对导入为绝对导入',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python fix_imports.py --dry-run              # 预览
  python fix_imports.py                        # 执行修复
  python fix_imports.py --dir trace_generation # 只修复特定目录
        """
    )
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不实际修改文件')
    parser.add_argument('--dir', type=str, default='trace_generation', help='要处理的目录')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.dir)
    
    if not os.path.exists(root_dir):
        print(f"❌ 错误：目录不存在: {root_dir}")
        return
    
    print(f"🔍 扫描目录: {root_dir}")
    python_files = find_python_files(root_dir)
    print(f"📁 找到 {len(python_files)} 个Python文件")
    print()
    
    # 分析所有文件
    files_to_fix = {}
    for filepath in python_files:
        changes = analyze_file(filepath)
        if changes:
            files_to_fix[filepath] = changes
    
    if not files_to_fix:
        print("✅ 没有发现需要修复的文件！")
        return
    
    print(f"📋 发现 {len(files_to_fix)} 个文件需要修复：")
    print()
    
    for filepath, changes in files_to_fix.items():
        rel_path = os.path.relpath(filepath, root_dir)
        print(f"📄 {rel_path}")
        for line_no, old_import, new_import in changes:
            print(f"   行 {line_no + 1}:")
            print(f"     - {old_import}")
            print(f"     + {new_import}")
        print()
    
    if args.dry_run:
        print("💡 这是预览模式，没有实际修改文件")
        print("💡 运行 'python fix_imports.py' 执行修复")
        return
    
    # 询问确认
    confirm = input(f"\n⚠️  将要修改 {len(files_to_fix)} 个文件，是否继续？(y/N): ")
    if confirm.lower() != 'y':
        print("❌ 已取消")
        return
    
    # 执行修复
    print("\n🔧 修复中...")
    success_count = 0
    for filepath in files_to_fix.keys():
        modified, changes = fix_file(filepath, dry_run=False)
        if modified:
            success_count += 1
            rel_path = os.path.relpath(filepath, root_dir)
            print(f"  ✓ {rel_path}")
    
    print(f"\n✅ 成功修复 {success_count} 个文件！")
    print("💾 备份文件已保存（.backup_* 后缀）")
    print("\n💡 提示：运行以下命令验证修改：")
    print("  python -c \"from trace_generation.core.collision.obb_detector import OBBCollisionEnv; print('导入成功！')\"")


if __name__ == '__main__':
    main()
