#!/usr/bin/env python3
"""
清理项目中不必要的 sys.path 修改代码

这个脚本会：
1. 扫描所有 .py 文件
2. 识别手动添加路径的代码模式
3. 创建备份
4. 移除不必要的路径配置代码

使用方法：
    python cleanup_imports.py --dry-run  # 预览将要修改的文件
    python cleanup_imports.py            # 执行清理
    python cleanup_imports.py --restore  # 恢复备份
"""

import os
import re
import argparse
import shutil
from datetime import datetime

# 需要移除的代码模式
PATTERNS_TO_REMOVE = [
    # sys.path.insert(0, ...)
    r'sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\),\s*[\'"][.\\/]+[\'"]\)\)',
    r'sys\.path\.insert\(0,\s*os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\),\s*[\'"][.\\/]+[\'"]\)\)\)',
    r'sys\.path\.insert\(0,\s*str\(project_root\)\)',
    
    # sys.path.append(...)
    r'sys\.path\.append\(os\.path\.join\(os\.path\.dirname\(__file__\),\s*[\'"][.\\/]+[\'"]\)\)',
    r'sys\.path\.append\([\'"][.]+[\'"]\)',
    
    # project_root = Path(...).resolve().parents[N]
    r'project_root\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[\d+\]',
]

# 需要保留特定导入的文件（如果移除 sys.path 后还需要 sys 模块）
KEEP_SYS_IMPORT = True


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
    """分析文件，返回需要移除的行"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines_to_remove = []
    for i, line in enumerate(lines):
        for pattern in PATTERNS_TO_REMOVE:
            if re.search(pattern, line):
                lines_to_remove.append((i, line.strip()))
                break
    
    return lines_to_remove


def clean_file(filepath, dry_run=False):
    """清理文件中的路径配置代码"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 标记需要移除的行
    lines_to_remove = set()
    for i, line in enumerate(lines):
        for pattern in PATTERNS_TO_REMOVE:
            if re.search(pattern, line):
                lines_to_remove.add(i)
                break
    
    if not lines_to_remove:
        return False, []
    
    # 构建新内容
    new_lines = []
    removed_lines = []
    for i, line in enumerate(lines):
        if i in lines_to_remove:
            removed_lines.append(line.strip())
        else:
            new_lines.append(line)
    
    # 检查是否还需要保留 import sys 和 import os
    needs_sys = any('sys.' in line for line in new_lines)
    needs_os = any('os.' in line for line in new_lines)
    
    # 清理不必要的空行（连续的空行只保留一个）
    final_lines = []
    prev_blank = False
    for line in new_lines:
        is_blank = line.strip() == ''
        
        # 移除不需要的 import
        if not needs_sys and line.strip() == 'import sys':
            removed_lines.append(line.strip())
            continue
        if not needs_os and line.strip() == 'import os':
            removed_lines.append(line.strip())
            continue
        
        if is_blank and prev_blank:
            continue
        
        final_lines.append(line)
        prev_blank = is_blank
    
    if not dry_run:
        # 创建备份
        backup_path = filepath + '.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(filepath, backup_path)
        
        # 写入新内容
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
    
    return True, removed_lines


def restore_backup(filepath):
    """恢复最新的备份"""
    backup_files = sorted([f for f in os.listdir(os.path.dirname(filepath)) 
                          if f.startswith(os.path.basename(filepath) + '.backup_')])
    
    if not backup_files:
        return False
    
    latest_backup = os.path.join(os.path.dirname(filepath), backup_files[-1])
    shutil.copy2(latest_backup, filepath)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='清理项目中不必要的 sys.path 修改代码',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python cleanup_imports.py --dry-run        # 预览将要修改的文件
  python cleanup_imports.py                  # 执行清理
  python cleanup_imports.py --restore        # 恢复所有备份
  python cleanup_imports.py --dir ./trace_generation  # 只清理特定目录
        """
    )
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不实际修改文件')
    parser.add_argument('--restore', action='store_true', help='恢复所有备份文件')
    parser.add_argument('--dir', type=str, default='.', help='要处理的目录（默认为当前目录）')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.dir)
    
    if not os.path.exists(root_dir):
        print(f"❌ 错误：目录不存在: {root_dir}")
        return
    
    print(f"🔍 扫描目录: {root_dir}")
    python_files = find_python_files(root_dir)
    print(f"📁 找到 {len(python_files)} 个Python文件")
    print()
    
    if args.restore:
        print("🔄 恢复备份文件...")
        restored_count = 0
        for filepath in python_files:
            if restore_backup(filepath):
                restored_count += 1
                print(f"  ✓ 恢复: {os.path.relpath(filepath, root_dir)}")
        print(f"\n✅ 恢复了 {restored_count} 个文件")
        return
    
    # 分析所有文件
    files_to_clean = {}
    for filepath in python_files:
        lines_to_remove = analyze_file(filepath)
        if lines_to_remove:
            files_to_clean[filepath] = lines_to_remove
    
    if not files_to_clean:
        print("✅ 没有发现需要清理的文件！")
        return
    
    print(f"📋 发现 {len(files_to_clean)} 个文件需要清理：")
    print()
    
    for filepath, lines in files_to_clean.items():
        rel_path = os.path.relpath(filepath, root_dir)
        print(f"📄 {rel_path}")
        for line_no, line_content in lines:
            print(f"   行 {line_no + 1}: {line_content}")
        print()
    
    if args.dry_run:
        print("💡 这是预览模式，没有实际修改文件")
        print("💡 运行 'python cleanup_imports.py' 执行清理")
        return
    
    # 询问确认
    confirm = input(f"\n⚠️  将要修改 {len(files_to_clean)} 个文件，是否继续？(y/N): ")
    if confirm.lower() != 'y':
        print("❌ 已取消")
        return
    
    # 执行清理
    print("\n🔧 清理中...")
    success_count = 0
    for filepath in files_to_clean.keys():
        modified, removed = clean_file(filepath, dry_run=False)
        if modified:
            success_count += 1
            rel_path = os.path.relpath(filepath, root_dir)
            print(f"  ✓ {rel_path}")
            for line in removed:
                print(f"      移除: {line}")
    
    print(f"\n✅ 成功清理 {success_count} 个文件！")
    print("💾 备份文件已保存（.backup_* 后缀）")
    print("\n💡 提示：运行以下命令验证修改：")
    print("  python -c \"import trace_generation; print('导入成功！')\"")
    print("\n💡 如需恢复，运行：")
    print("  python cleanup_imports.py --restore")


if __name__ == '__main__':
    main()
