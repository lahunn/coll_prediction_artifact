#!/bin/bash
# 从Windows OneDrive同步机器人数据到trace_files目录
# 用法: ./re_rsync.sh

# Windows路径转换为WSL路径
SOURCE_DIR="/mnt/d/onedrive/wsl/robot_data/"
TARGET_DIR="$(dirname "$0")"

echo "=== 开始同步数据 ==="
echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo ""

# 使用rsync同步，保持权限和时间戳
rsync -avh --progress \
    --exclude='*.tmp' \
    --exclude='.git' \
    "$SOURCE_DIR" "$TARGET_DIR/"

echo ""
echo "=== 同步完成 ==="
