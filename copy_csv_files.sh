#!/bin/bash

# 源文件夹
source_dir="csv_history"
# 目标文件夹
destination_dir="csv_origin"

# 创建目标文件夹，如果不存在
mkdir -p "$destination_dir"

# 循环遍历日期范围
for date in $(seq -w 20250731 20250731); do
    sub_dir="$source_dir/$date"
    # 检查子文件夹是否存在
    if [ -d "$sub_dir" ]; then
        # 查找并复制 CSV 文件
        find "$sub_dir" -name "*.csv" -exec cp {} "$destination_dir" \;
    fi
done

echo "复制完成。"
    
