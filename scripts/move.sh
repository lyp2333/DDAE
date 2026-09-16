#!/bin/bash
num_samples=1000
original_dir=/home/lyp/Code/DiffuseGAE/Recon/step_20/original
target_dir1=/home/lyp/Code/DiffuseGAE/ppl_eval/folder1
target_dir2=/home/lyp/Code/DiffuseGAE/ppl_eval/folder2
rm -rf $target_dir1 $target_dir2
mkdir -p $target_dir1 $target_dir2 # 如果b文件夹不存在，则创建它
count=0 # 初始化计数器
for file in ${original_dir}/*.png; do
  if [ "$count" -lt $num_samples ]; then # 判断是否达到复制数量上限
    cp "$file" "$target_dir1/" # 复制文件到folder1文件夹
  elif [[ "$count" -ge $num_samples && "$count" -lt $((num_samples * 2)) ]]; then
    cp "$file" "$target_dir2/" # 复制文件到folder2文件夹
  else
    break
  fi
  count=$((count + 1)) # 计数器加1
done
echo "复制完成！共复制了 $count 个文件。"
