#!/bin/bash

# 获取当前时间作为日志文件名的一部分
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="training_${timestamp}.log"

echo "正在启动后台训练任务..."
echo "日志将输出到: $log_file"

# 使用 nohup 在后台运行命令
# -n marvel: 指定 conda 环境
# --no-capture-output: 确保输出即时显示
# > "$log_file" 2>&1: 将标准输出和错误输出重定向到日志文件
# &: 在后台运行
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup conda run -n marvel --no-capture-output python3 -u src/semantic_training/driver.py > "$log_file" 2>&1 &

pid=$!
echo "训练已在后台启动，进程 ID (PID): $pid"
echo "即使 SSH 断开，训练也会继续运行。"
echo "----------------------------------------"
echo "查看实时日志命令: tail -f $log_file"
echo "停止训练命令: kill $pid"
