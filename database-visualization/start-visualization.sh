#!/bin/bash

# K12数学教育系统 - 数据库可视化启动脚本

echo "🚀 启动K12数学教育系统数据库可视化"
echo "=================================="

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未安装，请先安装pip3"
    exit 1
fi

echo "📦 安装Python依赖..."
pip3 install flask flask-cors pymysql

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

echo "🔧 启动API服务器..."
echo "API地址: http://localhost:5001"
echo "可视化界面: 请在浏览器中打开 index.html"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "=================================="

# 启动API服务器
python3 api-server.py
