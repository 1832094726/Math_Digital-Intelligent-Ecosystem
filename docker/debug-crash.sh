#!/bin/bash

# 调试Flask应用崩溃问题

echo "🔍 调试Flask应用崩溃问题..."
echo "=================================="

# 1. 检查容器状态
echo "📦 检查容器状态:"
docker-compose -f docker-compose-linux.yml ps

# 2. 查看Flask应用的完整日志
echo ""
echo "📋 Flask应用完整日志:"
docker-compose -f docker-compose-linux.yml logs app

# 3. 尝试手动启动容器查看错误
echo ""
echo "🔧 尝试手动启动容器查看错误:"
docker-compose -f docker-compose-linux.yml up app --no-deps

echo ""
echo "=================================="
echo "请查看上述日志找出崩溃原因"
