#!/bin/bash

# Docker空间清理脚本

echo "🧹 Docker空间清理开始..."
echo "=================================="

# 显示清理前的空间使用情况
echo "📊 清理前空间使用情况:"
df -h | grep -E '(Filesystem|/$)'
echo ""
docker system df

echo ""
echo "🗑️ 开始清理..."

# 1. 停止所有容器
echo "🛑 停止所有容器..."
docker stop $(docker ps -aq) 2>/dev/null || echo "没有运行的容器"

# 2. 删除所有停止的容器
echo "🗑️ 删除停止的容器..."
docker container prune -f

# 3. 删除未使用的镜像
echo "🗑️ 删除未使用的镜像..."
docker image prune -a -f

# 4. 删除未使用的卷
echo "🗑️ 删除未使用的卷..."
docker volume prune -f

# 5. 删除未使用的网络
echo "🗑️ 删除未使用的网络..."
docker network prune -f

# 6. 删除构建缓存
echo "🗑️ 删除构建缓存..."
docker builder prune -a -f

# 7. 系统全面清理
echo "🗑️ 系统全面清理..."
docker system prune -a -f --volumes

# 8. 清理日志文件
echo "🗑️ 清理Docker日志..."
sudo find /var/lib/docker/containers/ -name "*.log" -exec truncate -s 0 {} \; 2>/dev/null || echo "需要root权限清理日志"

# 显示清理后的空间使用情况
echo ""
echo "📊 清理后空间使用情况:"
df -h | grep -E '(Filesystem|/$)'
echo ""
docker system df

echo ""
echo "=================================="
echo "✅ Docker空间清理完成！"
echo ""
echo "💡 提示:"
echo "- 如需重新部署，运行: ./deploy-linux.sh"
echo "- 定期运行此脚本可保持系统清洁"
echo "- 清理后需要重新构建镜像"
