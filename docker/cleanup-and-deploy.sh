#!/bin/bash

echo "🧹 清理Docker缓存并重新部署..."
echo "=================================="

# 1. 显示当前磁盘使用情况
echo "📊 当前磁盘使用情况:"
df -h | grep -E '(Filesystem|/$|/var)'
echo ""
docker system df

# 2. 停止所有服务
echo ""
echo "🛑 停止所有服务..."
cd /root/Math_Digital-Intelligent-Ecosystem/docker
docker-compose -f docker-compose-linux.yml down

# 3. 清理Docker缓存和未使用资源
echo ""
echo "🗑️ 清理Docker缓存..."
docker system prune -a -f --volumes
docker builder prune -a -f
docker image prune -a -f
docker container prune -f
docker volume prune -f
docker network prune -f

# 4. 显示清理后的空间
echo ""
echo "📊 清理后磁盘使用情况:"
df -h | grep -E '(Filesystem|/$|/var)'
echo ""
docker system df

# 5. 拉取最新代码
echo ""
echo "📥 拉取最新代码..."
cd /root/Math_Digital-Intelligent-Ecosystem
git pull origin main

# 6. 重新构建并部署
echo ""
echo "🔨 重新构建并部署..."
cd docker
docker-compose -f docker-compose-linux.yml build --no-cache
docker-compose -f docker-compose-linux.yml up -d

# 7. 等待服务启动
echo ""
echo "⏳ 等待服务启动..."
sleep 30

# 8. 检查服务状态
echo ""
echo "📊 检查服务状态..."
docker-compose -f docker-compose-linux.yml ps

# 9. 测试访问
echo ""
echo "🌐 测试访问..."
curl -s -I http://localhost:8080/homework | head -3
curl -s http://localhost:8081/api/health

echo ""
echo "=================================="
echo "✅ 清理和部署完成！"
echo ""
echo "🌐 访问地址:"
echo "   - 主页: http://172.104.172.5:8080"
echo "   - 作业页面: http://172.104.172.5:8080/homework"
echo "   - API健康检查: http://172.104.172.5:8081/api/health"
echo ""
echo "💾 磁盘空间已清理，系统已更新到最新版本！"
