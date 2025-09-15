#!/bin/bash

# 修复Flask网络监听问题

echo "🔧 修复Flask网络监听问题..."
echo "=================================="

# 1. 停止应用容器
echo "🛑 停止应用容器..."
docker-compose -f docker-compose-linux.yml stop app

# 2. 删除应用容器
echo "🗑️ 删除应用容器..."
docker-compose -f docker-compose-linux.yml rm -f app

# 3. 重新构建应用镜像
echo "🔨 重新构建应用镜像..."
docker-compose -f docker-compose-linux.yml build --no-cache app

# 4. 启动应用容器
echo "🚀 启动应用容器..."
docker-compose -f docker-compose-linux.yml up -d app

# 5. 等待应用启动
echo "⏳ 等待应用启动..."
sleep 20

# 6. 检查应用状态
echo "📊 检查应用状态..."
docker-compose -f docker-compose-linux.yml ps app

# 7. 检查应用日志
echo ""
echo "📋 应用启动日志:"
docker-compose -f docker-compose-linux.yml logs --tail=10 app

# 8. 测试网络连接
echo ""
echo "🌐 测试网络连接..."
echo "测试健康检查 (直接访问):"
curl -s http://localhost:8081/api/health | head -3

echo ""
echo "测试容器内网络:"
docker exec math_ecosystem_nginx wget -q -O - http://app:5000/api/health | head -3

# 9. 重启nginx以确保连接
echo ""
echo "🔄 重启nginx..."
docker-compose -f docker-compose-linux.yml restart nginx

# 10. 最终测试
echo ""
echo "🎯 最终测试..."
sleep 5
curl -s -I http://localhost:8080/ | head -3

echo ""
echo "=================================="
echo "✅ 网络修复完成！"
echo ""
echo "🌐 访问地址:"
echo "   - 主页: http://172.104.172.5:8080"
echo "   - API: http://172.104.172.5:8081/api/health"
