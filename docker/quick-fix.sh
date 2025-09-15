#!/bin/bash

echo "🚀 快速修复Flask应用崩溃问题..."

# 停止所有服务
docker-compose -f docker-compose-linux.yml down

# 清理Docker空间
echo "🧹 清理Docker空间..."
docker system prune -f
docker image prune -f

# 重新构建应用
docker-compose -f docker-compose-linux.yml build --no-cache app

# 启动服务
docker-compose -f docker-compose-linux.yml up -d

# 等待启动
sleep 15

# 检查状态
echo "📊 服务状态:"
docker-compose -f docker-compose-linux.yml ps

echo ""
echo "📋 Flask应用日志:"
docker-compose -f docker-compose-linux.yml logs --tail=10 app

echo ""
echo "🌐 测试访问:"
curl -s http://localhost:8081/api/health || echo "API测试失败"
curl -s -I http://localhost:8080/ | head -2 || echo "Web测试失败"

echo ""
echo "✅ 修复完成！访问: http://172.104.172.5:8080/homework"
