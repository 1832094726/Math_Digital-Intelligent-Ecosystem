#!/bin/bash

# 修复Python依赖问题的脚本

echo "🔧 修复Python依赖问题..."
echo "=================================="

# 1. 停止现有服务
echo "🛑 停止现有服务..."
docker-compose -f docker-compose-linux.yml down

# 2. 删除旧的应用镜像
echo "🗑️ 删除旧的应用镜像..."
docker rmi docker_app 2>/dev/null || echo "镜像不存在，跳过删除"

# 3. 清理构建缓存
echo "🧹 清理Docker构建缓存..."
docker builder prune -f

# 4. 重新构建应用镜像（无缓存）
echo "🔨 重新构建应用镜像（包含所有依赖）..."
docker-compose -f docker-compose-linux.yml build --no-cache app

# 5. 启动服务
echo "🚀 启动服务..."
docker-compose -f docker-compose-linux.yml up -d

# 6. 等待服务启动
echo "⏳ 等待服务启动..."
sleep 45

# 7. 检查Flask应用状态
echo "🔍 检查Flask应用状态..."
echo "容器状态:"
docker-compose -f docker-compose-linux.yml ps app

echo ""
echo "Flask应用日志 (最后10行):"
docker-compose -f docker-compose-linux.yml logs --tail=10 app

# 8. 测试健康检查
echo ""
echo "🏥 测试健康检查..."
sleep 5
curl -s http://localhost:8081/api/health || echo "❌ 健康检查失败"

# 9. 测试前端访问
echo ""
echo "🌐 测试前端访问..."
curl -s -I http://localhost:8080/ | head -3

echo ""
echo "=================================="
echo "✅ 依赖修复完成！"
echo ""
echo "🌐 访问地址:"
echo "   - 主页: http://172.104.172.5:8080"
echo "   - API健康检查: http://172.104.172.5:8081/api/health"
echo ""
echo "如果仍有问题，请检查上述日志输出"
