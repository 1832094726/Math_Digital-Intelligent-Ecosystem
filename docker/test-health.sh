#!/bin/bash

# 测试健康检查脚本

echo "🏥 测试服务健康状态..."

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 测试Flask应用健康检查
echo "🔍 测试Flask应用健康检查..."
curl -f http://localhost:8081/api/health || echo "❌ Flask健康检查失败"

# 测试Nginx代理
echo "🔍 测试Nginx代理..."
curl -f http://localhost:8080/ || echo "❌ Nginx代理失败"

# 测试Redis
echo "🔍 测试Redis连接..."
docker exec math_ecosystem_redis redis-cli ping || echo "❌ Redis连接失败"

echo "✅ 健康检查完成"
