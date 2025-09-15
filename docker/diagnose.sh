#!/bin/bash

# 诊断脚本 - 检查502错误原因

echo "🔍 开始诊断502错误..."
echo "=================================="

# 1. 检查容器状态
echo "📦 检查容器状态:"
docker-compose -f docker-compose-linux.yml ps

echo ""
echo "🔍 检查所有容器:"
docker ps -a

# 2. 检查Flask应用日志
echo ""
echo "📋 Flask应用日志 (最后20行):"
docker-compose -f docker-compose-linux.yml logs --tail=20 app

# 3. 检查nginx日志
echo ""
echo "📋 Nginx日志 (最后10行):"
docker-compose -f docker-compose-linux.yml logs --tail=10 nginx

# 4. 测试Flask应用直接访问
echo ""
echo "🔍 测试Flask应用直接访问:"
echo "测试健康检查端点..."
curl -v http://localhost:8081/api/health 2>&1 | head -10

# 5. 测试容器内网络
echo ""
echo "🌐 测试Docker网络连接:"
docker exec math_ecosystem_nginx ping -c 2 app 2>/dev/null || echo "❌ nginx无法ping通app容器"

# 6. 检查端口占用
echo ""
echo "🔌 检查端口占用:"
netstat -tlnp | grep -E ':(8080|8081|5000)' || echo "未找到相关端口监听"

# 7. 检查nginx配置
echo ""
echo "⚙️ 检查nginx配置:"
docker exec math_ecosystem_nginx nginx -t 2>&1 || echo "❌ nginx配置测试失败"

echo ""
echo "=================================="
echo "🎯 诊断建议:"
echo "1. 如果Flask应用容器未运行，检查构建日志"
echo "2. 如果Flask应用无法访问，检查端口配置"
echo "3. 如果nginx无法连接app，检查Docker网络"
echo "4. 查看上述日志输出寻找具体错误信息"
