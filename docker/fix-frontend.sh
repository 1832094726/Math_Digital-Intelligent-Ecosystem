#!/bin/bash

# 修复前端静态文件问题

echo "🔧 修复前端静态文件问题..."
echo "=================================="

# 1. 检查当前文件结构
echo "🔍 检查容器内文件结构..."
./check-files.sh

echo ""
echo "🛠️ 开始修复..."

# 2. 停止服务
echo "🛑 停止服务..."
docker-compose -f docker-compose-linux.yml down

# 3. 清理空间
echo "🧹 清理Docker空间..."
docker system prune -f

# 4. 检查Vue项目结构
echo "📁 检查Vue项目结构..."
ls -la ../homework_system/

# 5. 重新构建（确保前端构建成功）
echo "🔨 重新构建镜像（包含前端构建）..."
docker-compose -f docker-compose-linux.yml build --no-cache app

# 6. 启动服务
echo "🚀 启动服务..."
docker-compose -f docker-compose-linux.yml up -d

# 7. 等待启动
echo "⏳ 等待服务启动..."
sleep 30

# 8. 再次检查文件结构
echo "🔍 检查修复后的文件结构..."
./check-files.sh

# 9. 测试访问
echo ""
echo "🌐 测试访问..."
echo "API健康检查:"
curl -s http://localhost:8081/api/health | head -3

echo ""
echo "前端页面测试:"
curl -s -I http://localhost:8080/homework | head -3

echo ""
echo "=================================="
echo "✅ 前端修复完成！"
echo ""
echo "🌐 访问地址:"
echo "   - 主页: http://172.104.172.5:8080"
echo "   - 作业页面: http://172.104.172.5:8080/homework"
echo "   - API健康检查: http://172.104.172.5:8081/api/health"
