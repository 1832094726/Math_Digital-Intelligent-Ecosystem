#!/bin/bash

echo "🧹 清理Docker缓存并重新部署..."
echo "=================================="

# 1. 检查磁盘空间
echo "📊 清理前磁盘空间:"
df -h | grep -E '(Filesystem|/$|/var)'

echo ""
echo "📊 Docker空间使用:"
docker system df

# 2. 停止所有服务
echo ""
echo "🛑 停止所有服务..."
docker-compose -f docker-compose-linux.yml down

# 3. 全面清理Docker
echo ""
echo "🗑️ 清理Docker缓存..."
echo "删除停止的容器..."
docker container prune -f

echo "删除未使用的镜像..."
docker image prune -a -f

echo "删除未使用的卷..."
docker volume prune -f

echo "删除未使用的网络..."
docker network prune -f

echo "删除构建缓存..."
docker builder prune -a -f

echo "系统全面清理..."
docker system prune -a -f --volumes

# 4. 拉取最新代码
echo ""
echo "📥 拉取最新代码..."
git pull origin main

# 5. 重新构建
echo ""
echo "🔨 重新构建应用..."
docker-compose -f docker-compose-linux.yml build --no-cache app

# 6. 启动服务
echo ""
echo "🚀 启动服务..."
docker-compose -f docker-compose-linux.yml up -d

# 7. 等待启动
echo ""
echo "⏳ 等待服务启动..."
sleep 30

# 8. 检查状态
echo ""
echo "📊 服务状态:"
docker-compose -f docker-compose-linux.yml ps

# 9. 检查清理后空间
echo ""
echo "📊 清理后磁盘空间:"
df -h | grep -E '(Filesystem|/$|/var)'

echo ""
echo "📊 清理后Docker空间:"
docker system df

# 10. 测试访问
echo ""
echo "🌐 测试访问..."
echo "API健康检查:"
curl -s http://localhost:8081/api/health | head -3

echo ""
echo "前端页面:"
curl -s -I http://localhost:8080/homework | head -3

echo ""
echo "静态资源测试:"
curl -s -I http://localhost:8080/static/css/app.css | head -2

echo ""
echo "=================================="
echo "✅ 清理和部署完成！"
echo ""
echo "🌐 访问地址:"
echo "   - 主页: http://172.104.172.5:8080"
echo "   - 作业页面: http://172.104.172.5:8080/homework"
echo "   - API: http://172.104.172.5:8081/api/health"
