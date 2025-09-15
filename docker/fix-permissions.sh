#!/bin/bash

# 修复Vue构建权限问题

echo "🔧 修复Vue构建权限问题..."
echo "=================================="

# 1. 停止服务
echo "🛑 停止服务..."
docker-compose -f docker-compose-linux.yml down

# 2. 删除旧镜像
echo "🗑️ 删除旧镜像..."
docker rmi docker_app 2>/dev/null || echo "镜像不存在"

# 3. 清理构建缓存
echo "🧹 清理构建缓存..."
docker builder prune -a -f

# 4. 检查本地Vue项目权限
echo "🔍 检查本地Vue项目权限..."
ls -la ../homework_system/node_modules/.bin/vue-cli-service 2>/dev/null || echo "本地vue-cli-service不存在"

# 5. 重新构建（显示详细日志）
echo "🔨 重新构建镜像（显示详细日志）..."
docker-compose -f docker-compose-linux.yml build --no-cache --progress=plain app

# 6. 如果构建失败，尝试替代方案
if [ $? -ne 0 ]; then
    echo "⚠️ 标准构建失败，尝试替代方案..."
    
    # 创建临时Dockerfile
    cat > Dockerfile.temp << 'EOF'
# 阶段1: Vue前端构建
FROM node:16-alpine AS frontend-builder

# 设置工作目录
WORKDIR /frontend

# 复制前端项目文件
COPY homework_system/package*.json ./
RUN npm install

# 复制前端源码
COPY homework_system/ ./

# 修复权限并构建
RUN chmod -R 755 node_modules/.bin/ && \
    npm config set unsafe-perm true && \
    npx vue-cli-service build

# 阶段2: 后端Flask + 前端静态文件
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    libffi-dev \
    libssl-dev \
    libmariadb-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制并安装Python依赖
COPY homework-backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY homework-backend/ ./

# 创建必要的目录
RUN mkdir -p logs data/uploads static/homework static/symbol

# 从前端构建阶段复制构建好的静态文件
COPY --from=frontend-builder /frontend/dist/ ./static/homework/

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PORT=5000

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# 启动命令
CMD ["python", "-c", "from app import app; app.run(host='0.0.0.0', port=5000, debug=False)"]
EOF

    # 使用临时Dockerfile构建
    docker build -f Dockerfile.temp -t docker_app ..
    
    # 删除临时文件
    rm Dockerfile.temp
fi

# 7. 启动服务
echo "🚀 启动服务..."
docker-compose -f docker-compose-linux.yml up -d

# 8. 等待启动
echo "⏳ 等待服务启动..."
sleep 30

# 9. 检查状态
echo "📊 检查服务状态..."
docker-compose -f docker-compose-linux.yml ps

# 10. 测试
echo ""
echo "🌐 测试访问..."
curl -s http://localhost:8081/api/health | head -3
curl -s -I http://localhost:8080/ | head -3

echo ""
echo "=================================="
echo "✅ 权限修复完成！"
echo "🌐 访问: http://172.104.172.5:8080/homework"
