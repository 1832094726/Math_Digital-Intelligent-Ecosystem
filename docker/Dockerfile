# K12数学教育生态系统 - 全栈生产就绪镜像
# 多阶段构建：前端Vue + 后端Flask

# ================================
# 阶段1: 前端Vue构建
# ================================
FROM node:16-alpine AS frontend-builder

# 设置工作目录
WORKDIR /frontend

# 复制前端项目文件
COPY homework_system/package*.json ./
RUN npm install

# 复制前端源码并构建
COPY homework_system/ ./
RUN chmod -R +x node_modules/.bin/ && \
    ls -la node_modules/.bin/vue-cli-service && \
    npm run build

# ================================
# 阶段2: 后端Flask + 前端静态文件
# ================================
FROM python:3.9-slim

# 设置标签
LABEL maintainer="K12 Math Ecosystem Team"
LABEL version="2.0"
LABEL description="K12 Math Education Full-Stack Digital Ecosystem"

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

# 复制后端应用代码
COPY homework-backend/ ./

# 创建必要目录
RUN mkdir -p logs data/uploads static/homework static/symbol

# 从前端构建阶段复制构建好的静态文件
COPY --from=frontend-builder /frontend/dist/ ./static/homework/

# 复制符号键盘前端（如果存在），否则创建默认页面
RUN if [ -d "../Subject_symbol_dynamic_keyboard/board-frontend/dist" ]; then \
        cp -r ../Subject_symbol_dynamic_keyboard/board-frontend/dist/* ./static/symbol/; \
    else \
        echo '<!DOCTYPE html><html><head><title>Symbol Keyboard</title></head><body><h1>数学符号键盘</h1><p>组件加载中...</p></body></html>' > static/symbol/index.html; \
    fi

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
CMD ["python", "app.py"]
