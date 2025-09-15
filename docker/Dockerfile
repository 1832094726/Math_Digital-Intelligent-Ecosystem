# K12数学教育生态系统 - 生产就绪镜像
FROM python:3.9-slim

# 设置标签
LABEL maintainer="K12 Math Ecosystem Team"
LABEL version="1.0"
LABEL description="K12 Math Education Digital Ecosystem"

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    libffi-dev \
    libssl-dev \
    libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制并安装Python依赖
COPY homework-backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY homework-backend/ ./

# 创建必要目录和默认静态文件
RUN mkdir -p logs data/uploads static/homework static/symbol && \
    echo '<!DOCTYPE html><html><head><title>K12 Math Ecosystem</title></head><body><h1>K12数学教育生态系统</h1><p>系统正在启动中...</p></body></html>' > static/homework/index.html && \
    echo '<!DOCTYPE html><html><head><title>Symbol Keyboard</title></head><body><h1>数学符号键盘</h1><p>组件加载中...</p></body></html>' > static/symbol/index.html

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
