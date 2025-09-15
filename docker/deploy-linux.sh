#!/bin/bash

# Linux服务器专用部署脚本
# 适用于CentOS 7 / RHEL 7

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[信息]${NC} $1"; }
print_success() { echo -e "${GREEN}[成功]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[警告]${NC} $1"; }
print_error() { echo -e "${RED}[错误]${NC} $1"; }

echo "🚀 K12数学教育生态系统 - Linux服务器部署"
echo "============================================="

# 检查系统
print_info "检查系统环境..."
if [[ ! -f /etc/redhat-release ]]; then
    print_warning "检测到非RHEL/CentOS系统，脚本可能需要调整"
fi

# 检查Docker
print_info "检查Docker安装..."
if ! command -v docker &> /dev/null; then
    print_error "Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_info "安装docker-compose..."
    curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# 启动Docker服务
print_info "启动Docker服务..."
systemctl start docker
systemctl enable docker

# 配置端口
print_info "配置服务端口..."
read -p "请输入Web端口 (默认8080): " WEB_PORT
WEB_PORT=${WEB_PORT:-8080}

read -p "请输入API端口 (默认8081): " API_PORT  
API_PORT=${API_PORT:-8081}

# 创建临时docker-compose文件
print_info "生成配置文件..."
cat > docker-compose-linux.yml << EOF
version: '3.8'

services:
  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: math_ecosystem_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - math_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Application (本地构建)
  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: math_ecosystem_app
    restart: unless-stopped
    ports:
      - "${API_PORT}:5000"
    environment:
      - FLASK_ENV=production
      # 远程OceanBase数据库配置
      - DB_HOST=obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud
      - DB_PORT=3306
      - DB_USER=hcj
      - DB_PASSWORD=Xv0Mu8_:
      - DB_NAME=testccnu
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=math_ecosystem_secret_key_2024
      - JWT_SECRET_KEY=jwt_secret_math_2024
    volumes:
      - app_data:/app/data
      - app_logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - math_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx Proxy
  nginx:
    image: nginx:alpine
    container_name: math_ecosystem_nginx
    restart: unless-stopped
    ports:
      - "${WEB_PORT}:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      app:
        condition: service_healthy
    networks:
      - math_network

volumes:
  redis_data:
  app_data:
  app_logs:

networks:
  math_network:
    driver: bridge
EOF

# 构建和启动服务
print_info "构建并启动服务..."
docker-compose -f docker-compose-linux.yml up -d --build

# 等待服务启动
print_info "等待服务启动..."
sleep 30

# 检查服务状态
print_info "检查服务状态..."
docker-compose -f docker-compose-linux.yml ps

# 显示访问信息
print_success "部署完成！"
echo ""
echo "🌐 访问地址:"
echo "   主页: http://172.104.172.5:${WEB_PORT}"
echo "   API:  http://172.104.172.5:${API_PORT}/api"
echo ""
echo "🔧 管理命令:"
echo "   查看日志: docker-compose -f docker-compose-linux.yml logs -f"
echo "   重启服务: docker-compose -f docker-compose-linux.yml restart"
echo "   停止服务: docker-compose -f docker-compose-linux.yml down"
echo ""
print_success "K12数学教育生态系统已成功部署到Linux服务器！"
