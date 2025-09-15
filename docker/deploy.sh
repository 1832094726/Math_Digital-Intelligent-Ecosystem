#!/bin/bash

# 🚀 K12数学教育生态系统 - 一键部署脚本
# 使用方法: ./deploy.sh

set -e

echo "🎯 K12数学教育数字化智能生态系统"
echo "=================================="
echo "🚀 开始一键部署..."

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

# 检查Docker
check_docker() {
    print_info "检查Docker环境..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        echo "安装命令: curl -fsSL https://get.docker.com | sh"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker未运行，请启动Docker"
        exit 1
    fi
    print_success "Docker环境正常"
}

# 检查Docker Compose
check_compose() {
    print_info "检查Docker Compose..."
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose未安装，尝试使用docker compose"
        if ! docker compose version &> /dev/null; then
            print_error "Docker Compose不可用，请安装Docker Compose"
            exit 1
        fi
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    print_success "Docker Compose可用"
}

# 创建必要目录
setup_dirs() {
    print_info "创建必要目录..."
    mkdir -p ../data/uploads ../logs
    print_success "目录创建完成"
}

# 拉取最新镜像
pull_images() {
    print_info "拉取最新镜像..."
    $COMPOSE_CMD pull
    print_success "镜像拉取完成"
}

# 启动服务
start_services() {
    print_info "启动服务..."
    $COMPOSE_CMD up -d
    print_success "服务启动完成"
}

# 等待服务就绪
wait_services() {
    print_info "等待服务就绪..."

    # 等待MySQL
    print_info "等待数据库启动..."
    for i in {1..30}; do
        if $COMPOSE_CMD exec -T mysql mysqladmin ping -h localhost --silent 2>/dev/null; then
            break
        fi
        sleep 2
        echo -n "."
    done
    echo ""

    # 等待应用
    print_info "等待应用启动..."
    for i in {1..30}; do
        if curl -f http://localhost:5000/api/health &>/dev/null; then
            break
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    print_success "所有服务已就绪"
}

# 显示部署结果
show_result() {
    echo ""
    echo "🎉 部署完成！"
    echo "===================="
    echo ""
    echo "📱 访问地址:"
    echo "   主应用: http://localhost"
    echo "   API接口: http://localhost:5000/api"
    echo "   作业系统: http://localhost/homework"
    echo ""
    echo "🔧 管理命令:"
    echo "   查看状态: $COMPOSE_CMD ps"
    echo "   查看日志: $COMPOSE_CMD logs -f"
    echo "   重启服务: $COMPOSE_CMD restart"
    echo "   停止服务: $COMPOSE_CMD down"
    echo ""
    echo "📊 服务状态:"
    $COMPOSE_CMD ps
}

# 主函数
main() {
    check_docker
    check_compose
    setup_dirs
    pull_images
    start_services
    wait_services
    show_result
}

# 执行部署
main
