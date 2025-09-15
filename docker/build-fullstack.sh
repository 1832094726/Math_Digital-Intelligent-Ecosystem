#!/bin/bash

# K12数学教育生态系统 - 全栈构建脚本
# 一键构建 Vue前端 + Flask后端

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[信息]${NC} $1"; }
print_success() { echo -e "${GREEN}[成功]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[警告]${NC} $1"; }
print_error() { echo -e "${RED}[错误]${NC} $1"; }

echo "🚀 K12数学教育生态系统 - 全栈构建"
echo "=================================="

# 检查Docker
print_info "检查Docker环境..."
if ! command -v docker &> /dev/null; then
    print_error "Docker未安装"
    exit 1
fi

# 设置镜像信息
IMAGE_NAME="matheco/k12-math-ecosystem"
VERSION="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${VERSION}"

print_info "构建全栈镜像: $FULL_IMAGE_NAME"

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查前端项目
print_info "检查前端项目..."
if [ ! -f "homework_system/package.json" ]; then
    print_error "未找到Vue前端项目"
    exit 1
fi

# 检查后端项目
print_info "检查后端项目..."
if [ ! -f "homework-backend/requirements.txt" ]; then
    print_error "未找到Flask后端项目"
    exit 1
fi

# 构建Docker镜像（多阶段构建）
print_info "开始多阶段构建..."
print_info "阶段1: 构建Vue前端..."
print_info "阶段2: 构建Flask后端并整合前端..."

docker build -t $FULL_IMAGE_NAME -f docker/Dockerfile .

if [ $? -eq 0 ]; then
    print_success "全栈镜像构建完成！"
else
    print_error "镜像构建失败"
    exit 1
fi

# 显示镜像信息
print_info "镜像信息:"
docker images | grep $IMAGE_NAME

# 测试镜像
print_info "测试镜像..."
CONTAINER_ID=$(docker run -d -p 5001:5000 $FULL_IMAGE_NAME)
sleep 10

if curl -f http://localhost:5001/api/health &>/dev/null; then
    print_success "镜像测试通过"
else
    print_warning "镜像测试失败，但镜像已构建完成"
fi

# 停止测试容器
docker stop $CONTAINER_ID >/dev/null 2>&1
docker rm $CONTAINER_ID >/dev/null 2>&1

# 询问是否推送
echo ""
read -p "是否推送到Docker Hub? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "推送镜像到Docker Hub..."
    docker push $FULL_IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        print_success "镜像推送完成"
        echo ""
        echo "🎉 全栈镜像已发布: $FULL_IMAGE_NAME"
        echo ""
        echo "📋 包含组件:"
        echo "   ✅ Vue.js前端 (homework_system)"
        echo "   ✅ Flask后端 (homework-backend)"
        echo "   ✅ 数学符号键盘"
        echo "   ✅ 静态资源"
        echo ""
        echo "🚀 现在可以在任何地方一键部署:"
        echo "   docker run -d -p 8080:5000 $FULL_IMAGE_NAME"
    else
        print_error "镜像推送失败"
        exit 1
    fi
else
    print_info "跳过推送"
    echo ""
    echo "💡 本地全栈镜像已构建完成"
    echo "   可以使用: docker run -d -p 8080:5000 $FULL_IMAGE_NAME"
fi

print_success "全栈构建流程完成！"
