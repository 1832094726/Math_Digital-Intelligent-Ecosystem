#!/bin/bash

# 构建并推送Docker镜像到Docker Hub

set -e

IMAGE_NAME="matheco/k12-math-ecosystem"
VERSION="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${VERSION}"

echo "🏗️ 构建K12数学教育生态系统Docker镜像"
echo "========================================="

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[信息]${NC} $1"; }
print_success() { echo -e "${GREEN}[成功]${NC} $1"; }

# 检查Docker
print_info "检查Docker环境..."
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    exit 1
fi

# 构建镜像
print_info "构建Docker镜像: $FULL_IMAGE_NAME"
docker build -t $FULL_IMAGE_NAME -f Dockerfile ..

print_success "镜像构建完成"

# 测试镜像
print_info "测试镜像..."
docker run --rm -d --name test-container -p 5001:5000 $FULL_IMAGE_NAME
sleep 10

if curl -f http://localhost:5001/api/health &>/dev/null; then
    print_success "镜像测试通过"
else
    echo "警告: 镜像测试失败，但继续推送"
fi

docker stop test-container 2>/dev/null || true

# 推送到Docker Hub
print_info "推送镜像到Docker Hub..."
echo "请确保已登录Docker Hub: docker login"
read -p "是否继续推送? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push $FULL_IMAGE_NAME
    print_success "镜像推送完成"
    echo ""
    echo "🎉 镜像已发布: $FULL_IMAGE_NAME"
    echo "用户现在可以直接使用预构建镜像部署！"
else
    print_info "跳过推送"
fi

echo ""
echo "📋 镜像信息:"
docker images | grep $IMAGE_NAME
