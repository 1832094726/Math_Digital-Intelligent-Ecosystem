#!/bin/bash

# 修复nginx配置文件挂载问题

echo "🔧 修复nginx配置文件挂载问题..."
echo "=================================="

# 1. 检查nginx配置文件是否存在
echo "🔍 检查nginx配置文件..."
if [ -f "nginx/nginx.conf" ]; then
    echo "✅ nginx.conf 文件存在"
    ls -la nginx/nginx.conf
else
    echo "❌ nginx.conf 文件不存在，创建默认配置..."
    
    mkdir -p nginx
    
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:5000;
    }
    
    server {
        listen 80;
        server_name _;
        
        # 静态文件
        location /static/ {
            proxy_pass http://app;
        }
        
        # API路由
        location /api/ {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 前端路由
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF
    
    echo "✅ 已创建默认nginx配置"
fi

# 2. 检查文件权限
echo ""
echo "🔍 检查文件权限..."
chmod 644 nginx/nginx.conf
ls -la nginx/nginx.conf

# 3. 停止服务
echo ""
echo "🛑 停止服务..."
docker-compose -f docker-compose-linux.yml down

# 4. 重新启动
echo ""
echo "🚀 重新启动服务..."
docker-compose -f docker-compose-linux.yml up -d

# 5. 等待启动
echo ""
echo "⏳ 等待服务启动..."
sleep 20

# 6. 检查状态
echo ""
echo "📊 检查服务状态..."
docker-compose -f docker-compose-linux.yml ps

# 7. 检查nginx日志
echo ""
echo "📋 nginx日志:"
docker-compose -f docker-compose-linux.yml logs --tail=5 nginx

# 8. 测试访问
echo ""
echo "🌐 测试访问..."
curl -s -I http://localhost:8080/ | head -3

echo ""
echo "=================================="
echo "✅ nginx修复完成！"
echo "🌐 访问: http://172.104.172.5:8080/homework"
