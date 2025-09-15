#!/bin/bash

# 创建nginx配置文件

echo "📝 创建nginx配置文件..."

# 确保nginx目录存在
mkdir -p nginx

# 创建nginx.conf
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

echo "✅ nginx配置文件已创建"
ls -la nginx/nginx.conf

echo ""
echo "🚀 现在可以运行部署脚本了:"
echo "./deploy-linux.sh"
