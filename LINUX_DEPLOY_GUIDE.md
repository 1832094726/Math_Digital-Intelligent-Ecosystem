# 🐧 Linux服务器部署指南

## 🎯 适用环境
- **服务器**: 172.104.172.5 (CentOS 7)
- **架构**: x86_64
- **Docker**: 已安装

## 🚀 方案一：Linux服务器直接构建（推荐）

### 1️⃣ 克隆项目
```bash
# SSH登录到Linux服务器
ssh root@172.104.172.5

# 克隆项目
git clone https://github.com/1832094726/Math_Digital-Intelligent-Ecosystem.git
cd Math_Digital-Intelligent-Ecosystem/docker
```

### 2️⃣ 运行Linux部署脚本
```bash
# 给脚本执行权限
chmod +x deploy-linux.sh

# 运行部署脚本
./deploy-linux.sh
```

### 3️⃣ 配置端口
脚本会询问端口配置：
- **Web端口**: 建议8080 (避免与现有服务冲突)
- **API端口**: 建议8081

### 4️⃣ 访问系统
```
主页: http://172.104.172.5:8080
API:  http://172.104.172.5:8081/api
```

---

## 🚀 方案二：Windows构建推送

### Windows端操作

#### 1️⃣ 安装Docker Desktop
- 下载: https://www.docker.com/products/docker-desktop
- 安装并启动Docker Desktop

#### 2️⃣ 构建镜像
```cmd
# 在Windows项目目录
cd "E:\program development\The Digital and Intelligent Ecosystem for K-12 Mathematics Education\docker"

# 运行构建脚本
build-windows.bat
```

#### 3️⃣ 推送到Docker Hub
- 脚本会询问是否推送
- 选择 `y` 推送到Docker Hub
- 需要Docker Hub账号登录

### Linux端操作

#### 1️⃣ 拉取镜像
```bash
# 拉取预构建镜像
docker pull matheco/k12-math-ecosystem:latest
```

#### 2️⃣ 创建部署配置
```bash
# 创建docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: math_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  app:
    image: matheco/k12-math-ecosystem:latest
    container_name: math_app
    restart: unless-stopped
    ports:
      - "8081:5000"  # API端口
    environment:
      - FLASK_ENV=production
      - DB_HOST=obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud
      - DB_PORT=3306
      - DB_USER=hcj
      - DB_PASSWORD=Xv0Mu8_:
      - DB_NAME=testccnu
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis

  nginx:
    image: nginx:alpine
    container_name: math_nginx
    restart: unless-stopped
    ports:
      - "8080:80"   # Web端口
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app

volumes:
  redis_data:
EOF
```

#### 3️⃣ 创建Nginx配置
```bash
cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server app:5000;
    }

    server {
        listen 80;
        server_name _;

        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /api/ {
            proxy_pass http://backend/api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
```

#### 4️⃣ 启动服务
```bash
docker-compose up -d
```

---

## 🔧 管理命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 更新镜像
docker-compose pull
docker-compose up -d
```

## 🌐 访问地址

部署完成后：
- **主页**: http://172.104.172.5:8080
- **作业系统**: http://172.104.172.5:8080/homework
- **API接口**: http://172.104.172.5:8081/api
- **健康检查**: http://172.104.172.5:8081/api/health

## 🆘 故障排除

### 端口冲突
```bash
# 检查端口占用
netstat -tlnp | grep :8080
netstat -tlnp | grep :8081

# 修改docker-compose.yml中的端口映射
```

### 服务无法启动
```bash
# 查看详细日志
docker-compose logs app
docker logs math_app

# 检查数据库连接
docker exec -it math_app curl http://localhost:5000/api/health
```

### 防火墙设置
```bash
# CentOS 7 开放端口
firewall-cmd --permanent --add-port=8080/tcp
firewall-cmd --permanent --add-port=8081/tcp
firewall-cmd --reload
```

---

**推荐使用方案一（Linux直接构建），更简单快捷！** 🎯
