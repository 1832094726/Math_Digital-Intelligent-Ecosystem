# K12数学教育数字化智能生态系统 - Docker部署指南

## 📋 概述

本指南提供了完整的Docker容器化部署方案，支持本地开发和远程服务器部署。

## 🏗️ 架构组件

### 服务组件
- **MySQL 8.0**: 主数据库
- **Redis 7**: 缓存和会话存储
- **Flask应用**: Python后端API服务
- **Nginx**: 反向代理和负载均衡
- **Vue.js前端**: 作业管理系统界面
- **符号键盘**: 数学符号输入组件

### 网络架构
```
Internet → Nginx (80/443) → Flask App (5000) → MySQL (3306) + Redis (6379)
```

## 🚀 快速开始

### 1. 本地部署

#### 前置要求
- Docker 20.10+
- Docker Compose 2.0+
- 至少4GB可用内存
- 至少10GB可用磁盘空间

#### 部署步骤
```bash
# 1. 克隆项目
git clone https://github.com/1832094726/Math_Digital-Intelligent-Ecosystem.git
cd Math_Digital-Intelligent-Ecosystem

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置数据库密码等配置

# 3. 执行部署
chmod +x deploy.sh
./deploy.sh

# 4. 访问应用
# 主应用: http://localhost
# API文档: http://localhost/api/docs
```

### 2. 远程服务器部署

#### 服务器信息
- **IP地址**: 172.104.172.5
- **用户名**: root
- **密码**: CCNU_rqmWLlqDmx^XF6bOLhF%vSNe*7cYPwk

#### 自动部署
```bash
# 执行远程部署脚本
chmod +x deploy-remote.sh
./deploy-remote.sh
```

#### 手动部署步骤
```bash
# 1. 连接服务器
ssh root@172.104.172.5

# 2. 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl start docker
systemctl enable docker

# 3. 安装Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 4. 上传项目文件
scp -r . root@172.104.172.5:/opt/math-ecosystem/

# 5. 部署应用
cd /opt/math-ecosystem
cp .env.example .env
./deploy.sh prod
```

## ⚙️ 配置说明

### 环境变量配置 (.env)
```bash
# 数据库配置
MYSQL_ROOT_PASSWORD=root123456
MYSQL_DATABASE=math_ecosystem
MYSQL_USER=mathuser
MYSQL_PASSWORD=mathpass123

# 应用配置
SECRET_KEY=your-very-secret-key-change-this-in-production
FLASK_ENV=production

# 安全配置
JWT_SECRET_KEY=your-jwt-secret-key-change-this
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Nginx配置
- 反向代理配置: `nginx/nginx.conf`
- SSL证书路径: `nginx/ssl/`
- 静态文件缓存: 1年
- API请求限制: 10req/s
- 登录请求限制: 1req/s

## 🔧 运维命令

### Docker Compose命令
```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f [service_name]

# 重启服务
docker-compose restart [service_name]

# 停止所有服务
docker-compose down

# 完全清理（包括数据卷）
docker-compose down -v --remove-orphans

# 重新构建镜像
docker-compose build --no-cache

# 更新服务
docker-compose pull
docker-compose up -d
```

### 数据库管理
```bash
# 连接MySQL
docker-compose exec mysql mysql -u mathuser -p math_ecosystem

# 备份数据库
docker-compose exec mysql mysqldump -u mathuser -p math_ecosystem > backup.sql

# 恢复数据库
docker-compose exec -T mysql mysql -u mathuser -p math_ecosystem < backup.sql

# 查看Redis
docker-compose exec redis redis-cli
```

### 应用管理
```bash
# 查看应用日志
docker-compose logs -f app

# 进入应用容器
docker-compose exec app bash

# 运行数据库迁移
docker-compose exec app python scripts/migrate.py

# 重启应用
docker-compose restart app
```

## 🔒 安全配置

### 防火墙设置
```bash
# Ubuntu/Debian
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable

# CentOS/RHEL
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --permanent --add-port=80/tcp
firewall-cmd --permanent --add-port=443/tcp
firewall-cmd --reload
```

### SSL证书配置
```bash
# 安装Certbot
apt-get install certbot

# 获取SSL证书
certbot certonly --webroot -w /var/www/certbot -d your-domain.com

# 自动续期
echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -
```

## 📊 监控和日志

### 健康检查
- 应用健康检查: `http://localhost:5000/api/health`
- 数据库连接检查: `docker-compose exec mysql mysqladmin ping`
- Redis连接检查: `docker-compose exec redis redis-cli ping`

### 日志位置
- 应用日志: `logs/app.log`
- Nginx访问日志: `/var/log/nginx/access.log`
- Nginx错误日志: `/var/log/nginx/error.log`
- MySQL日志: Docker容器内 `/var/log/mysql/`

## 🚨 故障排除

### 常见问题

#### 1. 端口冲突
```bash
# 检查端口占用
netstat -tulpn | grep :80
netstat -tulpn | grep :3306

# 修改docker-compose.yml中的端口映射
```

#### 2. 内存不足
```bash
# 检查内存使用
free -h
docker stats

# 增加swap空间
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
```

#### 3. 数据库连接失败
```bash
# 检查MySQL状态
docker-compose logs mysql

# 重置数据库
docker-compose down
docker volume rm math-ecosystem_mysql_data
docker-compose up -d
```

#### 4. 前端构建失败
```bash
# 检查Node.js版本
docker-compose exec app node --version

# 清理npm缓存
docker-compose exec app npm cache clean --force
```

## 📈 性能优化

### 数据库优化
- 启用查询缓存
- 配置适当的缓冲池大小
- 定期优化表结构

### 应用优化
- 启用Redis缓存
- 配置静态文件CDN
- 使用Gunicorn多进程部署

### Nginx优化
- 启用Gzip压缩
- 配置静态文件缓存
- 使用HTTP/2

## 📞 技术支持

如遇到部署问题，请检查：
1. Docker和Docker Compose版本
2. 系统资源使用情况
3. 网络连接状态
4. 日志文件中的错误信息

---

**部署完成后，请访问 http://your-server-ip 查看应用运行状态！**
