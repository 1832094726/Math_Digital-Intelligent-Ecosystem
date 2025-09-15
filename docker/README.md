# Docker 部署方案

## 📁 文件结构

```
docker/
├── Dockerfile              # 主应用容器构建文件
├── docker-compose.yml      # 服务编排配置
├── .env.example           # 环境变量模板
├── .dockerignore          # Docker构建忽略文件
├── deploy.sh              # 本地部署脚本
├── deploy-remote.sh       # 远程部署脚本
├── nginx/
│   └── nginx.conf         # Nginx配置文件
├── DEPLOYMENT_GUIDE.md    # 详细部署指南
└── README.md              # 本文件
```

## 🚀 快速开始

### 1. 本地部署

```bash
# 进入docker目录
cd docker

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件设置数据库密码等

# 执行部署
chmod +x deploy.sh
./deploy.sh
```

### 2. 远程服务器部署

```bash
# 进入docker目录
cd docker

# 执行远程部署
chmod +x deploy-remote.sh
./deploy-remote.sh
```

## 🏗️ 服务架构

- **MySQL 8.0**: 主数据库 (端口: 3306)
- **Redis 7**: 缓存服务 (端口: 6379)  
- **Flask App**: Python后端 (端口: 5000)
- **Nginx**: 反向代理 (端口: 80/443)

## 📋 环境变量

复制 `.env.example` 到 `.env` 并配置以下变量：

```bash
# 数据库配置
MYSQL_ROOT_PASSWORD=root123456
MYSQL_DATABASE=math_ecosystem
MYSQL_USER=mathuser
MYSQL_PASSWORD=mathpass123

# 应用配置
SECRET_KEY=your-secret-key
FLASK_ENV=production
```

## 🔧 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 完全清理
docker-compose down -v --remove-orphans
```

## 🌐 访问地址

部署成功后可通过以下地址访问：

- **主应用**: http://localhost
- **API接口**: http://localhost/api
- **数据库可视化**: http://localhost/database-visualization

## 📖 详细文档

更多详细信息请参考 [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)

## 🔒 服务器信息

- **IP**: 172.104.172.5
- **用户**: root
- **密码**: CCNU_rqmWLlqDmx^XF6bOLhF%vSNe*7cYPwk

## 🆘 故障排除

1. **端口冲突**: 检查端口占用 `netstat -tulpn | grep :80`
2. **内存不足**: 检查系统资源 `free -h`
3. **权限问题**: 确保脚本有执行权限 `chmod +x *.sh`
4. **网络问题**: 检查防火墙设置和网络连接

---

**注意**: 首次部署可能需要几分钟时间下载Docker镜像和构建应用。
