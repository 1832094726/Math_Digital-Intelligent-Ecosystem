# 🐧 K12数学教育生态系统 - Linux部署指南

## 🎯 服务器环境
- **服务器**: 172.104.172.5 (CentOS 7)
- **Docker**: 已安装

## 🚀 一键部署

### 1️⃣ 克隆项目
```bash
# SSH登录到Linux服务器
ssh root@172.104.172.5

# 克隆项目
git clone https://github.com/1832094726/Math_Digital-Intelligent-Ecosystem.git
cd Math_Digital-Intelligent-Ecosystem/docker
```

### 2️⃣ 运行Linux部署脚本（全栈构建）
```bash
# 给脚本执行权限
chmod +x deploy-linux.sh

# 运行部署脚本（自动构建Vue前端 + Flask后端）
./deploy-linux.sh
```

**构建过程**：
- 🔄 **阶段1**: 使用Node.js构建Vue前端
- 🔄 **阶段2**: 构建Flask后端并整合前端静态文件
- 🎯 **结果**: 单个Docker镜像包含完整的全栈应用

### 3️⃣ 配置端口
脚本会询问端口配置：
- **Web端口**: 建议8080 (避免与现有服务冲突)
- **API端口**: 建议8081

### 4️⃣ 访问系统
```
主页: http://172.104.172.5:8080
API:  http://172.104.172.5:8081/api
```



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

**一个脚本完成Vue前端 + Flask后端的完整部署！** 🎯
