# 🚀 K12数学教育生态系统 - 3分钟快速开始

## ⚡ 一键部署（真正的开箱即用）

### 前置要求
- 安装Docker（如果没有安装）:
  ```bash
  # Linux/Mac
  curl -fsSL https://get.docker.com | sh
  
  # Windows: 下载Docker Desktop
  ```

### 🎯 3步部署

```bash
# 1️⃣ 克隆项目
git clone https://github.com/1832094726/Math_Digital-Intelligent-Ecosystem.git
cd Math_Digital-Intelligent-Ecosystem/docker

# 2️⃣ 一键部署
./deploy.sh

# 3️⃣ 打开浏览器
# 访问: http://localhost
```

## 🎉 就是这么简单！

- ✅ **零配置** - 无需修改任何配置文件
- ✅ **自动化** - 自动下载镜像、启动服务、初始化数据库
- ✅ **预构建** - 使用预构建的Docker镜像，无需编译
- ✅ **健康检查** - 自动等待所有服务就绪

## 🌐 访问地址

部署完成后立即可用：

- 🏠 **主页**: http://localhost
- 📚 **作业系统**: http://localhost/homework
- 🔧 **API**: http://localhost:5000/api
- 📊 **数据库可视化**: http://localhost/database-visualization

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
```

## 🆘 遇到问题？

1. **端口被占用**: 修改 `docker-compose.yml` 中的端口
2. **Docker未启动**: 启动Docker服务
3. **权限问题**: 使用 `sudo` 或管理员权限

---

**从GitHub到运行只需要3个命令，真正的开箱即用！** 🎯
