# 🚀 K12数学教育生态系统 - 一键部署

## ⚡ 超简单部署（3步完成）

```bash
# 1. 克隆项目
git clone https://github.com/1832094726/Math_Digital-Intelligent-Ecosystem.git
cd Math_Digital-Intelligent-Ecosystem/docker

# 2. 一键部署
chmod +x deploy.sh
./deploy.sh

# 3. 访问应用
# 打开浏览器访问: http://localhost
```

## 🎯 零配置特性

- ✅ **无需配置环境变量** - 使用预设的安全默认值
- ✅ **无需构建镜像** - 使用预构建的Docker镜像
- ✅ **云端数据库** - 连接远程OceanBase，无需本地数据库
- ✅ **健康检查** - 自动等待所有服务就绪
- ✅ **中文界面** - 友好的中文提示信息

## 🏗️ 服务架构

```
浏览器 → Nginx (80) → Flask应用 (5000) → 远程OceanBase数据库 + 本地Redis (6379)
```

### 数据库配置
- **远程数据库**: OceanBase云数据库 (MySQL兼容)
- **连接地址**: obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud:3306
- **数据库名**: testccnu
- **优势**: 云端托管，高可用，无需本地维护

## 🌐 访问地址

部署完成后立即可用：

- 🏠 **主页**: http://localhost
- 📚 **作业系统**: http://localhost/homework
- 🔧 **API接口**: http://localhost:5000/api
- 📊 **数据库可视化**: http://localhost/database-visualization

## 🔧 管理命令

```bash
# 查看服务状态
docker-compose ps

# 查看实时日志
docker-compose logs -f

# 重启所有服务
docker-compose restart

# 停止所有服务
docker-compose down

# 完全清理（包括数据）
docker-compose down -v
```

## 📦 预构建镜像

使用预构建的Docker镜像，无需本地编译：

- `matheco/k12-math-ecosystem:latest` - 主应用镜像
- `redis:7-alpine` - 本地缓存
- `nginx:alpine` - 反向代理
- 远程OceanBase数据库 - 云端数据库服务

## 🆘 常见问题

**Q: 端口被占用怎么办？**
A: 修改 `docker-compose.yml` 中的端口映射

**Q: 如何重置数据库？**
A: 运行 `docker-compose down -v` 然后重新部署

**Q: 如何查看错误日志？**
A: 运行 `docker-compose logs app`

## 🎉 就是这么简单！

从GitHub克隆到运行只需要3个命令，真正的开箱即用！
