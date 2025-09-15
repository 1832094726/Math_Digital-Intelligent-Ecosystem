# 🚀 K12数学教育生态系统 - 一键部署

## ⚡ 一键部署（Linux服务器）

```bash
# 1. 克隆项目
git clone https://github.com/1832094726/Math_Digital-Intelligent-Ecosystem.git
cd Math_Digital-Intelligent-Ecosystem/docker

# 2. 一键部署（全栈构建）
chmod +x deploy-linux.sh
./deploy-linux.sh

# 3. 访问应用
# http://172.104.172.5:8080 (Web端口)
# http://172.104.172.5:8081 (API端口)
```

## 🎯 全栈构建特性

- ✅ **Vue前端自动构建** - 自动构建homework_system前端项目
- ✅ **Flask后端集成** - 包含完整的API服务
- ✅ **云端数据库** - 连接远程OceanBase，无需本地数据库
- ✅ **端口自定义** - 避免与现有服务冲突
- ✅ **健康检查** - 自动等待所有服务就绪

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

## 🆘 常见问题

**Q: 端口被占用怎么办？**
A: 部署脚本会询问端口配置，选择其他端口即可

**Q: 如何查看错误日志？**
A: 运行 `docker-compose -f docker-compose-linux.yml logs -f`

**Q: 如何重启服务？**
A: 运行 `docker-compose -f docker-compose-linux.yml restart`

## 🎉 就是这么简单！

一个脚本完成Vue前端 + Flask后端的完整部署！
