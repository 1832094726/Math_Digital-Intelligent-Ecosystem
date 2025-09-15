# 🎓 K12数学教育数字化智能生态系统

基于推荐技术提升学生作业效率的全栈数学教育平台，包含Vue.js前端、Flask后端、智能符号推荐和云端数据库。

## 🚀 快速部署

### Linux服务器一键部署
```bash
git clone https://github.com/1832094726/Math_Digital-Intelligent-Ecosystem.git
cd Math_Digital-Intelligent-Ecosystem/docker
chmod +x deploy-linux.sh
./deploy-linux.sh
```

部署完成后访问：
- **主页**: http://172.104.172.5:8080
- **API**: http://172.104.172.5:8081/api

详细部署说明请参考：[LINUX_DEPLOY_GUIDE.md](LINUX_DEPLOY_GUIDE.md)

## 🏗️ 系统架构

```
用户浏览器 → Nginx → Flask后端 → 远程OceanBase数据库
                  ↗ Vue.js前端    ↘ Redis缓存
```

## 📁 项目结构

- **docker/**: Docker部署配置
- **homework_system/**: Vue.js前端应用
- **homework-backend/**: Flask后端API
- **Subject_symbol_dynamic_keyboard/**: 数学符号推荐系统
- **database-visualization/**: 数据库可视化工具
- **architect/**: 系统架构设计文档

## ✨ 核心功能

- 🎯 **作业管理系统** - 完整的作业发布、提交、批改流程
- 🔢 **智能符号推荐** - 基于深度学习的数学符号智能推荐
- 📊 **学习分析** - 学生学习行为分析和可视化
- 🌐 **云端数据库** - 基于OceanBase的高可用数据存储
- 📱 **响应式设计** - 支持多设备访问

## 🔧 技术栈

- **前端**: Vue.js 2.6, Element UI, ECharts
- **后端**: Flask, Python 3.9
- **数据库**: OceanBase (MySQL兼容)
- **缓存**: Redis
- **部署**: Docker, Nginx
- **AI**: BERT, 协同过滤, 深度学习推荐
