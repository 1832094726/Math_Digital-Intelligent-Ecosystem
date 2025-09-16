# 传统方式部署信息

## 🎉 部署成功！

### 服务状态
- ✅ 后端服务：运行在 http://0.0.0.0:8081
- ✅ 前端服务：运行在 http://0.0.0.0:8082
- ✅ API连接：正常

### 访问地址
- **前端应用**：http://172.104.172.5:8082
- **后端API**：http://172.104.172.5:8081/api/

### 启动命令
#### 后端启动
```bash
cd Math_Digital-Intelligent-Ecosystem/homework-backend
source venv/bin/activate
python simple_app.py
```

#### 前端启动
```bash
cd Math_Digital-Intelligent-Ecosystem/homework_system
npm run serve
```

### 系统架构
- **前端**：Vue.js 2.6.11 + Element UI
- **后端**：Flask 2.0.3 + Python 3.6.8
- **部署方式**：传统进程方式，无Docker

### 可用API端点
- GET /api/health - 健康检查
- GET /api/homework/list - 获取作业列表
- GET /api/homework/<id> - 获取作业详情
- POST /api/recommend/symbols - 获取符号推荐

### 注意事项
1. 服务器已清理Docker相关内容
2. 文件监听器限制已优化
3. 后端使用简化版本，包含基础功能
4. 前端代理已配置指向8081端口

### 进程管理
可以使用以下命令管理服务：
```bash
# 查看运行的服务
ps aux | grep -E "python|node"

# 停止服务
pkill -f "simple_app.py"
pkill -f "npm run serve"
```
