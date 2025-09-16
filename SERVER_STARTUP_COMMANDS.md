# K-12数学教育系统 - 服务器启动命令文档

## 📋 系统概述

本系统包含4个主要服务，运行在服务器 `172.104.172.5` 上：

| 服务名称 | 端口 | 技术栈 | 功能描述 |
|---------|------|--------|----------|
| 前端作业系统 | 8080 | Vue.js | 学生作业管理界面 |
| 后端API服务 | 8081 | Flask | 核心业务API |
| 数据库可视化 | 8082 | 静态文件 | 数据库关系图和API可视化 |
| 数据库API服务 | 5001 | Flask | 数据库查询API |
| 后端API服务(备用) | 5000 | Flask | 核心业务API备用端口 |

## 🚀 启动命令

### 1. 后端API服务 (端口8081)

```bash
# 进入后端目录
cd /root/Math_Digital-Intelligent-Ecosystem/homework-backend

# 激活虚拟环境
source venv/bin/activate

# 启动后端服务
python app.py
```

**后台运行方式：**
```bash
cd /root/Math_Digital-Intelligent-Ecosystem/homework-backend
source venv/bin/activate
nohup python app.py > backend.log 2>&1 &
```

### 2. 前端作业系统 (端口8080)

```bash
# 进入前端目录
cd /root/Math_Digital-Intelligent-Ecosystem/homework_system

# 启动前端开发服务器
npm run serve
```

**后台运行方式：**
```bash
cd /root/Math_Digital-Intelligent-Ecosystem/homework_system
nohup npm run serve > frontend.log 2>&1 &
```

### 3. 数据库可视化界面 (端口8082)

**注意**: 需要先将本地修改后的文件上传到服务器

```bash
# 进入数据库可视化目录
cd /root/Math_Digital-Intelligent-Ecosystem/database-visualization

# 启动静态文件服务器
python3 -m http.server 8082
```

**后台运行方式：**
```bash
cd /root/Math_Digital-Intelligent-Ecosystem/database-visualization
nohup python3 -m http.server 8082 > static_server.log 2>&1 &
```

### 4. 数据库API服务 (端口5001)

```bash
# 进入数据库可视化目录
cd /root/Math_Digital-Intelligent-Ecosystem/database-visualization

# 启动数据库API服务
python3 api-server.py
```

**后台运行方式：**
```bash
cd /root/Math_Digital-Intelligent-Ecosystem/database-visualization
nohup python3 api-server.py > db_visualization.log 2>&1 &
```


## 🔧 一键启动脚本

### 创建启动脚本

```bash
# 创建启动脚本
cat > /root/start_all_services.sh << 'EOF'
#!/bin/bash

echo "🚀 启动K-12数学教育系统所有服务"
echo "=================================="

# 1. 启动后端API服务
echo "📡 启动后端API服务 (端口8081)..."
cd /root/Math_Digital-Intelligent-Ecosystem/homework-backend
source venv/bin/activate
nohup python app.py > backend.log 2>&1 &
echo "✅ 后端API服务已启动"

# 2. 启动前端作业系统
echo "🎨 启动前端作业系统 (端口8080)..."
cd /root/Math_Digital-Intelligent-Ecosystem/homework_system
nohup npm run serve > frontend.log 2>&1 &
echo "✅ 前端作业系统已启动"

# 3. 启动数据库可视化界面
echo "📊 启动数据库可视化界面 (端口8082)..."
cd /root/Math_Digital-Intelligent-Ecosystem/database-visualization
nohup python3 -m http.server 8082 > static_server.log 2>&1 &
echo "✅ 数据库可视化界面已启动"

# 4. 启动数据库API服务
echo "🗄️ 启动数据库API服务 (端口5001)..."
cd /root/Math_Digital-Intelligent-Ecosystem/database-visualization
nohup python3 api-server.py > db_visualization.log 2>&1 &
echo "✅ 数据库API服务已启动"


echo "=================================="
echo "🎉 所有服务启动完成！"
echo ""
echo "访问地址："
echo "  - 前端作业系统: http://172.104.172.5:8080"
echo "  - 后端API服务: http://172.104.172.5:8081"
echo "  - 数据库可视化: http://172.104.172.5:8082"
echo "  - 数据库API: http://172.104.172.5:5001"
echo ""
echo "查看日志："
echo "  - 后端日志: tail -f /root/Math_Digital-Intelligent-Ecosystem/homework-backend/backend.log"
echo "  - 前端日志: tail -f /root/Math_Digital-Intelligent-Ecosystem/homework_system/frontend.log"
echo "  - 数据库可视化日志: tail -f /root/Math_Digital-Intelligent-Ecosystem/database-visualization/static_server.log"
echo "  - 数据库API日志: tail -f /root/Math_Digital-Intelligent-Ecosystem/database-visualization/db_visualization.log"
EOF

# 给脚本执行权限
chmod +x /root/start_all_services.sh
```

### 创建停止脚本

```bash
# 创建停止脚本
cat > /root/stop_all_services.sh << 'EOF'
#!/bin/bash

echo "🛑 停止K-12数学教育系统所有服务"
echo "=================================="

# 停止后端API服务
echo "📡 停止后端API服务..."
pkill -f "python app.py"
echo "✅ 后端API服务已停止"

# 停止前端作业系统
echo "🎨 停止前端作业系统..."
pkill -f "npm run serve"
echo "✅ 前端作业系统已停止"

# 停止数据库可视化界面
echo "📊 停止数据库可视化界面..."
pkill -f "python3 -m http.server 8082"
echo "✅ 数据库可视化界面已停止"

# 停止数据库API服务
echo "🗄️ 停止数据库API服务..."
pkill -f "api-server.py"
echo "✅ 数据库API服务已停止"


echo "=================================="
echo "🎉 所有服务已停止！"
EOF

# 给脚本执行权限
chmod +x /root/stop_all_services.sh
```

## 📊 服务状态检查

### 检查所有服务状态

```bash
# 检查端口占用情况
netstat -tlnp | grep -E ":(8080|8081|8082|5000|5001)"

# 检查进程状态
ps aux | grep -E "(python app.py|npm run serve|api-server.py|http.server 8082)"
```

### 检查服务健康状态

```bash
# 检查后端API健康状态
curl -s http://172.104.172.5:8081/api/health

# 检查数据库API健康状态
curl -s http://172.104.172.5:5001/api/health

# 检查前端服务
curl -s -I http://172.104.172.5:8080

```

## 📝 日志管理

### 查看实时日志

```bash
# 查看后端日志
tail -f /root/Math_Digital-Intelligent-Ecosystem/homework-backend/backend.log

# 查看前端日志
tail -f /root/Math_Digital-Intelligent-Ecosystem/homework_system/frontend.log

# 查看数据库API日志
tail -f /root/Math_Digital-Intelligent-Ecosystem/database-visualization/db_visualization.log

```

### 查看所有日志

```bash
# 查看所有服务日志
tail -f /root/Math_Digital-Intelligent-Ecosystem/*/backend.log \
      /root/Math_Digital-Intelligent-Ecosystem/*/frontend.log \
      /root/Math_Digital-Intelligent-Ecosystem/*/db_visualization.log
```

## 🔧 故障排除

### 端口冲突解决

```bash
# 查看端口占用
lsof -i :8080
lsof -i :8081
lsof -i :5000
lsof -i :5001

# 杀死占用端口的进程
kill -9 <PID>
```

### 服务重启

```bash
# 重启单个服务
pkill -f "python app.py" && cd /root/Math_Digital-Intelligent-Ecosystem/homework-backend && source venv/bin/activate && nohup python app.py > backend.log 2>&1 &

# 重启所有服务
/root/stop_all_services.sh && sleep 5 && /root/start_all_services.sh
```

## 📋 环境要求

### 系统要求
- CentOS 7+ / Ubuntu 18+
- Python 3.6+
- Node.js 16+
- npm 8+

### Python依赖
```bash
# 后端依赖
pip3 install flask==2.0.3 flask-cors==3.0.10 werkzeug==2.0.3 pymysql==1.0.2

# 数据库可视化依赖
pip3 install flask flask-cors pymysql
```

### Node.js依赖
```bash
# 前端依赖
cd /root/Math_Digital-Intelligent-Ecosystem/homework_system
npm install
```

## 🌐 访问地址总结

| 服务 | 访问地址 | 功能 |
|------|----------|------|
| 前端作业系统 | http://172.104.172.5:8080 | 学生作业管理界面 |
| 后端API服务 | http://172.104.172.5:8081 | 核心业务API |
| 数据库API | http://172.104.172.5:5001 | 数据库查询API |

---

**创建时间**: 2024年9月16日  
**服务器**: 172.104.172.5  
**维护者**: AI Assistant
