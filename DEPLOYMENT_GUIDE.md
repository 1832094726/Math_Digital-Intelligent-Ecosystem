# K12数学教育系统 - 完整部署指南

## 🚨 当前问题
- Docker缓存占用40GB空间
- 网页无法访问 (ERR_HTTP_RESPONSE_CODE_FAILURE)
- 需要清理缓存并重新部署

## 🧹 第一步：清理Docker缓存

### 连接服务器
```bash
ssh root@172.104.172.5
```

### 检查空间使用
```bash
# 检查磁盘空间
df -h

# 检查Docker空间使用
docker system df
```

### 全面清理Docker
```bash
# 进入项目目录
cd Math_Digital-Intelligent-Ecosystem/docker

# 停止所有服务
docker-compose -f docker-compose-linux.yml down

# 删除所有停止的容器
docker container prune -f

# 删除所有未使用的镜像（这会释放大量空间）
docker image prune -a -f

# 删除所有未使用的卷
docker volume prune -f

# 删除所有未使用的网络
docker network prune -f

# 删除构建缓存
docker builder prune -a -f

# 系统全面清理
docker system prune -a -f --volumes
```

## 🔧 第二步：修复并重新部署

### 拉取最新代码
```bash
# 拉取包含修复的最新代码
git pull origin main
```

### 重新构建和部署
```bash
# 重新构建（无缓存）
docker-compose -f docker-compose-linux.yml build --no-cache

# 启动服务
docker-compose -f docker-compose-linux.yml up -d
```

## 🔍 第三步：验证修复

### 检查服务状态
```bash
# 查看容器状态
docker-compose -f docker-compose-linux.yml ps

# 查看应用日志
docker-compose -f docker-compose-linux.yml logs app

# 查看nginx日志
docker-compose -f docker-compose-linux.yml logs nginx
```

### 测试访问
```bash
# 测试API
curl http://localhost:8081/api/health

# 测试前端
curl -I http://localhost:8080/homework

# 测试静态资源
curl -I http://localhost:8080/static/css/app.css
```

## 🎯 关键修复内容

### 1. Vue模板文件
已添加 `homework_system/public/index.html`：
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>K12数学教育系统</title>
</head>
<body>
    <div id="app"></div>
</body>
</html>
```

### 2. Flask静态文件路由
已修复 `homework-backend/app.py`：
```python
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static/homework', filename)
```

## 📊 预期结果

清理后应该释放约35-40GB空间，重新部署后：
- ✅ 网页正常显示Vue应用界面
- ✅ CSS样式正确加载
- ✅ JavaScript功能完全可用
- ✅ API接口正常响应

## 🌐 访问地址
- 主页: http://172.104.172.5:8080
- 作业系统: http://172.104.172.5:8080/homework
- API健康检查: http://172.104.172.5:8081/api/health
