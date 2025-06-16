# K-12数学教育数字化智能生态系统

## 项目概述

本项目旨在构建一个全面的K-12数学教育数字化智能生态系统，集成作业管理、知识图谱、智能推荐、数学符号输入等功能，为中小学数学教育提供全方位的数字化支持。

## 技术架构

### 前端架构
- 框架：Vue.js 2.x
- 状态管理：Vuex
- 路由管理：Vue Router
- UI组件库：Element UI
- 数学公式渲染：MathJax/KaTeX
- 图表可视化：ECharts

### 后端架构
- 语言：Python
- Web框架：Flask
- 数据库：SQLite/MongoDB（开发阶段）
- API设计：RESTful API
- 认证：JWT

### 智能推荐系统
- 基于知识图谱的内容推荐
- 基于用户行为的个性化推荐
- 基于学习路径的进阶推荐

### 数据存储
- 用户数据：用户信息、学习记录、作业记录
- 内容数据：题库、知识点、学习资源
- 关系数据：知识点关联、学习路径

## 核心功能模块

### 1. 作业系统
- 作业发布与管理
- 作业完成与提交
- 自动批改与反馈
- 作业分析与报告

### 2. 知识图谱
- 数学知识点体系构建
- 知识点关联与依赖
- 学习路径规划
- 知识掌握度评估
- 题目知识点自动提取与标注

### 3. 智能推荐
- 个性化学习内容推荐
- 知识点巩固练习推荐
- 薄弱环节针对性训练
- 学习资源智能匹配

### 4. 数学符号输入系统
- 常用数学符号快速输入
- 手写公式识别
- 公式编辑与渲染
- 符号推荐与联想

## 项目恢复与修复记录

### 2023-06-15: Vue项目恢复
1. 修复了package.json中的依赖版本问题，确保与Vue 2.x兼容
2. 重新安装了项目依赖
3. 启动开发服务器验证基本功能

### 2023-06-16: 前后端连接修复
1. 修改了前端服务`homeworkService.js`，将模拟数据改为使用axios调用后端API
2. 修复了Vue配置文件中的API代理设置，确保API请求正确转发到后端
3. 确保前后端API路径一致，前端保留`/api`前缀，与后端路由匹配
4. 同时启动前端和后端服务，验证数据流通

### 2023-06-17: 前后端路径匹配问题解决
1. 发现问题：前端通过`/api/homework/list`请求作业列表，但请求被转发到后端的`/api/api/homework/list`路径，导致404错误
2. 解决方案：修改Vue配置文件中的代理设置，将`pathRewrite`从`'^/api': ''`改为`'^/api': '/api'`，确保前端的`/api`前缀被正确保留并转发到后端
3. 验证：重启前端和后端服务，确认数据正常流通，作业列表可以正确显示

### 2023-06-18: 修复后端路由404错误
1. 发现问题：后端日志显示有请求直接访问`/homework/list`而不是`/api/homework/list`，导致404错误
2. 原因分析：可能是某些前端请求没有正确添加`/api`前缀，或者是代理配置未完全生效
3. 解决方案：在后端Flask应用中添加路由重定向，将`/homework/list`请求重定向到正确的`/api/homework/list`路由
4. 实现方法：在`app.py`中添加新的路由处理函数，使用Flask的`redirect`函数进行重定向
5. 验证：重启后端服务，确认直接访问`/homework/list`的请求能够正确重定向并返回数据

### 2023-06-19: 添加题目知识点提取功能
1. 需求：根据题目内容自动提取相关知识点，用于智能推荐和学习路径规划
2. 实现方法：
   - 创建新服务`knowledge_service.py`，实现知识点提取功能
   - 使用简单的关键词匹配方法，根据题目文本提取相关知识点
   - 支持通过题目ID或题目内容直接获取知识点
3. API设计：
   - 添加新的API端点`/api/knowledge/question`，支持GET和POST方法
   - GET方法：通过URL参数`questionId`或`text`获取知识点
   - POST方法：通过请求体提供题目ID或题目内容
4. 前端集成：
   - 创建新的服务`knowledgeService.js`，封装知识点API调用
   - 修改`KnowledgeRecommendation.vue`组件，使用API获取知识点
   - 添加数据转换逻辑，将后端返回的知识点数据转换为前端所需格式
5. 问题修复：
   - 修复前端作业详情页面显示问题，将`currentHomework.problems`改为`currentHomework.questions`，与后端数据结构保持一致
   - 添加对选择题的特殊处理，显示选项列表
6. 后续优化计划：
   - 引入机器学习模型，提高知识点提取的准确性
   - 建立更完善的知识点关联网络，支持知识点间的关系推理
   - 结合用户学习历史，提供个性化的知识点推荐

### 2023-06-20: 修复作业详情页面显示问题
1. 发现问题：作业详情页面加载后显示loading状态，无法正确显示作业内容
2. 原因分析：前端组件中使用了`problems`字段，而后端API返回的是`questions`字段，导致数据无法正确解析
3. 解决方案：修改前端组件`IntegratedHomeworkPage.vue`，将所有`problems`相关的引用改为`questions`
4. 实现方法：
   - 修改模板中的`v-for`循环，使用`currentHomework.questions`而不是`currentHomework.problems`
   - 更新计算属性`currentProblem`和`canSubmit`，使用`questions`而不是`problems`
   - 添加对选择题的特殊处理，显示单选按钮组
5. 验证：重启前端服务，确认作业详情页面可以正确显示题目内容和选项

## 运行项目

### 前端（作业系统）
```bash
cd homework_system
npm install
npm run serve
```

### 后端（API服务）
```bash
cd homework-backend
python app.py
```

## API文档

### 作业管理API
- `GET /api/homework/list?userId={userId}` - 获取作业列表
- `GET /api/homework/detail/{homeworkId}` - 获取作业详情
- `POST /api/homework/submit` - 提交作业答案
- `POST /api/homework/save` - 保存作业进度

### 知识点API
- `GET /api/knowledge/question?questionId={questionId}` - 根据题目ID获取知识点
- `GET /api/knowledge/question?text={questionText}` - 根据题目内容获取知识点
- `POST /api/knowledge/question` - 提交题目ID或内容获取知识点

### 推荐系统API
- `POST /api/recommend/symbols` - 获取推荐的数学符号
- `POST /api/recommend/knowledge` - 获取推荐的知识点
- `POST /api/recommend/exercises` - 获取推荐的练习题

### 用户模型API
- `GET /api/user/{userId}` - 获取用户信息
- `POST /api/user/update` - 更新用户模型

## 开发计划

1. 完善用户认证系统
2. 增强数据可视化功能
3. 优化移动端适配
4. 集成更多智能推荐算法
5. 扩展数学符号输入系统功能
6. 引入机器学习模型提高知识点提取准确性 