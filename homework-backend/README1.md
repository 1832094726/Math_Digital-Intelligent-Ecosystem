# K-12数学教育作业系统后端

这是K-12数学教育生态系统中的作业系统后端，提供作业管理、智能推荐和用户模型等功能的API服务。

## 项目结构

```
homework-backend/
├── app.py                 # 主应用入口
├── services/              # 服务模块
│   ├── homework_service.py    # 作业服务
│   ├── recommendation_service.py  # 推荐服务
│   └── user_service.py     # 用户服务
├── utils/                 # 工具模块
│   ├── data_utils.py      # 数据处理工具
│   └── ai_utils.py        # AI工具
├── data/                  # 数据文件
│   ├── homework.json      # 作业数据
│   ├── symbols.json       # 符号数据
│   ├── knowledge.json     # 知识点数据
│   ├── users.json         # 用户数据
│   └── user_models.json   # 用户模型数据
└── requirements.txt       # 项目依赖
```

## 功能模块

### 作业管理

- 获取作业列表
- 获取作业详情
- 提交作业答案
- 保存作业进度

### 智能推荐

- 推荐数学符号
- 推荐知识点
- 推荐练习题

### 用户模型

- 获取用户信息
- 更新用户模型

## API接口

### 作业接口

- `GET /api/homework/list?user_id=<user_id>` - 获取用户作业列表
- `GET /api/homework/<homework_id>` - 获取作业详情
- `POST /api/homework/submit` - 提交作业答案
- `POST /api/homework/save-progress` - 保存作业进度

### 推荐接口

- `POST /api/recommend/symbols` - 推荐数学符号
- `POST /api/recommend/knowledge` - 推荐知识点
- `POST /api/recommend/exercises` - 推荐练习题

### 用户接口

- `GET /api/user/<user_id>` - 获取用户信息
- `POST /api/user/update-model` - 更新用户模型

## 安装与运行

### 环境要求

- Python 3.8+
- Flask 2.1.3+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务

```bash
python app.py
```

服务将在 http://localhost:5000 启动。

## 数据模型

### 作业数据结构

```json
{
  "id": 1,
  "title": "代数基础练习",
  "subject": "数学",
  "grade": "初中二年级",
  "description": "本作业旨在巩固一元二次方程的解法和应用",
  "deadline": "2023-07-15 23:59:59",
  "teacher_id": 101,
  "teacher_name": "李老师",
  "status": "active",
  "questions": [
    {
      "id": 101,
      "type": "choice",
      "content": "解方程 x² - 5x + 6 = 0",
      "options": ["x=2,x=3", "x=1,x=6", "x=-2,x=-3", "x=2,x=-3"],
      "answer": "x=2,x=3",
      "score": 10,
      "knowledge_points": ["一元二次方程", "因式分解"]
    }
  ]
}
```

### 用户模型数据结构

```json
{
  "user_id": 1,
  "knowledge_mastery": {
    "三角形": 0.7,
    "等腰三角形": 0.5,
    "一元二次方程": 0.4
  },
  "learning_style": "visual",
  "difficulty_preference": "medium",
  "symbol_usage_frequency": {
    "+": 120,
    "-": 100,
    "×": 80
  },
  "recent_activities": [
    {
      "type": "homework",
      "id": 1,
      "timestamp": "2023-06-20 15:30:00",
      "score": 85
    }
  ]
}
```

## 开发说明

### 添加新服务

1. 在 `services` 目录下创建新的服务模块
2. 在 `app.py` 中导入并注册相应的路由

### 扩展数据模型

1. 在 `data` 目录中添加新的JSON数据文件
2. 在相应的服务模块中添加加载和处理数据的函数

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request 