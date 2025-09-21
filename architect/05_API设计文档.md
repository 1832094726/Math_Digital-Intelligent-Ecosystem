# K-12数学教育智能数字生态系统 - API详细文档

## 系统概述

本系统是一个基于推荐技术提升学生作业效率的K-12数学教育智能数字生态系统（DIEM），提供完整的作业管理、智能推荐、学习分析等功能。

### 技术架构
- **后端框架**: Flask 2.0.3
- **数据库**: MySQL (OceanBase云数据库)
- **API协议**: RESTful
- **数据格式**: JSON
- **认证方式**: JWT Token
- **跨域支持**: CORS

### 服务器配置
```
基础URL: http://172.104.172.5:5001
数据库: obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud:3306
数据库名: testccnu
```

## 通用响应格式

### 成功响应
```json
{
  "success": true,
  "data": {
    // 具体数据内容
  },
  "message": "操作成功"
}
```

### 错误响应
```json
{
  "success": false,
  "error": "错误信息",
  "message": "详细错误描述"
}
```

## 数据库可视化API

### 1. 健康检查

#### GET /api/health
检查API服务状态和数据库连接

**响应示例**
```json
{
  "status": "healthy",
  "database": "connected", 
  "message": "数据库API服务正常运行"
}
```

### 2. 数据库表管理

#### GET /api/database/tables
获取所有数据库表信息

#### GET /api/database/table/{table_name}
获取指定表的数据

#### GET /api/database/table/{table_name}/count
获取指定表的记录总数

#### GET /api/database/table/{table_name}/structure
获取指定表的结构信息

### 3. API分析功能

#### GET /api/apis
获取系统中所有API的详细信息

**响应示例**
```json
{
  "apis": {
    "blueprints_get_recommendation_stats": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations",
        "problem_recommendations"
      ],
      "description": "获取推荐系统统计信息，包括推荐准确率、使用频率等数据",
      "example_request": {},
      "example_response": {},
      "file": "recommendation_bp.py",
      "id": "blueprints_get_recommendation_stats",
      "methods": [
        "GET"
      ],
      "name": "get_recommendation_stats",
      "parameters": {},
      "path": "/stats",
      "responses": {
        "200": {
          "stats": "object",
          "success": "boolean"
        }
      },
      "technical_category": "blueprints"
    },
    "blueprints_recommend_exercises": {
      "category": "recommendation_system",
      "database_tables": [
        "users",
        "problem_recommendations",
        "questions"
      ],
      "description": "基于学生学习状态推荐练习题，支持难度自适应调整",
      "example_request": {},
      "example_response": {},
      "file": "recommendation_bp.py",
      "id": "blueprints_recommend_exercises",
      "methods": [
        "POST"
      ],
      "name": "recommend_exercises",
      "parameters": {
        "count": "推荐数量",
        "difficulty": "难度级别",
        "student_id": "学生ID",
        "subject": "学科"
      },
      "path": "/exercises",
      "responses": {
        "200": {
          "exercises": "array",
          "success": "boolean"
        }
      },
      "technical_category": "blueprints"
    },
    "blueprints_recommend_knowledge_points": {
      "category": "recommendation_system",
      "database_tables": [
        "users",
        "knowledge_points",
        "knowledge_relationships"
      ],
      "description": "基于AI的知识点推荐，根据用户学习状态和上下文推荐相关知识点",
      "example_request": {
        "context": "解一元二次方程",
        "limit": 3
      },
      "example_response": {
        "recommendations": [
          {
            "description": "用字母和数字表示的数学表达式",
            "difficulty_level": 2,
            "grade_level": 2,
            "id": 2,
            "name": "代数表达式",
            "recommendation_reason": "与输入内容相关",
            "relevance_score": 0.8
          }
        ],
        "success": true,
        "total": 1
      },
      "file": "recommendation_bp.py",
      "id": "blueprints_recommend_knowledge_points",
      "methods": [
        "POST"
      ],
      "name": "recommend_knowledge_points",
      "parameters": {
        "context": "学习上下文内容",
        "limit": "推荐数量限制，默认5",
        "question_id": "题目ID，基于题目推荐"
      },
      "path": "/knowledge",
      "responses": {
        "200": {
          "recommendations": "array",
          "success": "boolean",
          "timestamp": "string",
          "total": "number"
        }
      },
      "technical_category": "blueprints"
    },
    "blueprints_recommend_learning_path": {
      "category": "recommendation_system",
      "database_tables": [
        "learning_path_recommendations",
        "knowledge_points"
      ],
      "description": "为学生推荐个性化学习路径，基于知识图谱和学习进度",
      "example_request": {},
      "example_response": {},
      "file": "recommendation_bp.py",
      "id": "blueprints_recommend_learning_path",
      "methods": [
        "POST"
      ],
      "name": "recommend_learning_path",
      "parameters": {
        "current_level": "当前水平",
        "student_id": "学生ID",
        "target_knowledge": "目标知识点"
      },
      "path": "/learning-path",
      "responses": {
        "200": {
          "learning_path": "array",
          "success": "boolean"
        }
      },
      "technical_category": "blueprints"
    },
    "blueprints_recommend_symbols": {
      "category": "recommendation_system",
      "database_tables": [
        "users",
        "symbol_recommendations"
      ],
      "description": "数学符号智能推荐，基于上下文和用户习惯推荐合适的数学符号",
      "example_request": {},
      "example_response": {},
      "file": "recommendation_bp.py",
      "id": "blueprints_recommend_symbols",
      "methods": [
        "POST"
      ],
      "name": "recommend_symbols",
      "parameters": {
        "context": "当前输入上下文",
        "limit": "推荐数量，默认5",
        "question_text": "题目文本"
      },
      "path": "/symbols",
      "responses": {
        "200": {
          "context_analysis": "object",
          "recommendations": "array",
          "success": "boolean"
        }
      },
      "technical_category": "blueprints"
    },
    "blueprints_record_symbol_usage": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations",
        "interaction_logs"
      ],
      "description": "记录学生使用数学符号的行为数据，用于优化推荐算法",
      "example_request": {},
      "example_response": {},
      "file": "recommendation_bp.py",
      "id": "blueprints_record_symbol_usage",
      "methods": [
        "POST"
      ],
      "name": "record_symbol_usage",
      "parameters": {
        "context": "使用上下文",
        "symbol": "使用的符号",
        "user_id": "用户ID"
      },
      "path": "/symbols/usage",
      "responses": {
        "200": {
          "success": "boolean"
        }
      },
      "technical_category": "blueprints"
    },
    "db_viz_health": {
      "category": "database_visualization",
      "database_tables": [],
      "description": "健康检查接口",
      "example_request": {},
      "example_response": {
        "database": "connected",
        "message": "数据库API服务正常运行",
        "status": "healthy"
      },
      "file": "api-server.py",
      "id": "db_viz_health",
      "methods": [
        "GET"
      ],
      "name": "health_check",
      "parameters": {},
      "path": "/api/health",
      "responses": {
        "200": {
          "database": "string",
          "message": "string",
          "status": "string"
        }
      }
    },
    "db_viz_table_data": {
      "category": "database_visualization",
      "database_tables": [
        "dynamic"
      ],
      "description": "获取数据库表的实时数据，支持分页和筛选",
      "example_request": {
        "limit": 10,
        "offset": 0
      },
      "example_response": {
        "count": 0,
        "data": [],
        "limit": 10,
        "offset": 0
      },
      "file": "api-server.py",
      "id": "db_viz_table_data",
      "methods": [
        "GET"
      ],
      "name": "get_table_data",
      "parameters": {
        "limit": "查询数量限制，默认10",
        "offset": "偏移量，默认0",
        "table_name": "表名（路径参数）"
      },
      "path": "/api/database/table/<table_name>",
      "responses": {
        "200": {
          "count": "number",
          "data": "array",
          "limit": "number",
          "offset": "number",
          "source": "string",
          "table": "string"
        }
      }
    },
    "db_viz_tables": {
      "category": "database_visualization",
      "database_tables": [
        "INFORMATION_SCHEMA.TABLES"
      ],
      "description": "获取所有表信息",
      "example_request": {},
      "example_response": {
        "tables": [
          {
            "count": 100,
            "name": "users"
          }
        ],
        "total_tables": 1
      },
      "file": "api-server.py",
      "id": "db_viz_tables",
      "methods": [
        "GET"
      ],
      "name": "get_all_tables",
      "parameters": {},
      "path": "/api/database/tables",
      "responses": {
        "200": {
          "tables": "array",
          "total_tables": "number"
        }
      }
    },
    "main_health_check": {
      "category": "data_visualization",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_health_check",
      "methods": [
        "GET"
      ],
      "name": "health_check",
      "parameters": {},
      "path": "/api/health",
      "responses": {},
      "technical_category": "main"
    },
    "main_hello_world": {
      "category": "other",
      "database_tables": [],
      "description": "系统首页接口，返回系统基本信息和状态",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_hello_world",
      "methods": [
        "GET"
      ],
      "name": "hello_world",
      "parameters": {},
      "path": "/",
      "responses": {
        "200": {
          "message": "string",
          "system_info": "object"
        }
      },
      "technical_category": "main"
    },
    "main_homework_detail": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_submissions",
        "questions"
      ],
      "description": "获取指定作业的详细信息，包括题目、提交状态等",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_homework_detail",
      "methods": [
        "GET"
      ],
      "name": "homework_detail",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/api/homework/detail/<int:homework_id>",
      "responses": {
        "200": {
          "homework": "object",
          "questions": "array",
          "success": "boolean"
        }
      },
      "technical_category": "main"
    },
    "main_homework_list": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_assignments"
      ],
      "description": "获取作业列表，支持分页和筛选条件",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_homework_list",
      "methods": [
        "GET"
      ],
      "name": "homework_list",
      "parameters": {
        "limit": "每页数量",
        "page": "页码",
        "status": "作业状态筛选",
        "userId": "string"
      },
      "path": "/api/homework/list",
      "responses": {
        "200": {
          "homeworks": "array",
          "success": "boolean",
          "total": "number"
        }
      },
      "technical_category": "main"
    },
    "main_question_knowledge": {
      "category": "recommendation_system",
      "database_tables": [
        "knowledge_relationships",
        "knowledge_points",
        "questions"
      ],
      "description": "获取题目相关的知识点信息，支持知识点查询和关联分析",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_question_knowledge",
      "methods": [
        "GET",
        "POST"
      ],
      "name": "question_knowledge",
      "parameters": {
        "knowledge_point": "知识点名称",
        "questionId": "string",
        "question_id": "题目ID",
        "text": "string"
      },
      "path": "/api/knowledge/question",
      "responses": {
        "200": {
          "knowledge_points": "array",
          "relationships": "array",
          "success": "boolean"
        },
        "400": {
          "error": "string",
          "message": "string"
        }
      },
      "technical_category": "main"
    },
    "main_recommend_exercises": {
      "category": "recommendation_system",
      "database_tables": [
        "learning_behaviors",
        "problem_recommendations",
        "questions"
      ],
      "description": "主要的练习推荐接口，整合多种推荐算法",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_recommend_exercises",
      "methods": [
        "POST"
      ],
      "name": "recommend_exercises",
      "parameters": {
        "difficulty_range": "难度范围",
        "preferences": "用户偏好",
        "user_id": "用户ID"
      },
      "path": "/api/recommend/exercises",
      "responses": {
        "200": {
          "recommendations": "array",
          "success": "boolean"
        }
      },
      "technical_category": "main"
    },
    "main_recommend_knowledge": {
      "category": "recommendation_system",
      "database_tables": [
        "learning_behaviors",
        "knowledge_points",
        "knowledge_relationships"
      ],
      "description": "知识点推荐接口，基于学习进度推荐相关知识点",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_recommend_knowledge",
      "methods": [
        "POST"
      ],
      "name": "recommend_knowledge",
      "parameters": {
        "current_topic": "当前学习主题",
        "learning_goal": "学习目标",
        "user_id": "用户ID"
      },
      "path": "/api/recommend/knowledge",
      "responses": {
        "200": {
          "knowledge_recommendations": "array",
          "success": "boolean"
        }
      },
      "technical_category": "main"
    },
    "main_recommend_symbols": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations",
        "interaction_logs"
      ],
      "description": "数学符号推荐接口，根据输入上下文推荐合适的符号",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_recommend_symbols",
      "methods": [
        "POST"
      ],
      "name": "recommend_symbols",
      "parameters": {
        "context": "输入上下文",
        "subject": "学科领域",
        "user_level": "用户水平"
      },
      "path": "/api/recommend/symbols",
      "responses": {
        "200": {
          "success": "boolean",
          "symbols": "array"
        }
      },
      "technical_category": "main"
    },
    "main_redirect_homework_detail": {
      "category": "homework_management",
      "database_tables": [
        "homeworks"
      ],
      "description": "作业详情页面重定向接口，用于页面路由跳转",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_redirect_homework_detail",
      "methods": [
        "GET"
      ],
      "name": "redirect_homework_detail",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/homework/detail/<int:homework_id>",
      "responses": {
        "302": {
          "redirect_url": "string"
        }
      },
      "technical_category": "main"
    },
    "main_redirect_homework_list": {
      "category": "homework_management",
      "database_tables": [],
      "description": "作业列表页面重定向接口，用于页面路由跳转",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_redirect_homework_list",
      "methods": [
        "GET"
      ],
      "name": "redirect_homework_list",
      "parameters": {},
      "path": "/homework/list",
      "responses": {
        "302": {
          "redirect_url": "string"
        }
      },
      "technical_category": "main"
    },
    "main_redirect_knowledge_question": {
      "category": "recommendation_system",
      "database_tables": [],
      "description": "知识点题目页面重定向接口，用于页面路由跳转",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_redirect_knowledge_question",
      "methods": [
        "GET",
        "POST"
      ],
      "name": "redirect_knowledge_question",
      "parameters": {},
      "path": "/knowledge/question",
      "responses": {
        "302": {
          "redirect_url": "string"
        }
      },
      "technical_category": "main"
    },
    "main_save": {
      "category": "homework_management",
      "database_tables": [
        "homework_progress",
        "homework_submissions"
      ],
      "description": "保存作业进度接口，支持断点续做功能",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_save",
      "methods": [
        "POST"
      ],
      "name": "save",
      "parameters": {
        "answers": "当前答案数据",
        "homework_id": "作业ID",
        "progress": "完成进度"
      },
      "path": "/api/homework/save",
      "responses": {
        "200": {
          "saved_at": "string",
          "success": "boolean"
        }
      },
      "technical_category": "main"
    },
    "main_serve_frontend": {
      "category": "authentication",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_serve_frontend",
      "methods": [
        "GET"
      ],
      "name": "serve_frontend",
      "parameters": {},
      "path": "/register",
      "responses": {},
      "technical_category": "main"
    },
    "main_serve_homework_static": {
      "category": "homework_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_serve_homework_static",
      "methods": [
        "GET"
      ],
      "name": "serve_homework_static",
      "parameters": {},
      "path": "/static/homework/<path:filename>",
      "responses": {},
      "technical_category": "main"
    },
    "main_serve_static": {
      "category": "other",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_serve_static",
      "methods": [
        "GET"
      ],
      "name": "serve_static",
      "parameters": {},
      "path": "/static/<path:filename>",
      "responses": {},
      "technical_category": "main"
    },
    "main_serve_symbol_static": {
      "category": "recommendation_system",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_serve_symbol_static",
      "methods": [
        "GET"
      ],
      "name": "serve_symbol_static",
      "parameters": {},
      "path": "/static/symbol/<path:filename>",
      "responses": {},
      "technical_category": "main"
    },
    "main_submit": {
      "category": "homework_management",
      "database_tables": [
        "homework_submissions",
        "homeworks"
      ],
      "description": "提交作业答案接口，完成作业并触发自动评分",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_submit",
      "methods": [
        "POST"
      ],
      "name": "submit",
      "parameters": {
        "answers": "完整答案数据",
        "homework_id": "作业ID",
        "submit_time": "提交时间"
      },
      "path": "/api/homework/submit",
      "responses": {
        "200": {
          "score": "number",
          "submission_id": "number",
          "success": "boolean"
        }
      },
      "technical_category": "main"
    },
    "main_update_user": {
      "category": "data_visualization",
      "database_tables": [
        "users"
      ],
      "description": "更新用户信息接口，支持个人资料修改",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_update_user",
      "methods": [
        "POST"
      ],
      "name": "update_user",
      "parameters": {
        "avatar": "头像",
        "email": "邮箱",
        "name": "姓名",
        "phone": "电话"
      },
      "path": "/api/user/update",
      "responses": {
        "200": {
          "success": "boolean",
          "user": "object"
        }
      },
      "technical_category": "main"
    },
    "main_user_info": {
      "category": "data_visualization",
      "database_tables": [
        "users"
      ],
      "description": "获取指定用户的基本信息",
      "example_request": {},
      "example_response": {},
      "file": "app.py",
      "id": "main_user_info",
      "methods": [
        "GET"
      ],
      "name": "user_info",
      "parameters": {
        "user_id": "用户ID（路径参数）"
      },
      "path": "/api/user/<int:user_id>",
      "responses": {
        "200": {
          "success": "boolean",
          "user": "object"
        }
      },
      "technical_category": "main"
    },
    "routes_assign_homework": {
      "category": "homework_management",
      "database_tables": [
        "users",
        "classes",
        "homework_assignments"
      ],
      "description": "教师分配作业给班级或学生",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_assign_homework",
      "methods": [
        "POST"
      ],
      "name": "assign_homework",
      "parameters": {
        "due_date": "截止时间",
        "homework_id": "作业ID",
        "target_ids": "目标ID列表",
        "target_type": "分配类型（class/student）"
      },
      "path": "/assign",
      "responses": {
        "200": {
          "assignment_count": "number",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_batch_grade": {
      "category": "grading_system",
      "database_tables": [
        "homework_submissions",
        "homeworks"
      ],
      "description": "批量评分接口，支持多份作业同时评分",
      "example_request": {},
      "example_response": {},
      "file": "grading_routes.py",
      "id": "routes_batch_grade",
      "methods": [
        "POST"
      ],
      "name": "batch_grade",
      "parameters": {
        "grading_rules": "评分规则",
        "submission_ids": "提交ID列表"
      },
      "path": "/batch-grade",
      "responses": {
        "200": {
          "graded_count": "number",
          "results": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_create_homework": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_questions",
        "questions"
      ],
      "description": "创建新作业，教师可以创建包含多个题目的作业",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_create_homework",
      "methods": [
        "POST"
      ],
      "name": "create_homework",
      "parameters": {
        "description": "作业描述",
        "difficulty_level": "难度等级1-5",
        "due_date": "截止日期",
        "grade": "年级",
        "max_score": "总分",
        "questions": "题目列表",
        "subject": "学科",
        "title": "作业标题"
      },
      "path": "/create",
      "responses": {
        "201": {
          "homework_id": "number",
          "message": "string",
          "success": "boolean"
        },
        "400": {
          "errors": "array",
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_delete_homework": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_submissions",
        "questions"
      ],
      "description": "删除指定作业及其相关数据",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_delete_homework",
      "methods": [
        "DELETE"
      ],
      "name": "delete_homework",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/<int:homework_id>",
      "responses": {
        "200": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_export_analytics": {
      "category": "homework_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "analytics_routes.py",
      "id": "routes_export_analytics",
      "methods": [
        "POST"
      ],
      "name": "export_analytics",
      "parameters": {},
      "path": "/homework/<int:homework_id>/export",
      "responses": {},
      "technical_category": "routes"
    },
    "routes_get_adaptive_recommendations": {
      "category": "recommendation_system",
      "database_tables": [
        "learning_behaviors",
        "symbol_recommendations"
      ],
      "description": "获取自适应推荐结果，基于用户学习状态动态调整推荐内容",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_adaptive_recommendations",
      "methods": [
        "POST"
      ],
      "name": "get_adaptive_recommendations",
      "parameters": {
        "context": "当前学习上下文",
        "difficulty": "期望难度",
        "user_id": "用户ID"
      },
      "path": "/recommend/adaptive",
      "responses": {
        "200": {
          "recommendations": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_assignment_detail": {
      "category": "homework_management",
      "database_tables": [
        "homework_assignments"
      ],
      "description": "获取作业分配的详细信息，包括完成情况和统计数据",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_get_assignment_detail",
      "methods": [
        "GET"
      ],
      "name": "get_assignment_detail",
      "parameters": {
        "assignment_id": "分配ID（路径参数）"
      },
      "path": "/<int:assignment_id>",
      "responses": {
        "200": {
          "assignment": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_assignment_statistics": {
      "category": "homework_management",
      "database_tables": [
        "homework_submissions",
        "homework_assignments"
      ],
      "description": "获取作业分配的统计信息，包括完成率、平均分、提交时间分布等",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_get_assignment_statistics",
      "methods": [
        "GET"
      ],
      "name": "get_assignment_statistics",
      "parameters": {
        "assignment_id": "分配ID（路径参数）"
      },
      "path": "/statistics/<int:assignment_id>",
      "responses": {
        "200": {
          "statistics": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_class_assignments": {
      "category": "homework_management",
      "database_tables": [
        "classes",
        "homework_assignments"
      ],
      "description": "获取指定班级的作业分配情况，教师查看班级作业状态",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_get_class_assignments",
      "methods": [
        "GET"
      ],
      "name": "get_class_assignments",
      "parameters": {
        "class_id": "班级ID（路径参数）"
      },
      "path": "/class/<int:class_id>",
      "responses": {
        "200": {
          "assignments": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_class_students": {
      "category": "homework_management",
      "database_tables": [
        "class_students",
        "users"
      ],
      "description": "获取指定班级的学生名单，用于作业分配和管理",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_get_class_students",
      "methods": [
        "GET"
      ],
      "name": "get_class_students",
      "parameters": {
        "class_id": "班级ID（路径参数）"
      },
      "path": "/classes/<int:class_id>/students",
      "responses": {
        "200": {
          "students": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_context_aware_recommendations": {
      "category": "recommendation_system",
      "database_tables": [
        "users",
        "symbol_recommendations"
      ],
      "description": "获取上下文感知的符号推荐，基于当前题目和学习进度",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_context_aware_recommendations",
      "methods": [
        "POST"
      ],
      "name": "get_context_aware_recommendations",
      "parameters": {
        "context": "当前上下文",
        "subject": "学科",
        "user_level": "用户水平"
      },
      "path": "/context",
      "responses": {
        "200": {
          "recommendations": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_explained_symbol_recommendations": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations"
      ],
      "description": "获取带解释的符号推荐，包含推荐理由和使用说明",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_explained_symbol_recommendations",
      "methods": [
        "POST"
      ],
      "name": "get_explained_symbol_recommendations",
      "parameters": {
        "context": "输入上下文",
        "explain": "是否需要详细解释"
      },
      "path": "/recommend/explained",
      "responses": {
        "200": {
          "explanations": "array",
          "recommendations": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_favorite_homeworks": {
      "category": "homework_management",
      "database_tables": [
        "homework_favorites",
        "homework_assignments"
      ],
      "description": "获取用户收藏的作业列表",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_favorite_homeworks",
      "methods": [
        "GET"
      ],
      "name": "get_favorite_homeworks",
      "parameters": {},
      "path": "/favorites",
      "responses": {
        "200": {
          "favorites": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_filter_options": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "subjects",
        "grades"
      ],
      "description": "获取作业筛选选项，如可用的学科、年级等",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_filter_options",
      "methods": [
        "GET"
      ],
      "name": "get_filter_options",
      "parameters": {},
      "path": "/filters/options",
      "responses": {
        "200": {
          "options": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_grading_result": {
      "category": "student_features",
      "database_tables": [
        "homework_submissions"
      ],
      "description": "获取作业提交的评分结果，包括得分、错误分析、改进建议",
      "example_request": {},
      "example_response": {},
      "file": "grading_routes.py",
      "id": "routes_get_grading_result",
      "methods": [
        "GET"
      ],
      "name": "get_grading_result",
      "parameters": {
        "submission_id": "提交ID（路径参数）"
      },
      "path": "/result/<int:submission_id>",
      "responses": {
        "200": {
          "result": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_grading_rules": {
      "category": "homework_management",
      "database_tables": [
        "homeworks"
      ],
      "description": "获取指定作业的评分规则配置，包括评分标准和权重",
      "example_request": {},
      "example_response": {},
      "file": "grading_routes.py",
      "id": "routes_get_grading_rules",
      "methods": [
        "GET"
      ],
      "name": "get_grading_rules",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/rules/<int:homework_id>",
      "responses": {
        "200": {
          "rules": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "questions"
      ],
      "description": "获取指定作业的详细信息",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_get_homework",
      "methods": [
        "GET"
      ],
      "name": "get_homework",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/<int:homework_id>",
      "responses": {
        "200": {
          "homework": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework_analytics": {
      "category": "homework_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "analytics_routes.py",
      "id": "routes_get_homework_analytics",
      "methods": [
        "GET"
      ],
      "name": "get_homework_analytics",
      "parameters": {},
      "path": "/homework/<int:homework_id>",
      "responses": {},
      "technical_category": "routes"
    },
    "routes_get_homework_dashboard": {
      "category": "homework_management",
      "database_tables": [
        "homework_submissions",
        "homework_assignments"
      ],
      "description": "获取学生作业仪表板数据，包括统计信息",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_homework_dashboard",
      "methods": [
        "GET"
      ],
      "name": "get_homework_dashboard",
      "parameters": {},
      "path": "/dashboard",
      "responses": {
        "200": {
          "dashboard": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework_detail": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_submissions",
        "questions",
        "homework_assignments"
      ],
      "description": "获取作业详细信息，包括题目和学生提交状态",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_homework_detail",
      "methods": [
        "GET"
      ],
      "name": "get_homework_detail",
      "parameters": {
        "assignment_id": "作业分配ID（路径参数）"
      },
      "path": "/<int:assignment_id>",
      "responses": {
        "200": {
          "homework": "object",
          "questions": "array",
          "submission_status": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework_feedback": {
      "category": "homework_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "feedback_routes.py",
      "id": "routes_get_homework_feedback",
      "methods": [
        "GET"
      ],
      "name": "get_homework_feedback",
      "parameters": {},
      "path": "/homework/<int:homework_id>",
      "responses": {},
      "technical_category": "routes"
    },
    "routes_get_homework_list": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_assignments"
      ],
      "description": "获取学生可见的作业列表",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_homework_list",
      "methods": [
        "GET"
      ],
      "name": "get_homework_list",
      "parameters": {
        "limit": "每页数量",
        "page": "页码"
      },
      "path": "/list",
      "responses": {
        "200": {
          "homeworks": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework_progress": {
      "category": "homework_management",
      "database_tables": [
        "homework_progress"
      ],
      "description": "获取作业完成进度信息",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_homework_progress",
      "methods": [
        "GET"
      ],
      "name": "get_homework_progress",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/<int:homework_id>/progress",
      "responses": {
        "200": {
          "progress": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework_questions": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "questions"
      ],
      "description": "获取作业的所有题目列表",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_get_homework_questions",
      "methods": [
        "GET"
      ],
      "name": "get_homework_questions",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/<int:homework_id>/questions",
      "responses": {
        "200": {
          "questions": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework_reminders": {
      "category": "homework_management",
      "database_tables": [
        "homework_reminders",
        "homework_assignments"
      ],
      "description": "获取作业提醒列表，包括即将到期的作业",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_homework_reminders",
      "methods": [
        "GET"
      ],
      "name": "get_homework_reminders",
      "parameters": {},
      "path": "/reminders",
      "responses": {
        "200": {
          "reminders": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_homework_statistics": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_submissions"
      ],
      "description": "获取整体作业统计信息，教师查看所有作业的完成情况",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_get_homework_statistics",
      "methods": [
        "GET"
      ],
      "name": "get_homework_statistics",
      "parameters": {},
      "path": "/statistics",
      "responses": {
        "200": {
          "statistics": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_learning_insights": {
      "category": "recommendation_system",
      "database_tables": [
        "learning_behaviors",
        "engagement_metrics"
      ],
      "description": "获取学习洞察报告，分析用户学习模式和改进建议",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_learning_insights",
      "methods": [
        "GET"
      ],
      "name": "get_learning_insights",
      "parameters": {
        "user_id": "用户ID（路径参数）"
      },
      "path": "/learning-insights/<int:user_id>",
      "responses": {
        "200": {
          "insights": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_my_assignments": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_assignments"
      ],
      "description": "获取教师创建的所有作业分配，用于教师管理界面",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_get_my_assignments",
      "methods": [
        "GET"
      ],
      "name": "get_my_assignments",
      "parameters": {},
      "path": "/teacher/my",
      "responses": {
        "200": {
          "assignments": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_my_classes": {
      "category": "homework_management",
      "database_tables": [
        "classes"
      ],
      "description": "获取教师负责的班级列表，用于班级管理",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_get_my_classes",
      "methods": [
        "GET"
      ],
      "name": "get_my_classes",
      "parameters": {},
      "path": "/classes/my",
      "responses": {
        "200": {
          "classes": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_my_notifications": {
      "category": "homework_management",
      "database_tables": [
        "notifications"
      ],
      "description": "获取用户的通知消息列表，包括作业提醒、系统通知等",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_get_my_notifications",
      "methods": [
        "GET"
      ],
      "name": "get_my_notifications",
      "parameters": {},
      "path": "/notifications/my",
      "responses": {
        "200": {
          "notifications": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_profile": {
      "category": "authentication",
      "database_tables": [
        "users"
      ],
      "description": "获取当前用户的个人资料信息，包括基本信息和偏好设置",
      "example_request": {},
      "example_response": {},
      "file": "auth_routes.py",
      "id": "routes_get_profile",
      "methods": [
        "GET"
      ],
      "name": "get_profile",
      "parameters": {},
      "path": "/profile",
      "responses": {
        "200": {
          "profile": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_sessions": {
      "category": "authentication",
      "database_tables": [
        "user_sessions"
      ],
      "description": "获取用户的活跃会话列表，用于会话管理和安全监控",
      "example_request": {},
      "example_response": {},
      "file": "auth_routes.py",
      "id": "routes_get_sessions",
      "methods": [
        "GET"
      ],
      "name": "get_sessions",
      "parameters": {},
      "path": "/sessions",
      "responses": {
        "200": {
          "sessions": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_simple_homework_analytics": {
      "category": "homework_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "simple_analytics_routes.py",
      "id": "routes_get_simple_homework_analytics",
      "methods": [
        "GET"
      ],
      "name": "get_simple_homework_analytics",
      "parameters": {},
      "path": "/homework/<int:homework_id>",
      "responses": {},
      "technical_category": "routes"
    },
    "routes_get_simple_homework_feedback": {
      "category": "homework_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "simple_feedback_routes.py",
      "id": "routes_get_simple_homework_feedback",
      "methods": [
        "GET"
      ],
      "name": "get_simple_homework_feedback",
      "parameters": {},
      "path": "/homework/<int:homework_id>",
      "responses": {},
      "technical_category": "routes"
    },
    "routes_get_statistics": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "homework_submissions"
      ],
      "description": "获取作业统计信息，包括完成率、平均分等",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_get_statistics",
      "methods": [
        "GET"
      ],
      "name": "get_statistics",
      "parameters": {},
      "path": "/statistics",
      "responses": {
        "200": {
          "statistics": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_submission_result": {
      "category": "student_features",
      "database_tables": [
        "homework_submissions"
      ],
      "description": "获取学生作业提交的完整结果，包括答案、评分、反馈",
      "example_request": {},
      "example_response": {},
      "file": "submission_routes.py",
      "id": "routes_get_submission_result",
      "methods": [
        "GET"
      ],
      "name": "get_submission_result",
      "parameters": {
        "submission_id": "提交ID（路径参数）"
      },
      "path": "/<int:submission_id>/result",
      "responses": {
        "200": {
          "submission": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_symbol_categories": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations"
      ],
      "description": "获取数学符号分类列表，用于符号选择界面的分类显示",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_symbol_categories",
      "methods": [
        "GET"
      ],
      "name": "get_symbol_categories",
      "parameters": {},
      "path": "/categories",
      "responses": {
        "200": {
          "categories": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_symbol_completions": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations"
      ],
      "description": "获取符号自动补全建议，帮助用户快速输入数学表达式",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_symbol_completions",
      "methods": [
        "POST"
      ],
      "name": "get_symbol_completions",
      "parameters": {
        "limit": "返回数量限制",
        "partial_input": "部分输入内容"
      },
      "path": "/complete",
      "responses": {
        "200": {
          "completions": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_symbol_recommendations": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations"
      ],
      "description": "获取数学符号推荐，基于当前输入上下文推荐相关符号",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_symbol_recommendations",
      "methods": [
        "POST"
      ],
      "name": "get_symbol_recommendations",
      "parameters": {
        "context": "输入上下文",
        "subject": "学科领域"
      },
      "path": "/recommend",
      "responses": {
        "200": {
          "recommendations": "array",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_symbols_by_category": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations"
      ],
      "description": "获取指定分类下的所有数学符号，支持分类浏览",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_symbols_by_category",
      "methods": [
        "GET"
      ],
      "name": "get_symbols_by_category",
      "parameters": {
        "category_id": "分类ID（路径参数）"
      },
      "path": "/category/<category_id>",
      "responses": {
        "200": {
          "success": "boolean",
          "symbols": "array"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_teacher_overview": {
      "category": "class_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "analytics_routes.py",
      "id": "routes_get_teacher_overview",
      "methods": [
        "GET"
      ],
      "name": "get_teacher_overview",
      "parameters": {},
      "path": "/overview",
      "responses": {},
      "technical_category": "routes"
    },
    "routes_get_user_learning_analytics": {
      "category": "recommendation_system",
      "database_tables": [
        "learning_behaviors",
        "users"
      ],
      "description": "获取用户学习分析数据，包括学习行为、进度、偏好等",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_user_learning_analytics",
      "methods": [
        "GET"
      ],
      "name": "get_user_learning_analytics",
      "parameters": {
        "user_id": "用户ID（路径参数）"
      },
      "path": "/analytics/<int:user_id>",
      "responses": {
        "200": {
          "analytics": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_get_user_symbol_stats": {
      "category": "recommendation_system",
      "database_tables": [
        "users",
        "interaction_logs"
      ],
      "description": "获取用户的符号使用统计信息，包括常用符号、使用频率等",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_get_user_symbol_stats",
      "methods": [
        "GET"
      ],
      "name": "get_user_symbol_stats",
      "parameters": {
        "user_id": "用户ID（路径参数）"
      },
      "path": "/stats/<int:user_id>",
      "responses": {
        "200": {
          "stats": "object",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_grade_submission": {
      "category": "student_features",
      "database_tables": [
        "homework_submissions",
        "grading_results",
        "questions"
      ],
      "description": "自动评分学生作业提交，支持多种题型的智能评分",
      "example_request": {},
      "example_response": {},
      "file": "grading_routes.py",
      "id": "routes_grade_submission",
      "methods": [
        "POST"
      ],
      "name": "grade_submission",
      "parameters": {
        "submission_id": "提交ID（路径参数）"
      },
      "path": "/grade/<int:submission_id>",
      "responses": {
        "200": {
          "grading_result": "object",
          "max_score": "number",
          "success": "boolean",
          "total_score": "number"
        },
        "404": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_list_homeworks": {
      "category": "homework_management",
      "database_tables": [
        "homeworks",
        "users"
      ],
      "description": "获取作业列表，支持分页和筛选",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_list_homeworks",
      "methods": [
        "GET"
      ],
      "name": "list_homeworks",
      "parameters": {
        "category": "分类筛选",
        "grade": "年级筛选",
        "keyword": "关键词搜索",
        "limit": "每页数量，默认10",
        "page": "页码，默认1",
        "subject": "学科筛选"
      },
      "path": "/list",
      "responses": {
        "200": {
          "homeworks": "array",
          "page": "number",
          "success": "boolean",
          "total": "number",
          "total_pages": "number"
        }
      },
      "technical_category": "routes"
    },
    "routes_login": {
      "category": "authentication",
      "database_tables": [
        "users",
        "user_sessions"
      ],
      "description": "用户登录认证，支持学生、教师、管理员登录",
      "example_request": {
        "device_type": "web",
        "password": "student123",
        "username": "test_student_001"
      },
      "example_response": {
        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "success": true,
        "user": {
          "id": 2,
          "real_name": "测试学生",
          "role": "student",
          "username": "test_student_001"
        }
      },
      "file": "auth_routes.py",
      "id": "routes_login",
      "methods": [
        "POST"
      ],
      "name": "login",
      "parameters": {
        "device_id": "设备唯一标识",
        "device_type": "设备类型",
        "password": "用户密码",
        "username": "用户名或邮箱"
      },
      "path": "/login",
      "responses": {
        "200": {
          "access_token": "string",
          "expires_in": "number",
          "refresh_token": "string",
          "success": "boolean",
          "user": "object"
        },
        "401": {
          "error": "object",
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_logout": {
      "category": "authentication",
      "database_tables": [
        "user_sessions"
      ],
      "description": "用户登出，清除会话信息",
      "example_request": {},
      "example_response": {},
      "file": "auth_routes.py",
      "id": "routes_logout",
      "methods": [
        "POST"
      ],
      "name": "logout",
      "parameters": {},
      "path": "/logout",
      "responses": {
        "200": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_mark_notification_read": {
      "category": "homework_management",
      "database_tables": [
        "notifications"
      ],
      "description": "标记指定通知为已读状态，更新通知状态",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_mark_notification_read",
      "methods": [
        "PUT"
      ],
      "name": "mark_notification_read",
      "parameters": {
        "notification_id": "通知ID（路径参数）"
      },
      "path": "/notifications/<int:notification_id>/read",
      "responses": {
        "200": {
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_publish_homework": {
      "category": "homework_management",
      "database_tables": [
        "homeworks"
      ],
      "description": "发布作业，使学生可以看到并完成作业",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_publish_homework",
      "methods": [
        "POST"
      ],
      "name": "publish_homework",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/<int:homework_id>/publish",
      "responses": {
        "200": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_record_symbol_usage": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations",
        "interaction_logs"
      ],
      "description": "记录用户符号使用行为，用于优化推荐算法和学习分析",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_record_symbol_usage",
      "methods": [
        "POST"
      ],
      "name": "record_symbol_usage",
      "parameters": {
        "context": "使用上下文",
        "symbol": "使用的符号",
        "timestamp": "使用时间"
      },
      "path": "/usage",
      "responses": {
        "200": {
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_refresh": {
      "category": "authentication",
      "database_tables": [
        "user_sessions"
      ],
      "description": "刷新用户访问令牌，延长登录会话",
      "example_request": {},
      "example_response": {},
      "file": "auth_routes.py",
      "id": "routes_refresh",
      "methods": [
        "POST"
      ],
      "name": "refresh",
      "parameters": {
        "refresh_token": "刷新令牌"
      },
      "path": "/refresh",
      "responses": {
        "200": {
          "access_token": "string",
          "expires_in": "number",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_register": {
      "category": "authentication",
      "database_tables": [
        "users"
      ],
      "description": "用户注册，创建新的学生、教师或管理员账户",
      "example_request": {},
      "example_response": {},
      "file": "auth_routes.py",
      "id": "routes_register",
      "methods": [
        "POST"
      ],
      "name": "register",
      "parameters": {
        "class_name": "班级名称",
        "email": "邮箱地址",
        "grade": "年级（学生必填）",
        "password": "密码",
        "real_name": "真实姓名",
        "role": "用户角色：student/teacher/admin",
        "school": "学校名称",
        "username": "用户名"
      },
      "path": "/register",
      "responses": {
        "201": {
          "message": "string",
          "success": "boolean",
          "user_id": "number"
        },
        "400": {
          "errors": "object",
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_review_grading": {
      "category": "student_features",
      "database_tables": [
        "homework_submissions"
      ],
      "description": "教师复查自动评分结果，可以调整分数和添加评语",
      "example_request": {},
      "example_response": {},
      "file": "grading_routes.py",
      "id": "routes_review_grading",
      "methods": [
        "POST"
      ],
      "name": "review_grading",
      "parameters": {
        "adjustments": "评分调整",
        "comments": "教师评语",
        "submission_id": "提交ID（路径参数）"
      },
      "path": "/review/<int:submission_id>",
      "responses": {
        "200": {
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_save_homework_progress": {
      "category": "homework_management",
      "database_tables": [
        "homework_progress"
      ],
      "description": "保存作业完成进度，支持断点续做",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_save_homework_progress",
      "methods": [
        "POST"
      ],
      "name": "save_homework_progress",
      "parameters": {
        "answers": "答案数据",
        "homework_id": "作业ID（路径参数）",
        "progress": "完成进度"
      },
      "path": "/<int:homework_id>/progress",
      "responses": {
        "200": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_search_homeworks": {
      "category": "homework_management",
      "database_tables": [
        "homeworks"
      ],
      "description": "搜索作业，支持关键词、学科、年级等条件搜索",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_search_homeworks",
      "methods": [
        "GET"
      ],
      "name": "search_homeworks",
      "parameters": {
        "grade": "年级筛选",
        "keyword": "搜索关键词",
        "subject": "学科筛选"
      },
      "path": "/search",
      "responses": {
        "200": {
          "homeworks": "array",
          "success": "boolean",
          "total": "number"
        }
      },
      "technical_category": "routes"
    },
    "routes_search_symbols": {
      "category": "recommendation_system",
      "database_tables": [
        "symbol_recommendations"
      ],
      "description": "搜索数学符号，支持按名称、描述、LaTeX代码等条件搜索",
      "example_request": {},
      "example_response": {},
      "file": "enhanced_symbol_routes.py",
      "id": "routes_search_symbols",
      "methods": [
        "POST"
      ],
      "name": "search_symbols",
      "parameters": {
        "category": "分类筛选",
        "limit": "结果数量限制",
        "query": "搜索关键词"
      },
      "path": "/search",
      "responses": {
        "200": {
          "success": "boolean",
          "symbols": "array"
        }
      },
      "technical_category": "routes"
    },
    "routes_share_feedback": {
      "category": "homework_management",
      "database_tables": [],
      "description": "暂无描述",
      "example_request": {},
      "example_response": {},
      "file": "feedback_routes.py",
      "id": "routes_share_feedback",
      "methods": [
        "POST"
      ],
      "name": "share_feedback",
      "parameters": {},
      "path": "/homework/<int:homework_id>/share",
      "responses": {},
      "technical_category": "routes"
    },
    "routes_submit_homework": {
      "category": "homework_management",
      "database_tables": [
        "homework_submissions"
      ],
      "description": "提交作业答案，完成作业",
      "example_request": {},
      "example_response": {},
      "file": "submission_routes.py",
      "id": "routes_submit_homework",
      "methods": [
        "POST"
      ],
      "name": "submit_homework",
      "parameters": {
        "answers": "答案数据",
        "assignment_id": "作业分配ID（路径参数）"
      },
      "path": "/<int:assignment_id>",
      "responses": {
        "200": {
          "submission_id": "number",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_toggle_homework_favorite": {
      "category": "homework_management",
      "database_tables": [
        "homework_favorites"
      ],
      "description": "切换作业收藏状态，添加或移除收藏",
      "example_request": {},
      "example_response": {},
      "file": "student_homework_routes.py",
      "id": "routes_toggle_homework_favorite",
      "methods": [
        "POST"
      ],
      "name": "toggle_homework_favorite",
      "parameters": {
        "assignment_id": "作业分配ID（路径参数）"
      },
      "path": "/<int:assignment_id>/favorite",
      "responses": {
        "200": {
          "is_favorite": "boolean",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_unpublish_homework": {
      "category": "homework_management",
      "database_tables": [
        "homeworks"
      ],
      "description": "取消发布作业，隐藏作业不让学生看到",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_unpublish_homework",
      "methods": [
        "POST"
      ],
      "name": "unpublish_homework",
      "parameters": {
        "homework_id": "作业ID（路径参数）"
      },
      "path": "/<int:homework_id>/unpublish",
      "responses": {
        "200": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_update_assignment_status": {
      "category": "homework_management",
      "database_tables": [
        "homework_assignments"
      ],
      "description": "更新作业分配状态，如开启、关闭、延期等操作",
      "example_request": {},
      "example_response": {},
      "file": "assignment_routes.py",
      "id": "routes_update_assignment_status",
      "methods": [
        "PUT"
      ],
      "name": "update_assignment_status",
      "parameters": {
        "assignment_id": "分配ID（路径参数）",
        "status": "新状态"
      },
      "path": "/<int:assignment_id>/status",
      "responses": {
        "200": {
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_update_grading_rules": {
      "category": "homework_management",
      "database_tables": [
        "homeworks"
      ],
      "description": "更新作业的评分规则，教师可以自定义评分标准",
      "example_request": {},
      "example_response": {},
      "file": "grading_routes.py",
      "id": "routes_update_grading_rules",
      "methods": [
        "POST"
      ],
      "name": "update_grading_rules",
      "parameters": {
        "homework_id": "作业ID（路径参数）",
        "rules": "评分规则配置"
      },
      "path": "/rules/<int:homework_id>",
      "responses": {
        "200": {
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_update_homework": {
      "category": "homework_management",
      "database_tables": [
        "homeworks"
      ],
      "description": "更新作业信息，包括标题、描述、题目等",
      "example_request": {},
      "example_response": {},
      "file": "homework_routes.py",
      "id": "routes_update_homework",
      "methods": [
        "PUT"
      ],
      "name": "update_homework",
      "parameters": {
        "description": "作业描述",
        "due_date": "截止日期",
        "homework_id": "作业ID（路径参数）",
        "title": "作业标题"
      },
      "path": "/<int:homework_id>",
      "responses": {
        "200": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    },
    "routes_update_profile": {
      "category": "authentication",
      "database_tables": [
        "users"
      ],
      "description": "更新用户个人资料信息",
      "example_request": {},
      "example_response": {},
      "file": "auth_routes.py",
      "id": "routes_update_profile",
      "methods": [
        "PUT"
      ],
      "name": "update_profile",
      "parameters": {
        "email": "邮箱地址",
        "phone": "手机号码",
        "real_name": "真实姓名",
        "school": "学校名称"
      },
      "path": "/profile",
      "responses": {
        "200": {
          "message": "string",
          "success": "boolean"
        }
      },
      "technical_category": "routes"
    }
  },
  "categories": [
    "recommendation_system",
    "homework_management",
    "class_management",
    "student_features",
    "grading_system",
    "authentication",
    "other",
    "data_visualization",
    "database_visualization"
  ],
  "categorized_apis": {
    "authentication": [
      {
        "category": "authentication",
        "database_tables": [
          "users"
        ],
        "description": "用户注册，创建新的学生、教师或管理员账户",
        "example_request": {},
        "example_response": {},
        "file": "auth_routes.py",
        "id": "routes_register",
        "methods": [
          "POST"
        ],
        "name": "register",
        "parameters": {
          "class_name": "班级名称",
          "email": "邮箱地址",
          "grade": "年级（学生必填）",
          "password": "密码",
          "real_name": "真实姓名",
          "role": "用户角色：student/teacher/admin",
          "school": "学校名称",
          "username": "用户名"
        },
        "path": "/register",
        "responses": {
          "201": {
            "message": "string",
            "success": "boolean",
            "user_id": "number"
          },
          "400": {
            "errors": "object",
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "authentication",
        "database_tables": [
          "users",
          "user_sessions"
        ],
        "description": "用户登录认证，支持学生、教师、管理员登录",
        "example_request": {
          "device_type": "web",
          "password": "student123",
          "username": "test_student_001"
        },
        "example_response": {
          "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
          "success": true,
          "user": {
            "id": 2,
            "real_name": "测试学生",
            "role": "student",
            "username": "test_student_001"
          }
        },
        "file": "auth_routes.py",
        "id": "routes_login",
        "methods": [
          "POST"
        ],
        "name": "login",
        "parameters": {
          "device_id": "设备唯一标识",
          "device_type": "设备类型",
          "password": "用户密码",
          "username": "用户名或邮箱"
        },
        "path": "/login",
        "responses": {
          "200": {
            "access_token": "string",
            "expires_in": "number",
            "refresh_token": "string",
            "success": "boolean",
            "user": "object"
          },
          "401": {
            "error": "object",
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "authentication",
        "database_tables": [
          "user_sessions"
        ],
        "description": "刷新用户访问令牌，延长登录会话",
        "example_request": {},
        "example_response": {},
        "file": "auth_routes.py",
        "id": "routes_refresh",
        "methods": [
          "POST"
        ],
        "name": "refresh",
        "parameters": {
          "refresh_token": "刷新令牌"
        },
        "path": "/refresh",
        "responses": {
          "200": {
            "access_token": "string",
            "expires_in": "number",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "authentication",
        "database_tables": [
          "user_sessions"
        ],
        "description": "用户登出，清除会话信息",
        "example_request": {},
        "example_response": {},
        "file": "auth_routes.py",
        "id": "routes_logout",
        "methods": [
          "POST"
        ],
        "name": "logout",
        "parameters": {},
        "path": "/logout",
        "responses": {
          "200": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "authentication",
        "database_tables": [
          "users"
        ],
        "description": "获取当前用户的个人资料信息，包括基本信息和偏好设置",
        "example_request": {},
        "example_response": {},
        "file": "auth_routes.py",
        "id": "routes_get_profile",
        "methods": [
          "GET"
        ],
        "name": "get_profile",
        "parameters": {},
        "path": "/profile",
        "responses": {
          "200": {
            "profile": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "authentication",
        "database_tables": [
          "users"
        ],
        "description": "更新用户个人资料信息",
        "example_request": {},
        "example_response": {},
        "file": "auth_routes.py",
        "id": "routes_update_profile",
        "methods": [
          "PUT"
        ],
        "name": "update_profile",
        "parameters": {
          "email": "邮箱地址",
          "phone": "手机号码",
          "real_name": "真实姓名",
          "school": "学校名称"
        },
        "path": "/profile",
        "responses": {
          "200": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "authentication",
        "database_tables": [
          "user_sessions"
        ],
        "description": "获取用户的活跃会话列表，用于会话管理和安全监控",
        "example_request": {},
        "example_response": {},
        "file": "auth_routes.py",
        "id": "routes_get_sessions",
        "methods": [
          "GET"
        ],
        "name": "get_sessions",
        "parameters": {},
        "path": "/sessions",
        "responses": {
          "200": {
            "sessions": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "authentication",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_serve_frontend",
        "methods": [
          "GET"
        ],
        "name": "serve_frontend",
        "parameters": {},
        "path": "/register",
        "responses": {},
        "technical_category": "main"
      }
    ],
    "class_management": [
      {
        "category": "class_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "analytics_routes.py",
        "id": "routes_get_teacher_overview",
        "methods": [
          "GET"
        ],
        "name": "get_teacher_overview",
        "parameters": {},
        "path": "/overview",
        "responses": {},
        "technical_category": "routes"
      }
    ],
    "data_visualization": [
      {
        "category": "data_visualization",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_health_check",
        "methods": [
          "GET"
        ],
        "name": "health_check",
        "parameters": {},
        "path": "/api/health",
        "responses": {},
        "technical_category": "main"
      },
      {
        "category": "data_visualization",
        "database_tables": [
          "users"
        ],
        "description": "获取指定用户的基本信息",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_user_info",
        "methods": [
          "GET"
        ],
        "name": "user_info",
        "parameters": {
          "user_id": "用户ID（路径参数）"
        },
        "path": "/api/user/<int:user_id>",
        "responses": {
          "200": {
            "success": "boolean",
            "user": "object"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "data_visualization",
        "database_tables": [
          "users"
        ],
        "description": "更新用户信息接口，支持个人资料修改",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_update_user",
        "methods": [
          "POST"
        ],
        "name": "update_user",
        "parameters": {
          "avatar": "头像",
          "email": "邮箱",
          "name": "姓名",
          "phone": "电话"
        },
        "path": "/api/user/update",
        "responses": {
          "200": {
            "success": "boolean",
            "user": "object"
          }
        },
        "technical_category": "main"
      }
    ],
    "database_visualization": [
      {
        "category": "database_visualization",
        "database_tables": [],
        "description": "健康检查接口",
        "example_request": {},
        "example_response": {
          "database": "connected",
          "message": "数据库API服务正常运行",
          "status": "healthy"
        },
        "file": "api-server.py",
        "id": "db_viz_health",
        "methods": [
          "GET"
        ],
        "name": "health_check",
        "parameters": {},
        "path": "/api/health",
        "responses": {
          "200": {
            "database": "string",
            "message": "string",
            "status": "string"
          }
        }
      },
      {
        "category": "database_visualization",
        "database_tables": [
          "INFORMATION_SCHEMA.TABLES"
        ],
        "description": "获取所有表信息",
        "example_request": {},
        "example_response": {
          "tables": [
            {
              "count": 100,
              "name": "users"
            }
          ],
          "total_tables": 1
        },
        "file": "api-server.py",
        "id": "db_viz_tables",
        "methods": [
          "GET"
        ],
        "name": "get_all_tables",
        "parameters": {},
        "path": "/api/database/tables",
        "responses": {
          "200": {
            "tables": "array",
            "total_tables": "number"
          }
        }
      },
      {
        "category": "database_visualization",
        "database_tables": [
          "dynamic"
        ],
        "description": "获取数据库表的实时数据，支持分页和筛选",
        "example_request": {
          "limit": 10,
          "offset": 0
        },
        "example_response": {
          "count": 0,
          "data": [],
          "limit": 10,
          "offset": 0
        },
        "file": "api-server.py",
        "id": "db_viz_table_data",
        "methods": [
          "GET"
        ],
        "name": "get_table_data",
        "parameters": {
          "limit": "查询数量限制，默认10",
          "offset": "偏移量，默认0",
          "table_name": "表名（路径参数）"
        },
        "path": "/api/database/table/<table_name>",
        "responses": {
          "200": {
            "count": "number",
            "data": "array",
            "limit": "number",
            "offset": "number",
            "source": "string",
            "table": "string"
          }
        }
      }
    ],
    "grading_system": [
      {
        "category": "grading_system",
        "database_tables": [
          "homework_submissions",
          "homeworks"
        ],
        "description": "批量评分接口，支持多份作业同时评分",
        "example_request": {},
        "example_response": {},
        "file": "grading_routes.py",
        "id": "routes_batch_grade",
        "methods": [
          "POST"
        ],
        "name": "batch_grade",
        "parameters": {
          "grading_rules": "评分规则",
          "submission_ids": "提交ID列表"
        },
        "path": "/batch-grade",
        "responses": {
          "200": {
            "graded_count": "number",
            "results": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      }
    ],
    "homework_management": [
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "analytics_routes.py",
        "id": "routes_get_homework_analytics",
        "methods": [
          "GET"
        ],
        "name": "get_homework_analytics",
        "parameters": {},
        "path": "/homework/<int:homework_id>",
        "responses": {},
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "analytics_routes.py",
        "id": "routes_export_analytics",
        "methods": [
          "POST"
        ],
        "name": "export_analytics",
        "parameters": {},
        "path": "/homework/<int:homework_id>/export",
        "responses": {},
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_submissions"
        ],
        "description": "提交作业答案，完成作业",
        "example_request": {},
        "example_response": {},
        "file": "submission_routes.py",
        "id": "routes_submit_homework",
        "methods": [
          "POST"
        ],
        "name": "submit_homework",
        "parameters": {
          "answers": "答案数据",
          "assignment_id": "作业分配ID（路径参数）"
        },
        "path": "/<int:assignment_id>",
        "responses": {
          "200": {
            "submission_id": "number",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "simple_analytics_routes.py",
        "id": "routes_get_simple_homework_analytics",
        "methods": [
          "GET"
        ],
        "name": "get_simple_homework_analytics",
        "parameters": {},
        "path": "/homework/<int:homework_id>",
        "responses": {},
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "simple_feedback_routes.py",
        "id": "routes_get_simple_homework_feedback",
        "methods": [
          "GET"
        ],
        "name": "get_simple_homework_feedback",
        "parameters": {},
        "path": "/homework/<int:homework_id>",
        "responses": {},
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "users",
          "classes",
          "homework_assignments"
        ],
        "description": "教师分配作业给班级或学生",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_assign_homework",
        "methods": [
          "POST"
        ],
        "name": "assign_homework",
        "parameters": {
          "due_date": "截止时间",
          "homework_id": "作业ID",
          "target_ids": "目标ID列表",
          "target_type": "分配类型（class/student）"
        },
        "path": "/assign",
        "responses": {
          "200": {
            "assignment_count": "number",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "classes",
          "homework_assignments"
        ],
        "description": "获取指定班级的作业分配情况，教师查看班级作业状态",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_get_class_assignments",
        "methods": [
          "GET"
        ],
        "name": "get_class_assignments",
        "parameters": {
          "class_id": "班级ID（路径参数）"
        },
        "path": "/class/<int:class_id>",
        "responses": {
          "200": {
            "assignments": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_assignments"
        ],
        "description": "获取教师创建的所有作业分配，用于教师管理界面",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_get_my_assignments",
        "methods": [
          "GET"
        ],
        "name": "get_my_assignments",
        "parameters": {},
        "path": "/teacher/my",
        "responses": {
          "200": {
            "assignments": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_assignments"
        ],
        "description": "获取作业分配的详细信息，包括完成情况和统计数据",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_get_assignment_detail",
        "methods": [
          "GET"
        ],
        "name": "get_assignment_detail",
        "parameters": {
          "assignment_id": "分配ID（路径参数）"
        },
        "path": "/<int:assignment_id>",
        "responses": {
          "200": {
            "assignment": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_assignments"
        ],
        "description": "更新作业分配状态，如开启、关闭、延期等操作",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_update_assignment_status",
        "methods": [
          "PUT"
        ],
        "name": "update_assignment_status",
        "parameters": {
          "assignment_id": "分配ID（路径参数）",
          "status": "新状态"
        },
        "path": "/<int:assignment_id>/status",
        "responses": {
          "200": {
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "classes"
        ],
        "description": "获取教师负责的班级列表，用于班级管理",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_get_my_classes",
        "methods": [
          "GET"
        ],
        "name": "get_my_classes",
        "parameters": {},
        "path": "/classes/my",
        "responses": {
          "200": {
            "classes": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "class_students",
          "users"
        ],
        "description": "获取指定班级的学生名单，用于作业分配和管理",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_get_class_students",
        "methods": [
          "GET"
        ],
        "name": "get_class_students",
        "parameters": {
          "class_id": "班级ID（路径参数）"
        },
        "path": "/classes/<int:class_id>/students",
        "responses": {
          "200": {
            "students": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "notifications"
        ],
        "description": "获取用户的通知消息列表，包括作业提醒、系统通知等",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_get_my_notifications",
        "methods": [
          "GET"
        ],
        "name": "get_my_notifications",
        "parameters": {},
        "path": "/notifications/my",
        "responses": {
          "200": {
            "notifications": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "notifications"
        ],
        "description": "标记指定通知为已读状态，更新通知状态",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_mark_notification_read",
        "methods": [
          "PUT"
        ],
        "name": "mark_notification_read",
        "parameters": {
          "notification_id": "通知ID（路径参数）"
        },
        "path": "/notifications/<int:notification_id>/read",
        "responses": {
          "200": {
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_submissions",
          "homework_assignments"
        ],
        "description": "获取作业分配的统计信息，包括完成率、平均分、提交时间分布等",
        "example_request": {},
        "example_response": {},
        "file": "assignment_routes.py",
        "id": "routes_get_assignment_statistics",
        "methods": [
          "GET"
        ],
        "name": "get_assignment_statistics",
        "parameters": {
          "assignment_id": "分配ID（路径参数）"
        },
        "path": "/statistics/<int:assignment_id>",
        "responses": {
          "200": {
            "statistics": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "feedback_routes.py",
        "id": "routes_get_homework_feedback",
        "methods": [
          "GET"
        ],
        "name": "get_homework_feedback",
        "parameters": {},
        "path": "/homework/<int:homework_id>",
        "responses": {},
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "feedback_routes.py",
        "id": "routes_share_feedback",
        "methods": [
          "POST"
        ],
        "name": "share_feedback",
        "parameters": {},
        "path": "/homework/<int:homework_id>/share",
        "responses": {},
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_submissions"
        ],
        "description": "获取整体作业统计信息，教师查看所有作业的完成情况",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_homework_statistics",
        "methods": [
          "GET"
        ],
        "name": "get_homework_statistics",
        "parameters": {},
        "path": "/statistics",
        "responses": {
          "200": {
            "statistics": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks"
        ],
        "description": "获取指定作业的评分规则配置，包括评分标准和权重",
        "example_request": {},
        "example_response": {},
        "file": "grading_routes.py",
        "id": "routes_get_grading_rules",
        "methods": [
          "GET"
        ],
        "name": "get_grading_rules",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/rules/<int:homework_id>",
        "responses": {
          "200": {
            "rules": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks"
        ],
        "description": "更新作业的评分规则，教师可以自定义评分标准",
        "example_request": {},
        "example_response": {},
        "file": "grading_routes.py",
        "id": "routes_update_grading_rules",
        "methods": [
          "POST"
        ],
        "name": "update_grading_rules",
        "parameters": {
          "homework_id": "作业ID（路径参数）",
          "rules": "评分规则配置"
        },
        "path": "/rules/<int:homework_id>",
        "responses": {
          "200": {
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_assignments"
        ],
        "description": "获取学生可见的作业列表",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_homework_list",
        "methods": [
          "GET"
        ],
        "name": "get_homework_list",
        "parameters": {
          "limit": "每页数量",
          "page": "页码"
        },
        "path": "/list",
        "responses": {
          "200": {
            "homeworks": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_submissions",
          "questions",
          "homework_assignments"
        ],
        "description": "获取作业详细信息，包括题目和学生提交状态",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_homework_detail",
        "methods": [
          "GET"
        ],
        "name": "get_homework_detail",
        "parameters": {
          "assignment_id": "作业分配ID（路径参数）"
        },
        "path": "/<int:assignment_id>",
        "responses": {
          "200": {
            "homework": "object",
            "questions": "array",
            "submission_status": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_favorites"
        ],
        "description": "切换作业收藏状态，添加或移除收藏",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_toggle_homework_favorite",
        "methods": [
          "POST"
        ],
        "name": "toggle_homework_favorite",
        "parameters": {
          "assignment_id": "作业分配ID（路径参数）"
        },
        "path": "/<int:assignment_id>/favorite",
        "responses": {
          "200": {
            "is_favorite": "boolean",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_favorites",
          "homework_assignments"
        ],
        "description": "获取用户收藏的作业列表",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_favorite_homeworks",
        "methods": [
          "GET"
        ],
        "name": "get_favorite_homeworks",
        "parameters": {},
        "path": "/favorites",
        "responses": {
          "200": {
            "favorites": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_progress"
        ],
        "description": "获取作业完成进度信息",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_homework_progress",
        "methods": [
          "GET"
        ],
        "name": "get_homework_progress",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/<int:homework_id>/progress",
        "responses": {
          "200": {
            "progress": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_progress"
        ],
        "description": "保存作业完成进度，支持断点续做",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_save_homework_progress",
        "methods": [
          "POST"
        ],
        "name": "save_homework_progress",
        "parameters": {
          "answers": "答案数据",
          "homework_id": "作业ID（路径参数）",
          "progress": "完成进度"
        },
        "path": "/<int:homework_id>/progress",
        "responses": {
          "200": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_reminders",
          "homework_assignments"
        ],
        "description": "获取作业提醒列表，包括即将到期的作业",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_homework_reminders",
        "methods": [
          "GET"
        ],
        "name": "get_homework_reminders",
        "parameters": {},
        "path": "/reminders",
        "responses": {
          "200": {
            "reminders": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_submissions",
          "homework_assignments"
        ],
        "description": "获取学生作业仪表板数据，包括统计信息",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_homework_dashboard",
        "methods": [
          "GET"
        ],
        "name": "get_homework_dashboard",
        "parameters": {},
        "path": "/dashboard",
        "responses": {
          "200": {
            "dashboard": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks"
        ],
        "description": "搜索作业，支持关键词、学科、年级等条件搜索",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_search_homeworks",
        "methods": [
          "GET"
        ],
        "name": "search_homeworks",
        "parameters": {
          "grade": "年级筛选",
          "keyword": "搜索关键词",
          "subject": "学科筛选"
        },
        "path": "/search",
        "responses": {
          "200": {
            "homeworks": "array",
            "success": "boolean",
            "total": "number"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "subjects",
          "grades"
        ],
        "description": "获取作业筛选选项，如可用的学科、年级等",
        "example_request": {},
        "example_response": {},
        "file": "student_homework_routes.py",
        "id": "routes_get_filter_options",
        "methods": [
          "GET"
        ],
        "name": "get_filter_options",
        "parameters": {},
        "path": "/filters/options",
        "responses": {
          "200": {
            "options": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_questions",
          "questions"
        ],
        "description": "创建新作业，教师可以创建包含多个题目的作业",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_create_homework",
        "methods": [
          "POST"
        ],
        "name": "create_homework",
        "parameters": {
          "description": "作业描述",
          "difficulty_level": "难度等级1-5",
          "due_date": "截止日期",
          "grade": "年级",
          "max_score": "总分",
          "questions": "题目列表",
          "subject": "学科",
          "title": "作业标题"
        },
        "path": "/create",
        "responses": {
          "201": {
            "homework_id": "number",
            "message": "string",
            "success": "boolean"
          },
          "400": {
            "errors": "array",
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "users"
        ],
        "description": "获取作业列表，支持分页和筛选",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_list_homeworks",
        "methods": [
          "GET"
        ],
        "name": "list_homeworks",
        "parameters": {
          "category": "分类筛选",
          "grade": "年级筛选",
          "keyword": "关键词搜索",
          "limit": "每页数量，默认10",
          "page": "页码，默认1",
          "subject": "学科筛选"
        },
        "path": "/list",
        "responses": {
          "200": {
            "homeworks": "array",
            "page": "number",
            "success": "boolean",
            "total": "number",
            "total_pages": "number"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "questions"
        ],
        "description": "获取指定作业的详细信息",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_get_homework",
        "methods": [
          "GET"
        ],
        "name": "get_homework",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/<int:homework_id>",
        "responses": {
          "200": {
            "homework": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "questions"
        ],
        "description": "获取作业的所有题目列表",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_get_homework_questions",
        "methods": [
          "GET"
        ],
        "name": "get_homework_questions",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/<int:homework_id>/questions",
        "responses": {
          "200": {
            "questions": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks"
        ],
        "description": "更新作业信息，包括标题、描述、题目等",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_update_homework",
        "methods": [
          "PUT"
        ],
        "name": "update_homework",
        "parameters": {
          "description": "作业描述",
          "due_date": "截止日期",
          "homework_id": "作业ID（路径参数）",
          "title": "作业标题"
        },
        "path": "/<int:homework_id>",
        "responses": {
          "200": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_submissions",
          "questions"
        ],
        "description": "删除指定作业及其相关数据",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_delete_homework",
        "methods": [
          "DELETE"
        ],
        "name": "delete_homework",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/<int:homework_id>",
        "responses": {
          "200": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks"
        ],
        "description": "发布作业，使学生可以看到并完成作业",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_publish_homework",
        "methods": [
          "POST"
        ],
        "name": "publish_homework",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/<int:homework_id>/publish",
        "responses": {
          "200": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks"
        ],
        "description": "取消发布作业，隐藏作业不让学生看到",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_unpublish_homework",
        "methods": [
          "POST"
        ],
        "name": "unpublish_homework",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/<int:homework_id>/unpublish",
        "responses": {
          "200": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_submissions"
        ],
        "description": "获取作业统计信息，包括完成率、平均分等",
        "example_request": {},
        "example_response": {},
        "file": "homework_routes.py",
        "id": "routes_get_statistics",
        "methods": [
          "GET"
        ],
        "name": "get_statistics",
        "parameters": {},
        "path": "/statistics",
        "responses": {
          "200": {
            "statistics": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "作业列表页面重定向接口，用于页面路由跳转",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_redirect_homework_list",
        "methods": [
          "GET"
        ],
        "name": "redirect_homework_list",
        "parameters": {},
        "path": "/homework/list",
        "responses": {
          "302": {
            "redirect_url": "string"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks"
        ],
        "description": "作业详情页面重定向接口，用于页面路由跳转",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_redirect_homework_detail",
        "methods": [
          "GET"
        ],
        "name": "redirect_homework_detail",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/homework/detail/<int:homework_id>",
        "responses": {
          "302": {
            "redirect_url": "string"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_assignments"
        ],
        "description": "获取作业列表，支持分页和筛选条件",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_homework_list",
        "methods": [
          "GET"
        ],
        "name": "homework_list",
        "parameters": {
          "limit": "每页数量",
          "page": "页码",
          "status": "作业状态筛选",
          "userId": "string"
        },
        "path": "/api/homework/list",
        "responses": {
          "200": {
            "homeworks": "array",
            "success": "boolean",
            "total": "number"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homeworks",
          "homework_submissions",
          "questions"
        ],
        "description": "获取指定作业的详细信息，包括题目、提交状态等",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_homework_detail",
        "methods": [
          "GET"
        ],
        "name": "homework_detail",
        "parameters": {
          "homework_id": "作业ID（路径参数）"
        },
        "path": "/api/homework/detail/<int:homework_id>",
        "responses": {
          "200": {
            "homework": "object",
            "questions": "array",
            "success": "boolean"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_submissions",
          "homeworks"
        ],
        "description": "提交作业答案接口，完成作业并触发自动评分",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_submit",
        "methods": [
          "POST"
        ],
        "name": "submit",
        "parameters": {
          "answers": "完整答案数据",
          "homework_id": "作业ID",
          "submit_time": "提交时间"
        },
        "path": "/api/homework/submit",
        "responses": {
          "200": {
            "score": "number",
            "submission_id": "number",
            "success": "boolean"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "homework_management",
        "database_tables": [
          "homework_progress",
          "homework_submissions"
        ],
        "description": "保存作业进度接口，支持断点续做功能",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_save",
        "methods": [
          "POST"
        ],
        "name": "save",
        "parameters": {
          "answers": "当前答案数据",
          "homework_id": "作业ID",
          "progress": "完成进度"
        },
        "path": "/api/homework/save",
        "responses": {
          "200": {
            "saved_at": "string",
            "success": "boolean"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "homework_management",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_serve_homework_static",
        "methods": [
          "GET"
        ],
        "name": "serve_homework_static",
        "parameters": {},
        "path": "/static/homework/<path:filename>",
        "responses": {},
        "technical_category": "main"
      }
    ],
    "other": [
      {
        "category": "other",
        "database_tables": [],
        "description": "系统首页接口，返回系统基本信息和状态",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_hello_world",
        "methods": [
          "GET"
        ],
        "name": "hello_world",
        "parameters": {},
        "path": "/",
        "responses": {
          "200": {
            "message": "string",
            "system_info": "object"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "other",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_serve_static",
        "methods": [
          "GET"
        ],
        "name": "serve_static",
        "parameters": {},
        "path": "/static/<path:filename>",
        "responses": {},
        "technical_category": "main"
      }
    ],
    "recommendation_system": [
      {
        "category": "recommendation_system",
        "database_tables": [
          "users",
          "symbol_recommendations"
        ],
        "description": "数学符号智能推荐，基于上下文和用户习惯推荐合适的数学符号",
        "example_request": {},
        "example_response": {},
        "file": "recommendation_bp.py",
        "id": "blueprints_recommend_symbols",
        "methods": [
          "POST"
        ],
        "name": "recommend_symbols",
        "parameters": {
          "context": "当前输入上下文",
          "limit": "推荐数量，默认5",
          "question_text": "题目文本"
        },
        "path": "/symbols",
        "responses": {
          "200": {
            "context_analysis": "object",
            "recommendations": "array",
            "success": "boolean"
          }
        },
        "technical_category": "blueprints"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations",
          "interaction_logs"
        ],
        "description": "记录学生使用数学符号的行为数据，用于优化推荐算法",
        "example_request": {},
        "example_response": {},
        "file": "recommendation_bp.py",
        "id": "blueprints_record_symbol_usage",
        "methods": [
          "POST"
        ],
        "name": "record_symbol_usage",
        "parameters": {
          "context": "使用上下文",
          "symbol": "使用的符号",
          "user_id": "用户ID"
        },
        "path": "/symbols/usage",
        "responses": {
          "200": {
            "success": "boolean"
          }
        },
        "technical_category": "blueprints"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations",
          "problem_recommendations"
        ],
        "description": "获取推荐系统统计信息，包括推荐准确率、使用频率等数据",
        "example_request": {},
        "example_response": {},
        "file": "recommendation_bp.py",
        "id": "blueprints_get_recommendation_stats",
        "methods": [
          "GET"
        ],
        "name": "get_recommendation_stats",
        "parameters": {},
        "path": "/stats",
        "responses": {
          "200": {
            "stats": "object",
            "success": "boolean"
          }
        },
        "technical_category": "blueprints"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "users",
          "knowledge_points",
          "knowledge_relationships"
        ],
        "description": "基于AI的知识点推荐，根据用户学习状态和上下文推荐相关知识点",
        "example_request": {
          "context": "解一元二次方程",
          "limit": 3
        },
        "example_response": {
          "recommendations": [
            {
              "description": "用字母和数字表示的数学表达式",
              "difficulty_level": 2,
              "grade_level": 2,
              "id": 2,
              "name": "代数表达式",
              "recommendation_reason": "与输入内容相关",
              "relevance_score": 0.8
            }
          ],
          "success": true,
          "total": 1
        },
        "file": "recommendation_bp.py",
        "id": "blueprints_recommend_knowledge_points",
        "methods": [
          "POST"
        ],
        "name": "recommend_knowledge_points",
        "parameters": {
          "context": "学习上下文内容",
          "limit": "推荐数量限制，默认5",
          "question_id": "题目ID，基于题目推荐"
        },
        "path": "/knowledge",
        "responses": {
          "200": {
            "recommendations": "array",
            "success": "boolean",
            "timestamp": "string",
            "total": "number"
          }
        },
        "technical_category": "blueprints"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "users",
          "problem_recommendations",
          "questions"
        ],
        "description": "基于学生学习状态推荐练习题，支持难度自适应调整",
        "example_request": {},
        "example_response": {},
        "file": "recommendation_bp.py",
        "id": "blueprints_recommend_exercises",
        "methods": [
          "POST"
        ],
        "name": "recommend_exercises",
        "parameters": {
          "count": "推荐数量",
          "difficulty": "难度级别",
          "student_id": "学生ID",
          "subject": "学科"
        },
        "path": "/exercises",
        "responses": {
          "200": {
            "exercises": "array",
            "success": "boolean"
          }
        },
        "technical_category": "blueprints"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "learning_path_recommendations",
          "knowledge_points"
        ],
        "description": "为学生推荐个性化学习路径，基于知识图谱和学习进度",
        "example_request": {},
        "example_response": {},
        "file": "recommendation_bp.py",
        "id": "blueprints_recommend_learning_path",
        "methods": [
          "POST"
        ],
        "name": "recommend_learning_path",
        "parameters": {
          "current_level": "当前水平",
          "student_id": "学生ID",
          "target_knowledge": "目标知识点"
        },
        "path": "/learning-path",
        "responses": {
          "200": {
            "learning_path": "array",
            "success": "boolean"
          }
        },
        "technical_category": "blueprints"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations"
        ],
        "description": "获取数学符号推荐，基于当前输入上下文推荐相关符号",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_symbol_recommendations",
        "methods": [
          "POST"
        ],
        "name": "get_symbol_recommendations",
        "parameters": {
          "context": "输入上下文",
          "subject": "学科领域"
        },
        "path": "/recommend",
        "responses": {
          "200": {
            "recommendations": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations"
        ],
        "description": "获取带解释的符号推荐，包含推荐理由和使用说明",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_explained_symbol_recommendations",
        "methods": [
          "POST"
        ],
        "name": "get_explained_symbol_recommendations",
        "parameters": {
          "context": "输入上下文",
          "explain": "是否需要详细解释"
        },
        "path": "/recommend/explained",
        "responses": {
          "200": {
            "explanations": "array",
            "recommendations": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations"
        ],
        "description": "获取符号自动补全建议，帮助用户快速输入数学表达式",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_symbol_completions",
        "methods": [
          "POST"
        ],
        "name": "get_symbol_completions",
        "parameters": {
          "limit": "返回数量限制",
          "partial_input": "部分输入内容"
        },
        "path": "/complete",
        "responses": {
          "200": {
            "completions": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "users",
          "symbol_recommendations"
        ],
        "description": "获取上下文感知的符号推荐，基于当前题目和学习进度",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_context_aware_recommendations",
        "methods": [
          "POST"
        ],
        "name": "get_context_aware_recommendations",
        "parameters": {
          "context": "当前上下文",
          "subject": "学科",
          "user_level": "用户水平"
        },
        "path": "/context",
        "responses": {
          "200": {
            "recommendations": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations",
          "interaction_logs"
        ],
        "description": "记录用户符号使用行为，用于优化推荐算法和学习分析",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_record_symbol_usage",
        "methods": [
          "POST"
        ],
        "name": "record_symbol_usage",
        "parameters": {
          "context": "使用上下文",
          "symbol": "使用的符号",
          "timestamp": "使用时间"
        },
        "path": "/usage",
        "responses": {
          "200": {
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations"
        ],
        "description": "获取数学符号分类列表，用于符号选择界面的分类显示",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_symbol_categories",
        "methods": [
          "GET"
        ],
        "name": "get_symbol_categories",
        "parameters": {},
        "path": "/categories",
        "responses": {
          "200": {
            "categories": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations"
        ],
        "description": "获取指定分类下的所有数学符号，支持分类浏览",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_symbols_by_category",
        "methods": [
          "GET"
        ],
        "name": "get_symbols_by_category",
        "parameters": {
          "category_id": "分类ID（路径参数）"
        },
        "path": "/category/<category_id>",
        "responses": {
          "200": {
            "success": "boolean",
            "symbols": "array"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations"
        ],
        "description": "搜索数学符号，支持按名称、描述、LaTeX代码等条件搜索",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_search_symbols",
        "methods": [
          "POST"
        ],
        "name": "search_symbols",
        "parameters": {
          "category": "分类筛选",
          "limit": "结果数量限制",
          "query": "搜索关键词"
        },
        "path": "/search",
        "responses": {
          "200": {
            "success": "boolean",
            "symbols": "array"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "users",
          "interaction_logs"
        ],
        "description": "获取用户的符号使用统计信息，包括常用符号、使用频率等",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_user_symbol_stats",
        "methods": [
          "GET"
        ],
        "name": "get_user_symbol_stats",
        "parameters": {
          "user_id": "用户ID（路径参数）"
        },
        "path": "/stats/<int:user_id>",
        "responses": {
          "200": {
            "stats": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "learning_behaviors",
          "users"
        ],
        "description": "获取用户学习分析数据，包括学习行为、进度、偏好等",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_user_learning_analytics",
        "methods": [
          "GET"
        ],
        "name": "get_user_learning_analytics",
        "parameters": {
          "user_id": "用户ID（路径参数）"
        },
        "path": "/analytics/<int:user_id>",
        "responses": {
          "200": {
            "analytics": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "learning_behaviors",
          "symbol_recommendations"
        ],
        "description": "获取自适应推荐结果，基于用户学习状态动态调整推荐内容",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_adaptive_recommendations",
        "methods": [
          "POST"
        ],
        "name": "get_adaptive_recommendations",
        "parameters": {
          "context": "当前学习上下文",
          "difficulty": "期望难度",
          "user_id": "用户ID"
        },
        "path": "/recommend/adaptive",
        "responses": {
          "200": {
            "recommendations": "array",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "learning_behaviors",
          "engagement_metrics"
        ],
        "description": "获取学习洞察报告，分析用户学习模式和改进建议",
        "example_request": {},
        "example_response": {},
        "file": "enhanced_symbol_routes.py",
        "id": "routes_get_learning_insights",
        "methods": [
          "GET"
        ],
        "name": "get_learning_insights",
        "parameters": {
          "user_id": "用户ID（路径参数）"
        },
        "path": "/learning-insights/<int:user_id>",
        "responses": {
          "200": {
            "insights": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "recommendation_system",
        "database_tables": [],
        "description": "知识点题目页面重定向接口，用于页面路由跳转",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_redirect_knowledge_question",
        "methods": [
          "GET",
          "POST"
        ],
        "name": "redirect_knowledge_question",
        "parameters": {},
        "path": "/knowledge/question",
        "responses": {
          "302": {
            "redirect_url": "string"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "knowledge_relationships",
          "knowledge_points",
          "questions"
        ],
        "description": "获取题目相关的知识点信息，支持知识点查询和关联分析",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_question_knowledge",
        "methods": [
          "GET",
          "POST"
        ],
        "name": "question_knowledge",
        "parameters": {
          "knowledge_point": "知识点名称",
          "questionId": "string",
          "question_id": "题目ID",
          "text": "string"
        },
        "path": "/api/knowledge/question",
        "responses": {
          "200": {
            "knowledge_points": "array",
            "relationships": "array",
            "success": "boolean"
          },
          "400": {
            "error": "string",
            "message": "string"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "symbol_recommendations",
          "interaction_logs"
        ],
        "description": "数学符号推荐接口，根据输入上下文推荐合适的符号",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_recommend_symbols",
        "methods": [
          "POST"
        ],
        "name": "recommend_symbols",
        "parameters": {
          "context": "输入上下文",
          "subject": "学科领域",
          "user_level": "用户水平"
        },
        "path": "/api/recommend/symbols",
        "responses": {
          "200": {
            "success": "boolean",
            "symbols": "array"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "learning_behaviors",
          "knowledge_points",
          "knowledge_relationships"
        ],
        "description": "知识点推荐接口，基于学习进度推荐相关知识点",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_recommend_knowledge",
        "methods": [
          "POST"
        ],
        "name": "recommend_knowledge",
        "parameters": {
          "current_topic": "当前学习主题",
          "learning_goal": "学习目标",
          "user_id": "用户ID"
        },
        "path": "/api/recommend/knowledge",
        "responses": {
          "200": {
            "knowledge_recommendations": "array",
            "success": "boolean"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "recommendation_system",
        "database_tables": [
          "learning_behaviors",
          "problem_recommendations",
          "questions"
        ],
        "description": "主要的练习推荐接口，整合多种推荐算法",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_recommend_exercises",
        "methods": [
          "POST"
        ],
        "name": "recommend_exercises",
        "parameters": {
          "difficulty_range": "难度范围",
          "preferences": "用户偏好",
          "user_id": "用户ID"
        },
        "path": "/api/recommend/exercises",
        "responses": {
          "200": {
            "recommendations": "array",
            "success": "boolean"
          }
        },
        "technical_category": "main"
      },
      {
        "category": "recommendation_system",
        "database_tables": [],
        "description": "暂无描述",
        "example_request": {},
        "example_response": {},
        "file": "app.py",
        "id": "main_serve_symbol_static",
        "methods": [
          "GET"
        ],
        "name": "serve_symbol_static",
        "parameters": {},
        "path": "/static/symbol/<path:filename>",
        "responses": {},
        "technical_category": "main"
      }
    ],
    "student_features": [
      {
        "category": "student_features",
        "database_tables": [
          "homework_submissions"
        ],
        "description": "获取学生作业提交的完整结果，包括答案、评分、反馈",
        "example_request": {},
        "example_response": {},
        "file": "submission_routes.py",
        "id": "routes_get_submission_result",
        "methods": [
          "GET"
        ],
        "name": "get_submission_result",
        "parameters": {
          "submission_id": "提交ID（路径参数）"
        },
        "path": "/<int:submission_id>/result",
        "responses": {
          "200": {
            "submission": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "student_features",
        "database_tables": [
          "homework_submissions",
          "grading_results",
          "questions"
        ],
        "description": "自动评分学生作业提交，支持多种题型的智能评分",
        "example_request": {},
        "example_response": {},
        "file": "grading_routes.py",
        "id": "routes_grade_submission",
        "methods": [
          "POST"
        ],
        "name": "grade_submission",
        "parameters": {
          "submission_id": "提交ID（路径参数）"
        },
        "path": "/grade/<int:submission_id>",
        "responses": {
          "200": {
            "grading_result": "object",
            "max_score": "number",
            "success": "boolean",
            "total_score": "number"
          },
          "404": {
            "message": "string",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "student_features",
        "database_tables": [
          "homework_submissions"
        ],
        "description": "获取作业提交的评分结果，包括得分、错误分析、改进建议",
        "example_request": {},
        "example_response": {},
        "file": "grading_routes.py",
        "id": "routes_get_grading_result",
        "methods": [
          "GET"
        ],
        "name": "get_grading_result",
        "parameters": {
          "submission_id": "提交ID（路径参数）"
        },
        "path": "/result/<int:submission_id>",
        "responses": {
          "200": {
            "result": "object",
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      },
      {
        "category": "student_features",
        "database_tables": [
          "homework_submissions"
        ],
        "description": "教师复查自动评分结果，可以调整分数和添加评语",
        "example_request": {},
        "example_response": {},
        "file": "grading_routes.py",
        "id": "routes_review_grading",
        "methods": [
          "POST"
        ],
        "name": "review_grading",
        "parameters": {
          "adjustments": "评分调整",
          "comments": "教师评语",
          "submission_id": "提交ID（路径参数）"
        },
        "path": "/review/<int:submission_id>",
        "responses": {
          "200": {
            "success": "boolean"
          }
        },
        "technical_category": "routes"
      }
    ]
  },
  "success": true,
  "total_apis": 92
}
```

## 项目API详细列表

### API统计信息
- **总API数量**: 92
- **API分类**: recommendation_system, homework_management, class_management, student_features, grading_system, authentication, other, data_visualization, database_visualization
- **生成时间**: 2025-09-21 10:45:43


### 按功能分类的API

#### Authentication

**register**
- 路径: `POST /register`
- 描述: 用户注册，创建新的学生、教师或管理员账户
- 文件: auth_routes.py
- 关联表: users

**login**
- 路径: `POST /login`
- 描述: 用户登录认证，支持学生、教师、管理员登录
- 文件: auth_routes.py
- 关联表: users, user_sessions

**refresh**
- 路径: `POST /refresh`
- 描述: 刷新用户访问令牌，延长登录会话
- 文件: auth_routes.py
- 关联表: user_sessions

**logout**
- 路径: `POST /logout`
- 描述: 用户登出，清除会话信息
- 文件: auth_routes.py
- 关联表: user_sessions

**get_profile**
- 路径: `GET /profile`
- 描述: 获取当前用户的个人资料信息，包括基本信息和偏好设置
- 文件: auth_routes.py
- 关联表: users

**update_profile**
- 路径: `PUT /profile`
- 描述: 更新用户个人资料信息
- 文件: auth_routes.py
- 关联表: users

**get_sessions**
- 路径: `GET /sessions`
- 描述: 获取用户的活跃会话列表，用于会话管理和安全监控
- 文件: auth_routes.py
- 关联表: user_sessions

**serve_frontend**
- 路径: `GET /register`
- 描述: 暂无描述
- 文件: app.py

#### Class Management

**get_teacher_overview**
- 路径: `GET /overview`
- 描述: 暂无描述
- 文件: analytics_routes.py

#### Data Visualization

**health_check**
- 路径: `GET /api/health`
- 描述: 暂无描述
- 文件: app.py

**user_info**
- 路径: `GET /api/user/<int:user_id>`
- 描述: 获取指定用户的基本信息
- 文件: app.py
- 关联表: users

**update_user**
- 路径: `POST /api/user/update`
- 描述: 更新用户信息接口，支持个人资料修改
- 文件: app.py
- 关联表: users

#### Database Visualization

**health_check**
- 路径: `GET /api/health`
- 描述: 健康检查接口
- 文件: api-server.py

**get_all_tables**
- 路径: `GET /api/database/tables`
- 描述: 获取所有表信息
- 文件: api-server.py
- 关联表: INFORMATION_SCHEMA.TABLES

**get_table_data**
- 路径: `GET /api/database/table/<table_name>`
- 描述: 获取数据库表的实时数据，支持分页和筛选
- 文件: api-server.py
- 关联表: dynamic

#### Grading System

**batch_grade**
- 路径: `POST /batch-grade`
- 描述: 批量评分接口，支持多份作业同时评分
- 文件: grading_routes.py
- 关联表: homework_submissions, homeworks

#### Homework Management

**get_homework_analytics**
- 路径: `GET /homework/<int:homework_id>`
- 描述: 暂无描述
- 文件: analytics_routes.py

**export_analytics**
- 路径: `POST /homework/<int:homework_id>/export`
- 描述: 暂无描述
- 文件: analytics_routes.py

**submit_homework**
- 路径: `POST /<int:assignment_id>`
- 描述: 提交作业答案，完成作业
- 文件: submission_routes.py
- 关联表: homework_submissions

**get_simple_homework_analytics**
- 路径: `GET /homework/<int:homework_id>`
- 描述: 暂无描述
- 文件: simple_analytics_routes.py

**get_simple_homework_feedback**
- 路径: `GET /homework/<int:homework_id>`
- 描述: 暂无描述
- 文件: simple_feedback_routes.py

**assign_homework**
- 路径: `POST /assign`
- 描述: 教师分配作业给班级或学生
- 文件: assignment_routes.py
- 关联表: users, classes, homework_assignments

**get_class_assignments**
- 路径: `GET /class/<int:class_id>`
- 描述: 获取指定班级的作业分配情况，教师查看班级作业状态
- 文件: assignment_routes.py
- 关联表: classes, homework_assignments

**get_my_assignments**
- 路径: `GET /teacher/my`
- 描述: 获取教师创建的所有作业分配，用于教师管理界面
- 文件: assignment_routes.py
- 关联表: homeworks, homework_assignments

**get_assignment_detail**
- 路径: `GET /<int:assignment_id>`
- 描述: 获取作业分配的详细信息，包括完成情况和统计数据
- 文件: assignment_routes.py
- 关联表: homework_assignments

**update_assignment_status**
- 路径: `PUT /<int:assignment_id>/status`
- 描述: 更新作业分配状态，如开启、关闭、延期等操作
- 文件: assignment_routes.py
- 关联表: homework_assignments

**get_my_classes**
- 路径: `GET /classes/my`
- 描述: 获取教师负责的班级列表，用于班级管理
- 文件: assignment_routes.py
- 关联表: classes

**get_class_students**
- 路径: `GET /classes/<int:class_id>/students`
- 描述: 获取指定班级的学生名单，用于作业分配和管理
- 文件: assignment_routes.py
- 关联表: class_students, users

**get_my_notifications**
- 路径: `GET /notifications/my`
- 描述: 获取用户的通知消息列表，包括作业提醒、系统通知等
- 文件: assignment_routes.py
- 关联表: notifications

**mark_notification_read**
- 路径: `PUT /notifications/<int:notification_id>/read`
- 描述: 标记指定通知为已读状态，更新通知状态
- 文件: assignment_routes.py
- 关联表: notifications

**get_assignment_statistics**
- 路径: `GET /statistics/<int:assignment_id>`
- 描述: 获取作业分配的统计信息，包括完成率、平均分、提交时间分布等
- 文件: assignment_routes.py
- 关联表: homework_submissions, homework_assignments

**get_homework_feedback**
- 路径: `GET /homework/<int:homework_id>`
- 描述: 暂无描述
- 文件: feedback_routes.py

**share_feedback**
- 路径: `POST /homework/<int:homework_id>/share`
- 描述: 暂无描述
- 文件: feedback_routes.py

**get_homework_statistics**
- 路径: `GET /statistics`
- 描述: 获取整体作业统计信息，教师查看所有作业的完成情况
- 文件: student_homework_routes.py
- 关联表: homeworks, homework_submissions

**get_grading_rules**
- 路径: `GET /rules/<int:homework_id>`
- 描述: 获取指定作业的评分规则配置，包括评分标准和权重
- 文件: grading_routes.py
- 关联表: homeworks

**update_grading_rules**
- 路径: `POST /rules/<int:homework_id>`
- 描述: 更新作业的评分规则，教师可以自定义评分标准
- 文件: grading_routes.py
- 关联表: homeworks

**get_homework_list**
- 路径: `GET /list`
- 描述: 获取学生可见的作业列表
- 文件: student_homework_routes.py
- 关联表: homeworks, homework_assignments

**get_homework_detail**
- 路径: `GET /<int:assignment_id>`
- 描述: 获取作业详细信息，包括题目和学生提交状态
- 文件: student_homework_routes.py
- 关联表: homeworks, homework_submissions, questions, homework_assignments

**toggle_homework_favorite**
- 路径: `POST /<int:assignment_id>/favorite`
- 描述: 切换作业收藏状态，添加或移除收藏
- 文件: student_homework_routes.py
- 关联表: homework_favorites

**get_favorite_homeworks**
- 路径: `GET /favorites`
- 描述: 获取用户收藏的作业列表
- 文件: student_homework_routes.py
- 关联表: homework_favorites, homework_assignments

**get_homework_progress**
- 路径: `GET /<int:homework_id>/progress`
- 描述: 获取作业完成进度信息
- 文件: student_homework_routes.py
- 关联表: homework_progress

**save_homework_progress**
- 路径: `POST /<int:homework_id>/progress`
- 描述: 保存作业完成进度，支持断点续做
- 文件: student_homework_routes.py
- 关联表: homework_progress

**get_homework_reminders**
- 路径: `GET /reminders`
- 描述: 获取作业提醒列表，包括即将到期的作业
- 文件: student_homework_routes.py
- 关联表: homework_reminders, homework_assignments

**get_homework_dashboard**
- 路径: `GET /dashboard`
- 描述: 获取学生作业仪表板数据，包括统计信息
- 文件: student_homework_routes.py
- 关联表: homework_submissions, homework_assignments

**search_homeworks**
- 路径: `GET /search`
- 描述: 搜索作业，支持关键词、学科、年级等条件搜索
- 文件: homework_routes.py
- 关联表: homeworks

**get_filter_options**
- 路径: `GET /filters/options`
- 描述: 获取作业筛选选项，如可用的学科、年级等
- 文件: student_homework_routes.py
- 关联表: homeworks, subjects, grades

**create_homework**
- 路径: `POST /create`
- 描述: 创建新作业，教师可以创建包含多个题目的作业
- 文件: homework_routes.py
- 关联表: homeworks, homework_questions, questions

**list_homeworks**
- 路径: `GET /list`
- 描述: 获取作业列表，支持分页和筛选
- 文件: homework_routes.py
- 关联表: homeworks, users

**get_homework**
- 路径: `GET /<int:homework_id>`
- 描述: 获取指定作业的详细信息
- 文件: homework_routes.py
- 关联表: homeworks, questions

**get_homework_questions**
- 路径: `GET /<int:homework_id>/questions`
- 描述: 获取作业的所有题目列表
- 文件: homework_routes.py
- 关联表: homeworks, questions

**update_homework**
- 路径: `PUT /<int:homework_id>`
- 描述: 更新作业信息，包括标题、描述、题目等
- 文件: homework_routes.py
- 关联表: homeworks

**delete_homework**
- 路径: `DELETE /<int:homework_id>`
- 描述: 删除指定作业及其相关数据
- 文件: homework_routes.py
- 关联表: homeworks, homework_submissions, questions

**publish_homework**
- 路径: `POST /<int:homework_id>/publish`
- 描述: 发布作业，使学生可以看到并完成作业
- 文件: homework_routes.py
- 关联表: homeworks

**unpublish_homework**
- 路径: `POST /<int:homework_id>/unpublish`
- 描述: 取消发布作业，隐藏作业不让学生看到
- 文件: homework_routes.py
- 关联表: homeworks

**get_statistics**
- 路径: `GET /statistics`
- 描述: 获取作业统计信息，包括完成率、平均分等
- 文件: homework_routes.py
- 关联表: homeworks, homework_submissions

**redirect_homework_list**
- 路径: `GET /homework/list`
- 描述: 作业列表页面重定向接口，用于页面路由跳转
- 文件: app.py

**redirect_homework_detail**
- 路径: `GET /homework/detail/<int:homework_id>`
- 描述: 作业详情页面重定向接口，用于页面路由跳转
- 文件: app.py
- 关联表: homeworks

**homework_list**
- 路径: `GET /api/homework/list`
- 描述: 获取作业列表，支持分页和筛选条件
- 文件: app.py
- 关联表: homeworks, homework_assignments

**homework_detail**
- 路径: `GET /api/homework/detail/<int:homework_id>`
- 描述: 获取指定作业的详细信息，包括题目、提交状态等
- 文件: app.py
- 关联表: homeworks, homework_submissions, questions

**submit**
- 路径: `POST /api/homework/submit`
- 描述: 提交作业答案接口，完成作业并触发自动评分
- 文件: app.py
- 关联表: homework_submissions, homeworks

**save**
- 路径: `POST /api/homework/save`
- 描述: 保存作业进度接口，支持断点续做功能
- 文件: app.py
- 关联表: homework_progress, homework_submissions

**serve_homework_static**
- 路径: `GET /static/homework/<path:filename>`
- 描述: 暂无描述
- 文件: app.py

#### Other

**hello_world**
- 路径: `GET /`
- 描述: 系统首页接口，返回系统基本信息和状态
- 文件: app.py

**serve_static**
- 路径: `GET /static/<path:filename>`
- 描述: 暂无描述
- 文件: app.py

#### Recommendation System

**recommend_symbols**
- 路径: `POST /symbols`
- 描述: 数学符号智能推荐，基于上下文和用户习惯推荐合适的数学符号
- 文件: recommendation_bp.py
- 关联表: users, symbol_recommendations

**record_symbol_usage**
- 路径: `POST /symbols/usage`
- 描述: 记录学生使用数学符号的行为数据，用于优化推荐算法
- 文件: recommendation_bp.py
- 关联表: symbol_recommendations, interaction_logs

**get_recommendation_stats**
- 路径: `GET /stats`
- 描述: 获取推荐系统统计信息，包括推荐准确率、使用频率等数据
- 文件: recommendation_bp.py
- 关联表: symbol_recommendations, problem_recommendations

**recommend_knowledge_points**
- 路径: `POST /knowledge`
- 描述: 基于AI的知识点推荐，根据用户学习状态和上下文推荐相关知识点
- 文件: recommendation_bp.py
- 关联表: users, knowledge_points, knowledge_relationships

**recommend_exercises**
- 路径: `POST /exercises`
- 描述: 基于学生学习状态推荐练习题，支持难度自适应调整
- 文件: recommendation_bp.py
- 关联表: users, problem_recommendations, questions

**recommend_learning_path**
- 路径: `POST /learning-path`
- 描述: 为学生推荐个性化学习路径，基于知识图谱和学习进度
- 文件: recommendation_bp.py
- 关联表: learning_path_recommendations, knowledge_points

**get_symbol_recommendations**
- 路径: `POST /recommend`
- 描述: 获取数学符号推荐，基于当前输入上下文推荐相关符号
- 文件: enhanced_symbol_routes.py
- 关联表: symbol_recommendations

**get_explained_symbol_recommendations**
- 路径: `POST /recommend/explained`
- 描述: 获取带解释的符号推荐，包含推荐理由和使用说明
- 文件: enhanced_symbol_routes.py
- 关联表: symbol_recommendations

**get_symbol_completions**
- 路径: `POST /complete`
- 描述: 获取符号自动补全建议，帮助用户快速输入数学表达式
- 文件: enhanced_symbol_routes.py
- 关联表: symbol_recommendations

**get_context_aware_recommendations**
- 路径: `POST /context`
- 描述: 获取上下文感知的符号推荐，基于当前题目和学习进度
- 文件: enhanced_symbol_routes.py
- 关联表: users, symbol_recommendations

**record_symbol_usage**
- 路径: `POST /usage`
- 描述: 记录用户符号使用行为，用于优化推荐算法和学习分析
- 文件: enhanced_symbol_routes.py
- 关联表: symbol_recommendations, interaction_logs

**get_symbol_categories**
- 路径: `GET /categories`
- 描述: 获取数学符号分类列表，用于符号选择界面的分类显示
- 文件: enhanced_symbol_routes.py
- 关联表: symbol_recommendations

**get_symbols_by_category**
- 路径: `GET /category/<category_id>`
- 描述: 获取指定分类下的所有数学符号，支持分类浏览
- 文件: enhanced_symbol_routes.py
- 关联表: symbol_recommendations

**search_symbols**
- 路径: `POST /search`
- 描述: 搜索数学符号，支持按名称、描述、LaTeX代码等条件搜索
- 文件: enhanced_symbol_routes.py
- 关联表: symbol_recommendations

**get_user_symbol_stats**
- 路径: `GET /stats/<int:user_id>`
- 描述: 获取用户的符号使用统计信息，包括常用符号、使用频率等
- 文件: enhanced_symbol_routes.py
- 关联表: users, interaction_logs

**get_user_learning_analytics**
- 路径: `GET /analytics/<int:user_id>`
- 描述: 获取用户学习分析数据，包括学习行为、进度、偏好等
- 文件: enhanced_symbol_routes.py
- 关联表: learning_behaviors, users

**get_adaptive_recommendations**
- 路径: `POST /recommend/adaptive`
- 描述: 获取自适应推荐结果，基于用户学习状态动态调整推荐内容
- 文件: enhanced_symbol_routes.py
- 关联表: learning_behaviors, symbol_recommendations

**get_learning_insights**
- 路径: `GET /learning-insights/<int:user_id>`
- 描述: 获取学习洞察报告，分析用户学习模式和改进建议
- 文件: enhanced_symbol_routes.py
- 关联表: learning_behaviors, engagement_metrics

**redirect_knowledge_question**
- 路径: `GET /knowledge/question`
- 描述: 知识点题目页面重定向接口，用于页面路由跳转
- 文件: app.py

**question_knowledge**
- 路径: `GET /api/knowledge/question`
- 描述: 获取题目相关的知识点信息，支持知识点查询和关联分析
- 文件: app.py
- 关联表: knowledge_relationships, knowledge_points, questions

**recommend_symbols**
- 路径: `POST /api/recommend/symbols`
- 描述: 数学符号推荐接口，根据输入上下文推荐合适的符号
- 文件: app.py
- 关联表: symbol_recommendations, interaction_logs

**recommend_knowledge**
- 路径: `POST /api/recommend/knowledge`
- 描述: 知识点推荐接口，基于学习进度推荐相关知识点
- 文件: app.py
- 关联表: learning_behaviors, knowledge_points, knowledge_relationships

**recommend_exercises**
- 路径: `POST /api/recommend/exercises`
- 描述: 主要的练习推荐接口，整合多种推荐算法
- 文件: app.py
- 关联表: learning_behaviors, problem_recommendations, questions

**serve_symbol_static**
- 路径: `GET /static/symbol/<path:filename>`
- 描述: 暂无描述
- 文件: app.py

#### Student Features

**get_submission_result**
- 路径: `GET /<int:submission_id>/result`
- 描述: 获取学生作业提交的完整结果，包括答案、评分、反馈
- 文件: submission_routes.py
- 关联表: homework_submissions

**grade_submission**
- 路径: `POST /grade/<int:submission_id>`
- 描述: 自动评分学生作业提交，支持多种题型的智能评分
- 文件: grading_routes.py
- 关联表: homework_submissions, grading_results, questions

**get_grading_result**
- 路径: `GET /result/<int:submission_id>`
- 描述: 获取作业提交的评分结果，包括得分、错误分析、改进建议
- 文件: grading_routes.py
- 关联表: homework_submissions

**review_grading**
- 路径: `POST /review/<int:submission_id>`
- 描述: 教师复查自动评分结果，可以调整分数和添加评语
- 文件: grading_routes.py
- 关联表: homework_submissions


## 详细API信息

### blueprints_get_recommendation_stats

**基本信息**
- 名称: get_recommendation_stats
- 路径: /stats
- 方法: GET
- 分类: recommendation_system
- 文件: recommendation_bp.py
- 描述: 获取推荐系统统计信息，包括推荐准确率、使用频率等数据

**响应格式**
- 200:
  - stats: object
  - success: boolean

**关联数据库表**
- symbol_recommendations
- problem_recommendations

---

### blueprints_recommend_exercises

**基本信息**
- 名称: recommend_exercises
- 路径: /exercises
- 方法: POST
- 分类: recommendation_system
- 文件: recommendation_bp.py
- 描述: 基于学生学习状态推荐练习题，支持难度自适应调整

**请求参数**
- count: 推荐数量
- difficulty: 难度级别
- student_id: 学生ID
- subject: 学科

**响应格式**
- 200:
  - exercises: array
  - success: boolean

**关联数据库表**
- users
- problem_recommendations
- questions

---

### blueprints_recommend_knowledge_points

**基本信息**
- 名称: recommend_knowledge_points
- 路径: /knowledge
- 方法: POST
- 分类: recommendation_system
- 文件: recommendation_bp.py
- 描述: 基于AI的知识点推荐，根据用户学习状态和上下文推荐相关知识点

**请求参数**
- context: 学习上下文内容
- limit: 推荐数量限制，默认5
- question_id: 题目ID，基于题目推荐

**响应格式**
- 200:
  - recommendations: array
  - success: boolean
  - timestamp: string
  - total: number

**关联数据库表**
- users
- knowledge_points
- knowledge_relationships

---

### blueprints_recommend_learning_path

**基本信息**
- 名称: recommend_learning_path
- 路径: /learning-path
- 方法: POST
- 分类: recommendation_system
- 文件: recommendation_bp.py
- 描述: 为学生推荐个性化学习路径，基于知识图谱和学习进度

**请求参数**
- current_level: 当前水平
- student_id: 学生ID
- target_knowledge: 目标知识点

**响应格式**
- 200:
  - learning_path: array
  - success: boolean

**关联数据库表**
- learning_path_recommendations
- knowledge_points

---

### blueprints_recommend_symbols

**基本信息**
- 名称: recommend_symbols
- 路径: /symbols
- 方法: POST
- 分类: recommendation_system
- 文件: recommendation_bp.py
- 描述: 数学符号智能推荐，基于上下文和用户习惯推荐合适的数学符号

**请求参数**
- context: 当前输入上下文
- limit: 推荐数量，默认5
- question_text: 题目文本

**响应格式**
- 200:
  - context_analysis: object
  - recommendations: array
  - success: boolean

**关联数据库表**
- users
- symbol_recommendations

---

### blueprints_record_symbol_usage

**基本信息**
- 名称: record_symbol_usage
- 路径: /symbols/usage
- 方法: POST
- 分类: recommendation_system
- 文件: recommendation_bp.py
- 描述: 记录学生使用数学符号的行为数据，用于优化推荐算法

**请求参数**
- context: 使用上下文
- symbol: 使用的符号
- user_id: 用户ID

**响应格式**
- 200:
  - success: boolean

**关联数据库表**
- symbol_recommendations
- interaction_logs

---

### db_viz_health

**基本信息**
- 名称: health_check
- 路径: /api/health
- 方法: GET
- 分类: database_visualization
- 文件: api-server.py
- 描述: 健康检查接口

**响应格式**
- 200:
  - database: string
  - message: string
  - status: string

---

### db_viz_table_data

**基本信息**
- 名称: get_table_data
- 路径: /api/database/table/<table_name>
- 方法: GET
- 分类: database_visualization
- 文件: api-server.py
- 描述: 获取数据库表的实时数据，支持分页和筛选

**请求参数**
- limit: 查询数量限制，默认10
- offset: 偏移量，默认0
- table_name: 表名（路径参数）

**响应格式**
- 200:
  - count: number
  - data: array
  - limit: number
  - offset: number
  - source: string
  - table: string

**关联数据库表**
- dynamic

---

### db_viz_tables

**基本信息**
- 名称: get_all_tables
- 路径: /api/database/tables
- 方法: GET
- 分类: database_visualization
- 文件: api-server.py
- 描述: 获取所有表信息

**响应格式**
- 200:
  - tables: array
  - total_tables: number

**关联数据库表**
- INFORMATION_SCHEMA.TABLES

---

### main_health_check

**基本信息**
- 名称: health_check
- 路径: /api/health
- 方法: GET
- 分类: data_visualization
- 文件: app.py
- 描述: 暂无描述

---

### main_hello_world

**基本信息**
- 名称: hello_world
- 路径: /
- 方法: GET
- 分类: other
- 文件: app.py
- 描述: 系统首页接口，返回系统基本信息和状态

**响应格式**
- 200:
  - message: string
  - system_info: object

---

### main_homework_detail

**基本信息**
- 名称: homework_detail
- 路径: /api/homework/detail/<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: app.py
- 描述: 获取指定作业的详细信息，包括题目、提交状态等

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - homework: object
  - questions: array
  - success: boolean

**关联数据库表**
- homeworks
- homework_submissions
- questions

---

### main_homework_list

**基本信息**
- 名称: homework_list
- 路径: /api/homework/list
- 方法: GET
- 分类: homework_management
- 文件: app.py
- 描述: 获取作业列表，支持分页和筛选条件

**请求参数**
- limit: 每页数量
- page: 页码
- status: 作业状态筛选
- userId: string

**响应格式**
- 200:
  - homeworks: array
  - success: boolean
  - total: number

**关联数据库表**
- homeworks
- homework_assignments

---

### main_question_knowledge

**基本信息**
- 名称: question_knowledge
- 路径: /api/knowledge/question
- 方法: GET, POST
- 分类: recommendation_system
- 文件: app.py
- 描述: 获取题目相关的知识点信息，支持知识点查询和关联分析

**请求参数**
- knowledge_point: 知识点名称
- questionId: string
- question_id: 题目ID
- text: string

**响应格式**
- 200:
  - knowledge_points: array
  - relationships: array
  - success: boolean
- 400:
  - error: string
  - message: string

**关联数据库表**
- knowledge_relationships
- knowledge_points
- questions

---

### main_recommend_exercises

**基本信息**
- 名称: recommend_exercises
- 路径: /api/recommend/exercises
- 方法: POST
- 分类: recommendation_system
- 文件: app.py
- 描述: 主要的练习推荐接口，整合多种推荐算法

**请求参数**
- difficulty_range: 难度范围
- preferences: 用户偏好
- user_id: 用户ID

**响应格式**
- 200:
  - recommendations: array
  - success: boolean

**关联数据库表**
- learning_behaviors
- problem_recommendations
- questions

---

### main_recommend_knowledge

**基本信息**
- 名称: recommend_knowledge
- 路径: /api/recommend/knowledge
- 方法: POST
- 分类: recommendation_system
- 文件: app.py
- 描述: 知识点推荐接口，基于学习进度推荐相关知识点

**请求参数**
- current_topic: 当前学习主题
- learning_goal: 学习目标
- user_id: 用户ID

**响应格式**
- 200:
  - knowledge_recommendations: array
  - success: boolean

**关联数据库表**
- learning_behaviors
- knowledge_points
- knowledge_relationships

---

### main_recommend_symbols

**基本信息**
- 名称: recommend_symbols
- 路径: /api/recommend/symbols
- 方法: POST
- 分类: recommendation_system
- 文件: app.py
- 描述: 数学符号推荐接口，根据输入上下文推荐合适的符号

**请求参数**
- context: 输入上下文
- subject: 学科领域
- user_level: 用户水平

**响应格式**
- 200:
  - success: boolean
  - symbols: array

**关联数据库表**
- symbol_recommendations
- interaction_logs

---

### main_redirect_homework_detail

**基本信息**
- 名称: redirect_homework_detail
- 路径: /homework/detail/<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: app.py
- 描述: 作业详情页面重定向接口，用于页面路由跳转

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 302:
  - redirect_url: string

**关联数据库表**
- homeworks

---

### main_redirect_homework_list

**基本信息**
- 名称: redirect_homework_list
- 路径: /homework/list
- 方法: GET
- 分类: homework_management
- 文件: app.py
- 描述: 作业列表页面重定向接口，用于页面路由跳转

**响应格式**
- 302:
  - redirect_url: string

---

### main_redirect_knowledge_question

**基本信息**
- 名称: redirect_knowledge_question
- 路径: /knowledge/question
- 方法: GET, POST
- 分类: recommendation_system
- 文件: app.py
- 描述: 知识点题目页面重定向接口，用于页面路由跳转

**响应格式**
- 302:
  - redirect_url: string

---

### main_save

**基本信息**
- 名称: save
- 路径: /api/homework/save
- 方法: POST
- 分类: homework_management
- 文件: app.py
- 描述: 保存作业进度接口，支持断点续做功能

**请求参数**
- answers: 当前答案数据
- homework_id: 作业ID
- progress: 完成进度

**响应格式**
- 200:
  - saved_at: string
  - success: boolean

**关联数据库表**
- homework_progress
- homework_submissions

---

### main_serve_frontend

**基本信息**
- 名称: serve_frontend
- 路径: /register
- 方法: GET
- 分类: authentication
- 文件: app.py
- 描述: 暂无描述

---

### main_serve_homework_static

**基本信息**
- 名称: serve_homework_static
- 路径: /static/homework/<path:filename>
- 方法: GET
- 分类: homework_management
- 文件: app.py
- 描述: 暂无描述

---

### main_serve_static

**基本信息**
- 名称: serve_static
- 路径: /static/<path:filename>
- 方法: GET
- 分类: other
- 文件: app.py
- 描述: 暂无描述

---

### main_serve_symbol_static

**基本信息**
- 名称: serve_symbol_static
- 路径: /static/symbol/<path:filename>
- 方法: GET
- 分类: recommendation_system
- 文件: app.py
- 描述: 暂无描述

---

### main_submit

**基本信息**
- 名称: submit
- 路径: /api/homework/submit
- 方法: POST
- 分类: homework_management
- 文件: app.py
- 描述: 提交作业答案接口，完成作业并触发自动评分

**请求参数**
- answers: 完整答案数据
- homework_id: 作业ID
- submit_time: 提交时间

**响应格式**
- 200:
  - score: number
  - submission_id: number
  - success: boolean

**关联数据库表**
- homework_submissions
- homeworks

---

### main_update_user

**基本信息**
- 名称: update_user
- 路径: /api/user/update
- 方法: POST
- 分类: data_visualization
- 文件: app.py
- 描述: 更新用户信息接口，支持个人资料修改

**请求参数**
- avatar: 头像
- email: 邮箱
- name: 姓名
- phone: 电话

**响应格式**
- 200:
  - success: boolean
  - user: object

**关联数据库表**
- users

---

### main_user_info

**基本信息**
- 名称: user_info
- 路径: /api/user/<int:user_id>
- 方法: GET
- 分类: data_visualization
- 文件: app.py
- 描述: 获取指定用户的基本信息

**请求参数**
- user_id: 用户ID（路径参数）

**响应格式**
- 200:
  - success: boolean
  - user: object

**关联数据库表**
- users

---

### routes_assign_homework

**基本信息**
- 名称: assign_homework
- 路径: /assign
- 方法: POST
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 教师分配作业给班级或学生

**请求参数**
- due_date: 截止时间
- homework_id: 作业ID
- target_ids: 目标ID列表
- target_type: 分配类型（class/student）

**响应格式**
- 200:
  - assignment_count: number
  - success: boolean

**关联数据库表**
- users
- classes
- homework_assignments

---

### routes_batch_grade

**基本信息**
- 名称: batch_grade
- 路径: /batch-grade
- 方法: POST
- 分类: grading_system
- 文件: grading_routes.py
- 描述: 批量评分接口，支持多份作业同时评分

**请求参数**
- grading_rules: 评分规则
- submission_ids: 提交ID列表

**响应格式**
- 200:
  - graded_count: number
  - results: array
  - success: boolean

**关联数据库表**
- homework_submissions
- homeworks

---

### routes_create_homework

**基本信息**
- 名称: create_homework
- 路径: /create
- 方法: POST
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 创建新作业，教师可以创建包含多个题目的作业

**请求参数**
- description: 作业描述
- difficulty_level: 难度等级1-5
- due_date: 截止日期
- grade: 年级
- max_score: 总分
- questions: 题目列表
- subject: 学科
- title: 作业标题

**响应格式**
- 201:
  - homework_id: number
  - message: string
  - success: boolean
- 400:
  - errors: array
  - message: string
  - success: boolean

**关联数据库表**
- homeworks
- homework_questions
- questions

---

### routes_delete_homework

**基本信息**
- 名称: delete_homework
- 路径: /<int:homework_id>
- 方法: DELETE
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 删除指定作业及其相关数据

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - message: string
  - success: boolean

**关联数据库表**
- homeworks
- homework_submissions
- questions

---

### routes_export_analytics

**基本信息**
- 名称: export_analytics
- 路径: /homework/<int:homework_id>/export
- 方法: POST
- 分类: homework_management
- 文件: analytics_routes.py
- 描述: 暂无描述

---

### routes_get_adaptive_recommendations

**基本信息**
- 名称: get_adaptive_recommendations
- 路径: /recommend/adaptive
- 方法: POST
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取自适应推荐结果，基于用户学习状态动态调整推荐内容

**请求参数**
- context: 当前学习上下文
- difficulty: 期望难度
- user_id: 用户ID

**响应格式**
- 200:
  - recommendations: array
  - success: boolean

**关联数据库表**
- learning_behaviors
- symbol_recommendations

---

### routes_get_assignment_detail

**基本信息**
- 名称: get_assignment_detail
- 路径: /<int:assignment_id>
- 方法: GET
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 获取作业分配的详细信息，包括完成情况和统计数据

**请求参数**
- assignment_id: 分配ID（路径参数）

**响应格式**
- 200:
  - assignment: object
  - success: boolean

**关联数据库表**
- homework_assignments

---

### routes_get_assignment_statistics

**基本信息**
- 名称: get_assignment_statistics
- 路径: /statistics/<int:assignment_id>
- 方法: GET
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 获取作业分配的统计信息，包括完成率、平均分、提交时间分布等

**请求参数**
- assignment_id: 分配ID（路径参数）

**响应格式**
- 200:
  - statistics: object
  - success: boolean

**关联数据库表**
- homework_submissions
- homework_assignments

---

### routes_get_class_assignments

**基本信息**
- 名称: get_class_assignments
- 路径: /class/<int:class_id>
- 方法: GET
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 获取指定班级的作业分配情况，教师查看班级作业状态

**请求参数**
- class_id: 班级ID（路径参数）

**响应格式**
- 200:
  - assignments: array
  - success: boolean

**关联数据库表**
- classes
- homework_assignments

---

### routes_get_class_students

**基本信息**
- 名称: get_class_students
- 路径: /classes/<int:class_id>/students
- 方法: GET
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 获取指定班级的学生名单，用于作业分配和管理

**请求参数**
- class_id: 班级ID（路径参数）

**响应格式**
- 200:
  - students: array
  - success: boolean

**关联数据库表**
- class_students
- users

---

### routes_get_context_aware_recommendations

**基本信息**
- 名称: get_context_aware_recommendations
- 路径: /context
- 方法: POST
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取上下文感知的符号推荐，基于当前题目和学习进度

**请求参数**
- context: 当前上下文
- subject: 学科
- user_level: 用户水平

**响应格式**
- 200:
  - recommendations: array
  - success: boolean

**关联数据库表**
- users
- symbol_recommendations

---

### routes_get_explained_symbol_recommendations

**基本信息**
- 名称: get_explained_symbol_recommendations
- 路径: /recommend/explained
- 方法: POST
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取带解释的符号推荐，包含推荐理由和使用说明

**请求参数**
- context: 输入上下文
- explain: 是否需要详细解释

**响应格式**
- 200:
  - explanations: array
  - recommendations: array
  - success: boolean

**关联数据库表**
- symbol_recommendations

---

### routes_get_favorite_homeworks

**基本信息**
- 名称: get_favorite_homeworks
- 路径: /favorites
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取用户收藏的作业列表

**响应格式**
- 200:
  - favorites: array
  - success: boolean

**关联数据库表**
- homework_favorites
- homework_assignments

---

### routes_get_filter_options

**基本信息**
- 名称: get_filter_options
- 路径: /filters/options
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取作业筛选选项，如可用的学科、年级等

**响应格式**
- 200:
  - options: object
  - success: boolean

**关联数据库表**
- homeworks
- subjects
- grades

---

### routes_get_grading_result

**基本信息**
- 名称: get_grading_result
- 路径: /result/<int:submission_id>
- 方法: GET
- 分类: student_features
- 文件: grading_routes.py
- 描述: 获取作业提交的评分结果，包括得分、错误分析、改进建议

**请求参数**
- submission_id: 提交ID（路径参数）

**响应格式**
- 200:
  - result: object
  - success: boolean

**关联数据库表**
- homework_submissions

---

### routes_get_grading_rules

**基本信息**
- 名称: get_grading_rules
- 路径: /rules/<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: grading_routes.py
- 描述: 获取指定作业的评分规则配置，包括评分标准和权重

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - rules: object
  - success: boolean

**关联数据库表**
- homeworks

---

### routes_get_homework

**基本信息**
- 名称: get_homework
- 路径: /<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 获取指定作业的详细信息

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - homework: object
  - success: boolean

**关联数据库表**
- homeworks
- questions

---

### routes_get_homework_analytics

**基本信息**
- 名称: get_homework_analytics
- 路径: /homework/<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: analytics_routes.py
- 描述: 暂无描述

---

### routes_get_homework_dashboard

**基本信息**
- 名称: get_homework_dashboard
- 路径: /dashboard
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取学生作业仪表板数据，包括统计信息

**响应格式**
- 200:
  - dashboard: object
  - success: boolean

**关联数据库表**
- homework_submissions
- homework_assignments

---

### routes_get_homework_detail

**基本信息**
- 名称: get_homework_detail
- 路径: /<int:assignment_id>
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取作业详细信息，包括题目和学生提交状态

**请求参数**
- assignment_id: 作业分配ID（路径参数）

**响应格式**
- 200:
  - homework: object
  - questions: array
  - submission_status: object
  - success: boolean

**关联数据库表**
- homeworks
- homework_submissions
- questions
- homework_assignments

---

### routes_get_homework_feedback

**基本信息**
- 名称: get_homework_feedback
- 路径: /homework/<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: feedback_routes.py
- 描述: 暂无描述

---

### routes_get_homework_list

**基本信息**
- 名称: get_homework_list
- 路径: /list
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取学生可见的作业列表

**请求参数**
- limit: 每页数量
- page: 页码

**响应格式**
- 200:
  - homeworks: array
  - success: boolean

**关联数据库表**
- homeworks
- homework_assignments

---

### routes_get_homework_progress

**基本信息**
- 名称: get_homework_progress
- 路径: /<int:homework_id>/progress
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取作业完成进度信息

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - progress: object
  - success: boolean

**关联数据库表**
- homework_progress

---

### routes_get_homework_questions

**基本信息**
- 名称: get_homework_questions
- 路径: /<int:homework_id>/questions
- 方法: GET
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 获取作业的所有题目列表

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - questions: array
  - success: boolean

**关联数据库表**
- homeworks
- questions

---

### routes_get_homework_reminders

**基本信息**
- 名称: get_homework_reminders
- 路径: /reminders
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取作业提醒列表，包括即将到期的作业

**响应格式**
- 200:
  - reminders: array
  - success: boolean

**关联数据库表**
- homework_reminders
- homework_assignments

---

### routes_get_homework_statistics

**基本信息**
- 名称: get_homework_statistics
- 路径: /statistics
- 方法: GET
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 获取整体作业统计信息，教师查看所有作业的完成情况

**响应格式**
- 200:
  - statistics: object
  - success: boolean

**关联数据库表**
- homeworks
- homework_submissions

---

### routes_get_learning_insights

**基本信息**
- 名称: get_learning_insights
- 路径: /learning-insights/<int:user_id>
- 方法: GET
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取学习洞察报告，分析用户学习模式和改进建议

**请求参数**
- user_id: 用户ID（路径参数）

**响应格式**
- 200:
  - insights: object
  - success: boolean

**关联数据库表**
- learning_behaviors
- engagement_metrics

---

### routes_get_my_assignments

**基本信息**
- 名称: get_my_assignments
- 路径: /teacher/my
- 方法: GET
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 获取教师创建的所有作业分配，用于教师管理界面

**响应格式**
- 200:
  - assignments: array
  - success: boolean

**关联数据库表**
- homeworks
- homework_assignments

---

### routes_get_my_classes

**基本信息**
- 名称: get_my_classes
- 路径: /classes/my
- 方法: GET
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 获取教师负责的班级列表，用于班级管理

**响应格式**
- 200:
  - classes: array
  - success: boolean

**关联数据库表**
- classes

---

### routes_get_my_notifications

**基本信息**
- 名称: get_my_notifications
- 路径: /notifications/my
- 方法: GET
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 获取用户的通知消息列表，包括作业提醒、系统通知等

**响应格式**
- 200:
  - notifications: array
  - success: boolean

**关联数据库表**
- notifications

---

### routes_get_profile

**基本信息**
- 名称: get_profile
- 路径: /profile
- 方法: GET
- 分类: authentication
- 文件: auth_routes.py
- 描述: 获取当前用户的个人资料信息，包括基本信息和偏好设置

**响应格式**
- 200:
  - profile: object
  - success: boolean

**关联数据库表**
- users

---

### routes_get_sessions

**基本信息**
- 名称: get_sessions
- 路径: /sessions
- 方法: GET
- 分类: authentication
- 文件: auth_routes.py
- 描述: 获取用户的活跃会话列表，用于会话管理和安全监控

**响应格式**
- 200:
  - sessions: array
  - success: boolean

**关联数据库表**
- user_sessions

---

### routes_get_simple_homework_analytics

**基本信息**
- 名称: get_simple_homework_analytics
- 路径: /homework/<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: simple_analytics_routes.py
- 描述: 暂无描述

---

### routes_get_simple_homework_feedback

**基本信息**
- 名称: get_simple_homework_feedback
- 路径: /homework/<int:homework_id>
- 方法: GET
- 分类: homework_management
- 文件: simple_feedback_routes.py
- 描述: 暂无描述

---

### routes_get_statistics

**基本信息**
- 名称: get_statistics
- 路径: /statistics
- 方法: GET
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 获取作业统计信息，包括完成率、平均分等

**响应格式**
- 200:
  - statistics: object
  - success: boolean

**关联数据库表**
- homeworks
- homework_submissions

---

### routes_get_submission_result

**基本信息**
- 名称: get_submission_result
- 路径: /<int:submission_id>/result
- 方法: GET
- 分类: student_features
- 文件: submission_routes.py
- 描述: 获取学生作业提交的完整结果，包括答案、评分、反馈

**请求参数**
- submission_id: 提交ID（路径参数）

**响应格式**
- 200:
  - submission: object
  - success: boolean

**关联数据库表**
- homework_submissions

---

### routes_get_symbol_categories

**基本信息**
- 名称: get_symbol_categories
- 路径: /categories
- 方法: GET
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取数学符号分类列表，用于符号选择界面的分类显示

**响应格式**
- 200:
  - categories: array
  - success: boolean

**关联数据库表**
- symbol_recommendations

---

### routes_get_symbol_completions

**基本信息**
- 名称: get_symbol_completions
- 路径: /complete
- 方法: POST
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取符号自动补全建议，帮助用户快速输入数学表达式

**请求参数**
- limit: 返回数量限制
- partial_input: 部分输入内容

**响应格式**
- 200:
  - completions: array
  - success: boolean

**关联数据库表**
- symbol_recommendations

---

### routes_get_symbol_recommendations

**基本信息**
- 名称: get_symbol_recommendations
- 路径: /recommend
- 方法: POST
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取数学符号推荐，基于当前输入上下文推荐相关符号

**请求参数**
- context: 输入上下文
- subject: 学科领域

**响应格式**
- 200:
  - recommendations: array
  - success: boolean

**关联数据库表**
- symbol_recommendations

---

### routes_get_symbols_by_category

**基本信息**
- 名称: get_symbols_by_category
- 路径: /category/<category_id>
- 方法: GET
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取指定分类下的所有数学符号，支持分类浏览

**请求参数**
- category_id: 分类ID（路径参数）

**响应格式**
- 200:
  - success: boolean
  - symbols: array

**关联数据库表**
- symbol_recommendations

---

### routes_get_teacher_overview

**基本信息**
- 名称: get_teacher_overview
- 路径: /overview
- 方法: GET
- 分类: class_management
- 文件: analytics_routes.py
- 描述: 暂无描述

---

### routes_get_user_learning_analytics

**基本信息**
- 名称: get_user_learning_analytics
- 路径: /analytics/<int:user_id>
- 方法: GET
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取用户学习分析数据，包括学习行为、进度、偏好等

**请求参数**
- user_id: 用户ID（路径参数）

**响应格式**
- 200:
  - analytics: object
  - success: boolean

**关联数据库表**
- learning_behaviors
- users

---

### routes_get_user_symbol_stats

**基本信息**
- 名称: get_user_symbol_stats
- 路径: /stats/<int:user_id>
- 方法: GET
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 获取用户的符号使用统计信息，包括常用符号、使用频率等

**请求参数**
- user_id: 用户ID（路径参数）

**响应格式**
- 200:
  - stats: object
  - success: boolean

**关联数据库表**
- users
- interaction_logs

---

### routes_grade_submission

**基本信息**
- 名称: grade_submission
- 路径: /grade/<int:submission_id>
- 方法: POST
- 分类: student_features
- 文件: grading_routes.py
- 描述: 自动评分学生作业提交，支持多种题型的智能评分

**请求参数**
- submission_id: 提交ID（路径参数）

**响应格式**
- 200:
  - grading_result: object
  - max_score: number
  - success: boolean
  - total_score: number
- 404:
  - message: string
  - success: boolean

**关联数据库表**
- homework_submissions
- grading_results
- questions

---

### routes_list_homeworks

**基本信息**
- 名称: list_homeworks
- 路径: /list
- 方法: GET
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 获取作业列表，支持分页和筛选

**请求参数**
- category: 分类筛选
- grade: 年级筛选
- keyword: 关键词搜索
- limit: 每页数量，默认10
- page: 页码，默认1
- subject: 学科筛选

**响应格式**
- 200:
  - homeworks: array
  - page: number
  - success: boolean
  - total: number
  - total_pages: number

**关联数据库表**
- homeworks
- users

---

### routes_login

**基本信息**
- 名称: login
- 路径: /login
- 方法: POST
- 分类: authentication
- 文件: auth_routes.py
- 描述: 用户登录认证，支持学生、教师、管理员登录

**请求参数**
- device_id: 设备唯一标识
- device_type: 设备类型
- password: 用户密码
- username: 用户名或邮箱

**响应格式**
- 200:
  - access_token: string
  - expires_in: number
  - refresh_token: string
  - success: boolean
  - user: object
- 401:
  - error: object
  - message: string
  - success: boolean

**关联数据库表**
- users
- user_sessions

---

### routes_logout

**基本信息**
- 名称: logout
- 路径: /logout
- 方法: POST
- 分类: authentication
- 文件: auth_routes.py
- 描述: 用户登出，清除会话信息

**响应格式**
- 200:
  - message: string
  - success: boolean

**关联数据库表**
- user_sessions

---

### routes_mark_notification_read

**基本信息**
- 名称: mark_notification_read
- 路径: /notifications/<int:notification_id>/read
- 方法: PUT
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 标记指定通知为已读状态，更新通知状态

**请求参数**
- notification_id: 通知ID（路径参数）

**响应格式**
- 200:
  - success: boolean

**关联数据库表**
- notifications

---

### routes_publish_homework

**基本信息**
- 名称: publish_homework
- 路径: /<int:homework_id>/publish
- 方法: POST
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 发布作业，使学生可以看到并完成作业

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - message: string
  - success: boolean

**关联数据库表**
- homeworks

---

### routes_record_symbol_usage

**基本信息**
- 名称: record_symbol_usage
- 路径: /usage
- 方法: POST
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 记录用户符号使用行为，用于优化推荐算法和学习分析

**请求参数**
- context: 使用上下文
- symbol: 使用的符号
- timestamp: 使用时间

**响应格式**
- 200:
  - success: boolean

**关联数据库表**
- symbol_recommendations
- interaction_logs

---

### routes_refresh

**基本信息**
- 名称: refresh
- 路径: /refresh
- 方法: POST
- 分类: authentication
- 文件: auth_routes.py
- 描述: 刷新用户访问令牌，延长登录会话

**请求参数**
- refresh_token: 刷新令牌

**响应格式**
- 200:
  - access_token: string
  - expires_in: number
  - success: boolean

**关联数据库表**
- user_sessions

---

### routes_register

**基本信息**
- 名称: register
- 路径: /register
- 方法: POST
- 分类: authentication
- 文件: auth_routes.py
- 描述: 用户注册，创建新的学生、教师或管理员账户

**请求参数**
- class_name: 班级名称
- email: 邮箱地址
- grade: 年级（学生必填）
- password: 密码
- real_name: 真实姓名
- role: 用户角色：student/teacher/admin
- school: 学校名称
- username: 用户名

**响应格式**
- 201:
  - message: string
  - success: boolean
  - user_id: number
- 400:
  - errors: object
  - message: string
  - success: boolean

**关联数据库表**
- users

---

### routes_review_grading

**基本信息**
- 名称: review_grading
- 路径: /review/<int:submission_id>
- 方法: POST
- 分类: student_features
- 文件: grading_routes.py
- 描述: 教师复查自动评分结果，可以调整分数和添加评语

**请求参数**
- adjustments: 评分调整
- comments: 教师评语
- submission_id: 提交ID（路径参数）

**响应格式**
- 200:
  - success: boolean

**关联数据库表**
- homework_submissions

---

### routes_save_homework_progress

**基本信息**
- 名称: save_homework_progress
- 路径: /<int:homework_id>/progress
- 方法: POST
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 保存作业完成进度，支持断点续做

**请求参数**
- answers: 答案数据
- homework_id: 作业ID（路径参数）
- progress: 完成进度

**响应格式**
- 200:
  - message: string
  - success: boolean

**关联数据库表**
- homework_progress

---

### routes_search_homeworks

**基本信息**
- 名称: search_homeworks
- 路径: /search
- 方法: GET
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 搜索作业，支持关键词、学科、年级等条件搜索

**请求参数**
- grade: 年级筛选
- keyword: 搜索关键词
- subject: 学科筛选

**响应格式**
- 200:
  - homeworks: array
  - success: boolean
  - total: number

**关联数据库表**
- homeworks

---

### routes_search_symbols

**基本信息**
- 名称: search_symbols
- 路径: /search
- 方法: POST
- 分类: recommendation_system
- 文件: enhanced_symbol_routes.py
- 描述: 搜索数学符号，支持按名称、描述、LaTeX代码等条件搜索

**请求参数**
- category: 分类筛选
- limit: 结果数量限制
- query: 搜索关键词

**响应格式**
- 200:
  - success: boolean
  - symbols: array

**关联数据库表**
- symbol_recommendations

---

### routes_share_feedback

**基本信息**
- 名称: share_feedback
- 路径: /homework/<int:homework_id>/share
- 方法: POST
- 分类: homework_management
- 文件: feedback_routes.py
- 描述: 暂无描述

---

### routes_submit_homework

**基本信息**
- 名称: submit_homework
- 路径: /<int:assignment_id>
- 方法: POST
- 分类: homework_management
- 文件: submission_routes.py
- 描述: 提交作业答案，完成作业

**请求参数**
- answers: 答案数据
- assignment_id: 作业分配ID（路径参数）

**响应格式**
- 200:
  - submission_id: number
  - success: boolean

**关联数据库表**
- homework_submissions

---

### routes_toggle_homework_favorite

**基本信息**
- 名称: toggle_homework_favorite
- 路径: /<int:assignment_id>/favorite
- 方法: POST
- 分类: homework_management
- 文件: student_homework_routes.py
- 描述: 切换作业收藏状态，添加或移除收藏

**请求参数**
- assignment_id: 作业分配ID（路径参数）

**响应格式**
- 200:
  - is_favorite: boolean
  - success: boolean

**关联数据库表**
- homework_favorites

---

### routes_unpublish_homework

**基本信息**
- 名称: unpublish_homework
- 路径: /<int:homework_id>/unpublish
- 方法: POST
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 取消发布作业，隐藏作业不让学生看到

**请求参数**
- homework_id: 作业ID（路径参数）

**响应格式**
- 200:
  - message: string
  - success: boolean

**关联数据库表**
- homeworks

---

### routes_update_assignment_status

**基本信息**
- 名称: update_assignment_status
- 路径: /<int:assignment_id>/status
- 方法: PUT
- 分类: homework_management
- 文件: assignment_routes.py
- 描述: 更新作业分配状态，如开启、关闭、延期等操作

**请求参数**
- assignment_id: 分配ID（路径参数）
- status: 新状态

**响应格式**
- 200:
  - success: boolean

**关联数据库表**
- homework_assignments

---

### routes_update_grading_rules

**基本信息**
- 名称: update_grading_rules
- 路径: /rules/<int:homework_id>
- 方法: POST
- 分类: homework_management
- 文件: grading_routes.py
- 描述: 更新作业的评分规则，教师可以自定义评分标准

**请求参数**
- homework_id: 作业ID（路径参数）
- rules: 评分规则配置

**响应格式**
- 200:
  - success: boolean

**关联数据库表**
- homeworks

---

### routes_update_homework

**基本信息**
- 名称: update_homework
- 路径: /<int:homework_id>
- 方法: PUT
- 分类: homework_management
- 文件: homework_routes.py
- 描述: 更新作业信息，包括标题、描述、题目等

**请求参数**
- description: 作业描述
- due_date: 截止日期
- homework_id: 作业ID（路径参数）
- title: 作业标题

**响应格式**
- 200:
  - message: string
  - success: boolean

**关联数据库表**
- homeworks

---

### routes_update_profile

**基本信息**
- 名称: update_profile
- 路径: /profile
- 方法: PUT
- 分类: authentication
- 文件: auth_routes.py
- 描述: 更新用户个人资料信息

**请求参数**
- email: 邮箱地址
- phone: 手机号码
- real_name: 真实姓名
- school: 学校名称

**响应格式**
- 200:
  - message: string
  - success: boolean

**关联数据库表**
- users

---


## 使用示例

### JavaScript调用示例
```javascript
// 获取所有API信息
fetch('http://172.104.172.5:5001/api/apis')
  .then(response => response.json())
  .then(data => console.log(data));

// 获取特定API详情
fetch('http://172.104.172.5:5001/api/apis/auth_login')
  .then(response => response.json())
  .then(data => console.log(data));
```

### Python调用示例
```python
import requests

# 获取所有API信息
response = requests.get('http://172.104.172.5:5001/api/apis')
apis = response.json()
print(f"总API数量: {apis['total_apis']}")

# 获取特定API详情
response = requests.get('http://172.104.172.5:5001/api/apis/auth_login')
api_detail = response.json()
print(api_detail)
```

## 总结

本文档基于实际运行的API服务自动生成，包含了系统中所有API的详细信息。文档生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

该API系统为K-12数学教育智能数字生态系统提供了完整的数据库可视化和API分析功能。
