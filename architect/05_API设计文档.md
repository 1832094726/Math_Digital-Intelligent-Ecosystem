# K-12数学教育智能数字生态系统 - API设计文档

## API概述

### 设计原则
- **RESTful架构**：遵循REST设计原则，资源化URL设计
- **统一响应格式**：标准化的JSON响应格式
- **版本管理**：API版本控制，向后兼容
- **安全认证**：JWT Token认证，权限分级控制
- **错误处理**：详细的错误码和错误信息
- **文档完整**：完整的API文档和示例

### 技术规范
- **协议**：HTTPS
- **数据格式**：JSON (Content-Type: application/json)
- **字符编码**：UTF-8
- **认证方式**：JWT Bearer Token
- **API版本**：v1 (URL path: /api/v1/)

## 基础配置

### 服务器配置
```
基础URL: https://diem.edu/api/v1
数据库: obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud:3306
数据库名: testccnu
```

### 通用响应格式

#### 成功响应
```json
{
  "success": true,
  "code": 200,
  "message": "操作成功",
  "data": {
    // 具体数据内容
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_1234567890"
}
```

#### 错误响应
```json
{
  "success": false,
  "code": 400,
  "message": "请求参数错误",
  "error": {
    "type": "ValidationError",
    "details": [
      {
        "field": "email",
        "message": "邮箱格式不正确"
      }
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_1234567890"
}
```

### 状态码定义
| 状态码 | 说明 | 使用场景 |
|--------|------|----------|
| 200 | 成功 | 请求成功处理 |
| 201 | 创建成功 | 资源创建成功 |
| 400 | 请求错误 | 参数错误、格式错误 |
| 401 | 未认证 | 缺少或无效的认证信息 |
| 403 | 权限不足 | 用户无权限访问资源 |
| 404 | 资源不存在 | 请求的资源不存在 |
| 409 | 资源冲突 | 资源已存在或状态冲突 |
| 422 | 参数验证失败 | 业务逻辑验证失败 |
| 500 | 服务器错误 | 系统内部错误 |

## 认证与授权

### 登录认证

#### POST /auth/login
用户登录获取访问令牌

**请求参数**
```json
{
  "username": "student001",
  "password": "password123",
  "device_type": "web",
  "device_id": "browser_fingerprint"
}
```

**响应示例**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IlJlZnJlc2gifQ...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "user": {
      "id": 1001,
      "username": "student001",
      "real_name": "张三",
      "role": "student",
      "grade": 7,
      "school": "实验中学",
      "permissions": ["homework.view", "homework.submit"]
    }
  }
}
```

#### POST /auth/refresh
刷新访问令牌

**请求参数**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IlJlZnJlc2gifQ..."
}
```

#### POST /auth/logout
用户登出

**请求头**
```
Authorization: Bearer {access_token}
```

### 用户注册

#### POST /auth/register
用户注册

**请求参数**
```json
{
  "username": "student002",
  "email": "student002@example.com",
  "password": "password123",
  "real_name": "李四",
  "role": "student",
  "grade": 8,
  "school": "实验中学",
  "phone": "13800138000"
}
```

### 用户信息

#### GET /auth/profile
获取当前用户信息

**响应示例**
```json
{
  "success": true,
  "data": {
    "id": 1001,
    "username": "student001",
    "email": "student001@example.com",
    "real_name": "张三",
    "role": "student",
    "grade": 7,
    "school": "实验中学",
    "class_name": "七年级(1)班",
    "avatar": "https://example.com/avatars/1001.jpg",
    "learning_preferences": {
      "difficulty_preference": 0.6,
      "subject_interests": ["algebra", "geometry"]
    },
    "created_at": "2024-01-01T10:00:00Z"
  }
}
```

#### PUT /auth/profile
更新用户信息

**请求参数**
```json
{
  "real_name": "张三丰",
  "phone": "13800138001",
  "learning_preferences": {
    "difficulty_preference": 0.7,
    "subject_interests": ["algebra", "geometry", "statistics"]
  }
}
```

## 作业管理

### 作业列表

#### GET /homework/list
获取作业列表

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| status | string | 否 | 作业状态: all, pending, completed, overdue |
| subject | string | 否 | 学科筛选 |
| grade | integer | 否 | 年级筛选 |
| page | integer | 否 | 页码，默认1 |
| limit | integer | 否 | 每页数量，默认20 |
| sort | string | 否 | 排序方式: created_time, due_date, difficulty |
| order | string | 否 | 排序顺序: asc, desc |

**请求示例**
```
GET /api/v1/homework/list?status=pending&subject=数学&page=1&limit=10
```

**响应示例**
```json
{
  "success": true,
  "data": {
    "homeworks": [
      {
        "id": 1001,
        "title": "一元一次方程练习",
        "description": "练习一元一次方程的基本解法",
        "subject": "数学",
        "grade": 7,
        "difficulty_level": 3,
        "question_count": 10,
        "max_score": 100,
        "time_limit": 60,
        "due_date": "2024-01-20T23:59:59Z",
        "status": "pending",
        "progress": {
          "completed_questions": 0,
          "total_questions": 10,
          "completion_rate": 0.0
        },
        "created_by": {
          "id": 2001,
          "name": "王老师"
        },
        "created_at": "2024-01-15T10:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 25,
      "total_pages": 3
    }
  }
}
```

### 作业详情

#### GET /homework/detail/{homework_id}
获取作业详细信息

**路径参数**
- homework_id: 作业ID

**响应示例**
```json
{
  "success": true,
  "data": {
    "id": 1001,
    "title": "一元一次方程练习",
    "description": "练习一元一次方程的基本解法",
    "subject": "数学",
    "grade": 7,
    "difficulty_level": 3,
    "max_score": 100,
    "time_limit": 60,
    "due_date": "2024-01-20T23:59:59Z",
    "instructions": "请仔细阅读题目，使用标准解题步骤",
    "questions": [
      {
        "id": 10001,
        "question_order": 1,
        "question_type": "calculation",
        "question_title": "解方程",
        "question_content": "解方程：2x + 3 = 7",
        "question_image": null,
        "options": null,
        "score": 10,
        "difficulty": 3,
        "knowledge_points": [
          {
            "id": 301,
            "name": "一元一次方程",
            "code": "MATH_LINEAR_EQ"
          }
        ],
        "symbols_used": ["+", "=", "x"],
        "solution_steps": [
          "移项：2x = 7 - 3",
          "计算：2x = 4",
          "系数化1：x = 2"
        ]
      }
    ],
    "submission": {
      "id": 20001,
      "status": "draft",
      "answers": {},
      "progress": 0.0,
      "created_at": "2024-01-15T14:00:00Z"
    },
    "recommendations": {
      "symbols": [
        {
          "id": 501,
          "symbol_text": "×",
          "latex_code": "\\times",
          "confidence": 0.85
        }
      ],
      "knowledge_points": [
        {
          "id": 302,
          "name": "移项法则",
          "relevance": 0.92
        }
      ]
    }
  }
}
```

### 作业提交

#### POST /homework/save
保存作业进度

**请求参数**
```json
{
  "homework_id": 1001,
  "answers": {
    "10001": {
      "content": "x = 2",
      "process": "2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2",
      "symbols_used": ["x", "=", "+", "-"],
      "time_spent": 120
    }
  },
  "current_question": 1,
  "auto_save": true
}
```

**响应示例**
```json
{
  "success": true,
  "data": {
    "submission_id": 20001,
    "saved_at": "2024-01-15T14:30:00Z",
    "progress": {
      "completed_questions": 1,
      "total_questions": 10,
      "completion_rate": 0.1
    }
  }
}
```

#### POST /homework/submit
提交作业

**请求参数**
```json
{
  "homework_id": 1001,
  "answers": {
    "10001": {
      "content": "x = 2",
      "process": "2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2",
      "symbols_used": ["x", "=", "+", "-"],
      "time_spent": 120
    }
  },
  "total_time_spent": 1800,
  "submit_type": "final"
}
```

**响应示例**
```json
{
  "success": true,
  "data": {
    "submission_id": 20001,
    "status": "submitted",
    "submitted_at": "2024-01-15T15:00:00Z",
    "estimated_grade_time": "2024-01-15T16:00:00Z",
    "message": "作业提交成功，系统正在自动评分中..."
  }
}
```

### 作业反馈

#### GET /homework/feedback/{submission_id}
获取作业评分反馈

**响应示例**
```json
{
  "success": true,
  "data": {
    "submission_id": 20001,
    "homework_id": 1001,
    "total_score": 85,
    "max_score": 100,
    "completion_rate": 100,
    "grade_percentage": 85,
    "grading_status": "completed",
    "graded_at": "2024-01-15T15:30:00Z",
    "question_results": [
      {
        "question_id": 10001,
        "student_answer": "x = 2",
        "correct_answer": "x = 2",
        "score": 10,
        "max_score": 10,
        "is_correct": true,
        "feedback": "解答正确，步骤清晰",
        "error_analysis": null,
        "suggestions": [
          "步骤写得很好，继续保持"
        ]
      }
    ],
    "overall_feedback": {
      "strengths": ["解题步骤清晰", "计算准确"],
      "weaknesses": ["可以写得更详细"],
      "suggestions": ["继续练习类似题目"]
    },
    "knowledge_mastery": [
      {
        "knowledge_point_id": 301,
        "knowledge_point_name": "一元一次方程",
        "mastery_level": 0.85,
        "improvement_suggestions": ["加强练习"]
      }
    ]
  }
}
```

## 智能推荐

### 符号推荐

#### POST /recommend/symbols
获取符号推荐

**请求参数**
```json
{
  "context": "解方程：2x + 3 = 7",
  "current_symbols": ["2", "x", "+", "3"],
  "question_id": 10001,
  "limit": 10
}
```

**响应示例**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "id": 501,
        "symbol_text": "=",
        "latex_code": "=",
        "symbol_name": "等号",
        "category": "关系符号",
        "confidence": 0.95,
        "reason": "方程中必需的等号",
        "usage_examples": ["x = 5", "2 + 3 = 5"]
      },
      {
        "id": 502,
        "symbol_text": "-",
        "latex_code": "-",
        "symbol_name": "减号",
        "category": "运算符号",
        "confidence": 0.88,
        "reason": "移项时常用的减法运算",
        "usage_examples": ["7 - 3", "x - 2"]
      }
    ],
    "context_analysis": {
      "equation_type": "linear_equation",
      "difficulty_level": 3,
      "required_operations": ["addition", "subtraction", "division"]
    }
  }
}
```

### 知识点推荐

#### POST /recommend/knowledge
获取知识点推荐

**请求参数**
```json
{
  "question_id": 10001,
  "user_id": 1001,
  "current_knowledge_points": [301],
  "limit": 5
}
```

**响应示例**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "id": 302,
        "name": "移项法则",
        "code": "MATH_TRANSPOSE",
        "description": "等式两边同时加减同一个数或代数式",
        "relevance_score": 0.92,
        "reason": "解一元一次方程的核心方法",
        "examples": [
          {
            "title": "基础移项",
            "content": "x + 3 = 7 → x = 7 - 3"
          }
        ],
        "related_videos": [
          {
            "id": "video_001",
            "title": "移项法则详解",
            "duration": 300,
            "url": "https://example.com/videos/transpose.mp4"
          }
        ]
      }
    ],
    "learning_path": [
      {
        "step": 1,
        "knowledge_point_id": 301,
        "name": "一元一次方程基础"
      },
      {
        "step": 2,
        "knowledge_point_id": 302,
        "name": "移项法则"
      },
      {
        "step": 3,
        "knowledge_point_id": 303,
        "name": "合并同类项"
      }
    ]
  }
}
```

### 练习推荐

#### POST /recommend/exercises
获取练习题推荐

**请求参数**
```json
{
  "user_id": 1001,
  "subject": "数学",
  "difficulty_level": 3,
  "topic": "一元一次方程",
  "limit": 5
}
```

**响应示例**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "id": 50001,
        "title": "一元一次方程基础练习",
        "question_count": 8,
        "difficulty_level": 3,
        "estimated_time": 20,
        "success_rate": 0.78,
        "topics": ["移项", "合并同类项"],
        "preview_questions": [
          {
            "question": "解方程：3x - 5 = 7",
            "type": "calculation"
          }
        ]
      }
    ],
    "personalization": {
      "based_on": ["学习历史", "错误模式", "知识点掌握度"],
      "difficulty_adjustment": "根据您的表现，推荐中等难度题目"
    }
  }
}
```

## 知识图谱

### 知识点查询

#### GET /knowledge/points
获取知识点列表

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| subject | string | 否 | 学科筛选 |
| grade | integer | 否 | 年级筛选 |
| category | string | 否 | 分类筛选 |
| parent_id | integer | 否 | 父知识点ID |
| search | string | 否 | 搜索关键词 |

**响应示例**
```json
{
  "success": true,
  "data": {
    "knowledge_points": [
      {
        "id": 301,
        "name": "一元一次方程",
        "code": "MATH_LINEAR_EQ",
        "description": "含有一个未知数，且未知数的最高次数是1的方程",
        "subject": "数学",
        "grade": 7,
        "category": "数与代数",
        "difficulty_level": 4,
        "is_core": true,
        "prerequisites": [
          {
            "id": 201,
            "name": "有理数运算"
          }
        ],
        "children": [
          {
            "id": 302,
            "name": "移项法则"
          }
        ]
      }
    ]
  }
}
```

#### GET /knowledge/points/{point_id}
获取知识点详情

**响应示例**
```json
{
  "success": true,
  "data": {
    "id": 301,
    "name": "一元一次方程",
    "code": "MATH_LINEAR_EQ",
    "description": "含有一个未知数，且未知数的最高次数是1的方程",
    "subject": "数学",
    "grade": 7,
    "category": "数与代数",
    "difficulty_level": 4,
    "learning_objectives": [
      "理解一元一次方程的概念",
      "掌握一元一次方程的解法",
      "能够应用一元一次方程解决实际问题"
    ],
    "prerequisites": [
      {
        "id": 201,
        "name": "有理数运算",
        "relation_type": "prerequisite",
        "strength": 0.9
      }
    ],
    "related_points": [
      {
        "id": 302,
        "name": "移项法则",
        "relation_type": "contains",
        "strength": 0.95
      }
    ],
    "examples": [
      {
        "title": "基础一元一次方程",
        "equation": "2x + 3 = 7",
        "solution": "x = 2",
        "steps": ["移项：2x = 7 - 3", "计算：2x = 4", "系数化1：x = 2"]
      }
    ],
    "common_errors": [
      {
        "error_type": "移项符号错误",
        "description": "移项时没有变号",
        "example": "x + 3 = 7 错误写成 x = 7 + 3"
      }
    ],
    "related_symbols": [
      {
        "id": 501,
        "symbol_text": "x",
        "description": "未知数变量"
      }
    ]
  }
}
```

### 知识图谱关系

#### GET /knowledge/relations
获取知识点关系

**查询参数**
- source_id: 源知识点ID
- target_id: 目标知识点ID
- relation_type: 关系类型

**响应示例**
```json
{
  "success": true,
  "data": {
    "relations": [
      {
        "id": 1001,
        "source": {
          "id": 201,
          "name": "有理数运算"
        },
        "target": {
          "id": 301,
          "name": "一元一次方程"
        },
        "relation_type": "prerequisite",
        "strength": 0.9,
        "description": "有理数运算是学习一元一次方程的基础"
      }
    ]
  }
}
```

## 学习分析

### 学习统计

#### GET /analytics/summary
获取学习概览统计

**查询参数**
- time_range: 时间范围 (week, month, semester, year)
- subject: 学科筛选

**响应示例**
```json
{
  "success": true,
  "data": {
    "overview": {
      "total_homeworks": 25,
      "completed_homeworks": 22,
      "average_score": 86.5,
      "total_study_time": 1800,
      "completion_rate": 0.88
    },
    "recent_activity": [
      {
        "date": "2024-01-15",
        "activity_type": "homework",
        "activity_name": "一元一次方程练习",
        "score": 85,
        "time_spent": 60
      }
    ],
    "subject_performance": [
      {
        "subject": "数学",
        "homework_count": 15,
        "average_score": 88.2,
        "improvement_trend": 0.05
      }
    ],
    "knowledge_mastery": [
      {
        "knowledge_point": "一元一次方程",
        "mastery_level": 0.85,
        "practice_count": 12,
        "accuracy_rate": 0.83
      }
    ]
  }
}
```

### 学习进度

#### GET /analytics/progress
获取学习进度分析

**响应示例**
```json
{
  "success": true,
  "data": {
    "timeline": [
      {
        "date": "2024-01-01",
        "homeworks_completed": 2,
        "total_score": 170,
        "study_time": 120
      }
    ],
    "knowledge_progress": [
      {
        "knowledge_point_id": 301,
        "knowledge_point_name": "一元一次方程",
        "initial_level": 0.2,
        "current_level": 0.85,
        "improvement": 0.65,
        "milestones": [
          {
            "date": "2024-01-05",
            "level": 0.4,
            "event": "完成基础练习"
          }
        ]
      }
    ],
    "skill_radar": {
      "computation": 0.85,
      "reasoning": 0.72,
      "problem_solving": 0.68,
      "communication": 0.75
    }
  }
}
```

### 错误分析

#### GET /analytics/errors
获取错误分析报告

**响应示例**
```json
{
  "success": true,
  "data": {
    "error_summary": {
      "total_errors": 45,
      "error_rate": 0.15,
      "improvement_rate": 0.08
    },
    "error_categories": [
      {
        "category": "计算错误",
        "count": 20,
        "percentage": 0.44,
        "trend": "decreasing",
        "common_mistakes": [
          {
            "mistake": "移项时符号错误",
            "frequency": 12,
            "examples": ["x + 3 = 7 写成 x = 7 + 3"]
          }
        ]
      }
    ],
    "knowledge_gaps": [
      {
        "knowledge_point": "移项法则",
        "gap_level": 0.3,
        "recommended_actions": [
          "观看相关教学视频",
          "完成专项练习",
          "寻求教师指导"
        ]
      }
    ],
    "improvement_suggestions": [
      {
        "priority": "high",
        "suggestion": "加强移项法则的理解和练习",
        "estimated_improvement": 0.15
      }
    ]
  }
}
```

## 系统管理

### 系统配置

#### GET /system/config
获取系统配置（仅管理员）

**响应示例**
```json
{
  "success": true,
  "data": {
    "configs": [
      {
        "key": "homework.default_time_limit",
        "value": "60",
        "type": "number",
        "description": "作业默认时间限制(分钟)"
      }
    ]
  }
}
```

#### PUT /system/config
更新系统配置（仅管理员）

**请求参数**
```json
{
  "configs": [
    {
      "key": "homework.default_time_limit",
      "value": "90"
    }
  ]
}
```

### 日志查询

#### GET /system/logs
获取操作日志（仅管理员）

**查询参数**
- operation_type: 操作类型
- user_id: 用户ID
- start_date: 开始时间
- end_date: 结束时间
- page: 页码
- limit: 每页数量

**响应示例**
```json
{
  "success": true,
  "data": {
    "logs": [
      {
        "id": 100001,
        "user_id": 1001,
        "username": "student001",
        "operation_type": "homework_submit",
        "resource_type": "homework",
        "resource_id": 1001,
        "operation_detail": "提交作业：一元一次方程练习",
        "ip_address": "192.168.1.100",
        "status": "success",
        "execution_time": 250,
        "created_at": "2024-01-15T15:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 1000,
      "total_pages": 50
    }
  }
}
```

## 错误码参考

### 认证相关错误
| 错误码 | HTTP状态码 | 错误信息 | 说明 |
|--------|------------|----------|------|
| AUTH001 | 401 | 用户名或密码错误 | 登录凭证无效 |
| AUTH002 | 401 | 访问令牌已过期 | Token过期需要刷新 |
| AUTH003 | 401 | 访问令牌无效 | Token格式错误或被篡改 |
| AUTH004 | 403 | 权限不足 | 用户无权限访问资源 |
| AUTH005 | 409 | 用户名已存在 | 注册时用户名冲突 |

### 作业相关错误
| 错误码 | HTTP状态码 | 错误信息 | 说明 |
|--------|------------|----------|------|
| HW001 | 404 | 作业不存在 | 作业ID无效 |
| HW002 | 403 | 作业已截止 | 超过截止时间 |
| HW003 | 409 | 作业已提交 | 不能重复提交 |
| HW004 | 422 | 答案格式错误 | 答案数据格式不正确 |
| HW005 | 422 | 必答题未完成 | 存在未回答的必答题 |

### 推荐相关错误
| 错误码 | HTTP状态码 | 错误信息 | 说明 |
|--------|------------|----------|------|
| REC001 | 400 | 上下文信息不足 | 推荐所需上下文缺失 |
| REC002 | 404 | 无可推荐内容 | 没有找到合适的推荐内容 |
| REC003 | 500 | 推荐引擎错误 | 推荐算法执行失败 |

## SDK示例

### JavaScript/TypeScript SDK

```typescript
class DiemAPI {
  private baseURL = 'https://diem.edu/api/v1';
  private token: string | null = null;

  // 设置认证令牌
  setToken(token: string) {
    this.token = token;
  }

  // 通用请求方法
  private async request(endpoint: string, options: RequestInit = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(this.token && { Authorization: `Bearer ${this.token}` }),
      ...options.headers,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.message || '请求失败');
    }
    
    return data.data;
  }

  // 用户登录
  async login(username: string, password: string) {
    const data = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    
    this.setToken(data.access_token);
    return data;
  }

  // 获取作业列表
  async getHomeworkList(params: any = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/homework/list?${query}`);
  }

  // 获取作业详情
  async getHomeworkDetail(homeworkId: number) {
    return this.request(`/homework/detail/${homeworkId}`);
  }

  // 提交作业
  async submitHomework(homeworkId: number, answers: any) {
    return this.request('/homework/submit', {
      method: 'POST',
      body: JSON.stringify({ homework_id: homeworkId, answers }),
    });
  }

  // 获取符号推荐
  async getSymbolRecommendations(context: string, questionId?: number) {
    return this.request('/recommend/symbols', {
      method: 'POST',
      body: JSON.stringify({ context, question_id: questionId }),
    });
  }
}

// 使用示例
const api = new DiemAPI();

// 登录
const loginData = await api.login('student001', 'password123');
console.log('登录成功', loginData.user);

// 获取作业列表
const homeworks = await api.getHomeworkList({ status: 'pending' });
console.log('待完成作业', homeworks);

// 获取符号推荐
const symbols = await api.getSymbolRecommendations('解方程：2x + 3 = 7');
console.log('推荐符号', symbols);
```

### Python SDK

```python
import requests
from typing import Dict, Any, Optional

class DiemAPI:
    def __init__(self, base_url: str = 'https://diem.edu/api/v1'):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.session = requests.Session()
    
    def set_token(self, token: str):
        """设置认证令牌"""
        self.token = token
        self.session.headers.update({'Authorization': f'Bearer {token}'})
    
    def _request(self, endpoint: str, method: str = 'GET', **kwargs) -> Dict[str, Any]:
        """通用请求方法"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        data = response.json()
        if not data.get('success'):
            raise Exception(data.get('message', '请求失败'))
        
        return data.get('data')
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """用户登录"""
        data = self._request('/auth/login', 'POST', json={
            'username': username,
            'password': password
        })
        
        self.set_token(data['access_token'])
        return data
    
    def get_homework_list(self, **params) -> Dict[str, Any]:
        """获取作业列表"""
        return self._request('/homework/list', params=params)
    
    def get_homework_detail(self, homework_id: int) -> Dict[str, Any]:
        """获取作业详情"""
        return self._request(f'/homework/detail/{homework_id}')
    
    def submit_homework(self, homework_id: int, answers: Dict[str, Any]) -> Dict[str, Any]:
        """提交作业"""
        return self._request('/homework/submit', 'POST', json={
            'homework_id': homework_id,
            'answers': answers
        })
    
    def get_symbol_recommendations(self, context: str, question_id: Optional[int] = None) -> Dict[str, Any]:
        """获取符号推荐"""
        return self._request('/recommend/symbols', 'POST', json={
            'context': context,
            'question_id': question_id
        })

# 使用示例
api = DiemAPI()

# 登录
login_data = api.login('student001', 'password123')
print('登录成功', login_data['user'])

# 获取作业列表
homeworks = api.get_homework_list(status='pending')
print('待完成作业', homeworks)

# 获取符号推荐
symbols = api.get_symbol_recommendations('解方程：2x + 3 = 7')
print('推荐符号', symbols)
```

## 总结

本API设计文档提供了K-12数学教育智能数字生态系统的完整API接口规范。主要特点包括：

1. **RESTful设计**：标准化的资源导向API设计
2. **统一响应格式**：一致的JSON响应结构
3. **完善的认证体系**：JWT Token认证和权限控制
4. **智能推荐接口**：符号、知识点、练习题推荐
5. **学习分析功能**：进度跟踪、错误分析、统计报告
6. **详细错误处理**：标准化错误码和错误信息
7. **SDK支持**：多语言SDK示例代码

该API设计能够很好地支持前端应用和第三方系统的集成，为K-12数学教育提供强大的数据服务支持。
