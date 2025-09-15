# Epic 01: 用户认证与管理系统

## 史诗概述

### 史诗标题
用户认证与管理系统

### 史诗描述
构建完整的用户认证、授权和用户信息管理系统，支持学生、教师、管理员和家长四种角色，提供安全可靠的登录认证机制和用户信息管理功能。

### 业务价值
- 为系统提供基础的用户身份认证和授权服务
- 支持多角色用户的差异化功能访问
- 建立用户画像和个性化偏好设置基础
- 确保系统数据安全和用户隐私保护

### 验收标准
- [x] 用户可以成功注册、登录和退出
- [x] 系统支持四种用户角色的权限管理
- [ ] 用户可以管理个人信息和学习偏好
- [x] 会话管理和token认证机制正常工作
- [x] 支持多设备登录和会话管理

## 用户故事

### Story 1.1: 用户注册功能

**作为** 新用户  
**我希望** 能够在系统中注册账户  
**以便** 开始使用K-12数学教育系统

#### 验收标准
- [ ] 用户可以选择角色类型（学生/教师/家长）
- [ ] 必填字段：用户名、邮箱、密码、真实姓名、角色
- [ ] 学生角色需要额外填写：年级、学校、班级、学号
- [ ] 教师角色需要额外填写：学校、教授年级
- [ ] 家长角色需要额外填写：孩子信息关联
- [ ] 系统验证邮箱格式和用户名唯一性
- [ ] 密码强度验证（至少8位，包含字母和数字）
- [ ] 注册成功后发送激活邮件
- [ ] 重复注册时给出友好提示

#### 技术任务
- [x] 创建用户注册API接口 `POST /api/auth/register`
- [ ] 实现前端注册表单组件 `RegisterForm.vue`
- [x] 添加表单验证规则和错误处理
- [x] 实现密码加密存储（bcrypt）
- [x] 设计用户角色权限体系
- [ ] 添加邮箱验证功能

#### 数据库变更
- [x] 确认users表结构符合需求
- [x] 添加用户角色枚举值
- [x] 设置必要的索引和约束

### Story 1.2: 用户登录功能

**作为** 已注册用户  
**我希望** 能够安全地登录系统  
**以便** 访问个性化的教育内容和功能

#### 验收标准
- [x] 用户可以使用用户名/邮箱和密码登录
- [x] 登录成功后生成JWT访问令牌
- [ ] 支持"记住我"功能（延长token有效期）
- [x] 记录设备信息和登录时间
- [ ] 错误密码尝试次数限制（5次后锁定30分钟）
- [x] 登录成功后跳转到用户角色对应的首页
- [x] 支持多设备同时登录

#### 技术任务
- [x] 创建用户登录API接口 `POST /api/auth/login`
- [x] 实现JWT token生成和验证逻辑
- [x] 创建前端登录组件 `LoginPage.vue`
- [x] 实现Vuex认证状态管理
- [x] 添加路由守卫进行权限控制
- [x] 实现会话管理和token刷新机制
- [x] 创建用户会话记录表

#### 数据库变更
- [x] 创建user_sessions表记录用户会话
- [x] 添加设备信息和IP地址字段
- [x] 设置会话过期时间索引

### Story 1.3: 用户信息管理

**作为** 已登录用户  
**我希望** 能够查看和修改个人信息  
**以便** 保持信息准确性和个性化设置

#### 验收标准
- [ ] 用户可以查看完整的个人信息
- [ ] 可修改字段：真实姓名、手机号、头像、学习偏好
- [ ] 不可修改字段：用户名、邮箱、角色（需要管理员权限）
- [ ] 学习偏好包括：难度偏好、学科兴趣、学习风格
- [ ] 头像上传支持常见图片格式（jpg, png, gif）
- [ ] 修改后实时更新界面显示
- [ ] 数据验证和错误提示

#### 技术任务
- [x] 创建用户信息API接口 `GET/PUT /api/auth/profile`
- [ ] 实现前端用户信息组件 `UserProfile.vue`
- [ ] 添加头像上传功能
- [ ] 实现学习偏好设置界面
- [x] 添加表单验证和数据绑定
- [x] 实现信息修改的乐观锁机制

#### 数据库变更
- [x] 为users表添加profile和learning_preferences JSON字段
- [ ] 创建头像文件存储策略

### Story 1.4: 角色权限管理

**作为** 系统管理员  
**我希望** 能够管理用户角色和权限  
**以便** 确保不同用户只能访问相应的功能

#### 验收标准
- [ ] 定义四种基础角色：学生、教师、管理员、家长
- [ ] 学生权限：查看作业、提交作业、查看反馈、个人分析
- [ ] 教师权限：创建作业、批改作业、查看学生数据、班级管理
- [ ] 管理员权限：用户管理、系统配置、数据分析、权限分配
- [ ] 家长权限：查看孩子学习情况、接收学习报告
- [ ] API接口根据用户角色进行权限验证
- [ ] 前端界面根据权限动态显示菜单和功能

#### 技术任务
- [ ] 设计RBAC权限模型
- [ ] 实现权限装饰器和中间件
- [ ] 创建权限验证工具函数
- [ ] 实现前端权限指令 `v-permission`
- [ ] 添加路由级权限控制
- [ ] 创建权限管理后台界面

#### 数据库变更
- [ ] 定义角色权限映射关系
- [ ] 添加权限验证相关索引

### Story 1.5: 会话管理和安全

**作为** 系统用户  
**我希望** 系统能够安全地管理我的登录会话  
**以便** 保护我的账户安全

#### 验收标准
- [ ] JWT token包含用户ID、角色和权限信息
- [ ] token过期时间设置（默认1小时）
- [ ] 支持token刷新机制
- [ ] 用户可以查看当前活跃会话列表
- [ ] 用户可以主动退出指定设备的会话
- [ ] 异常登录时发送安全提醒
- [ ] 支持强制下线功能（管理员）

#### 技术任务
- [ ] 实现JWT token生成和验证服务
- [ ] 创建token刷新API `POST /api/auth/refresh`
- [ ] 实现会话管理API `GET /api/auth/sessions`
- [ ] 添加设备指纹识别
- [ ] 实现退出登录API `POST /api/auth/logout`
- [ ] 创建安全日志记录机制
- [ ] 实现前端自动token刷新

#### 数据库变更
- [ ] 优化user_sessions表结构
- [ ] 添加安全日志表
- [ ] 设置会话清理定时任务

## 技术实现要点

### 后端实现
```python
# 用户认证服务
class AuthService:
    def register_user(self, user_data):
        # 验证用户数据
        # 加密密码
        # 创建用户记录
        # 发送激活邮件
        pass
    
    def authenticate_user(self, username, password):
        # 验证用户凭证
        # 生成JWT token
        # 记录登录会话
        pass
    
    def refresh_token(self, refresh_token):
        # 验证refresh token
        # 生成新的access token
        pass
```

### 前端实现
```vue
<!-- 登录组件 -->
<template>
  <div class="login-form">
    <el-form ref="loginForm" :model="loginData" :rules="loginRules">
      <el-form-item prop="username">
        <el-input v-model="loginData.username" placeholder="用户名/邮箱" />
      </el-form-item>
      <el-form-item prop="password">
        <el-input type="password" v-model="loginData.password" placeholder="密码" />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="handleLogin" :loading="loading">
          登录
        </el-button>
      </el-form-item>
    </el-form>
  </div>
</template>
```

### API设计
```yaml
# 用户认证API
/api/auth/register:
  post:
    summary: 用户注册
    parameters:
      - username: string
      - email: string
      - password: string
      - role: enum[student,teacher,admin,parent]
    responses:
      201: 注册成功
      400: 参数错误
      409: 用户已存在

/api/auth/login:
  post:
    summary: 用户登录
    parameters:
      - username: string
      - password: string
    responses:
      200: 登录成功，返回token
      401: 认证失败
```

## 测试策略

### 单元测试
- [ ] 用户注册逻辑测试
- [ ] 密码加密和验证测试
- [ ] JWT token生成和验证测试
- [ ] 权限验证逻辑测试

### 集成测试
- [ ] 注册登录完整流程测试
- [ ] 不同角色权限访问测试
- [ ] 会话管理和token刷新测试

### 用户验收测试
- [ ] 用户注册流程UAT
- [ ] 登录和权限控制UAT
- [ ] 用户信息管理UAT

## 部署考虑

### 数据库连接
- 连接OceanBase数据库：`obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud:3306`
- 使用数据库：`testccnu`
- 确保用户表和会话表正确创建

### 环境配置
- JWT密钥配置
- 邮件服务配置（用户激活）
- Redis配置（会话缓存）
- 文件上传路径配置（头像存储）

### 监控指标
- 用户注册成功率
- 登录失败率
- token刷新频率
- 会话并发数
- 安全事件数量

## 风险和依赖

### 技术风险
- JWT token安全性
- 数据库连接稳定性
- 邮件服务可用性

### 业务风险
- 用户隐私保护合规
- 角色权限设计复杂度

### 依赖关系
- 依赖邮件服务进行用户激活
- 依赖Redis进行会话缓存
- 依赖文件存储服务进行头像管理

## 开发者记录

### 已完成实现
- [x] 配置文件 (`config.py`) - 数据库连接、JWT配置等
- [x] 数据库管理器 (`models/database.py`) - 连接池和基础CRUD操作
- [x] 用户模型 (`models/user.py`) - 用户类，包含验证、加密、CRUD方法
- [x] 认证服务 (`services/auth_service.py`) - JWT令牌生成、用户注册登录等
- [x] 认证路由 (`routes/auth_routes.py`) - REST API端点和权限装饰器
- [x] 数据库初始化脚本 (`init_database.py`) - 连接测试和表结构检查
- [x] 认证测试脚本 (`test_auth.py`) - 完整的API测试用例
- [x] 依赖更新 (`requirements.txt`) - 添加PyMySQL, bcrypt, PyJWT

### 文件列表
```
homework-backend/
├── config.py                    # 系统配置文件
├── models/
│   ├── database.py              # 数据库连接管理
│   └── user.py                  # 用户模型类
├── services/
│   └── auth_service.py          # 认证服务
├── routes/
│   └── auth_routes.py           # 认证API路由
├── init_database.py             # 数据库初始化脚本
├── test_auth.py                 # 认证功能测试
├── app.py                       # 主应用(已更新)
└── requirements.txt             # 依赖(已更新)
```

### API端点
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录  
- `POST /api/auth/refresh` - 刷新令牌
- `POST /api/auth/logout` - 用户登出
- `GET /api/auth/profile` - 获取用户信息
- `PUT /api/auth/profile` - 更新用户信息
- `GET /api/auth/sessions` - 获取用户会话列表

### 状态
**实现完成度**: 90%
- ✅ 核心认证功能已实现
- ✅ 数据库模型已创建
- ✅ API接口已实现
- ✅ 前端登录组件已实现
- ✅ 路由守卫和权限控制已实现
- ✅ JWT token验证和会话管理已实现
- ⏳ 邮箱验证功能待实现
- ⏳ 登录失败次数限制待实现
- ⏳ 用户信息管理界面待实现

### 最新更新 (2025-09-13)
- ✅ 修复了session_token字段长度问题，支持长JWT token
- ✅ 创建了完整的登录页面 `LoginPage.vue`
- ✅ 实现了路由守卫，自动跳转到登录页
- ✅ 修复了前端token字段名问题（access_token）
- ✅ 添加了用户信息显示和退出登录功能
- ✅ 完成了前后端登录流程的完整集成
