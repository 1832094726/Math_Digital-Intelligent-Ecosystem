// K-12数学教育系统数据库结构数据 - 实际版本
// 基于当前运行系统的真实数据库结构
const databaseSchema = {
    // 数据表定义 - 只包含实际存在的22个表
    tables: {
        // 用户权限系统
        users: {
            name: 'users',
            displayName: '用户基础信息',
            category: 'user',
            description: '存储所有用户的基础信息，包括学生、教师、管理员、家长等',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '用户ID' },
                { name: 'username', type: 'varchar(50)', isUnique: true, description: '用户名' },
                { name: 'email', type: 'varchar(100)', isUnique: true, description: '邮箱' },
                { name: 'password_hash', type: 'varchar(255)', description: '密码哈希' },
                { name: 'role', type: 'enum', values: ['student','teacher','admin','parent'], description: '用户角色' },
                { name: 'real_name', type: 'varchar(50)', description: '真实姓名' },
                { name: 'grade', type: 'int(11)', description: '年级' },
                { name: 'school', type: 'varchar(100)', description: '学校' },
                { name: 'class_name', type: 'varchar(50)', description: '班级' },
                { name: 'student_id', type: 'varchar(20)', description: '学号' },
                { name: 'phone', type: 'varchar(20)', description: '手机号' },
                { name: 'avatar', type: 'varchar(255)', description: '头像URL' },
                { name: 'profile', type: 'json', description: '用户配置信息' },
                { name: 'learning_preferences', type: 'json', description: '学习偏好设置' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否激活' },
                { name: 'last_login_time', type: 'datetime', description: '最后登录时间' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'hasMany',
                    table: 'user_sessions',
                    localKey: 'id',
                    foreignKey: 'user_id',
                    description: '用户有多个会话记录',
                    designReason: {
                        fieldPurpose: 'user_id字段标识会话所属的用户',
                        businessLogic: '一个用户可以在多个设备（手机、电脑、平板）上同时登录',
                        dataIntegrity: '确保每个会话都有明确的用户归属，支持会话管理和安全审计',
                        performanceBenefit: '便于按用户查询所有会话，支持强制下线和异常登录检测'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'notifications',
                    localKey: 'id',
                    foreignKey: 'user_id',
                    description: '用户有多个通知记录',
                    designReason: {
                        fieldPurpose: 'user_id字段标识通知的接收用户',
                        businessLogic: '系统需要向用户发送作业提醒、成绩通知、系统公告等多种通知',
                        dataIntegrity: '确保通知能准确送达目标用户，避免通知错发或丢失',
                        performanceBenefit: '支持按用户查询未读通知，便于消息推送和批量操作'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'homework_submissions',
                    localKey: 'id',
                    foreignKey: 'student_id',
                    description: '用户（学生）有多个作业提交记录',
                    designReason: {
                        fieldPurpose: 'student_id字段标识提交作业的学生',
                        businessLogic: '一个学生在学期内需要提交多次作业，每次提交都是独立记录',
                        dataIntegrity: '确保作业提交与学生身份绑定，支持成绩统计和学习分析',
                        performanceBenefit: '便于查询学生的所有作业记录，支持学习进度跟踪'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'homework_progress',
                    localKey: 'id',
                    foreignKey: 'student_id',
                    description: '用户（学生）有多个答题进度记录',
                    designReason: {
                        fieldPurpose: 'student_id字段标识答题进度所属的学生',
                        businessLogic: '学生可能同时进行多个作业，每个作业都有独立的答题进度',
                        dataIntegrity: '确保答题进度与学生绑定，支持断点续答和进度恢复',
                        performanceBenefit: '便于实时保存和恢复学生的答题状态，提升用户体验'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'class_students',
                    localKey: 'id',
                    foreignKey: 'student_id',
                    description: '用户（学生）可以属于多个班级',
                    designReason: {
                        fieldPurpose: 'student_id字段标识班级中的学生成员',
                        businessLogic: '学生可能因为选修课、兴趣班等原因同时属于多个班级',
                        dataIntegrity: '支持灵活的班级管理，避免学生信息重复存储',
                        performanceBenefit: '便于按班级查询学生列表，支持批量作业分发和成绩统计'
                    }
                }
            ]
        },

        user_sessions: {
            name: 'user_sessions',
            displayName: '用户会话',
            category: 'user',
            description: '用户登录会话管理',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '会话ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'session_token', type: 'text', description: '会话令牌' },
                { name: 'device_type', type: 'varchar(20)', description: '设备类型' },
                { name: 'device_id', type: 'varchar(100)', description: '设备ID' },
                { name: 'ip_address', type: 'varchar(45)', description: 'IP地址' },
                { name: 'user_agent', type: 'text', description: '用户代理' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否活跃' },
                { name: 'expires_at', type: 'datetime', description: '过期时间' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '会话属于特定用户',
                    designReason: {
                        fieldPurpose: 'user_id外键确保每个会话都关联到具体的用户账户',
                        businessLogic: '会话是用户登录后的状态记录，必须与用户身份绑定',
                        dataIntegrity: '防止孤立的会话记录，确保会话安全性和可追溯性',
                        performanceBenefit: '支持快速验证用户身份，便于会话管理和权限控制'
                    }
                }
            ]
        },

        notifications: {
            name: 'notifications',
            displayName: '通知消息',
            category: 'user',
            description: '系统通知和消息管理',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '通知ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'notification_type', type: 'varchar(50)', description: '通知类型' },
                { name: 'title', type: 'varchar(200)', description: '通知标题' },
                { name: 'content', type: 'text', description: '通知内容' },
                { name: 'related_type', type: 'varchar(50)', description: '关联类型' },
                { name: 'related_id', type: 'bigint(20)', description: '关联ID' },
                { name: 'is_read', type: 'tinyint(1)', defaultValue: '0', description: '是否已读' },
                { name: 'priority', type: 'enum', values: ['low','normal','high','urgent'], description: '优先级' },
                { name: 'expires_at', type: 'datetime', description: '过期时间' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '通知发送给特定用户',
                    designReason: {
                        fieldPurpose: 'user_id外键指定通知的接收用户',
                        businessLogic: '通知系统需要精确投递消息给目标用户，如作业截止提醒、成绩发布通知',
                        dataIntegrity: '确保通知不会错发给其他用户，保护用户隐私和信息安全',
                        performanceBenefit: '支持按用户查询未读通知数量，便于消息中心和推送服务'
                    }
                }
            ]
        },

        // 学校组织架构
        schools: {
            name: 'schools',
            displayName: '学校基础信息',
            category: 'school',
            description: '学校基本信息和配置',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '学校ID' },
                { name: 'school_name', type: 'varchar(200)', description: '学校名称' },
                { name: 'school_code', type: 'varchar(50)', isUnique: true, description: '学校代码' },
                { name: 'school_type', type: 'enum', values: ['primary','middle','high','mixed'], description: '学校类型' },
                { name: 'education_level', type: 'varchar(50)', description: '教育层次' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否启用' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                { type: 'hasMany', table: 'grades', localKey: 'id', description: '学校有多个年级' },
                { type: 'hasMany', table: 'curriculum_standards', localKey: 'id', description: '学校有多个课程标准' }
            ]
        },

        grades: {
            name: 'grades',
            displayName: '年级信息',
            category: 'school',
            description: '年级基本信息',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '年级ID' },
                { name: 'school_id', type: 'bigint(20)', isForeignKey: true, references: 'schools.id', description: '学校ID' },
                { name: 'grade_name', type: 'varchar(50)', description: '年级名称' },
                { name: 'grade_level', type: 'int(11)', description: '年级数字' },
                { name: 'academic_year', type: 'varchar(20)', description: '学年' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否启用' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'schools',
                    foreignKey: 'school_id',
                    description: '年级属于特定学校',
                    designReason: {
                        fieldPurpose: 'school_id外键标识年级所属的学校',
                        businessLogic: '年级是学校组织架构的重要组成部分，必须归属于具体学校',
                        dataIntegrity: '确保年级与学校的正确关联，支持多校区管理',
                        performanceBenefit: '便于按学校查询年级列表，支持学校级别的统计分析'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'classes',
                    localKey: 'id',
                    foreignKey: 'grade_id',
                    description: '年级包含多个班级',
                    designReason: {
                        fieldPurpose: 'grade_id外键将班级归属到具体年级',
                        businessLogic: '一个年级通常包含多个平行班级，如一年级1班、2班等',
                        dataIntegrity: '确保班级与年级的层级关系正确，支持年级管理',
                        performanceBenefit: '便于按年级查询所有班级，支持年级级别的批量操作'
                    }
                }
            ]
        },

        classes: {
            name: 'classes',
            displayName: '班级信息',
            category: 'school',
            description: '班级基本信息和配置',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '班级ID' },
                { name: 'school_id', type: 'bigint(20)', isForeignKey: true, references: 'schools.id', description: '学校ID' },
                { name: 'grade_id', type: 'bigint(20)', isForeignKey: true, references: 'grades.id', description: '年级ID' },
                { name: 'class_name', type: 'varchar(100)', description: '班级名称' },
                { name: 'class_code', type: 'varchar(50)', description: '班级代码' },
                { name: 'head_teacher_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '班主任ID' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否启用' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'schools',
                    foreignKey: 'school_id',
                    description: '班级属于特定学校',
                    designReason: {
                        fieldPurpose: 'school_id外键标识班级所属的学校',
                        businessLogic: '班级是学校的基本教学单位，必须归属于具体学校',
                        dataIntegrity: '确保班级与学校的正确关联，支持多校区管理',
                        performanceBenefit: '便于按学校查询班级列表，支持学校级别的管理'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'grades',
                    foreignKey: 'grade_id',
                    description: '班级属于特定年级',
                    designReason: {
                        fieldPurpose: 'grade_id外键标识班级所属的年级',
                        businessLogic: '班级必须归属于具体年级，如一年级、二年级等',
                        dataIntegrity: '确保班级与年级的层级关系正确，支持年级管理',
                        performanceBenefit: '便于按年级查询班级，支持年级级别的统计分析'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'head_teacher_id',
                    description: '班级配置班主任教师',
                    designReason: {
                        fieldPurpose: 'head_teacher_id外键指定班级的班主任教师',
                        businessLogic: '每个班级需要配置班主任负责班级管理和学生指导',
                        dataIntegrity: '确保班主任是系统中的有效教师用户',
                        performanceBenefit: '便于查询教师负责的班级，支持班主任工作管理'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'class_students',
                    localKey: 'id',
                    foreignKey: 'class_id',
                    description: '班级包含多个学生关联记录',
                    designReason: {
                        fieldPurpose: 'class_id外键将学生关联到具体班级',
                        businessLogic: '一个班级包含多名学生，需要维护班级成员关系',
                        dataIntegrity: '支持学生转班、退学等状态变更管理',
                        performanceBenefit: '便于查询班级学生名单，支持班级管理操作'
                    }
                }
            ]
        },

        class_students: {
            name: 'class_students',
            displayName: '班级学生关联',
            category: 'school',
            description: '班级与学生的关联关系',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '关联ID' },
                { name: 'class_id', type: 'bigint(20)', isForeignKey: true, references: 'classes.id', description: '班级ID' },
                { name: 'student_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '学生ID' },
                { name: 'student_number', type: 'varchar(50)', description: '班级内学号' },
                { name: 'joined_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '加入时间' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否活跃' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'classes',
                    foreignKey: 'class_id',
                    description: '学生关联记录属于特定班级',
                    designReason: {
                        fieldPurpose: 'class_id外键标识学生所属的班级',
                        businessLogic: '学生必须归属于具体的班级，便于班级管理和课程安排',
                        dataIntegrity: '确保学生与班级的正确关联，支持班级学生名单管理',
                        performanceBenefit: '便于按班级查询学生列表，支持批量操作和统计分析'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'student_id',
                    description: '班级关联记录对应特定学生',
                    designReason: {
                        fieldPurpose: 'student_id外键标识班级中的具体学生',
                        businessLogic: '班级成员必须是系统中的注册学生，确保身份有效性',
                        dataIntegrity: '防止无效的学生记录，确保班级成员的真实性',
                        performanceBenefit: '便于查询学生的班级归属，支持跨班级学生管理'
                    }
                }
            ]
        },

        curriculum_standards: {
            name: 'curriculum_standards',
            displayName: '课程标准',
            category: 'school',
            description: '课程标准定义',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '标准ID' },
                { name: 'school_id', type: 'bigint(20)', isForeignKey: true, references: 'schools.id', description: '学校ID' },
                { name: 'standard_name', type: 'varchar(200)', description: '标准名称' },
                { name: 'grade_range', type: 'varchar(20)', description: '适用年级范围' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否启用' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'schools', foreignKey: 'school_id', description: '标准属于学校' },
                { type: 'hasMany', table: 'subjects', localKey: 'id', description: '标准有多个学科' }
            ]
        },

        subjects: {
            name: 'subjects',
            displayName: '学科信息',
            category: 'school',
            description: '学科基本信息',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '学科ID' },
                { name: 'standard_id', type: 'bigint(20)', isForeignKey: true, references: 'curriculum_standards.id', description: '课程标准ID' },
                { name: 'subject_name', type: 'varchar(100)', description: '学科名称' },
                { name: 'subject_code', type: 'varchar(50)', description: '学科代码' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否启用' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'curriculum_standards', foreignKey: 'standard_id', description: '学科属于课程标准' },
                { type: 'hasMany', table: 'courses', localKey: 'id', description: '学科有多个课程' }
            ]
        },

        courses: {
            name: 'courses',
            displayName: '课程信息',
            category: 'course',
            description: '课程基本信息',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '课程ID' },
                { name: 'subject_id', type: 'bigint(20)', isForeignKey: true, references: 'subjects.id', description: '学科ID' },
                { name: 'course_code', type: 'varchar(50)', description: '课程代码' },
                { name: 'course_name', type: 'varchar(200)', description: '课程名称' },
                { name: 'grade_level', type: 'int(11)', description: '年级' },
                { name: 'semester', type: 'int(11)', description: '学期' },
                { name: 'course_type', type: 'enum', values: ['standard','adaptive','self_paced','blended'], description: '课程类型' },
                { name: 'status', type: 'enum', values: ['draft','active','archived'], description: '状态' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'subjects', foreignKey: 'subject_id', description: '课程属于学科' },
                { type: 'hasMany', table: 'course_modules', localKey: 'id', description: '课程有多个模块' },
                { type: 'hasMany', table: 'class_schedules', localKey: 'id', description: '课程有多个安排' }
            ]
        },

        course_modules: {
            name: 'course_modules',
            displayName: '课程模块',
            category: 'course',
            description: '课程模块信息',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '模块ID' },
                { name: 'course_id', type: 'bigint(20)', isForeignKey: true, references: 'courses.id', description: '课程ID' },
                { name: 'module_name', type: 'varchar(200)', description: '模块名称' },
                { name: 'module_order', type: 'int(11)', description: '模块顺序' },
                { name: 'suggested_order', type: 'int(11)', description: '建议顺序' },
                { name: 'alternative_paths', type: 'json', description: '替代路径' },
                { name: 'difficulty_adaptations', type: 'json', description: '难度适应' },
                { name: 'content_formats', type: 'json', description: '内容格式' },
                { name: 'is_optional', type: 'tinyint(1)', defaultValue: '0', description: '是否可选' },
                { name: 'estimated_hours', type: 'int(11)', description: '预计学时' },
                { name: 'learning_goals', type: 'json', description: '学习目标' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'courses', foreignKey: 'course_id', description: '模块属于课程' },
                { type: 'hasMany', table: 'chapters', localKey: 'id', description: '模块有多个章节' }
            ]
        },

        chapters: {
            name: 'chapters',
            displayName: '章节信息',
            category: 'course',
            description: '章节基本信息',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '章节ID' },
                { name: 'module_id', type: 'bigint(20)', isForeignKey: true, references: 'course_modules.id', description: '模块ID' },
                { name: 'chapter_name', type: 'varchar(200)', description: '章节名称' },
                { name: 'chapter_order', type: 'int(11)', description: '章节顺序' },
                { name: 'estimated_hours', type: 'int(11)', description: '预计学时' },
                { name: 'learning_goals', type: 'json', description: '学习目标' },
                { name: 'difficulty_level', type: 'varchar(20)', description: '难度等级' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'course_modules', foreignKey: 'module_id', description: '章节属于模块' },
                { type: 'hasMany', table: 'lessons', localKey: 'id', description: '章节有多个课时' }
            ]
        },

        lessons: {
            name: 'lessons',
            displayName: '课时信息',
            category: 'course',
            description: '课时基本信息',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '课时ID' },
                { name: 'chapter_id', type: 'bigint(20)', isForeignKey: true, references: 'chapters.id', description: '章节ID' },
                { name: 'lesson_title', type: 'varchar(200)', description: '课时标题' },
                { name: 'lesson_order', type: 'int(11)', description: '课时顺序' },
                { name: 'duration_minutes', type: 'int(11)', defaultValue: '45', description: '时长(分钟)' },
                { name: 'lesson_type', type: 'enum', values: ['theory','practice','lab','discussion','assessment'], description: '课时类型' },
                { name: 'content_outline', type: 'json', description: '内容大纲' },
                { name: 'teaching_materials', type: 'json', description: '教学材料' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'chapters', foreignKey: 'chapter_id', description: '课时属于章节' }
            ]
        },

        class_schedules: {
            name: 'class_schedules',
            displayName: '班级课程安排',
            category: 'course',
            description: '班级课程时间安排',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '安排ID' },
                { name: 'course_id', type: 'bigint(20)', isForeignKey: true, references: 'courses.id', description: '课程ID' },
                { name: 'class_id', type: 'bigint(20)', isForeignKey: true, references: 'classes.id', description: '班级ID' },
                { name: 'teacher_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '教师ID' },
                { name: 'time_slot', type: 'varchar(50)', description: '时间段' },
                { name: 'classroom', type: 'varchar(100)', description: '教室' },
                { name: 'start_date', type: 'date', description: '开始日期' },
                { name: 'end_date', type: 'date', description: '结束日期' },
                { name: 'weekly_schedule', type: 'json', description: '周课程表' },
                { name: 'status', type: 'enum', values: ['active','cancelled','completed'], description: '状态' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'courses', foreignKey: 'course_id', description: '安排属于课程' },
                { type: 'belongsTo', table: 'classes', foreignKey: 'class_id', description: '安排属于班级' },
                { type: 'belongsTo', table: 'users', foreignKey: 'teacher_id', description: '安排属于教师' }
            ]
        },

        // 作业管理系统
        homeworks: {
            name: 'homeworks',
            displayName: '作业模板',
            category: 'homework',
            description: '作业基本信息和配置',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '作业ID' },
                { name: 'title', type: 'varchar(200)', description: '作业标题' },
                { name: 'description', type: 'text', description: '作业描述' },
                { name: 'subject', type: 'varchar(50)', description: '学科' },
                { name: 'grade', type: 'int(11)', description: '年级' },
                { name: 'difficulty_level', type: 'int(11)', description: '难度等级(1-5)' },
                { name: 'question_count', type: 'int(11)', description: '题目数量' },
                { name: 'max_score', type: 'int(11)', description: '总分' },
                { name: 'time_limit', type: 'int(11)', description: '时间限制(分钟)' },
                { name: 'due_date', type: 'datetime', description: '截止时间' },
                { name: 'start_date', type: 'datetime', description: '开始时间' },
                { name: 'is_published', type: 'tinyint(1)', defaultValue: '0', description: '是否发布' },
                { name: 'is_template', type: 'tinyint(1)', defaultValue: '0', description: '是否为模板' },
                { name: 'created_by', type: 'bigint(20)', description: '创建者ID' },
                { name: 'category', type: 'varchar(50)', description: '作业分类' },
                { name: 'tags', type: 'json', description: '标签列表' },
                { name: 'instructions', type: 'text', description: '作业说明' },
                { name: 'auto_grade', type: 'tinyint(1)', defaultValue: '1', description: '是否自动评分' },
                { name: 'max_attempts', type: 'int(11)', defaultValue: '1', description: '最大尝试次数' },
                { name: 'show_answers', type: 'tinyint(1)', defaultValue: '0', description: '是否显示答案' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'hasMany',
                    table: 'questions',
                    localKey: 'id',
                    foreignKey: 'homework_id',
                    description: '作业包含多个题目',
                    designReason: {
                        fieldPurpose: 'homework_id外键将题目归属到具体的作业',
                        businessLogic: '一份作业通常包含多道题目，每道题目都有独立的内容、分值和难度',
                        dataIntegrity: '确保题目与作业的强关联，删除作业时可以级联删除相关题目',
                        performanceBenefit: '便于按作业查询所有题目，支持作业内容的完整展示'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'homework_assignments',
                    localKey: 'id',
                    foreignKey: 'homework_id',
                    description: '作业有多个分配记录',
                    designReason: {
                        fieldPurpose: 'homework_id外键标识被分配的作业',
                        businessLogic: '同一份作业可以分配给不同的班级、学生或年级，每次分配都是独立记录',
                        dataIntegrity: '支持灵活的作业分发策略，避免重复创建相同内容的作业',
                        performanceBenefit: '便于统计作业的分配范围和完成情况'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'homework_submissions',
                    localKey: 'id',
                    foreignKey: 'homework_id',
                    description: '作业有多个学生提交记录',
                    designReason: {
                        fieldPurpose: 'homework_id外键标识提交的是哪份作业',
                        businessLogic: '每个学生对同一份作业可能有多次提交（如允许重做），需要独立记录',
                        dataIntegrity: '确保提交记录与作业内容匹配，支持成绩统计和分析',
                        performanceBenefit: '便于按作业查询所有提交情况，支持批量评分和统计分析'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'homework_progress',
                    localKey: 'id',
                    foreignKey: 'homework_id',
                    description: '作业有多个学生答题进度记录',
                    designReason: {
                        fieldPurpose: 'homework_id外键标识进度记录对应的作业',
                        businessLogic: '学生在答题过程中需要实时保存进度，支持断点续答',
                        dataIntegrity: '确保进度记录与作业内容同步，避免数据不一致',
                        performanceBenefit: '便于快速恢复学生的答题状态，提升用户体验'
                    }
                }
            ]
        },

        questions: {
            name: 'questions',
            displayName: '题目信息',
            category: 'homework',
            description: '作业题目详细信息',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '题目ID' },
                { name: 'homework_id', type: 'bigint(20)', isForeignKey: true, references: 'homeworks.id', description: '作业ID' },
                { name: 'content', type: 'text', description: '题目内容' },
                { name: 'question_type', type: 'enum', values: ['single_choice','multiple_choice','fill_blank','calculation','proof','application'], description: '题目类型' },
                { name: 'options', type: 'json', description: '选择题选项' },
                { name: 'correct_answer', type: 'text', description: '正确答案' },
                { name: 'score', type: 'decimal(5,2)', description: '题目分值' },
                { name: 'difficulty', type: 'int(11)', description: '难度等级(1-5)' },
                { name: 'order_index', type: 'int(11)', description: '题目顺序' },
                { name: 'knowledge_points', type: 'json', description: '关联知识点ID列表' },
                { name: 'explanation', type: 'text', description: '题目解析' },
                { name: 'created_at', type: 'timestamp', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'timestamp', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'homeworks',
                    foreignKey: 'homework_id',
                    description: '题目属于特定作业',
                    designReason: {
                        fieldPurpose: 'homework_id外键确保每道题目都归属于具体的作业',
                        businessLogic: '题目不能独立存在，必须作为作业的组成部分',
                        dataIntegrity: '防止孤立的题目记录，确保题目与作业内容的一致性',
                        performanceBenefit: '支持按作业快速查询所有题目，便于作业内容管理'
                    }
                }
            ]
        },

        homework_assignments: {
            name: 'homework_assignments',
            displayName: '作业分配',
            category: 'homework',
            description: '作业分配给学生或班级的记录',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '分配ID' },
                { name: 'homework_id', type: 'bigint(20)', isForeignKey: true, references: 'homeworks.id', description: '作业ID' },
                { name: 'assigned_to_type', type: 'enum', values: ['student','class','grade','all'], description: '分配目标类型' },
                { name: 'assigned_to_id', type: 'bigint(20)', description: '分配目标ID' },
                { name: 'assigned_by', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '分配者ID' },
                { name: 'assigned_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '分配时间' },
                { name: 'due_date_override', type: 'datetime', description: '截止时间覆盖' },
                { name: 'start_date_override', type: 'datetime', description: '开始时间覆盖' },
                { name: 'is_active', type: 'tinyint(1)', defaultValue: '1', description: '是否活跃' },
                { name: 'notes', type: 'text', description: '备注' },
                { name: 'max_attempts_override', type: 'int(11)', description: '最大尝试次数覆盖' },
                { name: 'notification_sent', type: 'tinyint(1)', defaultValue: '0', description: '是否已发送通知' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'homeworks', foreignKey: 'homework_id', description: '分配属于作业' },
                { type: 'belongsTo', table: 'users', foreignKey: 'assigned_by', description: '分配由教师执行' },
                { type: 'hasMany', table: 'homework_submissions', localKey: 'id', description: '分配有多个提交' }
            ]
        },

        homework_submissions: {
            name: 'homework_submissions',
            displayName: '作业提交',
            category: 'homework',
            description: '学生作业提交记录',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '提交ID' },
                { name: 'assignment_id', type: 'bigint(20)', isForeignKey: true, references: 'homework_assignments.id', description: '分配ID' },
                { name: 'student_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '学生ID' },
                { name: 'homework_id', type: 'bigint(20)', isForeignKey: true, references: 'homeworks.id', description: '作业ID' },
                { name: 'answers', type: 'json', description: '答案数据' },
                { name: 'submission_data', type: 'json', description: '提交数据' },
                { name: 'submitted_at', type: 'datetime', description: '提交时间' },
                { name: 'score', type: 'decimal(5,2)', description: '得分' },
                { name: 'max_score', type: 'decimal(5,2)', defaultValue: '100.00', description: '总分' },
                { name: 'status', type: 'enum', values: ['draft','submitted','graded','returned'], description: '状态' },
                { name: 'time_spent', type: 'int(11)', defaultValue: '0', description: '用时(秒)' },
                { name: 'attempt_count', type: 'int(11)', defaultValue: '1', description: '尝试次数' },
                { name: 'teacher_comments', type: 'text', description: '教师评语' },
                { name: 'auto_grade_data', type: 'json', description: '自动评分数据' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'homework_assignments',
                    foreignKey: 'assignment_id',
                    description: '提交记录关联到具体的作业分配',
                    designReason: {
                        fieldPurpose: 'assignment_id外键标识这次提交对应的作业分配记录',
                        businessLogic: '学生只能提交已分配给自己的作业，需要验证分配关系',
                        dataIntegrity: '确保提交的合法性，防止学生提交未分配的作业',
                        performanceBenefit: '便于按分配记录查询提交情况，支持班级作业统计'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'student_id',
                    description: '提交记录属于特定学生',
                    designReason: {
                        fieldPurpose: 'student_id外键标识提交作业的学生身份',
                        businessLogic: '每次作业提交都必须有明确的学生身份，用于成绩记录和学习分析',
                        dataIntegrity: '确保提交记录与学生账户绑定，支持个人成绩查询',
                        performanceBenefit: '便于按学生查询所有作业提交记录，支持学习进度跟踪'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'homeworks',
                    foreignKey: 'homework_id',
                    description: '提交记录对应特定作业',
                    designReason: {
                        fieldPurpose: 'homework_id外键标识提交的作业内容',
                        businessLogic: '提交记录需要关联到具体的作业，用于答案验证和评分',
                        dataIntegrity: '确保提交内容与作业要求匹配，支持自动评分系统',
                        performanceBenefit: '便于按作业查询所有提交记录，支持作业完成情况统计'
                    }
                }
            ]
        },

        homework_progress: {
            name: 'homework_progress',
            displayName: '作业进度',
            category: 'homework',
            description: '学生作业完成进度跟踪',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '进度ID' },
                { name: 'student_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '学生ID' },
                { name: 'homework_id', type: 'bigint(20)', isForeignKey: true, references: 'homeworks.id', description: '作业ID' },
                { name: 'progress_data', type: 'json', description: '进度数据' },
                { name: 'completion_rate', type: 'decimal(5,2)', defaultValue: '0.00', description: '完成率' },
                { name: 'last_saved_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '最后保存时间' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'student_id',
                    description: '答题进度属于特定学生',
                    designReason: {
                        fieldPurpose: 'student_id外键标识进度记录的所有者',
                        businessLogic: '每个学生的答题进度都是独立的，需要准确记录个人的答题状态',
                        dataIntegrity: '确保进度记录与学生身份绑定，支持个性化的断点续答功能',
                        performanceBenefit: '便于快速查询和恢复学生的答题状态，提升用户体验'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'homeworks',
                    foreignKey: 'homework_id',
                    description: '答题进度对应特定作业',
                    designReason: {
                        fieldPurpose: 'homework_id外键标识进度记录对应的作业内容',
                        businessLogic: '进度记录必须与具体的作业关联，确保答题内容的一致性',
                        dataIntegrity: '防止进度记录与作业内容不匹配，支持作业更新时的数据同步',
                        performanceBenefit: '便于按作业查询所有学生的答题进度，支持教师监控'
                    }
                }
            ]
        },

        homework_favorites: {
            name: 'homework_favorites',
            displayName: '作业收藏',
            category: 'homework',
            description: '学生收藏的作业',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '收藏ID' },
                { name: 'student_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '学生ID' },
                { name: 'assignment_id', type: 'bigint(20)', isForeignKey: true, references: 'homework_assignments.id', description: '分配ID' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '收藏时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'users', foreignKey: 'student_id', description: '收藏属于学生' },
                { type: 'belongsTo', table: 'homework_assignments', foreignKey: 'assignment_id', description: '收藏属于分配' }
            ]
        },

        homework_reminders: {
            name: 'homework_reminders',
            displayName: '作业提醒',
            category: 'homework',
            description: '作业截止提醒设置',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '提醒ID' },
                { name: 'student_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '学生ID' },
                { name: 'assignment_id', type: 'bigint(20)', isForeignKey: true, references: 'homework_assignments.id', description: '分配ID' },
                { name: 'reminder_type', type: 'enum', values: ['due_soon','overdue','custom'], description: '提醒类型' },
                { name: 'reminder_time', type: 'datetime', description: '提醒时间' },
                { name: 'message', type: 'text', description: '提醒消息' },
                { name: 'is_sent', type: 'tinyint(1)', defaultValue: '0', description: '是否已发送' },
                { name: 'sent_at', type: 'datetime', description: '发送时间' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'users', foreignKey: 'student_id', description: '提醒属于学生' },
                { type: 'belongsTo', table: 'homework_assignments', foreignKey: 'assignment_id', description: '提醒属于分配' }
            ]
        },

        // 学习路径系统
        learning_paths: {
            name: 'learning_paths',
            displayName: '个性化学习路径',
            category: 'learning',
            description: '学生个性化学习路径和进度',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '路径ID' },
                { name: 'student_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '学生ID' },
                { name: 'path_name', type: 'varchar(200)', description: '路径名称' },
                { name: 'path_structure', type: 'json', description: '路径结构' },
                { name: 'completion_rate', type: 'decimal(5,2)', defaultValue: '0.00', description: '完成率' },
                { name: 'target_completion', type: 'date', description: '目标完成日期' },
                { name: 'status', type: 'enum', values: ['active','paused','completed','cancelled'], description: '状态' },
                { name: 'milestones', type: 'json', description: '里程碑' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'student_id',
                    description: '学习路径属于特定学生',
                    designReason: {
                        fieldPurpose: 'student_id外键标识学习路径的所有者',
                        businessLogic: '每个学生都有个性化的学习路径，基于其学习能力、兴趣和目标定制',
                        dataIntegrity: '确保学习路径与学生身份绑定，支持个性化学习分析和进度跟踪',
                        performanceBenefit: '便于查询学生的学习路径，支持自适应学习算法和智能推荐'
                    }
                }
            ]
        },

        // 知识图谱系统
        knowledge_points: {
            name: 'knowledge_points',
            displayName: '知识点',
            category: 'knowledge',
            description: '知识点定义和认知层次管理',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '知识点ID' },
                { name: 'name', type: 'varchar(200)', description: '知识点名称' },
                { name: 'description', type: 'text', description: '详细描述' },
                { name: 'subject_id', type: 'bigint(20)', isForeignKey: true, references: 'subjects.id', description: '学科ID' },
                { name: 'grade_level', type: 'int(11)', description: '年级层次' },
                { name: 'difficulty_level', type: 'int(11)', description: '难度等级(1-5)' },
                { name: 'cognitive_type', type: 'enum', values: ['conceptual','procedural','metacognitive'], description: '认知类型' },
                { name: 'bloom_level', type: 'int(11)', description: '布鲁姆分类等级(1-6)' },
                { name: 'prerequisites', type: 'json', description: '前置知识点' },
                { name: 'learning_objectives', type: 'json', description: '学习目标' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'subjects',
                    foreignKey: 'subject_id',
                    description: '知识点属于特定学科',
                    designReason: {
                        fieldPurpose: 'subject_id外键标识知识点所属的学科领域',
                        businessLogic: '知识点必须归属于具体学科，如数学中的函数、几何等',
                        dataIntegrity: '确保知识点与学科的正确分类，支持学科知识体系构建',
                        performanceBenefit: '便于按学科查询知识点，支持学科级别的知识管理'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'knowledge_relationships',
                    localKey: 'id',
                    foreignKey: 'source_point_id',
                    description: '知识点作为源点有多个关系',
                    designReason: {
                        fieldPurpose: 'source_point_id外键标识关系的起始知识点',
                        businessLogic: '知识点之间存在前置、包含、相关等多种关系，构建知识图谱',
                        dataIntegrity: '建立完整的知识关系网络，支持学习路径规划',
                        performanceBenefit: '便于查询知识点的关联关系，支持智能推荐和学习路径生成'
                    }
                },
                {
                    type: 'hasMany',
                    table: 'mastery_tracking',
                    localKey: 'id',
                    foreignKey: 'knowledge_point_id',
                    description: '知识点有多个掌握度跟踪记录',
                    designReason: {
                        fieldPurpose: 'knowledge_point_id外键标识被跟踪的知识点',
                        businessLogic: '需要跟踪每个学生对各知识点的掌握程度，支持个性化学习',
                        dataIntegrity: '确保掌握度记录与知识点的正确关联，支持学习分析',
                        performanceBenefit: '便于查询知识点的掌握情况，支持学习效果评估'
                    }
                }
            ]
        },

        knowledge_relationships: {
            name: 'knowledge_relationships',
            displayName: '知识点关系',
            category: 'knowledge',
            description: '知识点之间的语义关系和依赖关系',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '关系ID' },
                { name: 'source_point_id', type: 'bigint(20)', isForeignKey: true, references: 'knowledge_points.id', description: '源知识点ID' },
                { name: 'target_point_id', type: 'bigint(20)', isForeignKey: true, references: 'knowledge_points.id', description: '目标知识点ID' },
                { name: 'relationship_type', type: 'enum', values: ['prerequisite','related','extends','applies_to','contradicts'], description: '关系类型' },
                { name: 'strength', type: 'decimal(3,2)', description: '关系强度(0-1)' },
                { name: 'confidence', type: 'decimal(3,2)', description: '置信度(0-1)' },
                { name: 'evidence_count', type: 'int(11)', defaultValue: '0', description: '证据数量' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'knowledge_points', foreignKey: 'source_point_id', description: '源知识点' },
                { type: 'belongsTo', table: 'knowledge_points', foreignKey: 'target_point_id', description: '目标知识点' }
            ]
        },

        concept_maps: {
            name: 'concept_maps',
            displayName: '概念图',
            category: 'knowledge',
            description: '概念图和思维导图的可视化表示',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '概念图ID' },
                { name: 'title', type: 'varchar(200)', description: '概念图标题' },
                { name: 'description', type: 'text', description: '描述' },
                { name: 'subject_id', type: 'bigint(20)', isForeignKey: true, references: 'subjects.id', description: '学科ID' },
                { name: 'grade_level', type: 'int(11)', description: '年级层次' },
                { name: 'map_data', type: 'json', description: '图结构数据(节点和边)' },
                { name: 'layout_config', type: 'json', description: '布局配置' },
                { name: 'created_by', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '创建者ID' },
                { name: 'is_public', type: 'tinyint(1)', defaultValue: '0', description: '是否公开' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                { 
                    type: 'belongsTo', 
                    table: 'subjects', 
                    foreignKey: 'subject_id', 
                    description: '记录属于特定学科',
                    designReason: {
                        fieldPurpose: 'subject_id外键标识记录所属的学科领域',
                        businessLogic: '需要按学科分类管理，支持学科级别的操作',
                        dataIntegrity: '确保记录与学科的正确分类，支持学科管理',
                        performanceBenefit: '便于按学科查询记录，支持学科级别的统计'
                    }
                },
                { type: 'belongsTo', table: 'users', foreignKey: 'created_by', description: '创建者' }
            ]
        },

        // 智能推荐系统
        symbol_recommendations: {
            name: 'symbol_recommendations',
            displayName: '符号推荐',
            category: 'ai_recommendation',
            description: '数学符号智能推荐和使用统计',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '推荐ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'context', type: 'text', description: '输入上下文' },
                { name: 'recommended_symbols', type: 'json', description: '推荐的符号列表' },
                { name: 'selected_symbol', type: 'varchar(50)', description: '用户选择的符号' },
                { name: 'usage_frequency', type: 'int(11)', defaultValue: '0', description: '使用频率' },
                { name: 'success_rate', type: 'decimal(5,2)', description: '推荐成功率' },
                { name: 'response_time', type: 'int(11)', description: '响应时间(毫秒)' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '符号推荐记录属于特定用户',
                    designReason: {
                        fieldPurpose: 'user_id外键标识接收推荐的用户',
                        businessLogic: '符号推荐需要基于用户的输入习惯和使用历史进行个性化推荐',
                        dataIntegrity: '确保推荐记录与用户身份绑定，支持个性化学习分析',
                        performanceBenefit: '便于查询用户的推荐历史，支持推荐算法优化和效果评估'
                    }
                }
            ]
        },

        problem_recommendations: {
            name: 'problem_recommendations',
            displayName: '题目推荐',
            category: 'ai_recommendation',
            description: '个性化题目推荐和难度适配',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '推荐ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'question_id', type: 'bigint(20)', isForeignKey: true, references: 'questions.id', description: '题目ID' },
                { name: 'recommendation_reason', type: 'text', description: '推荐理由' },
                { name: 'difficulty_match', type: 'decimal(3,2)', description: '难度匹配度(0-1)' },
                { name: 'knowledge_gap_target', type: 'json', description: '目标知识缺口' },
                { name: 'predicted_success_rate', type: 'decimal(3,2)', description: '预测成功率' },
                { name: 'actual_result', type: 'enum', values: ['correct','incorrect','skipped','timeout'], description: '实际结果' },
                { name: 'user_feedback', type: 'int(11)', description: '用户反馈评分(1-5)' },
                { name: 'recommended_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '推荐时间' },
                { name: 'completed_at', type: 'datetime', description: '完成时间' }
            ],
            relationships: [
                { 
                    type: 'belongsTo', 
                    table: 'users', 
                    foreignKey: 'user_id', 
                    description: '记录属于特定用户',
                    designReason: {
                        fieldPurpose: 'user_id外键标识记录所属的用户',
                        businessLogic: '需要将数据与具体用户关联，支持个性化分析',
                        dataIntegrity: '确保数据与用户身份绑定，保护用户隐私',
                        performanceBenefit: '便于按用户查询相关数据，支持个性化服务'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'questions',
                    foreignKey: 'question_id',
                    description: '推荐记录关联到具体题目',
                    designReason: {
                        fieldPurpose: 'question_id外键标识被推荐的具体题目',
                        businessLogic: 'AI推荐系统需要基于学生的知识掌握情况推荐合适的题目，实现个性化学习',
                        dataIntegrity: '确保推荐记录与题目内容匹配，支持推荐效果分析和算法优化',
                        performanceBenefit: '便于分析题目的推荐效果，支持智能推荐算法的持续改进'
                    }
                }
            ]
        },

        learning_path_recommendations: {
            name: 'learning_path_recommendations',
            displayName: '学习路径推荐',
            category: 'ai_recommendation',
            description: '自适应学习路径推荐和优化',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '推荐ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'current_knowledge_state', type: 'json', description: '当前知识状态' },
                { name: 'recommended_path', type: 'json', description: '推荐路径' },
                { name: 'path_type', type: 'enum', values: ['remedial','advancement','review','exploration'], description: '路径类型' },
                { name: 'estimated_duration', type: 'int(11)', description: '预计时长(分钟)' },
                { name: 'success_prediction', type: 'decimal(3,2)', description: '成功预测概率' },
                { name: 'adaptation_triggers', type: 'json', description: '适应触发条件' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '学习路径推荐属于特定学生',
                    designReason: {
                        fieldPurpose: 'user_id外键标识接收路径推荐的学生',
                        businessLogic: 'AI系统基于学生的知识状态、学习习惯和目标生成个性化学习路径推荐',
                        dataIntegrity: '确保推荐路径与学生身份绑定，支持自适应学习算法的持续优化',
                        performanceBenefit: '便于查询学生的路径推荐历史，支持学习效果分析和路径调整'
                    }
                }
            ]
        },

        // 学习分析系统
        learning_behaviors: {
            name: 'learning_behaviors',
            displayName: '学习行为',
            category: 'learning_analytics',
            description: '学习行为数据采集和分析',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '行为ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'session_id', type: 'varchar(100)', description: '会话ID' },
                { name: 'behavior_type', type: 'enum', values: ['click','hover','input','submit','pause','review','help_seek'], description: '行为类型' },
                { name: 'behavior_data', type: 'json', description: '行为详细数据' },
                { name: 'context_info', type: 'json', description: '上下文信息' },
                { name: 'timestamp', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '时间戳' },
                { name: 'duration', type: 'int(11)', description: '持续时间(毫秒)' },
                { name: 'device_info', type: 'json', description: '设备信息' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '学习行为记录属于特定学生',
                    designReason: {
                        fieldPurpose: 'user_id外键标识产生学习行为的学生',
                        businessLogic: '系统需要采集和分析学生的学习行为模式，为智能推荐和个性化教学提供数据支持',
                        dataIntegrity: '确保行为数据与学生身份绑定，保护学生隐私并支持行为分析的准确性',
                        performanceBenefit: '便于分析学生的学习习惯和行为模式，支持学习效果预测和干预策略'
                    }
                }
            ]
        },

        interaction_logs: {
            name: 'interaction_logs',
            displayName: '交互日志',
            category: 'learning_analytics',
            description: '详细的用户交互日志和响应分析',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '日志ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'homework_id', type: 'bigint(20)', isForeignKey: true, references: 'homeworks.id', description: '作业ID' },
                { name: 'question_id', type: 'bigint(20)', isForeignKey: true, references: 'questions.id', description: '题目ID' },
                { name: 'interaction_type', type: 'enum', values: ['view','attempt','submit','hint','skip','review'], description: '交互类型' },
                { name: 'interaction_data', type: 'json', description: '交互数据' },
                { name: 'response_time', type: 'int(11)', description: '响应时间(毫秒)' },
                { name: 'accuracy', type: 'decimal(3,2)', description: '准确率' },
                { name: 'hint_used', type: 'tinyint(1)', defaultValue: '0', description: '是否使用提示' },
                { name: 'attempts_count', type: 'int(11)', defaultValue: '1', description: '尝试次数' },
                { name: 'final_answer', type: 'text', description: '最终答案' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '交互日志属于特定学生',
                    designReason: {
                        fieldPurpose: 'user_id外键标识产生交互行为的学生',
                        businessLogic: '系统需要详细记录学生与题目的交互过程，分析学习模式和解题策略',
                        dataIntegrity: '确保交互数据与学生身份绑定，支持个性化学习分析和行为建模',
                        performanceBenefit: '便于分析学生的解题过程和思维模式，支持智能辅导和错误诊断'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'homeworks',
                    foreignKey: 'homework_id',
                    description: '交互日志关联到特定作业',
                    designReason: {
                        fieldPurpose: 'homework_id外键标识交互发生的作业上下文',
                        businessLogic: '需要在作业维度分析学生的交互行为，评估作业设计的有效性',
                        dataIntegrity: '确保交互记录与作业内容匹配，支持作业完成情况的准确统计',
                        performanceBenefit: '便于按作业分析学生表现，支持作业难度评估和优化'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'questions',
                    foreignKey: 'question_id',
                    description: '交互日志关联到具体题目',
                    designReason: {
                        fieldPurpose: 'question_id外键标识交互涉及的具体题目',
                        businessLogic: '需要在题目级别分析学生的解题过程，识别题目的难点和学生的薄弱环节',
                        dataIntegrity: '确保交互数据与题目内容匹配，支持题目质量评估和改进',
                        performanceBenefit: '便于分析题目的解答情况，支持题目推荐和个性化学习路径'
                    }
                }
            ]
        },

        engagement_metrics: {
            name: 'engagement_metrics',
            displayName: '参与度指标',
            category: 'learning_analytics',
            description: '学习参与度和专注度分析',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '指标ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'date', type: 'date', description: '日期' },
                { name: 'session_duration', type: 'int(11)', description: '会话时长(分钟)' },
                { name: 'questions_attempted', type: 'int(11)', description: '尝试题目数' },
                { name: 'completion_rate', type: 'decimal(5,2)', description: '完成率' },
                { name: 'focus_score', type: 'decimal(3,2)', description: '专注度评分(0-1)' },
                { name: 'persistence_score', type: 'decimal(3,2)', description: '坚持度评分(0-1)' },
                { name: 'help_seeking_frequency', type: 'decimal(5,2)', description: '求助频率' },
                { name: 'self_regulation_score', type: 'decimal(3,2)', description: '自我调节评分(0-1)' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' }
            ],
            relationships: [
                { 
                    type: 'belongsTo', 
                    table: 'users', 
                    foreignKey: 'user_id', 
                    description: '记录属于特定用户',
                    designReason: {
                        fieldPurpose: 'user_id外键标识记录所属的用户',
                        businessLogic: '需要将数据与具体用户关联，支持个性化分析',
                        dataIntegrity: '确保数据与用户身份绑定，保护用户隐私',
                        performanceBenefit: '便于按用户查询相关数据，支持个性化服务'
                    }
                }
            ]
        },

        // 错误分析系统
        error_patterns: {
            name: 'error_patterns',
            displayName: '错误模式',
            category: 'error_analysis',
            description: '常见错误模式识别和分析',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '错误模式ID' },
                { name: 'question_id', type: 'bigint(20)', isForeignKey: true, references: 'questions.id', description: '题目ID' },
                { name: 'error_type', type: 'enum', values: ['conceptual','procedural','computational','careless'], description: '错误类型' },
                { name: 'error_description', type: 'text', description: '错误描述' },
                { name: 'frequency', type: 'int(11)', defaultValue: '0', description: '出现频率' },
                { name: 'common_misconceptions', type: 'json', description: '常见误解' },
                { name: 'difficulty_indicators', type: 'json', description: '难点指标' },
                { name: 'remediation_strategies', type: 'json', description: '补救策略' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                { type: 'belongsTo', table: 'questions', foreignKey: 'question_id', description: '属于题目' },
                { type: 'hasMany', table: 'misconception_analysis', description: '误解分析' }
            ]
        },

        misconception_analysis: {
            name: 'misconception_analysis',
            displayName: '误解分析',
            category: 'error_analysis',
            description: '学生误解检测和干预建议',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '分析ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'knowledge_point_id', type: 'bigint(20)', isForeignKey: true, references: 'knowledge_points.id', description: '知识点ID' },
                { name: 'error_pattern_id', type: 'bigint(20)', isForeignKey: true, references: 'error_patterns.id', description: '错误模式ID' },
                { name: 'misconception_type', type: 'varchar(100)', description: '误解类型' },
                { name: 'evidence_data', type: 'json', description: '证据数据' },
                { name: 'confidence_level', type: 'decimal(3,2)', description: '置信度(0-1)' },
                { name: 'detected_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '检测时间' },
                { name: 'intervention_suggested', type: 'json', description: '建议干预措施' },
                { name: 'resolved_at', type: 'datetime', description: '解决时间' }
            ],
            relationships: [
                { 
                    type: 'belongsTo', 
                    table: 'users', 
                    foreignKey: 'user_id', 
                    description: '记录属于特定用户',
                    designReason: {
                        fieldPurpose: 'user_id外键标识记录所属的用户',
                        businessLogic: '需要将数据与具体用户关联，支持个性化分析',
                        dataIntegrity: '确保数据与用户身份绑定，保护用户隐私',
                        performanceBenefit: '便于按用户查询相关数据，支持个性化服务'
                    }
                },
                { type: 'belongsTo', table: 'knowledge_points', foreignKey: 'knowledge_point_id', description: '相关知识点' },
                { type: 'belongsTo', table: 'error_patterns', foreignKey: 'error_pattern_id', description: '错误模式' }
            ]
        },

        // 自适应学习系统
        adaptive_paths: {
            name: 'adaptive_paths',
            displayName: '自适应路径',
            category: 'adaptive_learning',
            description: '动态调整的个性化学习路径',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '路径ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'current_state', type: 'json', description: '当前学习状态' },
                { name: 'target_state', type: 'json', description: '目标状态' },
                { name: 'path_steps', type: 'json', description: '路径步骤' },
                { name: 'adaptation_reason', type: 'text', description: '适应原因' },
                { name: 'difficulty_adjustment', type: 'decimal(3,2)', description: '难度调整系数' },
                { name: 'estimated_completion', type: 'int(11)', description: '预计完成时间(分钟)' },
                { name: 'success_rate', type: 'decimal(3,2)', description: '成功率预测' },
                { name: 'last_updated', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '最后更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '自适应路径属于特定学生',
                    designReason: {
                        fieldPurpose: 'user_id外键标识自适应路径的所有者',
                        businessLogic: '系统根据学生的实时学习表现动态调整学习路径，实现真正的个性化教育',
                        dataIntegrity: '确保自适应路径与学生身份绑定，支持学习轨迹的连续性和一致性',
                        performanceBenefit: '便于实时查询和更新学生的学习路径，支持智能化的学习指导'
                    }
                }
            ]
        },

        mastery_tracking: {
            name: 'mastery_tracking',
            displayName: '掌握度跟踪',
            category: 'adaptive_learning',
            description: '知识点掌握度实时跟踪和预测',
            fields: [
                { name: 'id', type: 'bigint(20)', isPrimary: true, description: '跟踪ID' },
                { name: 'user_id', type: 'bigint(20)', isForeignKey: true, references: 'users.id', description: '用户ID' },
                { name: 'knowledge_point_id', type: 'bigint(20)', isForeignKey: true, references: 'knowledge_points.id', description: '知识点ID' },
                { name: 'mastery_level', type: 'decimal(3,2)', description: '掌握度(0-1)' },
                { name: 'confidence_interval', type: 'json', description: '置信区间' },
                { name: 'evidence_count', type: 'int(11)', defaultValue: '0', description: '证据数量' },
                { name: 'last_assessment', type: 'datetime', description: '最后评估时间' },
                { name: 'decay_rate', type: 'decimal(5,4)', description: '遗忘衰减率' },
                { name: 'next_review_due', type: 'datetime', description: '下次复习时间' },
                { name: 'mastery_achieved_at', type: 'datetime', description: '掌握达成时间' },
                { name: 'created_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP', description: '创建时间' },
                { name: 'updated_at', type: 'datetime', defaultValue: 'CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP', description: '更新时间' }
            ],
            relationships: [
                {
                    type: 'belongsTo',
                    table: 'users',
                    foreignKey: 'user_id',
                    description: '掌握度跟踪属于特定学生',
                    designReason: {
                        fieldPurpose: 'user_id外键标识被跟踪学生的身份',
                        businessLogic: '系统需要实时跟踪每个学生对各知识点的掌握程度，支持个性化学习分析',
                        dataIntegrity: '确保掌握度数据与学生身份绑定，支持学习轨迹的连续性跟踪',
                        performanceBenefit: '便于查询学生的知识掌握状况，支持自适应学习算法和智能推荐'
                    }
                },
                {
                    type: 'belongsTo',
                    table: 'knowledge_points',
                    foreignKey: 'knowledge_point_id',
                    description: '掌握度跟踪对应特定知识点',
                    designReason: {
                        fieldPurpose: 'knowledge_point_id外键标识被跟踪的具体知识点',
                        businessLogic: '需要精确跟踪学生对每个知识点的掌握程度，构建完整的知识掌握图谱',
                        dataIntegrity: '确保掌握度记录与知识点内容匹配，支持知识体系的完整性',
                        performanceBenefit: '便于按知识点分析掌握情况，支持知识点难度评估和学习路径优化'
                    }
                }
            ]
        }
    },

    // 关系定义
    relationships: [
        // 用户权限系统内部关系
        { from: 'users', to: 'user_sessions', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'notifications', type: 'one-to-many', label: '1:N' },

        // 学校组织架构关系
        { from: 'schools', to: 'grades', type: 'one-to-many', label: '1:N' },
        { from: 'grades', to: 'classes', type: 'one-to-many', label: '1:N' },
        { from: 'classes', to: 'class_students', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'class_students', type: 'one-to-many', label: '1:N' },
        { from: 'schools', to: 'curriculum_standards', type: 'one-to-many', label: '1:N' },
        { from: 'curriculum_standards', to: 'subjects', type: 'one-to-many', label: '1:N' },

        // 课程体系关系
        { from: 'subjects', to: 'courses', type: 'one-to-many', label: '1:N' },
        { from: 'courses', to: 'course_modules', type: 'one-to-many', label: '1:N' },
        { from: 'course_modules', to: 'chapters', type: 'one-to-many', label: '1:N' },
        { from: 'chapters', to: 'lessons', type: 'one-to-many', label: '1:N' },
        { from: 'courses', to: 'class_schedules', type: 'one-to-many', label: '1:N' },
        { from: 'classes', to: 'class_schedules', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'class_schedules', type: 'one-to-many', label: '1:N' },

        // 作业管理系统关系
        { from: 'homeworks', to: 'questions', type: 'one-to-many', label: '1:N' },
        { from: 'homeworks', to: 'homework_assignments', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'homework_assignments', type: 'one-to-many', label: '1:N' },
        { from: 'homework_assignments', to: 'homework_submissions', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'homework_submissions', type: 'one-to-many', label: '1:N' },
        { from: 'homeworks', to: 'homework_submissions', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'homework_progress', type: 'one-to-many', label: '1:N' },
        { from: 'homeworks', to: 'homework_progress', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'homework_favorites', type: 'one-to-many', label: '1:N' },
        { from: 'homework_assignments', to: 'homework_favorites', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'homework_reminders', type: 'one-to-many', label: '1:N' },
        { from: 'homework_assignments', to: 'homework_reminders', type: 'one-to-many', label: '1:N' },

        // 学习路径关系
        { from: 'users', to: 'learning_paths', type: 'one-to-many', label: '1:N' },

        // 知识图谱系统关系
        { from: 'subjects', to: 'knowledge_points', type: 'one-to-many', label: '1:N' },
        { from: 'knowledge_points', to: 'knowledge_relationships', type: 'one-to-many', label: '1:N' },
        { from: 'subjects', to: 'concept_maps', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'concept_maps', type: 'one-to-many', label: '1:N' },

        // 智能推荐系统关系
        { from: 'users', to: 'symbol_recommendations', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'problem_recommendations', type: 'one-to-many', label: '1:N' },
        { from: 'questions', to: 'problem_recommendations', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'learning_path_recommendations', type: 'one-to-many', label: '1:N' },

        // 学习分析系统关系
        { from: 'users', to: 'learning_behaviors', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'interaction_logs', type: 'one-to-many', label: '1:N' },
        { from: 'homeworks', to: 'interaction_logs', type: 'one-to-many', label: '1:N' },
        { from: 'questions', to: 'interaction_logs', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'engagement_metrics', type: 'one-to-many', label: '1:N' },

        // 错误分析系统关系
        { from: 'questions', to: 'error_patterns', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'misconception_analysis', type: 'one-to-many', label: '1:N' },
        { from: 'knowledge_points', to: 'misconception_analysis', type: 'one-to-many', label: '1:N' },
        { from: 'error_patterns', to: 'misconception_analysis', type: 'one-to-many', label: '1:N' },

        // 自适应学习系统关系
        { from: 'users', to: 'adaptive_paths', type: 'one-to-many', label: '1:N' },
        { from: 'users', to: 'mastery_tracking', type: 'one-to-many', label: '1:N' },
        { from: 'knowledge_points', to: 'mastery_tracking', type: 'one-to-many', label: '1:N' }
    ],

    // 模块分类
    modules: {
        user: {
            name: '用户权限系统',
            color: '#ff6b6b',
            description: '管理用户认证、授权和会话',
            tables: ['users', 'user_sessions', 'notifications']
        },
        school: {
            name: '学校组织架构',
            color: '#4ecdc4',
            description: '管理学校、年级、班级等组织结构',
            tables: ['schools', 'grades', 'classes', 'class_students', 'curriculum_standards', 'subjects']
        },
        course: {
            name: '课程体系',
            color: '#45b7d1',
            description: '管理课程、模块、章节和课时',
            tables: ['courses', 'course_modules', 'chapters', 'lessons', 'class_schedules']
        },
        homework: {
            name: '作业管理系统',
            color: '#96ceb4',
            description: '管理作业创建、分发、提交和评分',
            tables: ['homeworks', 'questions', 'homework_assignments', 'homework_submissions', 'homework_progress', 'homework_favorites', 'homework_reminders']
        },
        learning: {
            name: '学习路径系统',
            color: '#feca57',
            description: '管理个性化学习路径',
            tables: ['learning_paths']
        },
        knowledge: {
            name: '知识图谱系统',
            color: '#a29bfe',
            description: '知识点关系和概念图管理',
            tables: ['knowledge_points', 'knowledge_relationships', 'concept_maps']
        },
        ai_recommendation: {
            name: 'AI智能推荐',
            color: '#fd79a8',
            description: '智能符号推荐和个性化题目推荐',
            tables: ['symbol_recommendations', 'problem_recommendations', 'learning_path_recommendations']
        },
        learning_analytics: {
            name: '学习分析系统',
            color: '#00b894',
            description: '学习行为分析和参与度监测',
            tables: ['learning_behaviors', 'interaction_logs', 'engagement_metrics']
        },
        error_analysis: {
            name: '错误分析系统',
            color: '#e17055',
            description: '错误模式识别和误解分析',
            tables: ['error_patterns', 'misconception_analysis']
        },
        adaptive_learning: {
            name: '自适应学习',
            color: '#6c5ce7',
            description: '自适应路径和掌握度跟踪',
            tables: ['adaptive_paths', 'mastery_tracking']
        }
    }
};

// 导出数据
if (typeof module !== 'undefined' && module.exports) {
    module.exports = databaseSchema;
}
