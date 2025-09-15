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
                { type: 'hasMany', table: 'user_sessions', localKey: 'id', description: '用户有多个会话' },
                { type: 'hasMany', table: 'notifications', localKey: 'id', description: '用户有多个通知' },
                { type: 'hasMany', table: 'homework_submissions', localKey: 'id', description: '用户有多个作业提交' },
                { type: 'hasMany', table: 'homework_progress', localKey: 'id', description: '用户有多个答题进度' },
                { type: 'hasMany', table: 'class_students', localKey: 'id', description: '用户可以是多个班级的学生' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '会话属于用户' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '通知属于用户' }
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
                { type: 'belongsTo', table: 'schools', foreignKey: 'school_id', description: '年级属于学校' },
                { type: 'hasMany', table: 'classes', localKey: 'id', description: '年级有多个班级' }
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
                { type: 'belongsTo', table: 'schools', foreignKey: 'school_id', description: '班级属于学校' },
                { type: 'belongsTo', table: 'grades', foreignKey: 'grade_id', description: '班级属于年级' },
                { type: 'belongsTo', table: 'users', foreignKey: 'head_teacher_id', description: '班级有班主任' },
                { type: 'hasMany', table: 'class_students', localKey: 'id', description: '班级有多个学生关联' }
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
                { type: 'belongsTo', table: 'classes', foreignKey: 'class_id', description: '关联属于班级' },
                { type: 'belongsTo', table: 'users', foreignKey: 'student_id', description: '关联属于学生' }
            ]
        },

        curriculum_standards: {
            name: 'curriculum_standards',
            displayName: '课程标准',
            category: 'course',
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
            category: 'course',
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
                { type: 'hasMany', table: 'questions', localKey: 'id', description: '作业有多个题目' },
                { type: 'hasMany', table: 'homework_assignments', localKey: 'id', description: '作业有多个分配记录' },
                { type: 'hasMany', table: 'homework_submissions', localKey: 'id', description: '作业有多个提交记录' },
                { type: 'hasMany', table: 'homework_progress', localKey: 'id', description: '作业有多个进度记录' }
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
                { type: 'belongsTo', table: 'homeworks', foreignKey: 'homework_id', description: '题目属于作业' }
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
                { type: 'belongsTo', table: 'homework_assignments', foreignKey: 'assignment_id', description: '提交属于分配' },
                { type: 'belongsTo', table: 'users', foreignKey: 'student_id', description: '提交属于学生' },
                { type: 'belongsTo', table: 'homeworks', foreignKey: 'homework_id', description: '提交属于作业' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'student_id', description: '进度属于学生' },
                { type: 'belongsTo', table: 'homeworks', foreignKey: 'homework_id', description: '进度属于作业' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'student_id', description: '路径属于学生' }
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
                { type: 'belongsTo', table: 'subjects', foreignKey: 'subject_id', description: '属于学科' },
                { type: 'hasMany', table: 'knowledge_relationships', description: '知识点关系' },
                { type: 'hasMany', table: 'mastery_tracking', description: '掌握度跟踪' }
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
                { type: 'belongsTo', table: 'subjects', foreignKey: 'subject_id', description: '属于学科' },
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' },
                { type: 'belongsTo', table: 'questions', foreignKey: 'question_id', description: '推荐题目' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' },
                { type: 'belongsTo', table: 'homeworks', foreignKey: 'homework_id', description: '属于作业' },
                { type: 'belongsTo', table: 'questions', foreignKey: 'question_id', description: '属于题目' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' },
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' }
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
                { type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' },
                { type: 'belongsTo', table: 'knowledge_points', foreignKey: 'knowledge_point_id', description: '相关知识点' }
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
