-- K-12数学教育智能数字生态系统 - 实际数据库结构
-- 基于当前运行系统的真实数据库结构
-- 数据库：testccnu
-- 主机：obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud:3306
-- 更新日期：2024年

-- =============================================================================
-- 基础配置
-- =============================================================================

-- 设置字符集
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- 使用数据库
USE testccnu;

-- =============================================================================
-- 学校与组织架构模块
-- =============================================================================

-- 学校表
DROP TABLE IF EXISTS `schools`;
CREATE TABLE `schools` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学校ID',
  `school_name` varchar(200) NOT NULL COMMENT '学校名称',
  `school_code` varchar(50) NOT NULL UNIQUE COMMENT '学校代码',
  `school_type` enum('primary','middle','high','mixed') NOT NULL COMMENT '学校类型',
  `education_level` varchar(50) DEFAULT NULL COMMENT '教育层次',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_school_code` (`school_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学校基础信息表';

-- 年级表
DROP TABLE IF EXISTS `grades`;
CREATE TABLE `grades` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '年级ID',
  `school_id` bigint(20) NOT NULL COMMENT '学校ID',
  `grade_name` varchar(50) NOT NULL COMMENT '年级名称',
  `grade_level` int(11) NOT NULL COMMENT '年级数字',
  `academic_year` varchar(20) NOT NULL COMMENT '学年',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_school_id` (`school_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='年级信息表';

-- 班级表
DROP TABLE IF EXISTS `classes`;
CREATE TABLE `classes` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '班级ID',
  `school_id` bigint(20) NOT NULL COMMENT '学校ID',
  `grade_id` bigint(20) NOT NULL COMMENT '年级ID',
  `class_name` varchar(100) NOT NULL COMMENT '班级名称',
  `class_code` varchar(50) NOT NULL COMMENT '班级代码',
  `head_teacher_id` bigint(20) DEFAULT NULL COMMENT '班主任ID',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_school_id` (`school_id`),
  KEY `idx_grade_id` (`grade_id`),
  KEY `idx_head_teacher_id` (`head_teacher_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='班级信息表';

-- 班级学生关联表
DROP TABLE IF EXISTS `class_students`;
CREATE TABLE `class_students` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '关联ID',
  `class_id` bigint(20) NOT NULL COMMENT '班级ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `student_number` varchar(50) DEFAULT NULL COMMENT '班级内学号',
  `joined_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '加入时间',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否活跃',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_class_student` (`class_id`,`student_id`),
  KEY `idx_class_id` (`class_id`),
  KEY `idx_student_id` (`student_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='班级学生关联表';

-- =============================================================================
-- 用户与认证模块
-- =============================================================================

-- 用户表 (基于实际使用的字段)
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(50) NOT NULL UNIQUE COMMENT '用户名',
  `email` varchar(100) NOT NULL UNIQUE COMMENT '邮箱',
  `password_hash` varchar(255) NOT NULL COMMENT '密码哈希',
  `role` enum('student','teacher','admin','parent') NOT NULL DEFAULT 'student' COMMENT '用户角色',
  `real_name` varchar(50) DEFAULT NULL COMMENT '真实姓名',
  `grade` int(11) DEFAULT NULL COMMENT '年级',
  `school` varchar(100) DEFAULT NULL COMMENT '学校',
  `class_name` varchar(50) DEFAULT NULL COMMENT '班级',
  `student_id` varchar(20) DEFAULT NULL COMMENT '学号',
  `phone` varchar(20) DEFAULT NULL COMMENT '手机号',
  `avatar` varchar(255) DEFAULT NULL COMMENT '头像URL',
  `profile` json DEFAULT NULL COMMENT '用户配置信息',
  `learning_preferences` json DEFAULT NULL COMMENT '学习偏好设置',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否激活',
  `last_login_time` datetime DEFAULT NULL COMMENT '最后登录时间',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`),
  KEY `idx_role` (`role`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户基础信息表';

-- 用户会话表
DROP TABLE IF EXISTS `user_sessions`;
CREATE TABLE `user_sessions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '会话ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `session_token` text NOT NULL COMMENT '会话令牌',
  `device_type` varchar(20) DEFAULT NULL COMMENT '设备类型',
  `device_id` varchar(100) DEFAULT NULL COMMENT '设备ID',
  `ip_address` varchar(45) DEFAULT NULL COMMENT 'IP地址',
  `user_agent` text DEFAULT NULL COMMENT '用户代理',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否活跃',
  `expires_at` datetime NOT NULL COMMENT '过期时间',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_expires_at` (`expires_at`),
  KEY `idx_session_token` (`session_token`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户会话表';

-- 通知表
DROP TABLE IF EXISTS `notifications`;
CREATE TABLE `notifications` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '通知ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `notification_type` varchar(50) NOT NULL COMMENT '通知类型',
  `title` varchar(200) NOT NULL COMMENT '通知标题',
  `content` text NOT NULL COMMENT '通知内容',
  `related_type` varchar(50) DEFAULT NULL COMMENT '关联类型',
  `related_id` bigint(20) DEFAULT NULL COMMENT '关联ID',
  `is_read` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否已读',
  `priority` enum('low','normal','high','urgent') NOT NULL DEFAULT 'normal' COMMENT '优先级',
  `expires_at` datetime DEFAULT NULL COMMENT '过期时间',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_notification_type` (`notification_type`),
  KEY `idx_is_read` (`is_read`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='通知消息表';

-- =============================================================================
-- 课程体系模块
-- =============================================================================

-- 课程标准表
DROP TABLE IF EXISTS `curriculum_standards`;
CREATE TABLE `curriculum_standards` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '标准ID',
  `school_id` bigint(20) NOT NULL COMMENT '学校ID',
  `standard_name` varchar(200) NOT NULL COMMENT '标准名称',
  `grade_range` varchar(20) NOT NULL COMMENT '适用年级范围',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_school_id` (`school_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课程标准表';

-- 学科表
DROP TABLE IF EXISTS `subjects`;
CREATE TABLE `subjects` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学科ID',
  `standard_id` bigint(20) NOT NULL COMMENT '课程标准ID',
  `subject_name` varchar(100) NOT NULL COMMENT '学科名称',
  `subject_code` varchar(50) NOT NULL COMMENT '学科代码',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_standard_id` (`standard_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学科信息表';

-- 课程表
DROP TABLE IF EXISTS `courses`;
CREATE TABLE `courses` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `subject_id` bigint(20) NOT NULL COMMENT '学科ID',
  `course_code` varchar(50) NOT NULL COMMENT '课程代码',
  `course_name` varchar(200) NOT NULL COMMENT '课程名称',
  `grade_level` int(11) NOT NULL COMMENT '年级',
  `semester` int(11) NOT NULL COMMENT '学期',
  `course_type` enum('standard','adaptive','self_paced','blended') NOT NULL DEFAULT 'standard' COMMENT '课程类型',
  `status` enum('draft','active','archived') NOT NULL DEFAULT 'draft' COMMENT '状态',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_subject_id` (`subject_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课程信息表';

-- 课程模块表
DROP TABLE IF EXISTS `course_modules`;
CREATE TABLE `course_modules` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '模块ID',
  `course_id` bigint(20) NOT NULL COMMENT '课程ID',
  `module_name` varchar(200) NOT NULL COMMENT '模块名称',
  `module_order` int(11) NOT NULL COMMENT '模块顺序',
  `suggested_order` int(11) DEFAULT NULL COMMENT '建议顺序',
  `alternative_paths` json DEFAULT NULL COMMENT '替代路径',
  `difficulty_adaptations` json DEFAULT NULL COMMENT '难度适应',
  `content_formats` json DEFAULT NULL COMMENT '内容格式',
  `is_optional` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否可选',
  `estimated_hours` int(11) DEFAULT NULL COMMENT '预计学时',
  `learning_goals` json DEFAULT NULL COMMENT '学习目标',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_module_order` (`module_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课程模块表';

-- 章节表
DROP TABLE IF EXISTS `chapters`;
CREATE TABLE `chapters` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '章节ID',
  `module_id` bigint(20) NOT NULL COMMENT '模块ID',
  `chapter_name` varchar(200) NOT NULL COMMENT '章节名称',
  `chapter_order` int(11) NOT NULL COMMENT '章节顺序',
  `estimated_hours` int(11) DEFAULT NULL COMMENT '预计学时',
  `learning_goals` json DEFAULT NULL COMMENT '学习目标',
  `difficulty_level` varchar(20) DEFAULT NULL COMMENT '难度等级',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_module_id` (`module_id`),
  KEY `idx_chapter_order` (`chapter_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='章节信息表';

-- 课时表
DROP TABLE IF EXISTS `lessons`;
CREATE TABLE `lessons` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '课时ID',
  `chapter_id` bigint(20) NOT NULL COMMENT '章节ID',
  `lesson_title` varchar(200) NOT NULL COMMENT '课时标题',
  `lesson_order` int(11) NOT NULL COMMENT '课时顺序',
  `duration_minutes` int(11) NOT NULL DEFAULT '45' COMMENT '时长(分钟)',
  `lesson_type` enum('theory','practice','lab','discussion','assessment') NOT NULL DEFAULT 'theory' COMMENT '课时类型',
  `content_outline` json DEFAULT NULL COMMENT '内容大纲',
  `teaching_materials` json DEFAULT NULL COMMENT '教学材料',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_chapter_id` (`chapter_id`),
  KEY `idx_lesson_order` (`lesson_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课时信息表';

-- 班级课程安排表
DROP TABLE IF EXISTS `class_schedules`;
CREATE TABLE `class_schedules` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '安排ID',
  `course_id` bigint(20) NOT NULL COMMENT '课程ID',
  `class_id` bigint(20) NOT NULL COMMENT '班级ID',
  `teacher_id` bigint(20) NOT NULL COMMENT '教师ID',
  `time_slot` varchar(50) NOT NULL COMMENT '时间段',
  `classroom` varchar(100) DEFAULT NULL COMMENT '教室',
  `start_date` date NOT NULL COMMENT '开始日期',
  `end_date` date NOT NULL COMMENT '结束日期',
  `weekly_schedule` json DEFAULT NULL COMMENT '周课程表',
  `status` enum('active','cancelled','completed') NOT NULL DEFAULT 'active' COMMENT '状态',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_class_id` (`class_id`),
  KEY `idx_teacher_id` (`teacher_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='班级课程安排表';

-- =============================================================================
-- 作业管理系统模块
-- =============================================================================

-- 作业表 (基于实际使用的字段)
DROP TABLE IF EXISTS `homeworks`;
CREATE TABLE `homeworks` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '作业ID',
  `title` varchar(200) NOT NULL COMMENT '作业标题',
  `description` text DEFAULT NULL COMMENT '作业描述',
  `subject` varchar(50) NOT NULL COMMENT '学科',
  `grade` int(11) NOT NULL COMMENT '年级',
  `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级(1-5)',
  `question_count` int(11) NOT NULL DEFAULT '0' COMMENT '题目数量',
  `max_score` int(11) NOT NULL DEFAULT '100' COMMENT '总分',
  `time_limit` int(11) DEFAULT NULL COMMENT '时间限制(分钟)',
  `due_date` datetime DEFAULT NULL COMMENT '截止时间',
  `start_date` datetime DEFAULT NULL COMMENT '开始时间',
  `is_published` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否发布',
  `is_template` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否为模板',
  `created_by` bigint(20) NOT NULL COMMENT '创建者ID',
  `category` varchar(50) DEFAULT NULL COMMENT '作业分类',
  `tags` json DEFAULT NULL COMMENT '标签列表',
  `instructions` text DEFAULT NULL COMMENT '作业说明',
  `auto_grade` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否自动评分',
  `max_attempts` int(11) NOT NULL DEFAULT '1' COMMENT '最大尝试次数',
  `show_answers` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否显示答案',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_created_by` (`created_by`),
  KEY `idx_subject_grade` (`subject`,`grade`),
  KEY `idx_due_date` (`due_date`),
  KEY `idx_is_published` (`is_published`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业基础信息表';

-- 题目表
DROP TABLE IF EXISTS `questions`;
CREATE TABLE `questions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
  `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
  `content` text NOT NULL COMMENT '题目内容',
  `question_type` enum('single_choice','multiple_choice','fill_blank','calculation','proof','application') NOT NULL DEFAULT 'single_choice' COMMENT '题目类型',
  `options` json DEFAULT NULL COMMENT '选择题选项',
  `correct_answer` text NOT NULL COMMENT '正确答案',
  `score` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '题目分值',
  `difficulty` int(11) NOT NULL DEFAULT '1' COMMENT '难度等级(1-5)',
  `order_index` int(11) NOT NULL DEFAULT '1' COMMENT '题目顺序',
  `knowledge_points` json DEFAULT NULL COMMENT '关联知识点',
  `explanation` text DEFAULT NULL COMMENT '题目解析',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_question_type` (`question_type`),
  KEY `idx_difficulty` (`difficulty`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目信息表';

-- 作业分配表
DROP TABLE IF EXISTS `homework_assignments`;
CREATE TABLE `homework_assignments` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '分配ID',
  `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
  `assigned_to_type` enum('student','class','grade','all') NOT NULL COMMENT '分配目标类型',
  `assigned_to_id` bigint(20) DEFAULT NULL COMMENT '分配目标ID',
  `assigned_by` bigint(20) NOT NULL COMMENT '分配者ID',
  `assigned_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '分配时间',
  `due_date_override` datetime DEFAULT NULL COMMENT '截止时间覆盖',
  `start_date_override` datetime DEFAULT NULL COMMENT '开始时间覆盖',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否活跃',
  `notes` text DEFAULT NULL COMMENT '备注',
  `max_attempts_override` int(11) DEFAULT NULL COMMENT '最大尝试次数覆盖',
  `notification_sent` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否已发送通知',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_assigned_to` (`assigned_to_type`,`assigned_to_id`),
  KEY `idx_assigned_by` (`assigned_by`),
  KEY `idx_is_active` (`is_active`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业分配表';

-- 作业提交表
DROP TABLE IF EXISTS `homework_submissions`;
CREATE TABLE `homework_submissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '提交ID',
  `assignment_id` bigint(20) NOT NULL COMMENT '分配ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
  `answers` json DEFAULT NULL COMMENT '答案数据',
  `submission_data` json DEFAULT NULL COMMENT '提交数据',
  `submitted_at` datetime DEFAULT NULL COMMENT '提交时间',
  `score` decimal(5,2) DEFAULT NULL COMMENT '得分',
  `max_score` decimal(5,2) DEFAULT '100.00' COMMENT '总分',
  `status` enum('draft','submitted','graded','returned') DEFAULT 'draft' COMMENT '状态',
  `time_spent` int(11) DEFAULT '0' COMMENT '用时(秒)',
  `attempt_count` int(11) DEFAULT '1' COMMENT '尝试次数',
  `teacher_comments` text DEFAULT NULL COMMENT '教师评语',
  `auto_grade_data` json DEFAULT NULL COMMENT '自动评分数据',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_assignment_student` (`assignment_id`,`student_id`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_submitted_at` (`submitted_at`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提交表';

-- 作业进度表
DROP TABLE IF EXISTS `homework_progress`;
CREATE TABLE `homework_progress` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '进度ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
  `progress_data` json DEFAULT NULL COMMENT '进度数据',
  `completion_rate` decimal(5,2) DEFAULT '0.00' COMMENT '完成率',
  `last_saved_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后保存时间',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_homework` (`student_id`,`homework_id`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_completion_rate` (`completion_rate`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业进度表';

-- 作业收藏表
DROP TABLE IF EXISTS `homework_favorites`;
CREATE TABLE `homework_favorites` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '收藏ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `assignment_id` bigint(20) NOT NULL COMMENT '分配ID',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '收藏时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_assignment` (`student_id`,`assignment_id`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业收藏表';

-- 作业提醒表
DROP TABLE IF EXISTS `homework_reminders`;
CREATE TABLE `homework_reminders` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '提醒ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `assignment_id` bigint(20) NOT NULL COMMENT '分配ID',
  `reminder_type` enum('due_soon','overdue','custom') DEFAULT 'due_soon' COMMENT '提醒类型',
  `reminder_time` datetime NOT NULL COMMENT '提醒时间',
  `message` text DEFAULT NULL COMMENT '提醒消息',
  `is_sent` tinyint(1) DEFAULT '0' COMMENT '是否已发送',
  `sent_at` datetime DEFAULT NULL COMMENT '发送时间',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_reminder_time` (`reminder_time`),
  KEY `idx_is_sent` (`is_sent`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提醒表';

-- =============================================================================
-- 学习路径系统模块
-- =============================================================================

-- 学习路径表
DROP TABLE IF EXISTS `learning_paths`;
CREATE TABLE `learning_paths` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '路径ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `path_name` varchar(200) NOT NULL COMMENT '路径名称',
  `path_structure` json NOT NULL COMMENT '路径结构',
  `completion_rate` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '完成率',
  `target_completion` date DEFAULT NULL COMMENT '目标完成日期',
  `status` enum('active','paused','completed','cancelled') NOT NULL DEFAULT 'active' COMMENT '状态',
  `milestones` json DEFAULT NULL COMMENT '里程碑',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_student_id` (`student_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学习路径表';

-- =============================================================================
-- 数据完整性约束和索引优化
-- =============================================================================

-- 重新启用外键检查
SET FOREIGN_KEY_CHECKS = 1;

-- =============================================================================
-- 初始化数据
-- =============================================================================

-- 插入默认学校数据
INSERT IGNORE INTO `schools` (`id`, `school_name`, `school_code`, `school_type`, `education_level`) VALUES
(1, '测试学校', 'TEST_SCHOOL_001', 'mixed', '小学-高中');

-- 插入默认年级数据
INSERT IGNORE INTO `grades` (`id`, `school_id`, `grade_name`, `grade_level`, `academic_year`) VALUES
(1, 1, '一年级', 1, '2024-2025'),
(2, 1, '二年级', 2, '2024-2025'),
(3, 1, '三年级', 3, '2024-2025'),
(4, 1, '四年级', 4, '2024-2025'),
(5, 1, '五年级', 5, '2024-2025'),
(6, 1, '六年级', 6, '2024-2025'),
(7, 1, '七年级', 7, '2024-2025'),
(8, 1, '八年级', 8, '2024-2025'),
(9, 1, '九年级', 9, '2024-2025');

-- 插入默认班级数据
INSERT IGNORE INTO `classes` (`id`, `school_id`, `grade_id`, `class_name`, `class_code`) VALUES
(1, 1, 7, '七年级一班', '7-1'),
(2, 1, 7, '七年级二班', '7-2'),
(3, 1, 8, '八年级一班', '8-1'),
(4, 1, 8, '八年级二班', '8-2');

-- 插入默认课程标准
INSERT IGNORE INTO `curriculum_standards` (`id`, `school_id`, `standard_name`, `grade_range`) VALUES
(1, 1, '义务教育数学课程标准', '1-9年级');

-- 插入默认学科
INSERT IGNORE INTO `subjects` (`id`, `standard_id`, `subject_name`, `subject_code`) VALUES
(1, 1, '数学', 'MATH'),
(2, 1, '语文', 'CHINESE'),
(3, 1, '英语', 'ENGLISH');

-- =============================================================================
-- 智能化系统扩展 - 知识图谱系统
-- =============================================================================

-- 知识点表
DROP TABLE IF EXISTS `knowledge_points`;
CREATE TABLE `knowledge_points` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '知识点ID',
  `name` varchar(200) NOT NULL COMMENT '知识点名称',
  `description` text DEFAULT NULL COMMENT '详细描述',
  `subject_id` bigint(20) DEFAULT NULL COMMENT '学科ID',
  `grade_level` int(11) NOT NULL COMMENT '年级层次',
  `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级(1-5)',
  `cognitive_type` enum('conceptual','procedural','metacognitive') DEFAULT 'conceptual' COMMENT '认知类型',
  `bloom_level` int(11) DEFAULT '1' COMMENT '布鲁姆分类等级(1-6)',
  `prerequisites` json DEFAULT NULL COMMENT '前置知识点',
  `learning_objectives` json DEFAULT NULL COMMENT '学习目标',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_subject_id` (`subject_id`),
  KEY `idx_grade_level` (`grade_level`),
  KEY `idx_difficulty_level` (`difficulty_level`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点定义表';

-- 知识点关系表
DROP TABLE IF EXISTS `knowledge_relationships`;
CREATE TABLE `knowledge_relationships` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '关系ID',
  `source_point_id` bigint(20) NOT NULL COMMENT '源知识点ID',
  `target_point_id` bigint(20) NOT NULL COMMENT '目标知识点ID',
  `relationship_type` enum('prerequisite','related','extends','applies_to','contradicts') NOT NULL COMMENT '关系类型',
  `strength` decimal(3,2) DEFAULT '0.50' COMMENT '关系强度(0-1)',
  `confidence` decimal(3,2) DEFAULT '0.50' COMMENT '置信度(0-1)',
  `evidence_count` int(11) DEFAULT '0' COMMENT '证据数量',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_source_point` (`source_point_id`),
  KEY `idx_target_point` (`target_point_id`),
  KEY `idx_relationship_type` (`relationship_type`),
  CONSTRAINT `fk_kr_source` FOREIGN KEY (`source_point_id`) REFERENCES `knowledge_points` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_kr_target` FOREIGN KEY (`target_point_id`) REFERENCES `knowledge_points` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点关系表';

-- 概念图表
DROP TABLE IF EXISTS `concept_maps`;
CREATE TABLE `concept_maps` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '概念图ID',
  `title` varchar(200) NOT NULL COMMENT '概念图标题',
  `description` text DEFAULT NULL COMMENT '描述',
  `subject_id` bigint(20) DEFAULT NULL COMMENT '学科ID',
  `grade_level` int(11) NOT NULL COMMENT '年级层次',
  `map_data` json DEFAULT NULL COMMENT '图结构数据(节点和边)',
  `layout_config` json DEFAULT NULL COMMENT '布局配置',
  `created_by` bigint(20) NOT NULL COMMENT '创建者ID',
  `is_public` tinyint(1) DEFAULT '0' COMMENT '是否公开',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_subject_id` (`subject_id`),
  KEY `idx_created_by` (`created_by`),
  KEY `idx_grade_level` (`grade_level`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='概念图表';

-- =============================================================================
-- 智能推荐系统
-- =============================================================================

-- 符号推荐表
DROP TABLE IF EXISTS `symbol_recommendations`;
CREATE TABLE `symbol_recommendations` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '推荐ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `context` text DEFAULT NULL COMMENT '输入上下文',
  `recommended_symbols` json DEFAULT NULL COMMENT '推荐的符号列表',
  `selected_symbol` varchar(50) DEFAULT NULL COMMENT '用户选择的符号',
  `usage_frequency` int(11) DEFAULT '0' COMMENT '使用频率',
  `success_rate` decimal(5,2) DEFAULT '0.00' COMMENT '推荐成功率',
  `response_time` int(11) DEFAULT '0' COMMENT '响应时间(毫秒)',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='符号推荐表';

-- 题目推荐表
DROP TABLE IF EXISTS `problem_recommendations`;
CREATE TABLE `problem_recommendations` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '推荐ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `recommendation_reason` text DEFAULT NULL COMMENT '推荐理由',
  `difficulty_match` decimal(3,2) DEFAULT '0.50' COMMENT '难度匹配度(0-1)',
  `knowledge_gap_target` json DEFAULT NULL COMMENT '目标知识缺口',
  `predicted_success_rate` decimal(3,2) DEFAULT '0.50' COMMENT '预测成功率',
  `actual_result` enum('correct','incorrect','skipped','timeout') DEFAULT NULL COMMENT '实际结果',
  `user_feedback` int(11) DEFAULT NULL COMMENT '用户反馈评分(1-5)',
  `recommended_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '推荐时间',
  `completed_at` datetime DEFAULT NULL COMMENT '完成时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_recommended_at` (`recommended_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目推荐表';

-- 学习路径推荐表
DROP TABLE IF EXISTS `learning_path_recommendations`;
CREATE TABLE `learning_path_recommendations` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '推荐ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `current_knowledge_state` json DEFAULT NULL COMMENT '当前知识状态',
  `recommended_path` json DEFAULT NULL COMMENT '推荐路径',
  `path_type` enum('remedial','advancement','review','exploration') DEFAULT 'advancement' COMMENT '路径类型',
  `estimated_duration` int(11) DEFAULT '0' COMMENT '预计时长(分钟)',
  `success_prediction` decimal(3,2) DEFAULT '0.50' COMMENT '成功预测概率',
  `adaptation_triggers` json DEFAULT NULL COMMENT '适应触发条件',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_path_type` (`path_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学习路径推荐表';

-- =============================================================================
-- 学习分析系统
-- =============================================================================

-- 学习行为表
DROP TABLE IF EXISTS `learning_behaviors`;
CREATE TABLE `learning_behaviors` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '行为ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `session_id` varchar(100) DEFAULT NULL COMMENT '会话ID',
  `behavior_type` enum('click','hover','input','submit','pause','review','help_seek') NOT NULL COMMENT '行为类型',
  `behavior_data` json DEFAULT NULL COMMENT '行为详细数据',
  `context_info` json DEFAULT NULL COMMENT '上下文信息',
  `timestamp` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '时间戳',
  `duration` int(11) DEFAULT '0' COMMENT '持续时间(毫秒)',
  `device_info` json DEFAULT NULL COMMENT '设备信息',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_session_id` (`session_id`),
  KEY `idx_behavior_type` (`behavior_type`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学习行为表';

-- 交互日志表
DROP TABLE IF EXISTS `interaction_logs`;
CREATE TABLE `interaction_logs` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '日志ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `homework_id` bigint(20) DEFAULT NULL COMMENT '作业ID',
  `question_id` bigint(20) DEFAULT NULL COMMENT '题目ID',
  `interaction_type` enum('view','attempt','submit','hint','skip','review') NOT NULL COMMENT '交互类型',
  `interaction_data` json DEFAULT NULL COMMENT '交互数据',
  `response_time` int(11) DEFAULT '0' COMMENT '响应时间(毫秒)',
  `accuracy` decimal(3,2) DEFAULT NULL COMMENT '准确率',
  `hint_used` tinyint(1) DEFAULT '0' COMMENT '是否使用提示',
  `attempts_count` int(11) DEFAULT '1' COMMENT '尝试次数',
  `final_answer` text DEFAULT NULL COMMENT '最终答案',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_interaction_type` (`interaction_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='交互日志表';

-- 参与度指标表
DROP TABLE IF EXISTS `engagement_metrics`;
CREATE TABLE `engagement_metrics` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '指标ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `date` date NOT NULL COMMENT '日期',
  `session_duration` int(11) DEFAULT '0' COMMENT '会话时长(分钟)',
  `questions_attempted` int(11) DEFAULT '0' COMMENT '尝试题目数',
  `completion_rate` decimal(5,2) DEFAULT '0.00' COMMENT '完成率',
  `focus_score` decimal(3,2) DEFAULT '0.00' COMMENT '专注度评分(0-1)',
  `persistence_score` decimal(3,2) DEFAULT '0.00' COMMENT '坚持度评分(0-1)',
  `help_seeking_frequency` decimal(5,2) DEFAULT '0.00' COMMENT '求助频率',
  `self_regulation_score` decimal(3,2) DEFAULT '0.00' COMMENT '自我调节评分(0-1)',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_date` (`user_id`,`date`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_date` (`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='参与度指标表';

-- =============================================================================
-- 错误分析系统
-- =============================================================================

-- 错误模式表
DROP TABLE IF EXISTS `error_patterns`;
CREATE TABLE `error_patterns` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '错误模式ID',
  `question_id` bigint(20) DEFAULT NULL COMMENT '题目ID',
  `error_type` enum('conceptual','procedural','computational','careless') NOT NULL COMMENT '错误类型',
  `error_description` text DEFAULT NULL COMMENT '错误描述',
  `frequency` int(11) DEFAULT '0' COMMENT '出现频率',
  `common_misconceptions` json DEFAULT NULL COMMENT '常见误解',
  `difficulty_indicators` json DEFAULT NULL COMMENT '难点指标',
  `remediation_strategies` json DEFAULT NULL COMMENT '补救策略',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_error_type` (`error_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='错误模式表';

-- 误解分析表
DROP TABLE IF EXISTS `misconception_analysis`;
CREATE TABLE `misconception_analysis` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '分析ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `knowledge_point_id` bigint(20) DEFAULT NULL COMMENT '知识点ID',
  `error_pattern_id` bigint(20) DEFAULT NULL COMMENT '错误模式ID',
  `misconception_type` varchar(100) DEFAULT NULL COMMENT '误解类型',
  `evidence_data` json DEFAULT NULL COMMENT '证据数据',
  `confidence_level` decimal(3,2) DEFAULT '0.50' COMMENT '置信度(0-1)',
  `detected_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '检测时间',
  `intervention_suggested` json DEFAULT NULL COMMENT '建议干预措施',
  `resolved_at` datetime DEFAULT NULL COMMENT '解决时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_knowledge_point_id` (`knowledge_point_id`),
  KEY `idx_error_pattern_id` (`error_pattern_id`),
  CONSTRAINT `fk_ma_knowledge_point` FOREIGN KEY (`knowledge_point_id`) REFERENCES `knowledge_points` (`id`) ON DELETE SET NULL,
  CONSTRAINT `fk_ma_error_pattern` FOREIGN KEY (`error_pattern_id`) REFERENCES `error_patterns` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='误解分析表';

-- =============================================================================
-- 自适应学习系统
-- =============================================================================

-- 自适应路径表
DROP TABLE IF EXISTS `adaptive_paths`;
CREATE TABLE `adaptive_paths` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '路径ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `current_state` json DEFAULT NULL COMMENT '当前学习状态',
  `target_state` json DEFAULT NULL COMMENT '目标状态',
  `path_steps` json DEFAULT NULL COMMENT '路径步骤',
  `adaptation_reason` text DEFAULT NULL COMMENT '适应原因',
  `difficulty_adjustment` decimal(3,2) DEFAULT '1.00' COMMENT '难度调整系数',
  `estimated_completion` int(11) DEFAULT '0' COMMENT '预计完成时间(分钟)',
  `success_rate` decimal(3,2) DEFAULT '0.50' COMMENT '成功率预测',
  `last_updated` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_last_updated` (`last_updated`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自适应路径表';

-- 掌握度跟踪表
DROP TABLE IF EXISTS `mastery_tracking`;
CREATE TABLE `mastery_tracking` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '跟踪ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `knowledge_point_id` bigint(20) NOT NULL COMMENT '知识点ID',
  `mastery_level` decimal(3,2) DEFAULT '0.00' COMMENT '掌握度(0-1)',
  `confidence_interval` json DEFAULT NULL COMMENT '置信区间',
  `evidence_count` int(11) DEFAULT '0' COMMENT '证据数量',
  `last_assessment` datetime DEFAULT NULL COMMENT '最后评估时间',
  `decay_rate` decimal(5,4) DEFAULT '0.0100' COMMENT '遗忘衰减率',
  `next_review_due` datetime DEFAULT NULL COMMENT '下次复习时间',
  `mastery_achieved_at` datetime DEFAULT NULL COMMENT '掌握达成时间',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_knowledge_point` (`user_id`,`knowledge_point_id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_knowledge_point_id` (`knowledge_point_id`),
  KEY `idx_mastery_level` (`mastery_level`),
  KEY `idx_next_review_due` (`next_review_due`),
  CONSTRAINT `fk_mt_knowledge_point` FOREIGN KEY (`knowledge_point_id`) REFERENCES `knowledge_points` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='掌握度跟踪表';

-- 题目知识点关联表 (支持智能推荐)
DROP TABLE IF EXISTS `question_knowledge_points`;
CREATE TABLE `question_knowledge_points` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '关联ID',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `knowledge_point_id` bigint(20) NOT NULL COMMENT '知识点ID',
  `relevance_score` decimal(3,2) DEFAULT '1.00' COMMENT '相关度分数(0-1)',
  `is_primary` tinyint(1) DEFAULT '0' COMMENT '是否为主要知识点',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_question_knowledge_point` (`question_id`,`knowledge_point_id`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_knowledge_point_id` (`knowledge_point_id`),
  CONSTRAINT `fk_qkp_knowledge_point` FOREIGN KEY (`knowledge_point_id`) REFERENCES `knowledge_points` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目知识点关联表';

-- =============================================================================
-- 数据库维护脚本
-- =============================================================================

-- 清理过期会话
DELIMITER $$
CREATE EVENT IF NOT EXISTS `cleanup_expired_sessions`
ON SCHEDULE EVERY 1 HOUR
DO
BEGIN
  DELETE FROM user_sessions WHERE expires_at < NOW();
END$$
DELIMITER ;

-- 清理过期通知
DELIMITER $$
CREATE EVENT IF NOT EXISTS `cleanup_expired_notifications`
ON SCHEDULE EVERY 1 DAY
DO
BEGIN
  DELETE FROM notifications WHERE expires_at IS NOT NULL AND expires_at < NOW();
END$$
DELIMITER ;

-- 启用事件调度器
SET GLOBAL event_scheduler = ON;

-- =============================================================================
-- 完成
-- =============================================================================

SELECT 'K-12数学教育系统数据库结构创建完成！' as message;
SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'testccnu';

-- 显示所有表的统计信息
SELECT
    table_name as '表名',
    table_comment as '表说明',
    table_rows as '行数'
FROM information_schema.tables
WHERE table_schema = 'testccnu'
ORDER BY table_name;
