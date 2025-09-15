# -*- coding: utf-8 -*-
"""
创建简化的数据库结构 - 分步骤创建
"""
import pymysql
from config import config

def create_simplified_schema():
    """创建简化的数据库结构"""
    
    # 数据库配置
    env = 'development'
    current_config = config[env]
    db_config = current_config.DATABASE_CONFIG
    
    connection = pymysql.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        passwd=db_config['password'],
        db=db_config['database'],
        charset='utf8mb4',
        autocommit=True
    )
    cursor = connection.cursor()
    
    try:
        print("=== 创建K-12数学教育系统数据库 ===\n")
        
        # 1. 清理现有表
        print("1. 清理现有表...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        tables_to_drop = [
            'question_answers', 'homework_submissions', 'questions', 'homeworks',
            'tutor_assignments', 'student_course_enrollments', 'courses', 'subjects',
            'curriculum_standards', 'classes', 'grades', 'schools', 'users', 
            'user_sessions', 'student_profiles', 'parent_student_relations'
        ]
        
        for table in tables_to_drop:
            cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
            
        print("   现有表已清理")
        
        # 2. 创建基础表 (无外键)
        print("\n2. 创建基础表...")
        
        # 学校表
        cursor.execute("""
        CREATE TABLE `schools` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学校ID',
          `school_name` varchar(200) NOT NULL COMMENT '学校名称',
          `school_code` varchar(50) UNIQUE NOT NULL COMMENT '学校代码',
          `school_type` enum('primary','middle','high','mixed') NOT NULL COMMENT '学校类型',
          `education_level` varchar(50) DEFAULT NULL COMMENT '教育层次',
          `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_school_code` (`school_code`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学校基础信息表'
        """)
        
        # 年级表
        cursor.execute("""
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='年级信息表'
        """)
        
        # 班级表
        cursor.execute("""
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='班级信息表'
        """)
        
        # 用户表
        cursor.execute("""
        CREATE TABLE `users` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
          `username` varchar(50) NOT NULL COMMENT '用户名',
          `email` varchar(100) NOT NULL COMMENT '邮箱',
          `password_hash` varchar(255) NOT NULL COMMENT '密码哈希',
          `role` enum('student','teacher','admin','parent','tutor') NOT NULL DEFAULT 'student' COMMENT '用户角色',
          `real_name` varchar(50) DEFAULT NULL COMMENT '真实姓名',
          `gender` enum('male','female','other') DEFAULT NULL COMMENT '性别',
          `birth_date` date DEFAULT NULL COMMENT '出生日期',
          `phone` varchar(20) DEFAULT NULL COMMENT '手机号',
          `avatar` varchar(255) DEFAULT NULL COMMENT '头像URL',
          `school_id` bigint(20) DEFAULT NULL COMMENT '所属学校ID',
          `grade_id` bigint(20) DEFAULT NULL COMMENT '年级ID(学生)',
          `class_id` bigint(20) DEFAULT NULL COMMENT '班级ID(学生)',
          `student_number` varchar(50) DEFAULT NULL COMMENT '学号',
          `teacher_number` varchar(50) DEFAULT NULL COMMENT '教师编号',
          `profile` json DEFAULT NULL COMMENT '用户配置信息',
          `learning_preferences` json DEFAULT NULL COMMENT '学习偏好设置',
          `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否激活',
          `last_login_time` datetime DEFAULT NULL COMMENT '最后登录时间',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_username` (`username`),
          UNIQUE KEY `uk_email` (`email`),
          UNIQUE KEY `uk_student_number` (`student_number`),
          UNIQUE KEY `uk_teacher_number` (`teacher_number`),
          KEY `idx_role` (`role`),
          KEY `idx_school_id` (`school_id`),
          KEY `idx_grade_id` (`grade_id`),
          KEY `idx_class_id` (`class_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户基础信息表'
        """)
        
        # 用户会话表
        cursor.execute("""
        CREATE TABLE `user_sessions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '会话ID',
          `user_id` bigint(20) NOT NULL COMMENT '用户ID',
          `session_token` varchar(255) NOT NULL COMMENT '会话令牌',
          `device_type` varchar(20) DEFAULT NULL COMMENT '设备类型',
          `device_id` varchar(100) DEFAULT NULL COMMENT '设备标识',
          `ip_address` varchar(45) DEFAULT NULL COMMENT 'IP地址',
          `user_agent` text DEFAULT NULL COMMENT '用户代理',
          `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否活跃',
          `expires_at` datetime NOT NULL COMMENT '过期时间',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_session_token` (`session_token`),
          KEY `idx_user_id` (`user_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户会话表'
        """)
        
        # 课程标准表
        cursor.execute("""
        CREATE TABLE `curriculum_standards` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '标准ID',
          `school_id` bigint(20) NOT NULL COMMENT '学校ID',
          `standard_name` varchar(200) NOT NULL COMMENT '标准名称',
          `grade_range` varchar(20) NOT NULL COMMENT '适用年级范围',
          `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_school_id` (`school_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课程标准表'
        """)
        
        # 学科表
        cursor.execute("""
        CREATE TABLE `subjects` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学科ID',
          `standard_id` bigint(20) NOT NULL COMMENT '课程标准ID',
          `subject_name` varchar(100) NOT NULL COMMENT '学科名称',
          `subject_code` varchar(50) NOT NULL COMMENT '学科代码',
          `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_standard_id` (`standard_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学科表'
        """)
        
        # 课程表
        cursor.execute("""
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课程表'
        """)
        
        # 作业表
        cursor.execute("""
        CREATE TABLE `homeworks` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '作业ID',
          `title` varchar(200) NOT NULL COMMENT '作业标题',
          `description` text DEFAULT NULL COMMENT '作业描述',
          `course_id` bigint(20) NOT NULL COMMENT '课程ID',
          `class_id` bigint(20) DEFAULT NULL COMMENT '班级ID',
          `subject` varchar(50) NOT NULL COMMENT '学科',
          `grade` int(11) NOT NULL COMMENT '年级',
          `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级(1-5)',
          `question_count` int(11) NOT NULL DEFAULT '0' COMMENT '题目数量',
          `max_score` int(11) NOT NULL DEFAULT '100' COMMENT '总分',
          `time_limit` int(11) DEFAULT NULL COMMENT '时间限制(分钟)',
          `due_date` datetime DEFAULT NULL COMMENT '截止时间',
          `start_date` datetime DEFAULT NULL COMMENT '开始时间',
          `is_published` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否发布',
          `created_by` bigint(20) NOT NULL COMMENT '创建者ID',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          KEY `idx_course_id` (`course_id`),
          KEY `idx_class_id` (`class_id`),
          KEY `idx_created_by` (`created_by`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业基础信息表'
        """)
        
        # 题目表 (无外键)
        cursor.execute("""
        CREATE TABLE `questions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `question_order` int(11) NOT NULL COMMENT '题目顺序',
          `question_type` enum('choice','fill','calculation','proof','application') NOT NULL COMMENT '题目类型',
          `question_title` varchar(500) NOT NULL COMMENT '题目标题',
          `question_content` text NOT NULL COMMENT '题目内容',
          `question_image` varchar(255) DEFAULT NULL COMMENT '题目图片',
          `options` json DEFAULT NULL COMMENT '选项(选择题使用)',
          `correct_answer` text NOT NULL COMMENT '正确答案',
          `answer_analysis` text DEFAULT NULL COMMENT '答案解析',
          `score` int(11) NOT NULL DEFAULT '10' COMMENT '分值',
          `difficulty` int(11) NOT NULL DEFAULT '3' COMMENT '难度系数(1-5)',
          `knowledge_points` json DEFAULT NULL COMMENT '关联知识点',
          `symbols_used` json DEFAULT NULL COMMENT '使用的数学符号',
          `solution_steps` json DEFAULT NULL COMMENT '解题步骤',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          KEY `idx_homework_id` (`homework_id`),
          KEY `idx_question_type` (`question_type`),
          KEY `idx_difficulty` (`difficulty`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目详情表'
        """)
        
        # 作业提交表
        cursor.execute("""
        CREATE TABLE `homework_submissions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '提交ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `student_id` bigint(20) NOT NULL COMMENT '学生ID',
          `submission_status` enum('draft','submitted','graded','returned') NOT NULL DEFAULT 'draft' COMMENT '提交状态',
          `total_score` decimal(5,2) DEFAULT NULL COMMENT '总得分',
          `max_score` decimal(5,2) NOT NULL COMMENT '满分',
          `completion_rate` decimal(5,2) DEFAULT NULL COMMENT '完成率',
          `time_spent` int(11) DEFAULT NULL COMMENT '用时(分钟)',
          `submit_count` int(11) NOT NULL DEFAULT '0' COMMENT '提交次数',
          `graded_by` bigint(20) DEFAULT NULL COMMENT '评分教师ID',
          `teacher_comment` text DEFAULT NULL COMMENT '教师评语',
          `submitted_at` datetime DEFAULT NULL COMMENT '提交时间',
          `graded_at` datetime DEFAULT NULL COMMENT '评分时间',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_homework_student` (`homework_id`,`student_id`),
          KEY `idx_student_id` (`student_id`),
          KEY `idx_graded_by` (`graded_by`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提交记录表'
        """)
        
        # 答题记录表
        cursor.execute("""
        CREATE TABLE `question_answers` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '答题记录ID',
          `submission_id` bigint(20) NOT NULL COMMENT '提交ID',
          `question_id` bigint(20) NOT NULL COMMENT '题目ID',
          `student_answer` text NOT NULL COMMENT '学生答案',
          `answer_process` text DEFAULT NULL COMMENT '解题过程',
          `answer_time` int(11) DEFAULT NULL COMMENT '答题用时(秒)',
          `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确',
          `score` decimal(5,2) DEFAULT NULL COMMENT '得分',
          `feedback` text DEFAULT NULL COMMENT '反馈信息',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_submission_question` (`submission_id`,`question_id`),
          KEY `idx_question_id` (`question_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学生答题记录表'
        """)
        
        print("   基础表创建完成")
        
        # 3. 添加外键约束
        print("\n3. 添加外键约束...")
        
        foreign_keys = [
            # 年级表
            "ALTER TABLE `grades` ADD CONSTRAINT `fk_grades_school_id` FOREIGN KEY (`school_id`) REFERENCES `schools` (`id`)",
            # 班级表
            "ALTER TABLE `classes` ADD CONSTRAINT `fk_classes_school_id` FOREIGN KEY (`school_id`) REFERENCES `schools` (`id`)",
            "ALTER TABLE `classes` ADD CONSTRAINT `fk_classes_grade_id` FOREIGN KEY (`grade_id`) REFERENCES `grades` (`id`)",
            "ALTER TABLE `classes` ADD CONSTRAINT `fk_classes_head_teacher_id` FOREIGN KEY (`head_teacher_id`) REFERENCES `users` (`id`)",
            # 用户表
            "ALTER TABLE `users` ADD CONSTRAINT `fk_users_school_id` FOREIGN KEY (`school_id`) REFERENCES `schools` (`id`)",
            "ALTER TABLE `users` ADD CONSTRAINT `fk_users_grade_id` FOREIGN KEY (`grade_id`) REFERENCES `grades` (`id`)",
            "ALTER TABLE `users` ADD CONSTRAINT `fk_users_class_id` FOREIGN KEY (`class_id`) REFERENCES `classes` (`id`)",
            # 用户会话表
            "ALTER TABLE `user_sessions` ADD CONSTRAINT `fk_user_sessions_user_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE",
            # 课程标准表
            "ALTER TABLE `curriculum_standards` ADD CONSTRAINT `fk_curriculum_standards_school_id` FOREIGN KEY (`school_id`) REFERENCES `schools` (`id`)",
            # 学科表
            "ALTER TABLE `subjects` ADD CONSTRAINT `fk_subjects_standard_id` FOREIGN KEY (`standard_id`) REFERENCES `curriculum_standards` (`id`)",
            # 课程表
            "ALTER TABLE `courses` ADD CONSTRAINT `fk_courses_subject_id` FOREIGN KEY (`subject_id`) REFERENCES `subjects` (`id`)",
            # 作业表
            "ALTER TABLE `homeworks` ADD CONSTRAINT `fk_homeworks_course_id` FOREIGN KEY (`course_id`) REFERENCES `courses` (`id`)",
            "ALTER TABLE `homeworks` ADD CONSTRAINT `fk_homeworks_class_id` FOREIGN KEY (`class_id`) REFERENCES `classes` (`id`)",
            "ALTER TABLE `homeworks` ADD CONSTRAINT `fk_homeworks_created_by` FOREIGN KEY (`created_by`) REFERENCES `users` (`id`)",
            # 题目表
            "ALTER TABLE `questions` ADD CONSTRAINT `fk_questions_homework_id` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE",
            # 作业提交表
            "ALTER TABLE `homework_submissions` ADD CONSTRAINT `fk_submissions_homework_id` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`)",
            "ALTER TABLE `homework_submissions` ADD CONSTRAINT `fk_submissions_student_id` FOREIGN KEY (`student_id`) REFERENCES `users` (`id`)",
            "ALTER TABLE `homework_submissions` ADD CONSTRAINT `fk_submissions_graded_by` FOREIGN KEY (`graded_by`) REFERENCES `users` (`id`)",
            # 答题记录表
            "ALTER TABLE `question_answers` ADD CONSTRAINT `fk_answers_submission_id` FOREIGN KEY (`submission_id`) REFERENCES `homework_submissions` (`id`) ON DELETE CASCADE",
            "ALTER TABLE `question_answers` ADD CONSTRAINT `fk_answers_question_id` FOREIGN KEY (`question_id`) REFERENCES `questions` (`id`)"
        ]
        
        success_count = 0
        for fk_sql in foreign_keys:
            try:
                cursor.execute(fk_sql)
                success_count += 1
                constraint_name = fk_sql.split("CONSTRAINT ")[1].split(" ")[0].strip("`")
                print(f"   ✅ {constraint_name}")
            except Exception as e:
                constraint_name = fk_sql.split("CONSTRAINT ")[1].split(" ")[0].strip("`")
                print(f"   ❌ {constraint_name}: {e}")
        
        print(f"\n   外键约束: {success_count}/{len(foreign_keys)} 成功")
        
        # 4. 启用外键检查
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        # 5. 插入初始数据
        print("\n4. 插入初始数据...")
        
        # 插入示例学校
        cursor.execute("""
        INSERT INTO `schools` (`school_name`, `school_code`, `school_type`, `education_level`) VALUES
        ('演示小学', 'DEMO_PRIMARY', 'primary', '小学'),
        ('演示中学', 'DEMO_MIDDLE', 'middle', '初中')
        """)
        
        # 插入管理员用户
        cursor.execute("""
        INSERT INTO `users` (`username`, `email`, `password_hash`, `role`, `real_name`, `is_active`) VALUES
        ('admin', 'admin@diem.edu', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewTH.iVEP0T/UEaa', 'admin', '系统管理员', 1)
        """)
        
        print("   初始数据插入完成")
        
        # 6. 检查结果
        print("\n5. 检查数据库状态...")
        
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"   创建的表: {len(tables)} 个")
        
        # 检查外键约束
        cursor.execute("""
        SELECT 
            TABLE_NAME,
            CONSTRAINT_NAME,
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME
        FROM information_schema.KEY_COLUMN_USAGE 
        WHERE TABLE_SCHEMA = 'testccnu' 
        AND REFERENCED_TABLE_NAME IS NOT NULL
        ORDER BY TABLE_NAME
        """)
        
        constraints = cursor.fetchall()
        print(f"   外键约束: {len(constraints)} 个")
        
        print("\n🎉 数据库创建成功！")
        print("现在可以运行以下测试:")
        print("  python init_database.py    # 检查数据库连接和表结构")
        print("  python test_auth.py        # 测试认证API")
        print("  python app.py              # 启动应用服务器")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据库创建失败: {e}")
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_simplified_schema()

