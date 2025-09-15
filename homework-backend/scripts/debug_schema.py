# -*- coding: utf-8 -*-
"""
调试数据库结构创建问题
"""
import pymysql
from config import config

def test_individual_tables():
    """逐个测试表的创建"""
    
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
        print("=== 调试数据库表创建 ===\n")
        
        # 1. 禁用外键检查
        print("1. 禁用外键检查...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        # 2. 删除已存在的表
        print("2. 清理现有表...")
        tables_to_drop = ['questions', 'homeworks', 'courses', 'subjects', 'curriculum_standards', 'classes', 'grades', 'schools', 'users']
        for table in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
                print(f"   删除表: {table}")
            except Exception as e:
                print(f"   删除表 {table} 失败: {e}")
        
        # 3. 创建基础表 (无外键依赖)
        print("\n3. 创建基础表...")
        
        # 创建学校表
        print("   创建 schools 表...")
        cursor.execute("""
        CREATE TABLE `schools` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学校ID',
          `school_name` varchar(200) NOT NULL COMMENT '学校名称',
          `school_code` varchar(50) UNIQUE NOT NULL COMMENT '学校代码',
          `school_type` enum('primary','middle','high','mixed') NOT NULL COMMENT '学校类型',
          `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学校基础信息表'
        """)
        
        # 创建年级表
        print("   创建 grades 表...")
        cursor.execute("""
        CREATE TABLE `grades` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '年级ID',
          `school_id` bigint(20) NOT NULL COMMENT '学校ID',
          `grade_name` varchar(50) NOT NULL COMMENT '年级名称',
          `grade_level` int(11) NOT NULL COMMENT '年级数字',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_school_id` (`school_id`),
          CONSTRAINT `fk_grades_school_id` FOREIGN KEY (`school_id`) REFERENCES `schools` (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='年级信息表'
        """)
        
        # 创建班级表
        print("   创建 classes 表...")
        cursor.execute("""
        CREATE TABLE `classes` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '班级ID',
          `school_id` bigint(20) NOT NULL COMMENT '学校ID',
          `grade_id` bigint(20) NOT NULL COMMENT '年级ID',
          `class_name` varchar(100) NOT NULL COMMENT '班级名称',
          `class_code` varchar(50) NOT NULL COMMENT '班级代码',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          CONSTRAINT `fk_classes_school_id` FOREIGN KEY (`school_id`) REFERENCES `schools` (`id`),
          CONSTRAINT `fk_classes_grade_id` FOREIGN KEY (`grade_id`) REFERENCES `grades` (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='班级信息表'
        """)
        
        # 创建用户表
        print("   创建 users 表...")
        cursor.execute("""
        CREATE TABLE `users` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
          `username` varchar(50) NOT NULL COMMENT '用户名',
          `email` varchar(100) NOT NULL COMMENT '邮箱',
          `password_hash` varchar(255) NOT NULL COMMENT '密码哈希',
          `role` enum('student','teacher','admin','parent') NOT NULL DEFAULT 'student' COMMENT '用户角色',
          `real_name` varchar(50) DEFAULT NULL COMMENT '真实姓名',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_username` (`username`),
          UNIQUE KEY `uk_email` (`email`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户基础信息表'
        """)
        
        # 创建课程标准表
        print("   创建 curriculum_standards 表...")
        cursor.execute("""
        CREATE TABLE `curriculum_standards` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '标准ID',
          `school_id` bigint(20) NOT NULL COMMENT '学校ID',
          `standard_name` varchar(200) NOT NULL COMMENT '标准名称',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          CONSTRAINT `fk_curriculum_standards_school_id` FOREIGN KEY (`school_id`) REFERENCES `schools` (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课程标准表'
        """)
        
        # 创建学科表
        print("   创建 subjects 表...")
        cursor.execute("""
        CREATE TABLE `subjects` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学科ID',
          `standard_id` bigint(20) NOT NULL COMMENT '课程标准ID',
          `subject_name` varchar(100) NOT NULL COMMENT '学科名称',
          `subject_code` varchar(50) NOT NULL COMMENT '学科代码',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          CONSTRAINT `fk_subjects_standard_id` FOREIGN KEY (`standard_id`) REFERENCES `curriculum_standards` (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学科表'
        """)
        
        # 创建课程表
        print("   创建 courses 表...")
        cursor.execute("""
        CREATE TABLE `courses` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '课程ID',
          `subject_id` bigint(20) NOT NULL COMMENT '学科ID',
          `course_name` varchar(200) NOT NULL COMMENT '课程名称',
          `grade_level` int(11) NOT NULL COMMENT '年级',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          CONSTRAINT `fk_courses_subject_id` FOREIGN KEY (`subject_id`) REFERENCES `subjects` (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='课程表'
        """)
        
        # 4. 测试作业表创建
        print("\n4. 创建作业表...")
        cursor.execute("""
        CREATE TABLE `homeworks` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '作业ID',
          `title` varchar(200) NOT NULL COMMENT '作业标题',
          `course_id` bigint(20) NOT NULL COMMENT '课程ID',
          `created_by` bigint(20) NOT NULL COMMENT '创建者ID',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          CONSTRAINT `fk_homeworks_course_id` FOREIGN KEY (`course_id`) REFERENCES `courses` (`id`),
          CONSTRAINT `fk_homeworks_created_by` FOREIGN KEY (`created_by`) REFERENCES `users` (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业基础信息表'
        """)
        print("   ✅ homeworks 表创建成功")
        
        # 5. 测试题目表创建 (问题所在)
        print("\n5. 创建题目表...")
        cursor.execute("""
        CREATE TABLE `questions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `question_title` varchar(500) NOT NULL COMMENT '题目标题',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_homework_id` (`homework_id`),
          CONSTRAINT `fk_questions_homework_id` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目详情表'
        """)
        print("   ✅ questions 表创建成功")
        
        # 6. 检查表结构
        print("\n6. 检查表结构...")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"   创建的表: {[table[0] for table in tables]}")
        
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
        print(f"\n   外键约束:")
        for constraint in constraints:
            print(f"     {constraint[0]}.{constraint[2]} -> {constraint[3]}.{constraint[4]} ({constraint[1]})")
        
        # 启用外键检查
        print("\n7. 重新启用外键检查...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        print("\n✅ 所有表创建成功！问题已解决。")
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        # 显示详细错误信息
        cursor.execute("SHOW ENGINE INNODB STATUS")
        status = cursor.fetchone()
        if status:
            print("\nInnoDB状态信息:")
            print(status[2][-2000:])  # 显示最后2000个字符
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    test_individual_tables()

