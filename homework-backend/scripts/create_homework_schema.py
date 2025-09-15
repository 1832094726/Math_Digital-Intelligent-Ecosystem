# -*- coding: utf-8 -*-
"""
创建作业管理系统的数据库结构
"""
import pymysql
from config import config

def create_homework_schema():
    """创建作业管理系统数据库结构"""
    
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
        print("=== 创建作业管理系统数据库 ===\n")
        
        # 1. 清理相关表
        print("1. 清理现有作业相关表...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        tables_to_drop = [
            'question_answers', 'homework_submissions', 'questions', 
            'homeworks', 'homework_assignments', 'knowledge_points',
            'question_knowledge_points', 'math_symbols'
        ]
        for table in tables_to_drop:
            cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
            
        print("   现有表已清理")
        
        # 2. 创建知识点表
        print("\n2. 创建知识点表...")
        cursor.execute("""
        CREATE TABLE `knowledge_points` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '知识点ID',
          `name` varchar(100) NOT NULL COMMENT '知识点名称',
          `code` varchar(50) NOT NULL COMMENT '知识点编码',
          `description` text DEFAULT NULL COMMENT '知识点描述',
          `subject` varchar(50) NOT NULL COMMENT '学科',
          `grade` int(11) NOT NULL COMMENT '年级',
          `category` varchar(50) DEFAULT NULL COMMENT '分类',
          `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级',
          `parent_id` bigint(20) DEFAULT NULL COMMENT '父知识点ID',
          `sort_order` int(11) NOT NULL DEFAULT '0' COMMENT '排序',
          `is_core` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否核心知识点',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_code` (`code`),
          KEY `idx_subject_grade` (`subject`,`grade`),
          KEY `idx_parent_id` (`parent_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点表'
        """)
        print("   ✅ knowledge_points 表创建成功")
        
        # 3. 创建数学符号表
        print("\n3. 创建数学符号表...")
        cursor.execute("""
        CREATE TABLE `math_symbols` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '符号ID',
          `symbol_name` varchar(100) NOT NULL COMMENT '符号名称',
          `symbol_text` varchar(50) NOT NULL COMMENT '符号字符',
          `latex_code` varchar(200) DEFAULT NULL COMMENT 'LaTeX代码',
          `unicode` varchar(20) DEFAULT NULL COMMENT 'Unicode编码',
          `category` varchar(50) NOT NULL COMMENT '符号分类',
          `subcategory` varchar(50) DEFAULT NULL COMMENT '子分类',
          `description` text DEFAULT NULL COMMENT '符号描述',
          `grade_range` varchar(20) DEFAULT NULL COMMENT '适用年级',
          `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级',
          `frequency_score` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '使用频率分数',
          `is_common` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否常用符号',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_symbol_text` (`symbol_text`),
          KEY `idx_category` (`category`),
          KEY `idx_grade_range` (`grade_range`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数学符号表'
        """)
        print("   ✅ math_symbols 表创建成功")
        
        # 4. 创建作业表
        print("\n4. 创建作业表...")
        cursor.execute("""
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
          `max_attempts` int(11) NOT NULL DEFAULT '1' COMMENT '最大提交次数',
          `show_answers` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否显示答案',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          KEY `idx_created_by` (`created_by`),
          KEY `idx_subject_grade` (`subject`,`grade`),
          KEY `idx_due_date` (`due_date`),
          KEY `idx_is_published` (`is_published`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业基础信息表'
        """)
        print("   ✅ homeworks 表创建成功")
        
        # 5. 创建题目表
        print("\n5. 创建题目表...")
        cursor.execute("""
        CREATE TABLE `questions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `question_order` int(11) NOT NULL COMMENT '题目顺序',
          `question_type` enum('choice','fill','calculation','proof','application') NOT NULL COMMENT '题目类型',
          `question_title` varchar(500) NOT NULL COMMENT '题目标题',
          `question_content` text NOT NULL COMMENT '题目内容',
          `question_latex` text DEFAULT NULL COMMENT '题目LaTeX代码',
          `question_image` varchar(255) DEFAULT NULL COMMENT '题目图片',
          `options` json DEFAULT NULL COMMENT '选项(选择题使用)',
          `correct_answer` text NOT NULL COMMENT '正确答案',
          `answer_analysis` text DEFAULT NULL COMMENT '答案解析',
          `score` int(11) NOT NULL DEFAULT '10' COMMENT '分值',
          `difficulty` int(11) NOT NULL DEFAULT '3' COMMENT '难度系数(1-5)',
          `solution_steps` json DEFAULT NULL COMMENT '解题步骤',
          `hints` json DEFAULT NULL COMMENT '提示信息',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          KEY `idx_homework_id` (`homework_id`),
          KEY `idx_question_type` (`question_type`),
          KEY `idx_difficulty` (`difficulty`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目详情表'
        """)
        print("   ✅ questions 表创建成功")
        
        # 6. 创建题目知识点关联表
        print("\n6. 创建题目知识点关联表...")
        cursor.execute("""
        CREATE TABLE `question_knowledge_points` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '关联ID',
          `question_id` bigint(20) NOT NULL COMMENT '题目ID',
          `knowledge_point_id` bigint(20) NOT NULL COMMENT '知识点ID',
          `relevance_score` decimal(3,2) NOT NULL DEFAULT '1.00' COMMENT '相关度分数',
          `is_primary` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否主要知识点',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_question_knowledge_point` (`question_id`,`knowledge_point_id`),
          KEY `idx_knowledge_point_id` (`knowledge_point_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目知识点关联表'
        """)
        print("   ✅ question_knowledge_points 表创建成功")
        
        # 7. 创建作业分发表
        print("\n7. 创建作业分发表...")
        cursor.execute("""
        CREATE TABLE `homework_assignments` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '分发ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `assigned_to_type` enum('student','class','grade','all') NOT NULL COMMENT '分发对象类型',
          `assigned_to_id` bigint(20) DEFAULT NULL COMMENT '分发对象ID',
          `assigned_by` bigint(20) NOT NULL COMMENT '分发者ID',
          `assigned_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '分发时间',
          `due_date_override` datetime DEFAULT NULL COMMENT '覆盖截止时间',
          `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否有效',
          `notes` text DEFAULT NULL COMMENT '分发备注',
          PRIMARY KEY (`id`),
          KEY `idx_homework_id` (`homework_id`),
          KEY `idx_assigned_to` (`assigned_to_type`,`assigned_to_id`),
          KEY `idx_assigned_by` (`assigned_by`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业分发记录表'
        """)
        print("   ✅ homework_assignments 表创建成功")
        
        # 8. 创建作业提交表
        print("\n8. 创建作业提交表...")
        cursor.execute("""
        CREATE TABLE `homework_submissions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '提交ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `student_id` bigint(20) NOT NULL COMMENT '学生ID',
          `assignment_id` bigint(20) DEFAULT NULL COMMENT '分发记录ID',
          `submission_status` enum('draft','submitted','graded','returned') NOT NULL DEFAULT 'draft' COMMENT '提交状态',
          `total_score` decimal(5,2) DEFAULT NULL COMMENT '总得分',
          `max_score` decimal(5,2) NOT NULL COMMENT '满分',
          `completion_rate` decimal(5,2) DEFAULT NULL COMMENT '完成率',
          `time_spent` int(11) DEFAULT NULL COMMENT '用时(分钟)',
          `submit_count` int(11) NOT NULL DEFAULT '0' COMMENT '提交次数',
          `auto_grade_score` decimal(5,2) DEFAULT NULL COMMENT '自动评分分数',
          `manual_grade_score` decimal(5,2) DEFAULT NULL COMMENT '人工评分分数',
          `graded_by` bigint(20) DEFAULT NULL COMMENT '评分教师ID',
          `teacher_comment` text DEFAULT NULL COMMENT '教师评语',
          `submitted_at` datetime DEFAULT NULL COMMENT '提交时间',
          `graded_at` datetime DEFAULT NULL COMMENT '评分时间',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_homework_student` (`homework_id`,`student_id`),
          KEY `idx_student_id` (`student_id`),
          KEY `idx_assignment_id` (`assignment_id`),
          KEY `idx_submission_status` (`submission_status`),
          KEY `idx_graded_by` (`graded_by`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提交记录表'
        """)
        print("   ✅ homework_submissions 表创建成功")
        
        # 9. 创建答题记录表
        print("\n9. 创建答题记录表...")
        cursor.execute("""
        CREATE TABLE `question_answers` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '答题记录ID',
          `submission_id` bigint(20) NOT NULL COMMENT '提交ID',
          `question_id` bigint(20) NOT NULL COMMENT '题目ID',
          `student_answer` text NOT NULL COMMENT '学生答案',
          `answer_process` text DEFAULT NULL COMMENT '解题过程',
          `answer_latex` text DEFAULT NULL COMMENT '答案LaTeX代码',
          `answer_time` int(11) DEFAULT NULL COMMENT '答题用时(秒)',
          `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确',
          `score` decimal(5,2) DEFAULT NULL COMMENT '得分',
          `auto_score` decimal(5,2) DEFAULT NULL COMMENT '自动评分',
          `manual_score` decimal(5,2) DEFAULT NULL COMMENT '人工评分',
          `feedback` text DEFAULT NULL COMMENT '反馈信息',
          `error_type` varchar(50) DEFAULT NULL COMMENT '错误类型',
          `symbols_used` json DEFAULT NULL COMMENT '使用的符号',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_submission_question` (`submission_id`,`question_id`),
          KEY `idx_question_id` (`question_id`),
          KEY `idx_is_correct` (`is_correct`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学生答题记录表'
        """)
        print("   ✅ question_answers 表创建成功")
        
        # 10. 添加外键约束
        print("\n10. 添加外键约束...")
        
        foreign_keys = [
            # 知识点表自引用
            "ALTER TABLE `knowledge_points` ADD CONSTRAINT `fk_knowledge_points_parent_id` FOREIGN KEY (`parent_id`) REFERENCES `knowledge_points` (`id`)",
            # 作业表
            "ALTER TABLE `homeworks` ADD CONSTRAINT `fk_homeworks_created_by` FOREIGN KEY (`created_by`) REFERENCES `users` (`id`)",
            # 题目表
            "ALTER TABLE `questions` ADD CONSTRAINT `fk_questions_homework_id` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE",
            # 题目知识点关联表
            "ALTER TABLE `question_knowledge_points` ADD CONSTRAINT `fk_qkp_question_id` FOREIGN KEY (`question_id`) REFERENCES `questions` (`id`) ON DELETE CASCADE",
            "ALTER TABLE `question_knowledge_points` ADD CONSTRAINT `fk_qkp_knowledge_point_id` FOREIGN KEY (`knowledge_point_id`) REFERENCES `knowledge_points` (`id`)",
            # 作业分发表
            "ALTER TABLE `homework_assignments` ADD CONSTRAINT `fk_assignments_homework_id` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE",
            "ALTER TABLE `homework_assignments` ADD CONSTRAINT `fk_assignments_assigned_by` FOREIGN KEY (`assigned_by`) REFERENCES `users` (`id`)",
            # 作业提交表
            "ALTER TABLE `homework_submissions` ADD CONSTRAINT `fk_submissions_homework_id` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`)",
            "ALTER TABLE `homework_submissions` ADD CONSTRAINT `fk_submissions_student_id` FOREIGN KEY (`student_id`) REFERENCES `users` (`id`)",
            "ALTER TABLE `homework_submissions` ADD CONSTRAINT `fk_submissions_assignment_id` FOREIGN KEY (`assignment_id`) REFERENCES `homework_assignments` (`id`)",
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
        
        # 11. 插入初始数据
        print("\n11. 插入初始数据...")
        
        # 插入基础知识点
        cursor.execute("""
        INSERT INTO `knowledge_points` (`name`, `code`, `description`, `subject`, `grade`, `category`, `difficulty_level`, `is_core`) VALUES
        ('整数运算', 'MATH_INT_CALC', '整数的加减乘除运算', '数学', 1, '数与代数', 1, 1),
        ('分数运算', 'MATH_FRAC_CALC', '分数的加减乘除运算', '数学', 3, '数与代数', 3, 1),
        ('小数运算', 'MATH_DEC_CALC', '小数的加减乘除运算', '数学', 4, '数与代数', 2, 1),
        ('一元一次方程', 'MATH_LINEAR_EQ', '一元一次方程的解法', '数学', 7, '数与代数', 4, 1),
        ('平面几何基础', 'MATH_PLANE_GEO', '点线面角的基本概念', '数学', 5, '图形与几何', 3, 1)
        """)
        
        # 插入常用数学符号
        cursor.execute("""
        INSERT INTO `math_symbols` (`symbol_name`, `symbol_text`, `latex_code`, `category`, `description`, `grade_range`, `difficulty_level`, `is_common`) VALUES
        ('加号', '+', '+', '运算符号', '加法运算符号', '1-12', 1, 1),
        ('减号', '-', '-', '运算符号', '减法运算符号', '1-12', 1, 1),
        ('乘号', '×', '\\times', '运算符号', '乘法运算符号', '1-12', 1, 1),
        ('除号', '÷', '\\div', '运算符号', '除法运算符号', '1-12', 1, 1),
        ('等号', '=', '=', '关系符号', '等于关系符号', '1-12', 1, 1),
        ('大于号', '>', '>', '关系符号', '大于关系符号', '1-12', 2, 1),
        ('小于号', '<', '<', '关系符号', '小于关系符号', '1-12', 2, 1),
        ('平方根', '√', '\\sqrt{}', '函数符号', '平方根符号', '6-12', 4, 0),
        ('分数线', '/', '\\frac{}{}', '运算符号', '分数表示符号', '3-12', 2, 1)
        """)
        
        print("   初始数据插入完成")
        
        # 12. 启用外键检查
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        # 13. 检查结果
        print("\n12. 检查数据库状态...")
        
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        homework_tables = [table[0] for table in tables if table[0] in [
            'homeworks', 'questions', 'homework_assignments', 'homework_submissions', 
            'question_answers', 'knowledge_points', 'question_knowledge_points', 'math_symbols'
        ]]
        print(f"   作业相关表: {homework_tables}")
        
        # 检查数据
        cursor.execute("SELECT COUNT(*) FROM knowledge_points")
        kp_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM math_symbols")
        symbol_count = cursor.fetchone()[0]
        print(f"   知识点数量: {kp_count}")
        print(f"   数学符号数量: {symbol_count}")
        
        print("\n🎉 作业管理系统数据库创建成功！")
        print("现在可以开始实现Story 2.1 作业创建功能")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据库创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_homework_schema()

