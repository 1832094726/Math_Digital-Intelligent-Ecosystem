# -*- coding: utf-8 -*-
"""
创建最基础的表结构
"""
import pymysql
from config import config

def create_basic_tables():
    """创建最基础的表结构"""
    
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
        print("=== 创建最基础的作业管理表 ===\n")
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        # 1. 创建题目表 (最简版本)
        print("1. 创建简化题目表...")
        cursor.execute("DROP TABLE IF EXISTS `questions`")
        cursor.execute("""
        CREATE TABLE `questions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `question_order` int(11) NOT NULL COMMENT '题目顺序',
          `question_type` varchar(20) NOT NULL COMMENT '题目类型',
          `question_title` varchar(500) NOT NULL COMMENT '题目标题',
          `question_content` text NOT NULL COMMENT '题目内容',
          `correct_answer` text NOT NULL COMMENT '正确答案',
          `score` int(11) NOT NULL DEFAULT '10' COMMENT '分值',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_homework_id` (`homework_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目详情表'
        """)
        print("   ✅ questions 表创建成功")
        
        # 2. 创建作业提交表 (最简版本)
        print("\n2. 创建简化作业提交表...")
        cursor.execute("DROP TABLE IF EXISTS `homework_submissions`")
        cursor.execute("""
        CREATE TABLE `homework_submissions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '提交ID',
          `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
          `student_id` bigint(20) NOT NULL COMMENT '学生ID',
          `submission_status` varchar(20) NOT NULL DEFAULT 'draft' COMMENT '提交状态',
          `total_score` decimal(5,2) DEFAULT NULL COMMENT '总得分',
          `max_score` decimal(5,2) NOT NULL COMMENT '满分',
          `submitted_at` datetime DEFAULT NULL COMMENT '提交时间',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_homework_id` (`homework_id`),
          KEY `idx_student_id` (`student_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提交记录表'
        """)
        print("   ✅ homework_submissions 表创建成功")
        
        # 3. 创建答题记录表 (最简版本)
        print("\n3. 创建简化答题记录表...")
        cursor.execute("DROP TABLE IF EXISTS `question_answers`")
        cursor.execute("""
        CREATE TABLE `question_answers` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '答题记录ID',
          `submission_id` bigint(20) NOT NULL COMMENT '提交ID',
          `question_id` bigint(20) NOT NULL COMMENT '题目ID',
          `student_answer` text NOT NULL COMMENT '学生答案',
          `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确',
          `score` decimal(5,2) DEFAULT NULL COMMENT '得分',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_submission_id` (`submission_id`),
          KEY `idx_question_id` (`question_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学生答题记录表'
        """)
        print("   ✅ question_answers 表创建成功")
        
        # 4. 创建知识点表 (最简版本)
        print("\n4. 创建简化知识点表...")
        cursor.execute("DROP TABLE IF EXISTS `knowledge_points`")
        cursor.execute("""
        CREATE TABLE `knowledge_points` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '知识点ID',
          `name` varchar(100) NOT NULL COMMENT '知识点名称',
          `code` varchar(50) NOT NULL COMMENT '知识点编码',
          `subject` varchar(50) NOT NULL COMMENT '学科',
          `grade` int(11) NOT NULL COMMENT '年级',
          `category` varchar(50) DEFAULT NULL COMMENT '分类',
          `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_code` (`code`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点表'
        """)
        print("   ✅ knowledge_points 表创建成功")
        
        # 5. 创建知识点关系表 (最简版本)
        print("\n5. 创建简化知识点关系表...")
        cursor.execute("DROP TABLE IF EXISTS `knowledge_relations`")
        cursor.execute("""
        CREATE TABLE `knowledge_relations` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '关系ID',
          `source_kp_id` bigint(20) NOT NULL COMMENT '源知识点ID',
          `target_kp_id` bigint(20) NOT NULL COMMENT '目标知识点ID',
          `relation_type` varchar(20) NOT NULL COMMENT '关系类型',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_source_kp_id` (`source_kp_id`),
          KEY `idx_target_kp_id` (`target_kp_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点关系表'
        """)
        print("   ✅ knowledge_relations 表创建成功")
        
        # 6. 创建数学符号表 (最简版本)
        print("\n6. 创建简化数学符号表...")
        cursor.execute("DROP TABLE IF EXISTS `math_symbols`")
        cursor.execute("""
        CREATE TABLE `math_symbols` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '符号ID',
          `symbol_name` varchar(100) NOT NULL COMMENT '符号名称',
          `symbol_text` varchar(50) NOT NULL COMMENT '符号字符',
          `latex_code` varchar(200) DEFAULT NULL COMMENT 'LaTeX代码',
          `category` varchar(50) NOT NULL COMMENT '符号分类',
          `grade_range` varchar(20) DEFAULT NULL COMMENT '适用年级',
          `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级',
          `is_common` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否常用符号',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_symbol_text` (`symbol_text`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数学符号表'
        """)
        print("   ✅ math_symbols 表创建成功")
        
        # 7. 插入基础数据
        print("\n7. 插入基础数据...")
        
        # 插入知识点
        cursor.execute("""
        INSERT INTO `knowledge_points` (`name`, `code`, `subject`, `grade`, `category`, `difficulty_level`) VALUES
        ('整数运算', 'MATH_INT_CALC', '数学', 1, '数与代数', 1),
        ('分数运算', 'MATH_FRAC_CALC', '数学', 3, '数与代数', 3),
        ('小数运算', 'MATH_DEC_CALC', '数学', 4, '数与代数', 2),
        ('一元一次方程', 'MATH_LINEAR_EQ', '数学', 7, '数与代数', 4),
        ('平面几何基础', 'MATH_PLANE_GEO', '数学', 5, '图形与几何', 3)
        """)
        
        # 插入数学符号
        cursor.execute("""
        INSERT INTO `math_symbols` (`symbol_name`, `symbol_text`, `latex_code`, `category`, `grade_range`, `difficulty_level`, `is_common`) VALUES
        ('加号', '+', '+', '运算符号', '1-12', 1, 1),
        ('减号', '-', '-', '运算符号', '1-12', 1, 1),
        ('乘号', '×', '\\\\times', '运算符号', '1-12', 1, 1),
        ('除号', '÷', '\\\\div', '运算符号', '1-12', 1, 1),
        ('等号', '=', '=', '关系符号', '1-12', 1, 1)
        """)
        
        print("   基础数据插入完成")
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        # 8. 检查结果
        print("\n8. 检查结果...")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        existing_tables = [table[0] for table in tables]
        required_tables = ['questions', 'homework_submissions', 'question_answers', 'knowledge_points', 'knowledge_relations', 'math_symbols']
        
        missing = [t for t in required_tables if t not in existing_tables]
        if missing:
            print(f"   ❌ 仍缺少: {missing}")
        else:
            print("   ✅ 所有基础表创建成功!")
            
        # 检查数据量
        for table in required_tables:
            if table in existing_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   - {table}: {count} 条记录")
        
        print("\n🎉 基础作业管理系统创建成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_basic_tables()

