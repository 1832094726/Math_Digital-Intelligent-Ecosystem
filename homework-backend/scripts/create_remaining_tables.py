# -*- coding: utf-8 -*-
"""
创建剩余的表（使用简单结构）
"""
import pymysql
from config import config

def create_remaining_tables():
    """创建剩余需要的表"""
    
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
        print("=== 创建剩余的表（简单版本）===\n")
        
        # 1. 创建questions表
        print("1. 创建questions表...")
        cursor.execute("DROP TABLE IF EXISTS questions")
        cursor.execute("""
        CREATE TABLE questions (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          homework_id bigint NOT NULL,
          question_order int NOT NULL DEFAULT 1,
          question_type varchar(20) NOT NULL DEFAULT 'choice',
          question_title varchar(500) NOT NULL,
          question_content text NOT NULL,
          question_image varchar(255) DEFAULT NULL,
          correct_answer text NOT NULL,
          answer_analysis text DEFAULT NULL,
          score int NOT NULL DEFAULT 10,
          difficulty int NOT NULL DEFAULT 3,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          INDEX idx_homework_id (homework_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目详情表'
        """)
        print("   ✅ questions表创建成功")
        
        # 2. 创建homework_submissions表
        print("\n2. 创建homework_submissions表...")
        cursor.execute("DROP TABLE IF EXISTS homework_submissions")
        cursor.execute("""
        CREATE TABLE homework_submissions (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          homework_id bigint NOT NULL,
          student_id bigint NOT NULL,
          submission_status varchar(20) NOT NULL DEFAULT 'draft',
          total_score decimal(5,2) DEFAULT NULL,
          max_score decimal(5,2) NOT NULL,
          completion_rate decimal(5,2) DEFAULT NULL,
          time_spent int DEFAULT NULL,
          submit_count int NOT NULL DEFAULT 0,
          teacher_comment text DEFAULT NULL,
          submitted_at datetime DEFAULT NULL,
          graded_at datetime DEFAULT NULL,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          UNIQUE KEY uk_homework_student (homework_id, student_id),
          INDEX idx_student_id (student_id),
          INDEX idx_submission_status (submission_status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提交记录表'
        """)
        print("   ✅ homework_submissions表创建成功")
        
        # 3. 创建question_answers表
        print("\n3. 创建question_answers表...")
        cursor.execute("DROP TABLE IF EXISTS question_answers")
        cursor.execute("""
        CREATE TABLE question_answers (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          submission_id bigint NOT NULL,
          question_id bigint NOT NULL,
          student_answer text NOT NULL,
          answer_process text DEFAULT NULL,
          answer_time int DEFAULT NULL,
          is_correct tinyint(1) DEFAULT NULL,
          score decimal(5,2) DEFAULT NULL,
          feedback text DEFAULT NULL,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          UNIQUE KEY uk_submission_question (submission_id, question_id),
          INDEX idx_question_id (question_id),
          INDEX idx_is_correct (is_correct)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学生答题记录表'
        """)
        print("   ✅ question_answers表创建成功")
        
        # 4. 创建knowledge_points表
        print("\n4. 创建knowledge_points表...")
        cursor.execute("DROP TABLE IF EXISTS knowledge_points")
        cursor.execute("""
        CREATE TABLE knowledge_points (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          name varchar(100) NOT NULL,
          code varchar(50) NOT NULL UNIQUE,
          description text DEFAULT NULL,
          subject varchar(50) NOT NULL,
          grade int NOT NULL,
          category varchar(50) DEFAULT NULL,
          difficulty_level int NOT NULL DEFAULT 3,
          parent_id bigint DEFAULT NULL,
          sort_order int NOT NULL DEFAULT 0,
          is_core tinyint(1) NOT NULL DEFAULT 0,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          INDEX idx_subject_grade (subject, grade),
          INDEX idx_parent_id (parent_id),
          INDEX idx_category (category)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点表'
        """)
        print("   ✅ knowledge_points表创建成功")
        
        # 5. 创建knowledge_relations表
        print("\n5. 创建knowledge_relations表...")
        cursor.execute("DROP TABLE IF EXISTS knowledge_relations")
        cursor.execute("""
        CREATE TABLE knowledge_relations (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          source_kp_id bigint NOT NULL,
          target_kp_id bigint NOT NULL,
          relation_type varchar(20) NOT NULL,
          strength decimal(3,2) NOT NULL DEFAULT 1.00,
          description varchar(200) DEFAULT NULL,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_source_target_type (source_kp_id, target_kp_id, relation_type),
          INDEX idx_target_kp_id (target_kp_id),
          INDEX idx_relation_type (relation_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点关系表'
        """)
        print("   ✅ knowledge_relations表创建成功")
        
        # 6. 创建math_symbols表
        print("\n6. 创建math_symbols表...")
        cursor.execute("DROP TABLE IF EXISTS math_symbols")
        cursor.execute("""
        CREATE TABLE math_symbols (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          symbol_name varchar(100) NOT NULL,
          symbol_text varchar(50) NOT NULL UNIQUE,
          latex_code varchar(200) DEFAULT NULL,
          unicode varchar(20) DEFAULT NULL,
          category varchar(50) NOT NULL,
          subcategory varchar(50) DEFAULT NULL,
          description text DEFAULT NULL,
          grade_range varchar(20) DEFAULT NULL,
          difficulty_level int NOT NULL DEFAULT 3,
          frequency_score decimal(5,2) NOT NULL DEFAULT 0.00,
          is_common tinyint(1) NOT NULL DEFAULT 0,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          INDEX idx_category (category),
          INDEX idx_grade_range (grade_range),
          INDEX idx_difficulty_level (difficulty_level)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数学符号表'
        """)
        print("   ✅ math_symbols表创建成功")
        
        # 7. 插入基础数据
        print("\n7. 插入基础数据...")
        
        # 插入知识点数据
        cursor.execute("""
        INSERT INTO knowledge_points (name, code, description, subject, grade, category, difficulty_level, is_core) VALUES
        ('整数运算', 'MATH_INT_CALC', '整数的加减乘除运算', '数学', 1, '数与代数', 1, 1),
        ('分数运算', 'MATH_FRAC_CALC', '分数的加减乘除运算', '数学', 3, '数与代数', 3, 1),
        ('小数运算', 'MATH_DEC_CALC', '小数的加减乘除运算', '数学', 4, '数与代数', 2, 1),
        ('一元一次方程', 'MATH_LINEAR_EQ', '一元一次方程的解法', '数学', 7, '数与代数', 4, 1),
        ('平面几何基础', 'MATH_PLANE_GEO', '点线面角的基本概念', '数学', 5, '图形与几何', 3, 1)
        """)
        
        # 插入数学符号数据
        cursor.execute("""
        INSERT INTO math_symbols (symbol_name, symbol_text, latex_code, category, description, grade_range, difficulty_level, is_common) VALUES
        ('加号', '+', '+', '运算符号', '加法运算符号', '1-12', 1, 1),
        ('减号', '-', '-', '运算符号', '减法运算符号', '1-12', 1, 1),
        ('乘号', '×', '\\\\times', '运算符号', '乘法运算符号', '1-12', 1, 1),
        ('除号', '÷', '\\\\div', '运算符号', '除法运算符号', '1-12', 1, 1),
        ('等号', '=', '=', '关系符号', '等于关系符号', '1-12', 1, 1),
        ('大于号', '>', '>', '关系符号', '大于关系符号', '1-12', 2, 1),
        ('小于号', '<', '<', '关系符号', '小于关系符号', '1-12', 2, 1),
        ('平方根', '√', '\\\\sqrt{}', '函数符号', '平方根符号', '6-12', 4, 0),
        ('分数线', '/', '\\\\frac{}{}', '运算符号', '分数表示符号', '3-12', 2, 1)
        """)
        
        print("   基础数据插入完成")
        
        # 8. 验证创建结果
        print("\n8. 验证创建结果...")
        
        required_tables = ['questions', 'homework_submissions', 'question_answers', 'knowledge_points', 'knowledge_relations', 'math_symbols']
        
        for table in required_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   ✅ {table}: {count} 条记录")
        
        print("\n🎉 所有作业管理表创建成功！")
        print("现在可以运行 python init_database.py 进行验证")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_remaining_tables()

