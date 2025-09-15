"""
创建学生作业相关数据库表
"""
import pymysql
from config import config

def create_student_homework_tables():
    """创建学生作业相关表"""
    try:
        # 获取当前配置
        current_config = config['development']
        
        # 连接数据库
        conn = pymysql.connect(
            host=current_config.DATABASE_CONFIG['host'],
            port=current_config.DATABASE_CONFIG['port'],
            user=current_config.DATABASE_CONFIG['user'],
            password=current_config.DATABASE_CONFIG['password'],
            database=current_config.DATABASE_CONFIG['database'],
            charset='utf8mb4',
            autocommit=True
        )
        
        cursor = conn.cursor()
        
        print("=== 创建学生作业相关表 ===")
        
        # 1. 创建homework_progress表
        print("\n1. 创建homework_progress表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS homework_progress (
          id bigint(20) NOT NULL AUTO_INCREMENT,
          student_id bigint(20) NOT NULL,
          homework_id bigint(20) NOT NULL,
          progress_data json DEFAULT NULL COMMENT '答题进度数据',
          completion_rate decimal(5,2) DEFAULT 0.00 COMMENT '完成率',
          last_saved_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (id),
          UNIQUE KEY uk_student_homework (student_id, homework_id),
          KEY idx_student_id (student_id),
          KEY idx_homework_id (homework_id),
          KEY idx_completion_rate (completion_rate)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业进度表'
        """)
        print("   ✅ homework_progress表创建成功")
        
        # 2. 创建homework_submissions表
        print("\n2. 创建homework_submissions表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS homework_submissions (
          id bigint(20) NOT NULL AUTO_INCREMENT,
          assignment_id bigint(20) NOT NULL COMMENT '作业分发ID',
          student_id bigint(20) NOT NULL,
          homework_id bigint(20) NOT NULL,
          answers json DEFAULT NULL COMMENT '学生答案',
          submission_data json DEFAULT NULL COMMENT '提交相关数据',
          submitted_at datetime DEFAULT NULL COMMENT '提交时间',
          score decimal(5,2) DEFAULT NULL COMMENT '分数',
          max_score decimal(5,2) DEFAULT 100.00 COMMENT '满分',
          status enum('draft','submitted','graded','returned') DEFAULT 'draft',
          time_spent int DEFAULT 0 COMMENT '用时(秒)',
          attempt_count int DEFAULT 1 COMMENT '尝试次数',
          teacher_comments text COMMENT '教师评语',
          auto_grade_data json DEFAULT NULL COMMENT '自动评分数据',
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (id),
          UNIQUE KEY uk_assignment_student (assignment_id, student_id),
          KEY idx_student_id (student_id),
          KEY idx_homework_id (homework_id),
          KEY idx_submitted_at (submitted_at),
          KEY idx_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提交表'
        """)
        print("   ✅ homework_submissions表创建成功")
        
        # 3. 创建homework_favorites表
        print("\n3. 创建homework_favorites表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS homework_favorites (
          id bigint(20) NOT NULL AUTO_INCREMENT,
          student_id bigint(20) NOT NULL,
          assignment_id bigint(20) NOT NULL COMMENT '作业分发ID',
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (id),
          UNIQUE KEY uk_student_assignment (student_id, assignment_id),
          KEY idx_student_id (student_id),
          KEY idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业收藏表'
        """)
        print("   ✅ homework_favorites表创建成功")
        
        # 4. 创建homework_reminders表
        print("\n4. 创建homework_reminders表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS homework_reminders (
          id bigint(20) NOT NULL AUTO_INCREMENT,
          student_id bigint(20) NOT NULL,
          assignment_id bigint(20) NOT NULL,
          reminder_type enum('due_soon','overdue','custom') DEFAULT 'due_soon',
          reminder_time datetime NOT NULL,
          message text,
          is_sent tinyint(1) DEFAULT 0,
          sent_at datetime DEFAULT NULL,
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (id),
          KEY idx_student_id (student_id),
          KEY idx_reminder_time (reminder_time),
          KEY idx_is_sent (is_sent)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业提醒表'
        """)
        print("   ✅ homework_reminders表创建成功")
        
        # 5. 插入测试数据
        print("\n5. 插入测试数据...")
        
        # 检查是否有现有的作业分发
        cursor.execute("SELECT COUNT(*) as count FROM homework_assignments")
        assignment_count = cursor.fetchone()[0]
        
        if assignment_count > 0:
            # 为test_student_001创建一些测试进度
            cursor.execute("""
            INSERT IGNORE INTO homework_progress (student_id, homework_id, progress_data, completion_rate)
            SELECT 2, ha.homework_id, 
                   JSON_OBJECT('answers', JSON_OBJECT('q1', '答案1', 'q2', ''), 'start_time', NOW()),
                   50.0
            FROM homework_assignments ha
            LIMIT 1
            """)
            
            print("   ✅ 测试进度数据插入完成")
        
        # 6. 检查创建结果
        print("\n6. 检查创建结果...")
        
        tables = ['homework_progress', 'homework_submissions', 'homework_favorites', 'homework_reminders']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   ✅ {table}: {count} 条记录")
        
        conn.close()
        
        print("\n🎉 学生作业相关表创建成功！")
        print("现在可以开始实现Student Homework API功能")
        
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_student_homework_tables()

