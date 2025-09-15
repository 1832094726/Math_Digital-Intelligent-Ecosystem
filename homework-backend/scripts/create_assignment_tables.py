# -*- coding: utf-8 -*-
"""
创建作业分发相关的数据库表
"""
import pymysql
from config import config

def create_assignment_tables():
    """创建作业分发相关表"""
    
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
        print("=== 创建作业分发相关表 ===\n")
        
        # 1. 创建作业分发表
        print("1. 创建homework_assignments表...")
        cursor.execute("DROP TABLE IF EXISTS homework_assignments")
        cursor.execute("""
        CREATE TABLE homework_assignments (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          homework_id bigint NOT NULL COMMENT '作业ID',
          assigned_to_type enum('student','class','grade','all') NOT NULL COMMENT '分发对象类型',
          assigned_to_id bigint DEFAULT NULL COMMENT '分发对象ID',
          assigned_by bigint NOT NULL COMMENT '分发者ID',
          assigned_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '分发时间',
          due_date_override datetime DEFAULT NULL COMMENT '覆盖截止时间',
          start_date_override datetime DEFAULT NULL COMMENT '覆盖开始时间',
          is_active tinyint(1) NOT NULL DEFAULT 1 COMMENT '是否有效',
          notes text DEFAULT NULL COMMENT '分发备注',
          max_attempts_override int DEFAULT NULL COMMENT '覆盖最大提交次数',
          notification_sent tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否已发送通知',
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          INDEX idx_homework_id (homework_id),
          INDEX idx_assigned_to (assigned_to_type, assigned_to_id),
          INDEX idx_assigned_by (assigned_by),
          INDEX idx_is_active (is_active)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作业分发记录表'
        """)
        print("   ✅ homework_assignments表创建成功")
        
        # 2. 创建班级学生关联表（简化版）
        print("\n2. 创建class_students表...")
        cursor.execute("DROP TABLE IF EXISTS class_students")
        cursor.execute("""
        CREATE TABLE class_students (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          class_id bigint NOT NULL COMMENT '班级ID',
          student_id bigint NOT NULL COMMENT '学生ID',
          student_number varchar(50) DEFAULT NULL COMMENT '学号',
          joined_at datetime DEFAULT CURRENT_TIMESTAMP COMMENT '加入时间',
          is_active tinyint(1) NOT NULL DEFAULT 1 COMMENT '是否有效',
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_class_student (class_id, student_id),
          INDEX idx_class_id (class_id),
          INDEX idx_student_id (student_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='班级学生关联表'
        """)
        print("   ✅ class_students表创建成功")
        
        # 3. 创建通知消息表
        print("\n3. 创建notifications表...")
        cursor.execute("DROP TABLE IF EXISTS notifications")
        cursor.execute("""
        CREATE TABLE notifications (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          user_id bigint NOT NULL COMMENT '接收用户ID',
          notification_type varchar(50) NOT NULL COMMENT '通知类型',
          title varchar(200) NOT NULL COMMENT '通知标题',
          content text NOT NULL COMMENT '通知内容',
          related_type varchar(50) DEFAULT NULL COMMENT '关联资源类型',
          related_id bigint DEFAULT NULL COMMENT '关联资源ID',
          is_read tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否已读',
          priority enum('low','normal','high','urgent') NOT NULL DEFAULT 'normal' COMMENT '优先级',
          expires_at datetime DEFAULT NULL COMMENT '过期时间',
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          INDEX idx_user_id (user_id),
          INDEX idx_notification_type (notification_type),
          INDEX idx_is_read (is_read),
          INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='通知消息表'
        """)
        print("   ✅ notifications表创建成功")
        
        # 4. 创建班级表（如果不存在）
        print("\n4. 创建classes表（如果不存在）...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS classes (
          id bigint AUTO_INCREMENT PRIMARY KEY,
          class_name varchar(100) NOT NULL COMMENT '班级名称',
          class_code varchar(50) NOT NULL COMMENT '班级代码',
          grade_level int NOT NULL COMMENT '年级',
          school_name varchar(100) DEFAULT NULL COMMENT '学校名称',
          head_teacher_id bigint DEFAULT NULL COMMENT '班主任ID',
          student_count int NOT NULL DEFAULT 0 COMMENT '学生数',
          is_active tinyint(1) NOT NULL DEFAULT 1 COMMENT '是否有效',
          created_at datetime DEFAULT CURRENT_TIMESTAMP,
          updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          UNIQUE KEY uk_class_code (class_code),
          INDEX idx_grade_level (grade_level),
          INDEX idx_head_teacher_id (head_teacher_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='班级信息表'
        """)
        print("   ✅ classes表检查/创建完成")
        
        # 5. 插入测试数据
        print("\n5. 插入测试数据...")
        
        # 首先插入学校数据
        cursor.execute("""
        INSERT IGNORE INTO schools (id, school_name, school_code, school_type, education_level) VALUES
        (1, '演示中学', 'DEMO_MS', 'middle', '初中')
        """)
        print("   ✅ 学校数据插入完成")
        
        # 插入年级数据
        cursor.execute("""
        INSERT IGNORE INTO grades (id, school_id, grade_name, grade_level, academic_year) VALUES
        (1, 1, '七年级', 7, '2024-2025'),
        (2, 1, '八年级', 8, '2024-2025')
        """)
        print("   ✅ 年级数据插入完成")
        
        # 插入测试班级（适配现有表结构）
        cursor.execute("""
        INSERT IGNORE INTO classes (class_name, class_code, school_id, grade_id, head_teacher_id) VALUES
        ('七年级1班', 'G7C1', 1, 1, 3),
        ('七年级2班', 'G7C2', 1, 1, 3),
        ('八年级1班', 'G8C1', 1, 2, 3)
        """)
        
        # 插入班级学生关联（假设student_id=2是学生）
        cursor.execute("""
        INSERT IGNORE INTO class_students (class_id, student_id, student_number) VALUES
        (1, 2, '20240001'),
        (1, 2, '20240001')
        """)
        
        print("   测试数据插入完成")
        
        # 6. 检查结果
        print("\n6. 检查创建结果...")
        
        assignment_tables = ['homework_assignments', 'class_students', 'notifications', 'classes']
        
        for table in assignment_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   ✅ {table}: {count} 条记录")
        
        print("\n🎉 作业分发相关表创建成功！")
        print("现在可以开始实现Story 2.2作业分发功能")
        
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
    create_assignment_tables()
