# -*- coding: utf-8 -*-
"""
创建认证系统的最小数据库结构
"""
import pymysql
from config import config

def create_auth_schema():
    """创建认证系统数据库结构"""
    
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
        print("=== 创建认证系统数据库 ===\n")
        
        # 1. 清理现有表
        print("1. 清理现有表...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        tables_to_drop = ['user_sessions', 'users']
        for table in tables_to_drop:
            cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
            
        print("   现有表已清理")
        
        # 2. 创建用户表
        print("\n2. 创建用户表...")
        cursor.execute("""
        CREATE TABLE `users` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
          `username` varchar(50) NOT NULL COMMENT '用户名',
          `email` varchar(100) NOT NULL COMMENT '邮箱',
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户基础信息表'
        """)
        print("   ✅ users 表创建成功")
        
        # 3. 创建用户会话表
        print("\n3. 创建用户会话表...")
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
          KEY `idx_user_id` (`user_id`),
          KEY `idx_expires_at` (`expires_at`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户会话表'
        """)
        print("   ✅ user_sessions 表创建成功")
        
        # 4. 添加外键约束
        print("\n4. 添加外键约束...")
        cursor.execute("""
        ALTER TABLE `user_sessions` 
        ADD CONSTRAINT `fk_user_sessions_user_id` 
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
        """)
        print("   ✅ 外键约束添加成功")
        
        # 5. 插入管理员用户
        print("\n5. 插入管理员用户...")
        cursor.execute("""
        INSERT INTO `users` (`username`, `email`, `password_hash`, `role`, `real_name`, `is_active`) VALUES
        ('admin', 'admin@diem.edu', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewTH.iVEP0T/UEaa', 'admin', '系统管理员', 1)
        """)
        print("   ✅ 管理员用户创建成功")
        
        # 6. 启用外键检查
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        # 7. 检查结果
        print("\n6. 检查数据库状态...")
        
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"   创建的表: {[table[0] for table in tables]}")
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"   用户数量: {user_count}")
        
        print("\n🎉 认证系统数据库创建成功！")
        print("现在可以运行以下测试:")
        print("  python test_auth.py        # 测试认证API")
        print("  python app.py              # 启动应用服务器")
        
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
    create_auth_schema()

