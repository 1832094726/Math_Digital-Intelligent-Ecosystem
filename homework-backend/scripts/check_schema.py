# -*- coding: utf-8 -*-
"""
检查数据库表结构
"""
import pymysql
from config import config

def check_table_structure():
    """检查表结构"""
    
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
        print("=== 检查数据库表结构 ===\n")
        
        # 检查homeworks表结构
        print("1. 检查 homeworks 表结构:")
        try:
            cursor.execute("DESC homeworks")
            columns = cursor.fetchall()
            for col in columns:
                print(f"   {col[0]}: {col[1]} {col[2]} {col[3]} {col[4]} {col[5]}")
        except Exception as e:
            print(f"   homeworks 表不存在: {e}")
        
        print()
        
        # 检查表的字符集和排序规则
        print("2. 检查表的字符集:")
        cursor.execute("""
        SELECT 
            TABLE_NAME,
            ENGINE,
            TABLE_COLLATION
        FROM information_schema.TABLES 
        WHERE TABLE_SCHEMA = 'testccnu' 
        AND TABLE_NAME IN ('homeworks', 'questions')
        """)
        
        tables_info = cursor.fetchall()
        for table_info in tables_info:
            print(f"   {table_info[0]}: {table_info[1]}, {table_info[2]}")
        
        print()
        
        # 检查列的字符集
        print("3. 检查 homeworks 表 id 列的详细信息:")
        cursor.execute("""
        SELECT 
            COLUMN_NAME,
            COLUMN_TYPE,
            CHARACTER_SET_NAME,
            COLLATION_NAME,
            COLUMN_KEY
        FROM information_schema.COLUMNS 
        WHERE TABLE_SCHEMA = 'testccnu' 
        AND TABLE_NAME = 'homeworks'
        AND COLUMN_NAME = 'id'
        """)
        
        id_info = cursor.fetchall()
        for info in id_info:
            print(f"   {info[0]}: {info[1]}, charset={info[2]}, collation={info[3]}, key={info[4]}")
        
        print()
        
        # 尝试手动创建外键约束
        print("4. 尝试手动创建外键约束:")
        
        # 先创建一个简单的questions表
        cursor.execute("DROP TABLE IF EXISTS test_questions")
        cursor.execute("""
        CREATE TABLE `test_questions` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT,
          `homework_id` bigint(20) NOT NULL,
          `title` varchar(100),
          PRIMARY KEY (`id`),
          KEY `idx_homework_id` (`homework_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("   test_questions 表创建成功")
        
        # 添加外键约束
        try:
            cursor.execute("""
            ALTER TABLE `test_questions` 
            ADD CONSTRAINT `fk_test_questions_homework_id` 
            FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE
            """)
            print("   ✅ 外键约束添加成功")
        except Exception as e:
            print(f"   ❌ 外键约束添加失败: {e}")
            
            # 显示详细的错误信息
            print("\n   检查参考表和列是否存在:")
            try:
                cursor.execute("SELECT id FROM homeworks LIMIT 1")
                print("     homeworks.id 列可以查询")
            except Exception as e2:
                print(f"     homeworks.id 列查询失败: {e2}")
        
        # 检查OceanBase版本和兼容性
        print("\n5. 检查数据库版本信息:")
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"   数据库版本: {version[0]}")
        
        # 检查是否支持外键
        cursor.execute("SHOW VARIABLES LIKE 'foreign_key_checks'")
        fk_check = cursor.fetchone()
        if fk_check:
            print(f"   外键检查状态: {fk_check[1]}")
        
        cursor.execute("SHOW VARIABLES LIKE 'sql_mode'")
        sql_mode = cursor.fetchone()
        if sql_mode:
            print(f"   SQL模式: {sql_mode[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    check_table_structure()

