# -*- coding: utf-8 -*-
"""
检查现有表结构
"""
import pymysql
from config import config

def check_existing_tables():
    """检查现有表结构"""
    
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
        print("=== 检查现有表结构 ===\n")
        
        # 检查homeworks表
        print("1. homeworks表结构:")
        cursor.execute("DESC homeworks")
        columns = cursor.fetchall()
        for col in columns:
            print(f"   {col[0]}: {col[1]} {col[2]} {col[3]} {col[4]} {col[5]}")
            
        print(f"\n2. homeworks表数据:")
        cursor.execute("SELECT * FROM homeworks LIMIT 3")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(f"   {row}")
        else:
            print("   (无数据)")
        
        # 检查是否有questions表
        print(f"\n3. 检查是否存在questions表:")
        try:
            cursor.execute("DESC questions")
            print("   ✅ questions表存在")
        except:
            print("   ❌ questions表不存在")
            
        # 检查users表
        print(f"\n4. users表中的用户:")
        cursor.execute("SELECT id, username, email, role FROM users LIMIT 5")
        users = cursor.fetchall()
        for user in users:
            print(f"   ID:{user[0]} 用户名:{user[1]} 邮箱:{user[2]} 角色:{user[3]}")
            
        # 尝试创建一个简单的questions表
        print(f"\n5. 尝试创建简单的questions表:")
        try:
            cursor.execute("DROP TABLE IF EXISTS simple_questions")
            cursor.execute("""
            CREATE TABLE simple_questions (
              id int AUTO_INCREMENT PRIMARY KEY,
              homework_id int,
              title varchar(200),
              content text,
              answer text,
              score int DEFAULT 10,
              created_at datetime DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            print("   ✅ 简单questions表创建成功")
        except Exception as e:
            print(f"   ❌ 创建失败: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    check_existing_tables()

