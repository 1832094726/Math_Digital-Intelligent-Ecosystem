# -*- coding: utf-8 -*-
"""
数据库初始化脚本
检查数据库连接并创建必要的表结构
"""
import pymysql
from config import Config
import sys

def test_database_connection():
    """测试数据库连接"""
    try:
        connection = pymysql.connect(
            host=Config.DATABASE_CONFIG['host'],
            port=Config.DATABASE_CONFIG['port'],
            user=Config.DATABASE_CONFIG['user'],
            password=Config.DATABASE_CONFIG['password'],
            database=Config.DATABASE_CONFIG['database'],
            charset=Config.DATABASE_CONFIG['charset']
        )
        
        print("✅ 数据库连接成功")
        
        # 测试查询
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"数据库版本: {version[0]}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def check_required_tables():
    """检查必需的表是否存在"""
    required_tables = [
        'users',
        'user_sessions',
        'homeworks',
        'questions',
        'homework_submissions',
        'question_answers',
        'knowledge_points',
        'knowledge_relations',
        'math_symbols'
    ]
    
    try:
        connection = pymysql.connect(
            host=Config.DATABASE_CONFIG['host'],
            port=Config.DATABASE_CONFIG['port'],
            user=Config.DATABASE_CONFIG['user'],
            password=Config.DATABASE_CONFIG['password'],
            database=Config.DATABASE_CONFIG['database'],
            charset=Config.DATABASE_CONFIG['charset']
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            existing_tables = [table[0] for table in cursor.fetchall()]
            
        print(f"数据库中现有表: {existing_tables}")
        
        missing_tables = []
        for table in required_tables:
            if table not in existing_tables:
                missing_tables.append(table)
        
        if missing_tables:
            print(f"❌ 缺少以下必需的表: {missing_tables}")
            print("请运行architect/04_数据模型设计.sql中的SQL脚本创建表结构")
            return False
        else:
            print("✅ 所有必需的表都存在")
            return True
            
        connection.close()
        
    except Exception as e:
        print(f"❌ 检查表结构失败: {e}")
        return False

def test_create_user():
    """测试创建用户功能"""
    try:
        from models.user import User
        
        # 测试用户数据
        test_user_data = {
            'username': 'init_test_user',
            'email': 'init_test@example.com',
            'password': 'TestPassword123!',
            'real_name': '初始化测试用户',
            'role': 'student',
            'grade': 7,
            'school': '测试学校'
        }
        
        # 检查用户是否已存在
        existing_user = User.get_by_username(test_user_data['username'])
        if existing_user:
            print("测试用户已存在，跳过创建")
            return True
        
        # 创建测试用户
        user = User.create(test_user_data)
        print(f"✅ 成功创建测试用户: {user.username} (ID: {user.id})")
        return True
        
    except Exception as e:
        print(f"❌ 创建测试用户失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 数据库初始化检查 ===\n")
    
    # 1. 测试数据库连接
    print("1. 测试数据库连接...")
    if not test_database_connection():
        print("数据库连接失败，请检查配置")
        sys.exit(1)
    
    print()
    
    # 2. 检查表结构
    print("2. 检查数据库表结构...")
    if not check_required_tables():
        print("数据库表结构不完整")
        sys.exit(1)
    
    print()
    
    # 3. 测试用户创建
    print("3. 测试用户创建功能...")
    if not test_create_user():
        print("用户创建功能测试失败")
        sys.exit(1)
    
    print("\n=== 数据库初始化检查完成 ===")
    print("✅ 所有检查项目都通过，系统可以正常运行")

if __name__ == "__main__":
    main()
