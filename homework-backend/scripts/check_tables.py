#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据库表结构
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def check_tables():
    """检查数据库表结构"""
    print("🔍 检查数据库表结构...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 查看所有表
                print("\n📋 数据库中的所有表:")
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                table_names = [list(table.values())[0] for table in tables]
                
                for i, table_name in enumerate(table_names, 1):
                    print(f"  {i:2d}. {table_name}")
                
                # 2. 检查questions相关的表
                print("\n🔍 检查questions相关表:")
                questions_tables = [t for t in table_names if 'question' in t.lower()]
                
                for table in questions_tables:
                    print(f"\n📝 表: {table}")
                    cursor.execute(f"DESCRIBE {table}")
                    columns = cursor.fetchall()
                    
                    for col in columns:
                        field = col['Field']
                        type_info = col['Type']
                        null_info = col['Null']
                        key_info = col['Key']
                        print(f"    {field:<20} {type_info:<30} {null_info:<5} {key_info}")
                
                # 3. 检查是否有数据
                print("\n📊 表数据统计:")
                for table in questions_tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    count = cursor.fetchone()['count']
                    print(f"  {table}: {count} 条记录")
                
                # 4. 检查homeworks表
                if 'homeworks' in table_names:
                    print(f"\n📚 homeworks表:")
                    cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                    count = cursor.fetchone()['count']
                    print(f"  homeworks: {count} 条记录")
                
                return table_names
                
    except Exception as e:
        print(f"❌ 检查表结构失败: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    check_tables()
