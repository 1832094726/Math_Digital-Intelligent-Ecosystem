#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入测试数据
"""
import mysql.connector
import json
from datetime import datetime, timedelta

def import_test_data():
    """导入测试数据"""
    print("🚀 开始导入测试数据...")
    
    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',
            database='homework_system',
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        
        print("✅ 数据库连接成功")
        
        # 读取并执行SQL文件
        sql_file = 'homework-backend/scripts/insert_test_data.sql'
        print(f"📖 读取SQL文件: {sql_file}")
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # 分割SQL语句并执行
        sql_statements = []
        current_statement = ""
        
        for line in sql_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                current_statement += line + " "
                if line.endswith(';'):
                    sql_statements.append(current_statement.strip())
                    current_statement = ""
        
        print(f"📝 准备执行 {len(sql_statements)} 条SQL语句")
        
        success_count = 0
        for i, statement in enumerate(sql_statements):
            if statement:
                try:
                    cursor.execute(statement)
                    success_count += 1
                    if i % 10 == 0:  # 每10条显示一次进度
                        print(f"⏳ 已执行 {i+1}/{len(sql_statements)} 条语句")
                except Exception as e:
                    if "Duplicate entry" in str(e) or "already exists" in str(e):
                        print(f"⚠️ 语句 {i+1} 数据已存在，跳过")
                        success_count += 1
                    else:
                        print(f"❌ 语句 {i+1} 执行失败: {e}")
                        print(f"SQL: {statement[:100]}...")
        
        conn.commit()
        print(f"✅ 成功执行 {success_count}/{len(sql_statements)} 条SQL语句")
        
        # 显示统计信息
        print("\n📊 数据导入统计:")
        
        tables_to_check = [
            ('users', '用户'),
            ('schools', '学校'),
            ('classes', '班级'),
            ('homeworks', '作业'),
            ('questions', '题目'),
            ('homework_assignments', '作业分配'),
            ('homework_submissions', '学生提交'),
            ('knowledge_points', '知识点')
        ]
        
        for table, name in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  {name}: {count} 条记录")
            except Exception as e:
                print(f"  {name}: 表不存在或查询失败")
        
        # 检查questions表结构
        print("\n🔍 检查questions表结构:")
        try:
            cursor.execute("DESCRIBE questions")
            columns = cursor.fetchall()
            print("  字段列表:")
            for col in columns:
                print(f"    {col[0]:<20} {col[1]:<30}")
        except Exception as e:
            print(f"  ❌ 无法查看questions表结构: {e}")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 数据导入完成！")
        return True
        
    except Exception as e:
        print(f"❌ 数据导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import_test_data()
