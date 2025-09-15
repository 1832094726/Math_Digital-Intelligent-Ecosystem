#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据库表结构脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
import sqlite3

def check_database_schema():
    """检查数据库中的所有表结构"""
    try:
        # 使用数据库管理器
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # 获取所有表名 (MySQL语法)
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()

                print('=== 数据库中的所有表 ===')
                print(f'总共 {len(tables)} 个表')

                for table_dict in tables:
                    # MySQL返回的是字典，获取表名
                    table_name = list(table_dict.values())[0]
                    print(f'\n表名: {table_name}')

                    # 获取表结构 (MySQL语法)
                    cursor.execute(f'DESCRIBE {table_name}')
                    columns = cursor.fetchall()

                    print('字段信息:')
                    for col in columns:
                        field = col['Field']
                        type_name = col['Type']
                        null = col['Null']
                        key = col['Key']
                        default = col['Default']
                        extra = col['Extra']

                        constraints = []
                        if null == 'NO':
                            constraints.append("NOT NULL")
                        if key == 'PRI':
                            constraints.append("PRIMARY KEY")
                        if key == 'UNI':
                            constraints.append("UNIQUE")
                        if default is not None:
                            constraints.append(f"DEFAULT {default}")
                        if extra:
                            constraints.append(extra)

                        constraint_str = " " + " ".join(constraints) if constraints else ""
                        print(f'  {field} {type_name}{constraint_str}')

                    # 获取索引信息
                    cursor.execute(f'SHOW INDEX FROM {table_name}')
                    indexes = cursor.fetchall()
                    if indexes:
                        print('索引信息:')
                        index_names = set()
                        for idx in indexes:
                            index_name = idx['Key_name']
                            if index_name not in index_names:
                                index_names.add(index_name)
                                non_unique = idx['Non_unique']
                                print(f'  {index_name} (unique: {not bool(non_unique)})')

        return True
        
    except Exception as e:
        print(f"检查数据库失败: {e}")
        return False

if __name__ == "__main__":
    check_database_schema()
