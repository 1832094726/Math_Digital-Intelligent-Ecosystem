#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据库用户脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def check_users():
    """检查数据库中的用户"""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, username, role FROM users LIMIT 5')
                users = cursor.fetchall()
                print('现有用户:')
                for user in users:
                    print(f'  ID: {user["id"]}, 用户名: {user["username"]}, 角色: {user["role"]}')
                
                if not users:
                    print('数据库中没有用户，需要创建测试用户')
                    return False
                return True
    except Exception as e:
        print(f'查询用户失败: {e}')
        return False

if __name__ == "__main__":
    check_users()
