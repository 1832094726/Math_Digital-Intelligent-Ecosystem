#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查用户密码脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from werkzeug.security import generate_password_hash, check_password_hash

def check_user_passwords():
    """检查用户密码"""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, username, password_hash FROM users LIMIT 5')
                users = cursor.fetchall()
                
                print('用户密码信息:')
                for user in users:
                    print(f'用户: {user["username"]}')
                    print(f'  密码哈希: {user["password_hash"][:50]}...')
                    
                    # 测试常见密码
                    test_passwords = ['password', '123456', 'admin', user["username"]]
                    for pwd in test_passwords:
                        if check_password_hash(user["password_hash"], pwd):
                            print(f'  ✅ 正确密码: {pwd}')
                            break
                    else:
                        print(f'  ❌ 未找到匹配的密码')
                    print()
                
                # 如果需要，创建新的测试用户
                print('=== 创建测试用户 ===')
                test_password = 'password'
                test_hash = generate_password_hash(test_password)
                print(f'测试密码 "{test_password}" 的哈希: {test_hash}')
                
    except Exception as e:
        print(f'检查失败: {e}')

if __name__ == "__main__":
    check_user_passwords()
