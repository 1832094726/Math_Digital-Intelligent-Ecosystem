#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查用户密码
"""

import sys
import os
from werkzeug.security import check_password_hash, generate_password_hash

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import get_db_connection

def check_user_password():
    """检查用户密码"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询test_student_001用户
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", ('test_student_001',))
        user_row = cursor.fetchone()

        if user_row:
            user = {
                'id': user_row[0],
                'username': user_row[1],
                'password_hash': user_row[2]
            }
            print(f"用户信息:")
            print(f"  ID: {user['id']}")
            print(f"  用户名: {user['username']}")
            print(f"  密码哈希: {user['password_hash']}")
            
            # 测试不同的密码
            test_passwords = ['student123', 'password123', '123456', 'test123']
            
            print(f"\n测试密码:")
            for password in test_passwords:
                is_valid = check_password_hash(user['password_hash'], password)
                print(f"  {password}: {'✅ 正确' if is_valid else '❌ 错误'}")
                
            # 如果都不对，重新设置密码
            if not any(check_password_hash(user['password_hash'], pwd) for pwd in test_passwords):
                print(f"\n🔧 重新设置密码为 'student123'...")
                new_hash = generate_password_hash('student123')
                cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, user['id']))
                conn.commit()
                print("✅ 密码重置成功")
                
        else:
            print("❌ 用户不存在")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")

if __name__ == '__main__':
    check_user_password()
