#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重置测试用户密码脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from models.user import User

def reset_test_user():
    """重置测试用户密码"""
    try:
        # 重置学生用户密码
        student_username = 'test_student_001'
        new_password = 'password'
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # 生成新的密码哈希
                password_hash = User.hash_password(new_password)
                
                # 更新密码
                cursor.execute(
                    'UPDATE users SET password_hash = %s WHERE username = %s',
                    (password_hash, student_username)
                )
                
                if cursor.rowcount > 0:
                    conn.commit()
                    print(f'✅ 用户 {student_username} 密码已重置为: {new_password}')
                    
                    # 验证密码
                    cursor.execute(
                        'SELECT password_hash FROM users WHERE username = %s',
                        (student_username,)
                    )
                    result = cursor.fetchone()
                    if result and User.verify_password(new_password, result['password_hash']):
                        print('✅ 密码验证成功')
                    else:
                        print('❌ 密码验证失败')
                else:
                    print(f'❌ 用户 {student_username} 不存在')
                    
    except Exception as e:
        print(f'重置密码失败: {e}')

if __name__ == "__main__":
    reset_test_user()
