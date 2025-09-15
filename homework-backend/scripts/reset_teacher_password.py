#!/usr/bin/env python3
"""
重置教师密码
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import DatabaseManager
import bcrypt

def main():
    db = DatabaseManager()
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # 重置teacher001的密码
                password = 'password'
                hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                
                cursor.execute("""
                    UPDATE users 
                    SET password_hash = %s 
                    WHERE username = 'teacher001'
                """, (hashed.decode('utf-8'),))
                
                conn.commit()
                
                print(f"✅ 教师 teacher001 密码已重置为: {password}")
                
                # 验证密码
                cursor.execute("SELECT password_hash FROM users WHERE username = 'teacher001'")
                result = cursor.fetchone()
                if result and bcrypt.checkpw(password.encode('utf-8'), result['password_hash'].encode('utf-8')):
                    print("✅ 密码验证成功")
                else:
                    print("❌ 密码验证失败")
                    
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()
