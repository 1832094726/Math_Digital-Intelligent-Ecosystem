#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复session_token字段长度脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def fix_session_token_length():
    """修复session_token字段长度"""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # 先删除session_token的唯一索引
                try:
                    cursor.execute("DROP INDEX uk_session_token ON user_sessions")
                    print("✅ 删除了session_token的唯一索引")
                except Exception as e:
                    print(f"删除索引失败(可能不存在): {e}")

                # 修改session_token字段长度为TEXT类型，支持更长的JWT token
                sql = """
                ALTER TABLE user_sessions
                MODIFY COLUMN session_token TEXT NOT NULL COMMENT '会话令牌'
                """

                cursor.execute(sql)

                # 重新创建索引，但不使用UNIQUE约束（因为TEXT字段不能做唯一索引）
                cursor.execute("CREATE INDEX idx_session_token ON user_sessions(session_token(255))")

                conn.commit()
                print("✅ session_token字段长度已修复为TEXT类型，并重新创建了索引")
                
                # 验证修改结果
                cursor.execute("DESCRIBE user_sessions")
                columns = cursor.fetchall()
                
                for col in columns:
                    if col['Field'] == 'session_token':
                        print(f"✅ 验证成功: session_token类型为 {col['Type']}")
                        break
                
    except Exception as e:
        print(f"修复失败: {e}")

if __name__ == "__main__":
    fix_session_token_length()
