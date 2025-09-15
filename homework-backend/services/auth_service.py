# -*- coding: utf-8 -*-
"""
认证服务
"""
import jwt
from datetime import datetime, timedelta
from flask import current_app
from models.user import User
from models.database import db
import json

class AuthService:
    """认证服务类"""
    
    @staticmethod
    def generate_tokens(user):
        """生成JWT令牌"""
        try:
            # 准备载荷数据
            payload = {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'permissions': user.get_permissions(),
                'exp': datetime.utcnow() + current_app.config['JWT_ACCESS_TOKEN_EXPIRES'],
                'iat': datetime.utcnow()
            }
            
            # 生成访问令牌
            access_token = jwt.encode(
                payload, 
                current_app.config['JWT_SECRET_KEY'], 
                algorithm='HS256'
            )
            
            # 生成刷新令牌
            refresh_payload = {
                'user_id': user.id,
                'type': 'refresh',
                'exp': datetime.utcnow() + current_app.config['JWT_REFRESH_TOKEN_EXPIRES'],
                'iat': datetime.utcnow()
            }
            
            refresh_token = jwt.encode(
                refresh_payload,
                current_app.config['JWT_SECRET_KEY'],
                algorithm='HS256'
            )
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'Bearer',
                'expires_in': int(current_app.config['JWT_ACCESS_TOKEN_EXPIRES'].total_seconds())
            }
            
        except Exception as e:
            raise Exception(f"生成令牌失败: {str(e)}")
    
    @staticmethod
    def verify_token(token):
        """验证JWT令牌"""
        try:
            payload = jwt.decode(
                token, 
                current_app.config['JWT_SECRET_KEY'], 
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("令牌已过期")
        except jwt.InvalidTokenError:
            raise Exception("令牌无效")
    
    @staticmethod
    def refresh_token(refresh_token):
        """刷新访问令牌"""
        try:
            # 验证刷新令牌
            payload = jwt.decode(
                refresh_token,
                current_app.config['JWT_SECRET_KEY'],
                algorithms=['HS256']
            )
            
            if payload.get('type') != 'refresh':
                raise Exception("令牌类型错误")
            
            # 获取用户信息
            user = User.get_by_id(payload['user_id'])
            if not user or not user.is_active:
                raise Exception("用户不存在或已被禁用")
            
            # 生成新的访问令牌
            return AuthService.generate_tokens(user)
            
        except jwt.ExpiredSignatureError:
            raise Exception("刷新令牌已过期")
        except jwt.InvalidTokenError:
            raise Exception("刷新令牌无效")
    
    @staticmethod
    def register_user(user_data):
        """用户注册"""
        try:
            # 创建用户
            user = User.create(user_data)
            
            # 生成令牌
            tokens = AuthService.generate_tokens(user)
            
            # 记录会话
            session_data = {
                'user_id': user.id,
                'device_type': user_data.get('device_type', 'web'),
                'device_id': user_data.get('device_id'),
                'ip_address': user_data.get('ip_address'),
                'user_agent': user_data.get('user_agent')
            }
            AuthService.create_session(tokens['access_token'], session_data)
            
            return {
                'success': True,
                'message': '注册成功',
                'data': {
                    **tokens,
                    'user': user.to_dict()
                }
            }
            
        except ValueError as e:
            return {
                'success': False,
                'message': str(e),
                'error': {
                    'type': 'ValidationError',
                    'details': str(e)
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': '注册失败',
                'error': {
                    'type': 'SystemError',
                    'details': str(e)
                }
            }
    
    @staticmethod
    def login_user(username_or_email, password, login_data=None):
        """用户登录"""
        try:
            # 检查登录尝试次数（简化实现，实际应该使用Redis等缓存）
            
            # 认证用户
            user = User.authenticate(username_or_email, password)
            if not user:
                return {
                    'success': False,
                    'message': '用户名或密码错误',
                    'error': {
                        'type': 'AuthenticationError'
                    }
                }
            
            # 生成令牌
            tokens = AuthService.generate_tokens(user)
            
            # 记录会话
            session_data = {
                'user_id': user.id,
                'device_type': login_data.get('device_type', 'web') if login_data else 'web',
                'device_id': login_data.get('device_id') if login_data else None,
                'ip_address': login_data.get('ip_address') if login_data else None,
                'user_agent': login_data.get('user_agent') if login_data else None
            }
            AuthService.create_session(tokens['access_token'], session_data)
            
            return {
                'success': True,
                'message': '登录成功',
                'data': {
                    **tokens,
                    'user': user.to_dict()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': '登录失败',
                'error': {
                    'type': 'SystemError',
                    'details': str(e)
                }
            }
    
    @staticmethod
    def create_session(access_token, session_data):
        """创建用户会话记录"""
        try:
            # 解析token获取过期时间
            payload = AuthService.verify_token(access_token)
            expires_at = datetime.fromtimestamp(payload['exp'])
            
            # 插入会话记录
            sql = """
                INSERT INTO user_sessions 
                (user_id, session_token, device_type, device_id, ip_address, user_agent, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            db.execute_insert(sql, (
                session_data['user_id'],
                access_token,
                session_data.get('device_type'),
                session_data.get('device_id'),
                session_data.get('ip_address'),
                session_data.get('user_agent'),
                expires_at
            ))
            
        except Exception as e:
            # 会话记录失败不应该影响登录流程
            print(f"创建会话记录失败: {e}")
    
    @staticmethod
    def logout_user(access_token):
        """用户登出"""
        try:
            # 将会话标记为非活跃
            sql = "UPDATE user_sessions SET is_active = 0 WHERE session_token = %s"
            db.execute_update(sql, (access_token,))
            
            return {
                'success': True,
                'message': '登出成功'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': '登出失败',
                'error': {
                    'type': 'SystemError',
                    'details': str(e)
                }
            }
    
    @staticmethod
    def get_user_sessions(user_id):
        """获取用户会话列表"""
        try:
            sql = """
                SELECT id, device_type, device_id, ip_address, 
                       is_active, created_at, expires_at
                FROM user_sessions 
                WHERE user_id = %s AND expires_at > NOW()
                ORDER BY created_at DESC
            """
            
            sessions = db.execute_query(sql, (user_id,))
            
            return {
                'success': True,
                'data': {
                    'sessions': sessions
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': '获取会话列表失败',
                'error': {
                    'type': 'SystemError',
                    'details': str(e)
                }
            }


