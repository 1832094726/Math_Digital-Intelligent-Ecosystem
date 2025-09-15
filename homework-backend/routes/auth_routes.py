# -*- coding: utf-8 -*-
"""
认证相关路由
"""
from flask import Blueprint, request, jsonify
from services.auth_service import AuthService
from models.user import User
from functools import wraps
import jwt
from datetime import datetime

# 创建蓝图
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

def jwt_required(f):
    """JWT认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # 从请求头获取token
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({
                    'success': False,
                    'message': '令牌格式错误',
                    'error': {'type': 'TokenFormatError'}
                }), 401
        
        if not token:
            return jsonify({
                'success': False,
                'message': '缺少认证令牌',
                'error': {'type': 'MissingTokenError'}
            }), 401
        
        try:
            # 验证token
            payload = AuthService.verify_token(token)
            request.current_user_id = payload['user_id']
            request.current_user_role = payload['role']
            request.current_user_permissions = payload['permissions']
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e),
                'error': {'type': 'TokenValidationError'}
            }), 401
            
        return f(*args, **kwargs)
    
    return decorated_function

def permission_required(required_permissions):
    """权限验证装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_permissions = getattr(request, 'current_user_permissions', [])
            
            # 检查是否有超级权限
            if '*' in user_permissions:
                return f(*args, **kwargs)
            
            # 检查具体权限
            for permission in required_permissions:
                if permission not in user_permissions:
                    return jsonify({
                        'success': False,
                        'message': '权限不足',
                        'error': {'type': 'PermissionDeniedError'}
                    }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route('/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据不能为空',
                'error': {'type': 'ValidationError'}
            }), 400
        
        # 添加客户端信息
        client_info = {
            'device_type': data.get('device_type', 'web'),
            'device_id': data.get('device_id'),
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent')
        }
        
        # 合并数据
        user_data = {**data, **client_info}
        
        # 调用认证服务进行注册
        result = AuthService.register_user(user_data)
        
        status_code = 201 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '注册请求处理失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据不能为空',
                'error': {'type': 'ValidationError'}
            }), 400
        
        username_or_email = data.get('username')
        password = data.get('password')
        
        if not username_or_email or not password:
            return jsonify({
                'success': False,
                'message': '用户名和密码不能为空',
                'error': {'type': 'ValidationError'}
            }), 400
        
        # 添加客户端信息
        login_data = {
            'device_type': data.get('device_type', 'web'),
            'device_id': data.get('device_id'),
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent')
        }
        
        # 调用认证服务进行登录
        result = AuthService.login_user(username_or_email, password, login_data)
        
        status_code = 200 if result['success'] else 401
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '登录请求处理失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@auth_bp.route('/refresh', methods=['POST'])
def refresh():
    """刷新访问令牌"""
    try:
        data = request.get_json()
        
        if not data or not data.get('refresh_token'):
            return jsonify({
                'success': False,
                'message': '缺少刷新令牌',
                'error': {'type': 'ValidationError'}
            }), 400
        
        # 调用认证服务刷新令牌
        result = AuthService.refresh_token(data['refresh_token'])
        
        return jsonify({
            'success': True,
            'message': '令牌刷新成功',
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'error': {'type': 'TokenRefreshError'}
        }), 401

@auth_bp.route('/logout', methods=['POST'])
@jwt_required
def logout():
    """用户登出"""
    try:
        # 从请求头获取token
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(" ")[1] if auth_header else None
        
        if token:
            result = AuthService.logout_user(token)
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'message': '无效的令牌',
                'error': {'type': 'InvalidTokenError'}
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '登出失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required
def get_profile():
    """获取当前用户信息"""
    try:
        user = User.get_by_id(request.current_user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'message': '用户不存在',
                'error': {'type': 'UserNotFoundError'}
            }), 404
        
        return jsonify({
            'success': True,
            'data': user.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '获取用户信息失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required
def update_profile():
    """更新用户信息"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据不能为空',
                'error': {'type': 'ValidationError'}
            }), 400
        
        user = User.get_by_id(request.current_user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'message': '用户不存在',
                'error': {'type': 'UserNotFoundError'}
            }), 404
        
        # 更新用户信息
        user.update_profile(data)
        
        return jsonify({
            'success': True,
            'message': '用户信息更新成功',
            'data': user.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '更新用户信息失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@auth_bp.route('/sessions', methods=['GET'])
@jwt_required
def get_sessions():
    """获取用户会话列表"""
    try:
        result = AuthService.get_user_sessions(request.current_user_id)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '获取会话列表失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500


