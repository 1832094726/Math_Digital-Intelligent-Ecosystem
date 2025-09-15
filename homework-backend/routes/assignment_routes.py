"""
作业分发管理路由
"""
from flask import Blueprint, request, jsonify
from models.assignment import Assignment, ClassManagement, Notification
from functools import wraps
import jwt
from flask import request
from datetime import datetime

assignment_bp = Blueprint('assignment', __name__, url_prefix='/api/assignment')

def token_required(f):
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
            from services.auth_service import AuthService
            payload = AuthService.verify_token(token)
            current_user = {
                'id': payload['user_id'],
                'role': payload['role'],
                'permissions': payload['permissions']
            }
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e),
                'error': {'type': 'TokenVerificationError'}
            }), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated_function

@assignment_bp.route('/assign', methods=['POST'])
@token_required
def assign_homework(current_user):
    """分发作业给班级"""
    if current_user['role'] not in ['teacher', 'admin']:
        return jsonify({
            "success": False,
            "message": "只有教师和管理员可以分发作业"
        }), 403
    
    try:
        data = request.get_json()
        homework_id = data.get('homework_id')
        class_id = data.get('class_id')
        due_date = data.get('due_date')
        instructions = data.get('instructions', '')
        
        # 验证必填字段
        if not all([homework_id, class_id, due_date]):
            return jsonify({
                "success": False,
                "message": "作业ID、班级ID和截止日期不能为空"
            }), 400
        
        # 验证日期格式
        try:
            datetime.strptime(due_date, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({
                "success": False,
                "message": "日期格式错误，请使用 YYYY-MM-DD HH:MM:SS 格式"
            }), 400
        
        # 分发作业
        result = Assignment.assign_homework_to_class(
            homework_id=homework_id,
            class_id=class_id,
            teacher_id=current_user['id'],
            due_date=due_date,
            instructions=instructions
        )
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"分发作业失败: {str(e)}"
        }), 500

@assignment_bp.route('/class/<int:class_id>', methods=['GET'])
@token_required
def get_class_assignments(current_user, class_id):
    """获取班级的作业分发列表"""
    try:
        status = request.args.get('status')  # 可选状态筛选
        
        assignments = Assignment.get_class_assignments(class_id, status)
        
        return jsonify({
            "success": True,
            "data": assignments,
            "total": len(assignments)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取班级作业失败: {str(e)}"
        }), 500

@assignment_bp.route('/teacher/my', methods=['GET'])
@token_required
def get_my_assignments(current_user):
    """获取教师的作业分发列表"""
    if current_user['role'] not in ['teacher', 'admin']:
        return jsonify({
            "success": False,
            "message": "只有教师和管理员可以查看作业分发"
        }), 403
    
    try:
        status = request.args.get('status')  # 可选状态筛选
        
        assignments = Assignment.get_teacher_assignments(current_user['id'], status)
        
        return jsonify({
            "success": True,
            "data": assignments,
            "total": len(assignments)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取作业分发失败: {str(e)}"
        }), 500

@assignment_bp.route('/<int:assignment_id>', methods=['GET'])
@token_required
def get_assignment_detail(current_user, assignment_id):
    """获取作业分发详情"""
    try:
        result = Assignment.get_assignment_detail(assignment_id)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取作业分发详情失败: {str(e)}"
        }), 500

@assignment_bp.route('/<int:assignment_id>/status', methods=['PUT'])
@token_required
def update_assignment_status(current_user, assignment_id):
    """更新作业分发状态"""
    if current_user['role'] not in ['teacher', 'admin']:
        return jsonify({
            "success": False,
            "message": "只有教师和管理员可以更新作业分发状态"
        }), 403
    
    try:
        data = request.get_json()
        status = data.get('status')
        
        if not status:
            return jsonify({
                "success": False,
                "message": "状态不能为空"
            }), 400
        
        if status not in ['active', 'paused', 'completed', 'cancelled']:
            return jsonify({
                "success": False,
                "message": "无效的状态值"
            }), 400
        
        result = Assignment.update_assignment_status(
            assignment_id=assignment_id,
            status=status,
            teacher_id=current_user['id']
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"更新状态失败: {str(e)}"
        }), 500

# 班级管理相关路由
@assignment_bp.route('/classes/my', methods=['GET'])
@token_required
def get_my_classes(current_user):
    """获取教师负责的班级"""
    if current_user['role'] not in ['teacher', 'admin']:
        return jsonify({
            "success": False,
            "message": "只有教师和管理员可以查看班级"
        }), 403
    
    try:
        classes = ClassManagement.get_teacher_classes(current_user['id'])
        
        return jsonify({
            "success": True,
            "data": classes,
            "total": len(classes)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取班级列表失败: {str(e)}"
        }), 500

@assignment_bp.route('/classes/<int:class_id>/students', methods=['GET'])
@token_required
def get_class_students(current_user, class_id):
    """获取班级学生列表"""
    try:
        students = ClassManagement.get_class_students(class_id)
        
        return jsonify({
            "success": True,
            "data": students,
            "total": len(students)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取班级学生失败: {str(e)}"
        }), 500

# 通知相关路由
@assignment_bp.route('/notifications/my', methods=['GET'])
@token_required
def get_my_notifications(current_user):
    """获取用户通知列表"""
    try:
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        
        notifications = Notification.get_user_notifications(
            user_id=current_user['id'],
            unread_only=unread_only
        )
        
        return jsonify({
            "success": True,
            "data": notifications,
            "total": len(notifications)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取通知失败: {str(e)}"
        }), 500

@assignment_bp.route('/notifications/<int:notification_id>/read', methods=['PUT'])
@token_required
def mark_notification_read(current_user, notification_id):
    """标记通知为已读"""
    try:
        success = Notification.mark_notification_read(
            notification_id=notification_id,
            user_id=current_user['id']
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "通知已标记为已读"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "标记失败"
            }), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"标记通知失败: {str(e)}"
        }), 500

@assignment_bp.route('/statistics/<int:assignment_id>', methods=['GET'])
@token_required
def get_assignment_statistics(current_user, assignment_id):
    """获取作业分发统计信息"""
    try:
        stats = Assignment.get_assignment_statistics(assignment_id)
        
        return jsonify({
            "success": True,
            "data": stats
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取统计信息失败: {str(e)}"
        }), 500
