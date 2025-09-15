"""
学生作业接收与展示路由
"""
from flask import Blueprint, request, jsonify
from models.student_homework import StudentHomework, HomeworkProgress, HomeworkReminder
from functools import wraps
import jwt
from datetime import datetime

student_homework_bp = Blueprint('student_homework', __name__, url_prefix='/api/student/homework')

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

@student_homework_bp.route('/list', methods=['GET'])
@token_required
def get_homework_list(current_user):
    """获取学生作业列表"""
    try:
        # 获取查询参数
        status = request.args.get('status')  # pending, in_progress, completed, overdue
        search_keyword = request.args.get('search', '')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
        
        # 限制页面大小
        page_size = min(page_size, 50)
        
        result = StudentHomework.get_student_homework_list(
            student_id=current_user['id'],
            status=status,
            search_keyword=search_keyword if search_keyword else None,
            page=page,
            page_size=page_size
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取作业列表失败: {str(e)}"
        }), 500

@student_homework_bp.route('/<int:assignment_id>', methods=['GET'])
@token_required
def get_homework_detail(current_user, assignment_id):
    """获取作业详细信息"""
    try:
        result = StudentHomework.get_homework_detail(
            student_id=current_user['id'],
            assignment_id=assignment_id
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取作业详情失败: {str(e)}"
        }), 500

@student_homework_bp.route('/statistics', methods=['GET'])
@token_required
def get_homework_statistics(current_user):
    """获取学生作业统计信息"""
    try:
        result = StudentHomework.get_homework_statistics(current_user['id'])
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取统计信息失败: {str(e)}"
        }), 500

@student_homework_bp.route('/<int:assignment_id>/favorite', methods=['POST'])
@token_required
def toggle_homework_favorite(current_user, assignment_id):
    """切换作业收藏状态"""
    try:
        data = request.get_json()
        is_favorite = data.get('is_favorite', True)
        
        result = StudentHomework.mark_homework_favorite(
            student_id=current_user['id'],
            assignment_id=assignment_id,
            is_favorite=is_favorite
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"收藏操作失败: {str(e)}"
        }), 500

@student_homework_bp.route('/favorites', methods=['GET'])
@token_required
def get_favorite_homeworks(current_user):
    """获取收藏的作业列表"""
    try:
        favorites = StudentHomework.get_favorite_homeworks(current_user['id'])
        
        return jsonify({
            "success": True,
            "data": favorites,
            "total": len(favorites)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取收藏列表失败: {str(e)}"
        }), 500

# 作业进度相关路由
@student_homework_bp.route('/<int:homework_id>/progress', methods=['GET'])
@token_required
def get_homework_progress(current_user, homework_id):
    """获取作业进度"""
    try:
        result = HomeworkProgress.get_progress(
            student_id=current_user['id'],
            homework_id=homework_id
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取进度失败: {str(e)}"
        }), 500

@student_homework_bp.route('/<int:homework_id>/progress', methods=['POST'])
@token_required
def save_homework_progress(current_user, homework_id):
    """保存作业进度"""
    try:
        data = request.get_json()
        progress_data = data.get('progress_data', {})
        
        if not progress_data:
            return jsonify({
                "success": False,
                "message": "进度数据不能为空"
            }), 400
        
        # 添加时间戳
        progress_data['last_updated'] = datetime.now().isoformat()
        
        result = HomeworkProgress.save_progress(
            student_id=current_user['id'],
            homework_id=homework_id,
            progress_data=progress_data
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"保存进度失败: {str(e)}"
        }), 500

@student_homework_bp.route('/reminders', methods=['GET'])
@token_required
def get_homework_reminders(current_user):
    """获取作业提醒"""
    try:
        hours_ahead = int(request.args.get('hours_ahead', 24))
        
        reminders = HomeworkReminder.get_due_reminders(
            student_id=current_user['id'],
            hours_ahead=hours_ahead
        )
        
        return jsonify({
            "success": True,
            "data": reminders,
            "total": len(reminders)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取提醒失败: {str(e)}"
        }), 500

@student_homework_bp.route('/dashboard', methods=['GET'])
@token_required
def get_homework_dashboard(current_user):
    """获取学生作业仪表板数据"""
    try:
        # 获取统计信息
        stats_result = StudentHomework.get_homework_statistics(current_user['id'])
        if not stats_result['success']:
            return jsonify(stats_result), 400
        
        # 获取最近的作业（待完成）
        recent_result = StudentHomework.get_student_homework_list(
            student_id=current_user['id'],
            status='pending',
            page=1,
            page_size=5
        )
        
        # 获取提醒
        reminders = HomeworkReminder.get_due_reminders(
            student_id=current_user['id'],
            hours_ahead=48
        )
        
        # 获取进行中的作业
        in_progress_result = StudentHomework.get_student_homework_list(
            student_id=current_user['id'],
            status='in_progress',
            page=1,
            page_size=5
        )
        
        dashboard_data = {
            "statistics": stats_result['data'],
            "recent_pending": recent_result['data'] if recent_result['success'] else [],
            "in_progress": in_progress_result['data'] if in_progress_result['success'] else [],
            "urgent_reminders": reminders,
            "last_updated": datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "data": dashboard_data
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取仪表板数据失败: {str(e)}"
        }), 500

# 过滤和搜索相关路由
@student_homework_bp.route('/search', methods=['GET'])
@token_required
def search_homeworks(current_user):
    """搜索作业"""
    try:
        keyword = request.args.get('keyword', '')
        difficulty = request.args.get('difficulty')  # easy, medium, hard
        class_name = request.args.get('class')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
        
        # 这里可以实现更复杂的搜索逻辑
        # 暂时使用基础的关键词搜索
        result = StudentHomework.get_student_homework_list(
            student_id=current_user['id'],
            search_keyword=keyword if keyword else None,
            page=page,
            page_size=min(page_size, 50)
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"搜索失败: {str(e)}"
        }), 500

@student_homework_bp.route('/filters/options', methods=['GET'])
@token_required
def get_filter_options(current_user):
    """获取筛选选项"""
    try:
        # 这里可以从数据库获取可用的筛选选项
        filter_options = {
            "statuses": [
                {"value": "pending", "label": "待完成"},
                {"value": "in_progress", "label": "进行中"},
                {"value": "completed", "label": "已完成"},
                {"value": "overdue", "label": "已过期"}
            ],
            "difficulties": [
                {"value": "easy", "label": "简单"},
                {"value": "medium", "label": "中等"},
                {"value": "hard", "label": "困难"}
            ],
            "sort_options": [
                {"value": "due_date_asc", "label": "截止时间升序"},
                {"value": "due_date_desc", "label": "截止时间降序"},
                {"value": "created_date_desc", "label": "创建时间降序"},
                {"value": "difficulty_asc", "label": "难度升序"},
                {"value": "difficulty_desc", "label": "难度降序"}
            ]
        }
        
        return jsonify({
            "success": True,
            "data": filter_options
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取筛选选项失败: {str(e)}"
        }), 500

