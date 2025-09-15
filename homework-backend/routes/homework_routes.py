# -*- coding: utf-8 -*-
"""
作业管理路由
"""
from flask import Blueprint, request, jsonify
from models.homework import Homework
from models.user import User
from routes.auth_routes import jwt_required
from datetime import datetime
import json

homework_bp = Blueprint('homework', __name__, url_prefix='/api/homework')

@homework_bp.route('/create', methods=['POST'])
@jwt_required
def create_homework():
    """创建作业"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据不能为空',
                'error': {'type': 'ValidationError'}
            }), 400
        
        # 验证必需字段
        required_fields = ['title', 'subject', 'grade']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'message': f'缺少必需字段: {field}',
                    'error': {'type': 'ValidationError'}
                }), 400
        
        # 验证用户权限（只有教师和管理员可以创建作业）
        user = User.get_by_id(request.current_user_id)
        if user.role not in ['teacher', 'admin']:
            return jsonify({
                'success': False,
                'message': '只有教师和管理员可以创建作业',
                'error': {'type': 'PermissionError'}
            }), 403
        
        # 设置创建者
        data['created_by'] = request.current_user_id
        
        # 处理日期字段
        for date_field in ['due_date', 'start_date']:
            if data.get(date_field):
                try:
                    # 尝试解析日期字符串
                    if isinstance(data[date_field], str):
                        data[date_field] = datetime.fromisoformat(data[date_field].replace('Z', '+00:00'))
                except ValueError:
                    return jsonify({
                        'success': False,
                        'message': f'日期格式错误: {date_field}',
                        'error': {'type': 'ValidationError'}
                    }), 400
        
        # 创建作业
        homework = Homework.create(data)
        
        return jsonify({
            'success': True,
            'message': '作业创建成功',
            'data': homework.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '创建作业失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/list', methods=['GET'])
@jwt_required
def list_homeworks():
    """获取作业列表"""
    try:
        user = User.get_by_id(request.current_user_id)
        
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        grade = request.args.get('grade', type=int)
        subject = request.args.get('subject')
        category = request.args.get('category')
        keyword = request.args.get('keyword')
        
        # 根据用户角色返回不同的作业列表
        if user.role in ['teacher', 'admin']:
            # 教师查看自己创建的作业
            if keyword or grade or subject or category:
                homeworks = Homework.search(
                    keyword=keyword, grade=grade, subject=subject, 
                    category=category, created_by=request.current_user_id, 
                    page=page, limit=limit
                )
            else:
                homeworks = Homework.get_by_teacher(request.current_user_id, page, limit)
        else:
            # 学生查看已发布的作业
            homeworks = Homework.get_published(grade=grade, subject=subject, page=page, limit=limit)
        
        return jsonify({
            'success': True,
            'data': {
                'homeworks': [hw.to_dict() for hw in homeworks],
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total': len(homeworks)  # 简化版本，实际应该查询总数
                }
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '获取作业列表失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/<int:homework_id>', methods=['GET'])
@jwt_required
def get_homework(homework_id):
    """获取作业详情"""
    try:
        homework = Homework.get_by_id(homework_id)
        
        if not homework:
            return jsonify({
                'success': False,
                'message': '作业不存在',
                'error': {'type': 'NotFoundError'}
            }), 404
        
        user = User.get_by_id(request.current_user_id)
        
        # 权限检查
        if user.role == 'student' and not homework.is_published:
            return jsonify({
                'success': False,
                'message': '作业未发布',
                'error': {'type': 'PermissionError'}
            }), 403
        
        if user.role == 'teacher' and homework.created_by != request.current_user_id:
            return jsonify({
                'success': False,
                'message': '无权访问此作业',
                'error': {'type': 'PermissionError'}
            }), 403
        
        return jsonify({
            'success': True,
            'data': homework.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '获取作业详情失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/<int:homework_id>/questions', methods=['GET'])
@jwt_required
def get_homework_questions(homework_id):
    """获取作业题目列表"""
    try:
        # 检查作业是否存在
        homework = Homework.get_by_id(homework_id)
        if not homework:
            return jsonify({
                'success': False,
                'message': '作业不存在',
                'error': {'type': 'NotFoundError'}
            }), 404

        # 检查用户权限
        user = User.get_by_id(request.current_user_id)
        if not user:
            return jsonify({
                'success': False,
                'message': '用户不存在',
                'error': {'type': 'AuthError'}
            }), 401

        # 权限检查
        if user.role == 'student' and not homework.is_published:
            return jsonify({
                'success': False,
                'message': '作业未发布',
                'error': {'type': 'PermissionError'}
            }), 403

        # 获取题目列表
        from models.database import db
        try:
            sql = """
            SELECT id, content, question_type, options, score, difficulty, order_index,
                   knowledge_points, explanation
            FROM questions
            WHERE homework_id = %s
            ORDER BY order_index ASC
            """

            questions = db.execute_query(sql, (homework_id,))

            # 处理每个题目的数据
            processed_questions = []
            for question in questions:
                try:
                    # 转换Decimal类型为float
                    if 'score' in question and question['score'] is not None:
                        question['score'] = float(question['score'])

                    # 确保其他数值字段也是正确类型
                    if 'difficulty' in question and question['difficulty'] is not None:
                        question['difficulty'] = int(question['difficulty'])

                    if 'order_index' in question and question['order_index'] is not None:
                        question['order_index'] = int(question['order_index'])

                    # 安全处理JSON字段
                    if question.get('options') and question['options']:
                        try:
                            question['options'] = json.loads(question['options'])
                        except (json.JSONDecodeError, TypeError):
                            question['options'] = []
                    else:
                        question['options'] = []

                    if question.get('knowledge_points') and question['knowledge_points']:
                        try:
                            question['knowledge_points'] = json.loads(question['knowledge_points'])
                        except (json.JSONDecodeError, TypeError):
                            question['knowledge_points'] = []
                    else:
                        question['knowledge_points'] = []

                    # 学生不显示正确答案
                    if user.role == 'student':
                        question.pop('correct_answer', None)

                    processed_questions.append(question)

                except Exception as question_error:
                    print(f"处理题目 {question.get('id', 'unknown')} 时出错: {question_error}")
                    # 跳过有问题的题目，继续处理其他题目
                    continue

            return jsonify({
                'success': True,
                'data': {
                    'homework_id': homework_id,
                    'questions': processed_questions,
                    'total_count': len(processed_questions)
                }
            }), 200

        except Exception as db_error:
            print(f"数据库查询错误: {db_error}")
            return jsonify({
                'success': False,
                'message': '数据库查询失败',
                'error': {
                    'type': 'DatabaseError',
                    'details': str(db_error)
                }
            }), 500

    except Exception as e:
        print(f"获取作业题目失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': '获取作业题目失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/<int:homework_id>', methods=['PUT'])
@jwt_required
def update_homework(homework_id):
    """更新作业"""
    try:
        homework = Homework.get_by_id(homework_id)
        
        if not homework:
            return jsonify({
                'success': False,
                'message': '作业不存在',
                'error': {'type': 'NotFoundError'}
            }), 404
        
        # 权限检查（只有作业创建者可以更新）
        if homework.created_by != request.current_user_id:
            return jsonify({
                'success': False,
                'message': '无权修改此作业',
                'error': {'type': 'PermissionError'}
            }), 403
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据不能为空',
                'error': {'type': 'ValidationError'}
            }), 400
        
        # 处理日期字段
        for date_field in ['due_date', 'start_date']:
            if data.get(date_field):
                try:
                    if isinstance(data[date_field], str):
                        data[date_field] = datetime.fromisoformat(data[date_field].replace('Z', '+00:00'))
                except ValueError:
                    return jsonify({
                        'success': False,
                        'message': f'日期格式错误: {date_field}',
                        'error': {'type': 'ValidationError'}
                    }), 400
        
        # 更新作业
        homework.update(data)
        
        return jsonify({
            'success': True,
            'message': '作业更新成功',
            'data': homework.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '更新作业失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/<int:homework_id>', methods=['DELETE'])
@jwt_required
def delete_homework(homework_id):
    """删除作业"""
    try:
        homework = Homework.get_by_id(homework_id)
        
        if not homework:
            return jsonify({
                'success': False,
                'message': '作业不存在',
                'error': {'type': 'NotFoundError'}
            }), 404
        
        # 权限检查（只有作业创建者可以删除）
        if homework.created_by != request.current_user_id:
            return jsonify({
                'success': False,
                'message': '无权删除此作业',
                'error': {'type': 'PermissionError'}
            }), 403
        
        homework.delete()
        
        return jsonify({
            'success': True,
            'message': '作业删除成功'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '删除作业失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/<int:homework_id>/publish', methods=['POST'])
@jwt_required
def publish_homework(homework_id):
    """发布作业"""
    try:
        homework = Homework.get_by_id(homework_id)
        
        if not homework:
            return jsonify({
                'success': False,
                'message': '作业不存在',
                'error': {'type': 'NotFoundError'}
            }), 404
        
        # 权限检查
        if homework.created_by != request.current_user_id:
            return jsonify({
                'success': False,
                'message': '无权发布此作业',
                'error': {'type': 'PermissionError'}
            }), 403
        
        homework.publish()
        
        return jsonify({
            'success': True,
            'message': '作业发布成功',
            'data': homework.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '发布作业失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/<int:homework_id>/unpublish', methods=['POST'])
@jwt_required
def unpublish_homework(homework_id):
    """取消发布作业"""
    try:
        homework = Homework.get_by_id(homework_id)
        
        if not homework:
            return jsonify({
                'success': False,
                'message': '作业不存在',
                'error': {'type': 'NotFoundError'}
            }), 404
        
        # 权限检查
        if homework.created_by != request.current_user_id:
            return jsonify({
                'success': False,
                'message': '无权取消发布此作业',
                'error': {'type': 'PermissionError'}
            }), 403
        
        homework.unpublish()
        
        return jsonify({
            'success': True,
            'message': '作业取消发布成功',
            'data': homework.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '取消发布作业失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/statistics', methods=['GET'])
@jwt_required
def get_statistics():
    """获取作业统计信息"""
    try:
        user = User.get_by_id(request.current_user_id)
        
        if user.role in ['teacher', 'admin']:
            # 教师查看自己的作业统计
            stats = Homework.get_statistics(teacher_id=request.current_user_id)
        else:
            # 学生查看全局统计（仅已发布的）
            stats = Homework.get_statistics()
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '获取统计信息失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

@homework_bp.route('/search', methods=['GET'])
@jwt_required
def search_homeworks():
    """搜索作业"""
    try:
        keyword = request.args.get('keyword')
        grade = request.args.get('grade', type=int)
        subject = request.args.get('subject')
        category = request.args.get('category')
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        
        user = User.get_by_id(request.current_user_id)
        
        # 根据用户角色搜索
        if user.role in ['teacher', 'admin']:
            # 教师搜索自己的作业
            created_by = request.current_user_id
        else:
            # 学生只能搜索已发布的作业（暂时不限制创建者）
            created_by = None
        
        homeworks = Homework.search(
            keyword=keyword, grade=grade, subject=subject,
            category=category, created_by=created_by,
            page=page, limit=limit
        )
        
        return jsonify({
            'success': True,
            'data': {
                'homeworks': [hw.to_dict() for hw in homeworks],
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total': len(homeworks)
                }
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '搜索作业失败',
            'error': {
                'type': 'SystemError',
                'details': str(e)
            }
        }), 500

