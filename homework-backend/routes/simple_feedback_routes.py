"""
简化版作业反馈路由
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import json
from datetime import datetime

from models.database import DatabaseManager

simple_feedback_bp = Blueprint('simple_feedback', __name__, url_prefix='/api/simple-feedback')
db = DatabaseManager()

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'success': False, 'message': '缺少认证令牌'}), 401
            
            token = auth_header.split(' ')[1]
            request.current_user = {'id': 2, 'role': 'student'}  # 模拟用户信息
            
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'success': False, 'message': '认证失败'}), 401
    return decorated_function

@simple_feedback_bp.route('/homework/<int:homework_id>', methods=['GET'])
@require_auth
def get_simple_homework_feedback(homework_id):
    """
    获取简化版作业反馈
    """
    try:
        user_id = request.current_user['id']
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 获取学生的作业提交记录
                cursor.execute("""
                    SELECT hs.*, h.title, h.max_score as homework_max_score, h.subject, h.grade as grade_level
                    FROM homework_submissions hs
                    JOIN homeworks h ON hs.homework_id = h.id
                    WHERE hs.homework_id = %s AND hs.student_id = %s
                    ORDER BY hs.submitted_at DESC
                    LIMIT 1
                """, (homework_id, user_id))
                
                submission = cursor.fetchone()
                if not submission:
                    return jsonify({
                        'success': False,
                        'message': '未找到作业提交记录'
                    }), 404
                
                # 获取班级统计信息
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_students,
                        AVG(hs.score) as avg_score,
                        MAX(hs.score) as max_score,
                        MIN(hs.score) as min_score
                    FROM homework_submissions hs
                    WHERE hs.homework_id = %s AND hs.status = 'graded'
                """, (homework_id,))
                
                class_stats = cursor.fetchone()
                
                # 计算学生排名
                cursor.execute("""
                    SELECT COUNT(*) + 1 as rank
                    FROM homework_submissions hs
                    WHERE hs.homework_id = %s AND hs.score > %s AND hs.status = 'graded'
                """, (homework_id, submission['score']))
                
                rank_result = cursor.fetchone()
                student_rank = rank_result['rank'] if rank_result else 1
                
                # 获取题目信息
                cursor.execute("""
                    SELECT q.*
                    FROM questions q
                    WHERE q.homework_id = %s
                    ORDER BY q.order_index
                """, (homework_id,))
                
                questions = cursor.fetchall()
                
                # 构建简化反馈响应
                feedback_data = {
                    'homework_info': {
                        'id': homework_id,
                        'title': submission['title'],
                        'subject': submission['subject'],
                        'grade_level': submission['grade_level']
                    },
                    'personal_performance': {
                        'total_score': float(submission['score']) if submission['score'] else 0,
                        'max_score': float(submission['homework_max_score']) if submission['homework_max_score'] else 100,
                        'percentage': round((float(submission['score']) / float(submission['homework_max_score'])) * 100, 1) if submission['score'] and submission['homework_max_score'] else 0,
                        'completion_time': submission.get('time_spent'),
                        'submitted_at': submission['submitted_at'].isoformat() if submission['submitted_at'] else None
                    },
                    'class_statistics': {
                        'total_students': class_stats['total_students'] if class_stats else 0,
                        'class_average': round(float(class_stats['avg_score']), 1) if class_stats and class_stats['avg_score'] else 0,
                        'class_max': float(class_stats['max_score']) if class_stats and class_stats['max_score'] else 0,
                        'class_min': float(class_stats['min_score']) if class_stats and class_stats['min_score'] else 0,
                        'student_rank': student_rank,
                        'percentile': round((1 - (student_rank - 1) / max(class_stats['total_students'] if class_stats else 1, 1)) * 100, 1) if class_stats else 100
                    },
                    'question_feedback': [
                        {
                            'question_id': q['id'],
                            'question_order': q['order_index'],
                            'question_type': q['question_type'],
                            'question_content': q['content'],
                            'max_score': float(q['score']) if q['score'] else 0,
                            'explanation': q.get('explanation', ''),
                            'knowledge_points': json.loads(q.get('knowledge_points', '[]')) if q.get('knowledge_points') else []
                        }
                        for q in questions
                    ],
                    'learning_suggestions': [
                        {
                            'type': 'encouragement',
                            'title': '继续努力',
                            'content': '你已经完成了这次作业，继续保持学习的积极性！',
                            'priority': 'medium'
                        }
                    ],
                    'error_analysis': {
                        'overall_performance': 'good',
                        'main_issues': [],
                        'improvement_areas': []
                    },
                    'generated_at': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'data': feedback_data
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取反馈失败: {str(e)}'
        }), 500
