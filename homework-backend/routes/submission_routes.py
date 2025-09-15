"""
作业提交相关路由
"""
from flask import Blueprint, request, jsonify
from models.submission import HomeworkSubmission
from routes.student_homework_routes import token_required

submission_bp = Blueprint('submission', __name__, url_prefix='/api/submission')

@submission_bp.route('/<int:assignment_id>', methods=['POST'])
@token_required
def submit_homework(current_user, assignment_id):
    """
    学生提交作业
    """
    if current_user.get('role') != 'student':
        return jsonify({"success": False, "message": "只有学生可以提交作业"}), 403

    try:
        data = request.get_json()
        answers = data.get('answers')
        time_spent = data.get('time_spent')

        if not isinstance(answers, dict) or time_spent is None:
            return jsonify({"success": False, "message": "请求参数不完整或格式错误"}), 400

        result = HomeworkSubmission.submit_homework(
            student_id=current_user['id'],
            assignment_id=assignment_id,
            answers=answers,
            time_spent=int(time_spent)
        )

        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({"success": False, "message": f"服务器内部错误: {str(e)}"}), 500


@submission_bp.route('/<int:submission_id>/result', methods=['GET'])
@token_required
def get_submission_result(current_user, submission_id):
    """
    获取作业提交结果
    """
    if current_user.get('role') != 'student':
        return jsonify({"success": False, "message": "只有学生可以查看提交结果"}), 403

    try:
        result = HomeworkSubmission.get_submission_result(
            student_id=current_user['id'],
            submission_id=submission_id
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        return jsonify({"success": False, "message": f"服务器内部错误: {str(e)}"}), 500

