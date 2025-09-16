from flask import Flask, jsonify, request, redirect, send_from_directory, send_file
from flask_cors import CORS
import os
import json
from services.homework_service import get_homework_list, get_homework_detail, submit_homework, save_progress
from services.recommendation_service import get_recommended_symbols, get_recommended_knowledge, get_recommended_exercises
from services.user_service import get_user_info, update_user_model
from services.knowledge_service import get_question_knowledge_points
from routes.enhanced_symbol_routes import enhanced_symbol_bp
from routes.auth_routes import auth_bp
from routes.homework_routes import homework_bp
from routes.assignment_routes import assignment_bp
from routes.student_homework_routes import student_homework_bp
from routes.submission_routes import submission_bp
from routes.grading_routes import grading_bp
from routes.feedback_routes import feedback_bp
from routes.analytics_routes import analytics_bp
from routes.simple_feedback_routes import simple_feedback_bp
from routes.simple_analytics_routes import simple_analytics_bp
from blueprints.recommendation_bp import recommendation_bp
from config import config

app = Flask(__name__)

# 加载配置
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

CORS(app, resources={r"/*": {'origins': '*'}})

# 注册蓝图
app.register_blueprint(enhanced_symbol_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(homework_bp)
app.register_blueprint(assignment_bp)
app.register_blueprint(student_homework_bp)
app.register_blueprint(submission_bp)
app.register_blueprint(grading_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(simple_feedback_bp)
app.register_blueprint(simple_analytics_bp)
app.register_blueprint(recommendation_bp)

# 数据路径
DATA_ROOT = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")

# 确保数据目录存在
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

@app.route('/')
def hello_world():
    return 'Homework System API'

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        # 简单的健康检查，返回服务状态
        return jsonify({
            'status': 'healthy',
            'service': 'homework-backend',
            'message': '作业管理系统API正常运行'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'homework-backend',
            'message': f'服务异常: {str(e)}'
        }), 500

# 重定向旧路由到新API路由
@app.route('/homework/list', methods=['GET'])
def redirect_homework_list():
    return redirect('/api/homework/list')

# 重定向旧路由到新API路由
@app.route('/homework/detail/<int:homework_id>', methods=['GET'])
def redirect_homework_detail(homework_id):
    return redirect(f'/api/homework/detail/{homework_id}')

# 重定向知识点路由
@app.route('/knowledge/question', methods=['GET', 'POST'])
def redirect_knowledge_question():
    if request.method == 'GET':
        # 获取所有GET参数并传递
        args = request.args.to_dict()
        # 构建重定向URL，保留所有查询参数
        query_string = '&'.join([f"{k}={v}" for k, v in args.items()])
        redirect_url = f'/api/knowledge/question?{query_string}' if query_string else '/api/knowledge/question'
        return redirect(redirect_url)
    else:
        # POST请求重定向
        return redirect('/api/knowledge/question')

# 作业管理API
@app.route('/api/homework/list', methods=['GET'])
def homework_list():
    user_id = request.args.get('userId', default=1, type=int)
    return jsonify(get_homework_list(user_id))

@app.route('/api/homework/detail/<int:homework_id>', methods=['GET'])
def homework_detail(homework_id):
    return jsonify(get_homework_detail(homework_id))

@app.route('/api/homework/submit', methods=['POST'])
def submit():
    data = request.get_json()
    return jsonify(submit_homework(data))

@app.route('/api/homework/save', methods=['POST'])
def save():
    data = request.get_json()
    return jsonify(save_progress(data))

# 知识点API
@app.route('/api/knowledge/question', methods=['GET', 'POST'])
def question_knowledge():
    if request.method == 'GET':
        # 通过URL参数获取题目ID或题目内容
        question_id = request.args.get('questionId', type=int)
        question_text = request.args.get('text')
        
        if not question_id and not question_text:
            return jsonify({"error": "请提供题目ID或题目内容"}), 400
            
        knowledge_points = get_question_knowledge_points(question_id, question_text)
        return jsonify({"knowledge_points": knowledge_points})
    else:
        # POST方法，从请求体获取数据
        data = request.get_json()
        question_id = data.get('questionId')
        question_text = data.get('text')
        
        if not question_id and not question_text:
            return jsonify({"error": "请提供题目ID或题目内容"}), 400
            
        knowledge_points = get_question_knowledge_points(question_id, question_text)
        return jsonify({"knowledge_points": knowledge_points})

# 推荐系统API
@app.route('/api/recommend/symbols', methods=['POST'])
def recommend_symbols():
    data = request.get_json()
    return jsonify(get_recommended_symbols(data))

@app.route('/api/recommend/knowledge', methods=['POST'])
def recommend_knowledge():
    data = request.get_json()
    return jsonify(get_recommended_knowledge(data))

@app.route('/api/recommend/exercises', methods=['POST'])
def recommend_exercises():
    data = request.get_json()
    return jsonify(get_recommended_exercises(data))

# 用户模型API
@app.route('/api/user/<int:user_id>', methods=['GET'])
def user_info(user_id):
    return jsonify(get_user_info(user_id))

@app.route('/api/user/update', methods=['POST'])
def update_user():
    data = request.get_json()
    return jsonify(update_user_model(data))

# 静态文件服务 - 服务Vue前端
@app.route('/static/<path:filename>')
def serve_static(filename):
    """服务所有静态文件"""
    return send_from_directory('static/homework', filename)

@app.route('/static/homework/<path:filename>')
def serve_homework_static(filename):
    """服务Vue前端静态文件"""
    return send_from_directory('static/homework', filename)

@app.route('/static/symbol/<path:filename>')
def serve_symbol_static(filename):
    """服务符号键盘静态文件"""
    return send_from_directory('static/symbol', filename)

# 前端路由处理 - 所有非API路由都返回Vue应用
@app.route('/')
@app.route('/homework')
@app.route('/login')
@app.route('/register')
def serve_frontend():
    """服务Vue前端应用"""
    try:
        return send_file('static/homework/index.html')
    except FileNotFoundError:
        return jsonify({
            'error': '前端文件未找到',
            'message': '请确保Vue前端已正确构建并复制到static/homework目录'
        }), 404

# 处理Vue Router的history模式 - 所有未匹配的路由都返回index.html
@app.errorhandler(404)
def not_found(error):
    # 如果是API请求，返回JSON错误
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API端点不存在'}), 404
    # 否则返回Vue应用（用于前端路由）
    try:
        return send_file('static/homework/index.html')
    except FileNotFoundError:
        return jsonify({
            'error': '前端文件未找到',
            'message': '请确保Vue前端已正确构建'
        }), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)