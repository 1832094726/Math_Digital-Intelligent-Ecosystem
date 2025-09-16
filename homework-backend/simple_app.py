#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的数学教育后端API服务
用于传统部署方式
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import traceback
from datetime import datetime

app = Flask(__name__, static_folder=None)

# 配置CORS
CORS(app, origins='*', allow_headers=['Content-Type', 'Authorization'], methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# 配置
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

# 数据存储路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_json_data(filename):
    """加载JSON数据文件"""
    try:
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"加载数据文件 {filename} 失败: {e}")
        return {}

def save_json_data(filename, data):
    """保存JSON数据文件"""
    try:
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存数据文件 {filename} 失败: {e}")
        return False

# 加载数据
homework_data = load_json_data('homework.json')
users_data = load_json_data('users.json')
knowledge_data = load_json_data('knowledge.json')

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': '数学教育后端服务运行正常'
    })

@app.route('/api/homework/list', methods=['GET'])
def get_homework_list():
    """获取作业列表"""
    try:
        # 返回示例作业数据
        homeworks = [
            {
                "id": "hw001",
                "title": "一元二次方程练习",
                "description": "练习解一元二次方程的基本方法",
                "difficulty": 2,
                "deadline": "2024-12-31T23:59:59",
                "status": "not_started",
                "questions": [
                    {
                        "id": "q1",
                        "content": "解方程：x² - 5x + 6 = 0",
                        "score": 10
                    },
                    {
                        "id": "q2", 
                        "content": "解方程：2x² + 3x - 5 = 0",
                        "score": 15
                    }
                ]
            },
            {
                "id": "hw002",
                "title": "几何图形面积计算",
                "description": "计算各种几何图形的面积",
                "difficulty": 1,
                "deadline": "2024-12-25T23:59:59",
                "status": "in_progress",
                "questions": [
                    {
                        "id": "q3",
                        "content": "计算半径为5cm的圆的面积",
                        "score": 8
                    }
                ]
            }
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'homeworks': homeworks
            },
            'message': '获取作业列表成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '获取作业列表失败'
        }), 500

@app.route('/api/homework/<homework_id>', methods=['GET'])
def get_homework_detail(homework_id):
    """获取作业详情"""
    try:
        # 返回示例作业详情
        homework = {
            "id": homework_id,
            "title": "一元二次方程练习",
            "description": "练习解一元二次方程的基本方法",
            "difficulty": 2,
            "deadline": "2024-12-31T23:59:59",
            "status": "not_started",
            "questions": [
                {
                    "id": "q1",
                    "content": "解方程：x² - 5x + 6 = 0",
                    "score": 10,
                    "type": "algebra"
                },
                {
                    "id": "q2",
                    "content": "解方程：2x² + 3x - 5 = 0", 
                    "score": 15,
                    "type": "algebra"
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'data': homework,
            'message': '获取作业详情成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '获取作业详情失败'
        }), 500

@app.route('/api/homework/<homework_id>/questions', methods=['GET'])
def get_homework_questions(homework_id):
    """获取作业题目"""
    try:
        questions = [
            {
                "id": "q1",
                "content": "解方程：x² - 5x + 6 = 0",
                "score": 10,
                "type": "algebra"
            },
            {
                "id": "q2",
                "content": "解方程：2x² + 3x - 5 = 0",
                "score": 15,
                "type": "algebra"
            }
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'questions': questions
            },
            'message': '获取题目成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '获取题目失败'
        }), 500

@app.route('/api/recommend/symbols', methods=['POST'])
def recommend_symbols():
    """推荐数学符号"""
    try:
        data = request.get_json() or {}
        
        # 基础符号推荐
        symbols = [
            {"id": 1, "symbol": "x", "description": "未知数x", "category": "variable", "relevance": 0.9},
            {"id": 2, "symbol": "²", "description": "平方", "category": "operator", "relevance": 0.8},
            {"id": 3, "symbol": "=", "description": "等号", "category": "operator", "relevance": 0.9},
            {"id": 4, "symbol": "±", "description": "正负号", "category": "operator", "relevance": 0.7},
            {"id": 5, "symbol": "√", "description": "根号", "category": "operator", "relevance": 0.6}
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'symbols': symbols
            },
            'message': '获取符号推荐成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '获取符号推荐失败'
        }), 500

# 静态文件服务
@app.route('/static/<path:filename>')
def serve_static(filename):
    """服务静态文件"""
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'homework_system', 'dist', 'static')
    if os.path.exists(os.path.join(static_dir, filename)):
        return send_from_directory(static_dir, filename)
    return jsonify({'error': '文件不存在'}), 404

# 前端路由
@app.route('/')
@app.route('/homework')
@app.route('/login')
@app.route('/register')
def serve_frontend():
    """服务前端页面"""
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'homework_system', 'dist')
    index_file = os.path.join(frontend_dir, 'index.html')
    if os.path.exists(index_file):
        return send_file(index_file)
    return jsonify({
        'error': '前端文件未找到',
        'message': '请确保Vue前端已正确构建'
    }), 404

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    # 如果是API请求，返回JSON错误
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API端点不存在'}), 404
    # 否则返回前端应用
    return serve_frontend()

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': '服务器内部错误'
    }), 500

if __name__ == '__main__':
    print("🚀 启动数学教育后端服务...")
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"🌐 服务地址: http://0.0.0.0:8081")
    
    app.run(debug=True, host='0.0.0.0', port=8081)
