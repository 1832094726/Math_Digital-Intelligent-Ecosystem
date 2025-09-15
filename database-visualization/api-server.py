#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库可视化API服务器
提供实时数据查询接口
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pymysql
import json
from decimal import Decimal
import os
import sys
import re
import glob
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 数据库配置（与作业系统保持一致）
DB_CONFIG = {
    'host': 'obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud',
    'port': 3306,
    'user': 'hcj',
    'password': 'Xv0Mu8_:',
    'database': 'testccnu',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """获取数据库连接"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None

def serialize_decimal(obj):
    """序列化Decimal对象"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class APIAnalyzer:
    """API分析器，扫描和分析项目中的所有API"""

    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.apis = {}
        self.table_api_mapping = {}

    def scan_apis(self):
        """扫描项目中的所有API"""
        print("🔍 开始扫描API...")

        # 扫描作业系统后端API
        homework_backend_path = os.path.join(self.project_root, 'homework-backend')
        self._scan_flask_apis(homework_backend_path)

        # 扫描数据库可视化API
        self._scan_current_apis()

        # 应用补充数据
        self._apply_api_supplements()

        print(f"✅ 扫描完成，发现 {len(self.apis)} 个API端点")
        return self.apis

    def _apply_api_supplements(self):
        """应用API补充数据"""
        try:
            # 读取补充数据文件
            supplements_file = os.path.join(os.path.dirname(__file__), 'api-supplements.json')
            if os.path.exists(supplements_file):
                with open(supplements_file, 'r', encoding='utf-8') as f:
                    supplements = json.load(f)

                for api_id, supplement_data in supplements.items():
                    if api_id in self.apis:
                        # 合并补充数据
                        api = self.apis[api_id]

                        # 更新描述
                        if supplement_data.get('description'):
                            api['description'] = supplement_data['description']

                        # 更新参数
                        if supplement_data.get('parameters'):
                            api['parameters'].update(supplement_data['parameters'])

                        # 更新响应格式
                        if supplement_data.get('responses'):
                            api['responses'].update(supplement_data['responses'])

                        # 更新数据库表
                        if supplement_data.get('database_tables'):
                            api['database_tables'].extend(supplement_data['database_tables'])
                            api['database_tables'] = list(set(api['database_tables']))  # 去重

                        # 更新示例
                        if supplement_data.get('example_request'):
                            api['example_request'] = supplement_data['example_request']

                        if supplement_data.get('example_response'):
                            api['example_response'] = supplement_data['example_response']

                print(f"✅ 应用了 {len(supplements)} 个API的补充数据")
            else:
                print("ℹ️ 未找到API补充数据文件")

        except Exception as e:
            print(f"⚠️ 应用API补充数据失败: {e}")

    def _scan_flask_apis(self, backend_path):
        """扫描Flask应用的API"""
        try:
            # 扫描blueprints目录
            blueprints_path = os.path.join(backend_path, 'blueprints')
            if os.path.exists(blueprints_path):
                for file_path in glob.glob(os.path.join(blueprints_path, '*.py')):
                    self._analyze_flask_file(file_path, 'blueprints')

            # 扫描routes目录
            routes_path = os.path.join(backend_path, 'routes')
            if os.path.exists(routes_path):
                for file_path in glob.glob(os.path.join(routes_path, '*.py')):
                    self._analyze_flask_file(file_path, 'routes')

            # 扫描主应用文件
            app_file = os.path.join(backend_path, 'app.py')
            if os.path.exists(app_file):
                self._analyze_flask_file(app_file, 'main')

        except Exception as e:
            print(f"扫描Flask API失败: {e}")

    def _analyze_flask_file(self, file_path, category):
        """分析Flask文件中的API端点"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找路由装饰器
            route_pattern = r'@(\w+\.)?route\([\'"]([^\'"]+)[\'"](?:,\s*methods\s*=\s*\[([^\]]+)\])?\)'
            function_pattern = r'def\s+(\w+)\s*\([^)]*\):'

            lines = content.split('\n')
            current_api = None

            for i, line in enumerate(lines):
                # 查找路由装饰器
                route_match = re.search(route_pattern, line)
                if route_match:
                    blueprint_name = route_match.group(1) if route_match.group(1) else 'app'
                    path = route_match.group(2)
                    methods = route_match.group(3)

                    if methods:
                        methods = [m.strip().strip('\'"') for m in methods.split(',')]
                    else:
                        methods = ['GET']

                    # 查找下一行的函数定义
                    for j in range(i + 1, min(i + 5, len(lines))):
                        func_match = re.search(function_pattern, lines[j])
                        if func_match:
                            func_name = func_match.group(1)

                            # 分析API详情
                            api_info = self._extract_api_details(content, func_name, i)

                            # 改进分类逻辑
                            functional_category = self._categorize_api_by_function(func_name, path, os.path.basename(file_path))

                            api_id = f"{category}_{func_name}"
                            self.apis[api_id] = {
                                'id': api_id,
                                'name': func_name,
                                'path': path,
                                'methods': methods,
                                'category': functional_category,  # 使用功能分类
                                'technical_category': category,   # 保留技术分类
                                'file': os.path.basename(file_path),
                                'description': api_info.get('description', ''),
                                'parameters': api_info.get('parameters', {}),
                                'responses': api_info.get('responses', {}),
                                'database_tables': api_info.get('tables', []),
                                'example_request': api_info.get('example_request', {}),
                                'example_response': api_info.get('example_response', {})
                            }

                            # 建立表与API的映射关系
                            for table in api_info.get('tables', []):
                                if table not in self.table_api_mapping:
                                    self.table_api_mapping[table] = []
                                self.table_api_mapping[table].append(api_id)

                            break

        except Exception as e:
            print(f"分析文件 {file_path} 失败: {e}")

    def _extract_api_details(self, content, func_name, route_line):
        """提取API详细信息"""
        details = {
            'description': '',
            'parameters': {},
            'responses': {},
            'tables': [],
            'example_request': {},
            'example_response': {}
        }

        try:
            # 查找函数定义位置
            func_pattern = f'def\\s+{func_name}\\s*\\([^)]*\\):'
            func_match = re.search(func_pattern, content)
            if not func_match:
                return details

            func_start = func_match.start()

            # 查找函数结束位置
            func_end = self._find_function_end(content, func_start)
            func_content = content[func_start:func_end]

            # 提取函数文档字符串 - 改进版
            details['description'] = self._extract_docstring(func_content)

            # 提取数据库表引用 - 改进版
            details['tables'] = self._extract_database_tables(func_content)

            # 提取请求参数 - 改进版
            details['parameters'] = self._extract_request_parameters(func_content)

            # 提取响应格式 - 改进版
            details['responses'] = self._extract_response_format(func_content)

            # 提取示例 - 改进版
            details['example_request'], details['example_response'] = self._extract_examples(func_content)

        except Exception as e:
            print(f"提取API详情失败: {e}")

        return details

    def _find_function_end(self, content, func_start):
        """查找函数结束位置"""
        lines = content[func_start:].split('\n')
        indent_level = None
        func_end = func_start

        for i, line in enumerate(lines):
            if i == 0:  # 跳过函数定义行
                continue

            if line.strip() == '':  # 跳过空行
                continue

            current_indent = len(line) - len(line.lstrip())

            if indent_level is None and line.strip():
                indent_level = current_indent

            # 如果遇到同级或更低级的缩进，说明函数结束
            if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                if i > 1:  # 确保不是函数的第一行
                    break

            func_end += len(line) + 1  # +1 for newline

        return min(func_end, len(content))

    def _extract_docstring(self, func_content):
        """提取函数文档字符串"""
        # 匹配三引号文档字符串
        docstring_patterns = [
            r'"""([^"]*(?:"[^"]*"[^"]*)*)"""',
            r"'''([^']*(?:'[^']*'[^']*)*)'''"
        ]

        for pattern in docstring_patterns:
            match = re.search(pattern, func_content, re.DOTALL)
            if match:
                docstring = match.group(1).strip()
                # 取第一行作为主要描述，去掉多余的引号
                first_line = docstring.split('\n')[0].strip().strip('"\'')
                return first_line if first_line else '暂无描述'

        return '暂无描述'

    def _extract_database_tables(self, func_content):
        """提取数据库表引用"""
        tables = set()

        # 更全面的表名匹配模式
        table_patterns = [
            # SQL语句中的表名
            r'FROM\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'INSERT\s+INTO\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'UPDATE\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'DELETE\s+FROM\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'JOIN\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            # cursor.execute中的表名
            r'cursor\.execute\([^)]*[\'"][^\'\"]*FROM\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'cursor\.execute\([^)]*[\'"][^\'\"]*INSERT\s+INTO\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'cursor\.execute\([^)]*[\'"][^\'\"]*UPDATE\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'cursor\.execute\([^)]*[\'"][^\'\"]*DELETE\s+FROM\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            # 直接的表名引用
            r'[\'"]SELECT[^\'\"]*FROM\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            r'[\'"]INSERT[^\'\"]*INTO\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?'
        ]

        for pattern in table_patterns:
            matches = re.findall(pattern, func_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match and len(match) > 2:  # 过滤太短的匹配
                    tables.add(match)

        # 过滤掉一些常见的非表名
        excluded = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'INTO', 'WHERE', 'ORDER', 'GROUP', 'HAVING', 'LIMIT'}
        tables = {t for t in tables if t.upper() not in excluded}

        return list(tables)

    def _extract_request_parameters(self, func_content):
        """提取请求参数"""
        parameters = {}

        # 匹配各种参数获取方式
        param_patterns = [
            # request.args.get('param')
            r'request\.args\.get\([\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]',
            # request.json.get('param') 或 data.get('param')
            r'(?:request\.json|data)\.get\([\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]',
            # request.form.get('param')
            r'request\.form\.get\([\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]',
            # 路径参数 <int:param>
            r'<(?:int|string|float):([a-zA-Z_][a-zA-Z0-9_]*)>',
        ]

        for pattern in param_patterns:
            matches = re.findall(pattern, func_content)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match:
                    parameters[match] = 'string'  # 默认类型

        # 尝试从函数参数中提取
        func_def_match = re.search(r'def\s+\w+\s*\(([^)]*)\):', func_content)
        if func_def_match:
            func_params = func_def_match.group(1)
            # 提取除了self和current_user_id之外的参数
            params = [p.strip() for p in func_params.split(',') if p.strip()]
            for param in params:
                if param and param not in ['self', 'current_user_id']:
                    param_name = param.split('=')[0].strip()
                    if param_name:
                        parameters[param_name] = 'string'

        return parameters

    def _extract_response_format(self, func_content):
        """提取响应格式"""
        responses = {}

        # 查找jsonify调用
        jsonify_pattern = r'jsonify\s*\(\s*\{([^}]*)\}'
        matches = re.findall(jsonify_pattern, func_content, re.DOTALL)

        if matches:
            # 解析第一个jsonify调用作为成功响应
            response_content = matches[0]
            response_fields = {}

            # 简单解析字段
            field_pattern = r'[\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]:\s*[\'"]?([^,}]+)[\'"]?'
            field_matches = re.findall(field_pattern, response_content)

            for field_name, field_value in field_matches:
                if field_name:
                    response_fields[field_name] = 'string'

            if response_fields:
                responses['200'] = response_fields

        # 查找错误响应
        error_patterns = [
            r'return\s+jsonify\([^)]*\),\s*(\d+)',
            r'jsonify\([^)]*\),\s*(\d+)'
        ]

        for pattern in error_patterns:
            error_matches = re.findall(pattern, func_content)
            for status_code in error_matches:
                if status_code != '200':
                    responses[status_code] = {'error': 'string', 'message': 'string'}

        return responses

    def _extract_examples(self, func_content):
        """提取示例请求和响应"""
        example_request = {}
        example_response = {}

        # 这里可以根据具体需要实现示例提取逻辑
        # 暂时返回空字典

        return example_request, example_response

    def _categorize_api_by_function(self, func_name, path, filename):
        """根据功能对API进行分类"""
        func_lower = func_name.lower()
        path_lower = path.lower()
        file_lower = filename.lower()

        # 用户认证相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['auth', 'login', 'logout', 'register', 'token', 'session', 'profile']):
            return 'authentication'

        # 作业管理相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['homework', 'assignment', 'create', 'publish', 'unpublish']):
            return 'homework_management'

        # 学生功能相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['student', 'submission', 'progress', 'favorite', 'dashboard']):
            return 'student_features'

        # 评分系统相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['grade', 'grading', 'score', 'review', 'result']):
            return 'grading_system'

        # 推荐系统相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['recommend', 'symbol', 'knowledge', 'exercise', 'learning', 'adaptive']):
            return 'recommendation_system'

        # 数据可视化相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['database', 'visualization', 'table', 'health', 'api']):
            return 'data_visualization'

        # 通知系统相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['notification', 'reminder', 'message']):
            return 'notification_system'

        # 班级管理相关
        if any(keyword in func_lower or keyword in path_lower or keyword in file_lower
               for keyword in ['class', 'teacher', 'statistics']):
            return 'class_management'

        # 默认分类
        return 'other'

    def _scan_current_apis(self):
        """扫描当前数据库可视化API"""
        current_apis = [
            {
                'id': 'db_viz_health',
                'name': 'health_check',
                'path': '/api/health',
                'methods': ['GET'],
                'category': 'database_visualization',
                'file': 'api-server.py',
                'description': '健康检查接口',
                'parameters': {},
                'responses': {'200': {'status': 'string', 'database': 'string', 'message': 'string'}},
                'database_tables': [],
                'example_request': {},
                'example_response': {'status': 'healthy', 'database': 'connected', 'message': '数据库API服务正常运行'}
            },
            {
                'id': 'db_viz_tables',
                'name': 'get_all_tables',
                'path': '/api/database/tables',
                'methods': ['GET'],
                'category': 'database_visualization',
                'file': 'api-server.py',
                'description': '获取所有表信息',
                'parameters': {},
                'responses': {'200': {'tables': 'array', 'total_tables': 'number'}},
                'database_tables': ['INFORMATION_SCHEMA.TABLES'],
                'example_request': {},
                'example_response': {'tables': [{'name': 'users', 'count': 100}], 'total_tables': 1}
            },
            {
                'id': 'db_viz_table_data',
                'name': 'get_table_data',
                'path': '/api/database/table/<table_name>',
                'methods': ['GET'],
                'category': 'database_visualization',
                'file': 'api-server.py',
                'description': '获取表数据',
                'parameters': {'table_name': 'string', 'limit': 'number', 'offset': 'number'},
                'responses': {'200': {'data': 'array', 'limit': 'number', 'offset': 'number', 'count': 'number'}},
                'database_tables': ['dynamic'],
                'example_request': {'limit': 10, 'offset': 0},
                'example_response': {'data': [], 'limit': 10, 'offset': 0, 'count': 0}
            }
        ]

        for api in current_apis:
            self.apis[api['id']] = api

# 全局API分析器实例
api_analyzer = APIAnalyzer()

def get_mock_data(table_name, limit=10, offset=0):
    """获取模拟数据"""
    mock_data = {
        'users': [
            {'id': 1, 'username': 'test_student_001', 'email': 'student@test.com', 'role': 'student', 'real_name': '测试学生', 'grade': 7, 'school': '测试中学', 'class_name': '七年级1班'},
            {'id': 2, 'username': 'test_teacher_001', 'email': 'teacher@test.com', 'role': 'teacher', 'real_name': '测试老师', 'grade': None, 'school': '测试中学', 'class_name': None},
            {'id': 3, 'username': 'test_admin', 'email': 'admin@test.com', 'role': 'admin', 'real_name': '系统管理员', 'grade': None, 'school': '测试中学', 'class_name': None}
        ],
        'schools': [
            {'id': 1, 'school_name': '北京市第一中学', 'school_code': 'BJ001', 'school_type': 'middle', 'education_level': '初中', 'is_active': 1, 'created_at': '2024-01-15 09:00:00', 'updated_at': '2024-01-15 09:00:00'},
            {'id': 2, 'school_name': '上海实验小学', 'school_code': 'SH002', 'school_type': 'primary', 'education_level': '小学', 'is_active': 1, 'created_at': '2024-01-16 10:30:00', 'updated_at': '2024-01-16 10:30:00'},
            {'id': 3, 'school_name': '深圳科技高中', 'school_code': 'SZ003', 'school_type': 'high', 'education_level': '高中', 'is_active': 1, 'created_at': '2024-01-17 14:20:00', 'updated_at': '2024-01-17 14:20:00'},
            {'id': 4, 'school_name': '广州外国语学校', 'school_code': 'GZ004', 'school_type': 'comprehensive', 'education_level': '九年一贯制', 'is_active': 1, 'created_at': '2024-01-18 08:45:00', 'updated_at': '2024-01-18 08:45:00'},
            {'id': 5, 'school_name': '杭州西湖中学', 'school_code': 'HZ005', 'school_type': 'middle', 'education_level': '初中', 'is_active': 0, 'created_at': '2024-01-19 16:15:00', 'updated_at': '2024-01-19 16:15:00'}
        ],
        'grades': [
            {'id': 1, 'school_id': 1, 'grade_name': '七年级', 'grade_level': 7, 'academic_year': '2024-2025', 'is_active': 1},
            {'id': 2, 'school_id': 1, 'grade_name': '八年级', 'grade_level': 8, 'academic_year': '2024-2025', 'is_active': 1},
            {'id': 3, 'school_id': 2, 'grade_name': '三年级', 'grade_level': 3, 'academic_year': '2024-2025', 'is_active': 1}
        ],
        'subjects': [
            {'id': 1, 'subject_name': '数学', 'subject_code': 'MATH', 'description': '数学学科', 'is_active': 1},
            {'id': 2, 'subject_name': '语文', 'subject_code': 'CHINESE', 'description': '语文学科', 'is_active': 1},
            {'id': 3, 'subject_name': '英语', 'subject_code': 'ENGLISH', 'description': '英语学科', 'is_active': 1},
            {'id': 4, 'subject_name': '物理', 'subject_code': 'PHYSICS', 'description': '物理学科', 'is_active': 1},
            {'id': 5, 'subject_name': '化学', 'subject_code': 'CHEMISTRY', 'description': '化学学科', 'is_active': 1}
        ],
        'homeworks': [
            {'id': 1, 'title': '七年级数学第一章练习', 'subject': '数学', 'grade': 7, 'difficulty_level': 3, 'question_count': 10, 'max_score': 100, 'is_published': 1},
            {'id': 2, 'title': '代数基础练习', 'subject': '数学', 'grade': 7, 'difficulty_level': 2, 'question_count': 8, 'max_score': 80, 'is_published': 1},
            {'id': 3, 'title': '几何图形认识', 'subject': '数学', 'grade': 7, 'difficulty_level': 4, 'question_count': 12, 'max_score': 120, 'is_published': 0}
        ]
    }

    data = mock_data.get(table_name, [])
    total = len(data)

    # 应用分页
    start = offset
    end = offset + limit
    paginated_data = data[start:end]

    return {
        'data': paginated_data,
        'limit': limit,
        'offset': offset,
        'table': table_name,
        'count': len(paginated_data),
        'total': total,
        'source': 'mock'
    }

@app.route('/api/database/table/<table_name>', methods=['GET'])
def get_table_data(table_name):
    """获取表数据"""
    try:
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))

        # 限制查询数量，防止过大查询
        limit = min(limit, 100)

        connection = get_db_connection()
        if not connection:
            print(f"数据库连接失败，使用模拟数据 - 表: {table_name}")
            return jsonify(get_mock_data(table_name, limit, offset))

        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # 检查表是否存在
                cursor.execute("SHOW TABLES LIKE %s", (table_name,))
                if not cursor.fetchone():
                    print(f"表 {table_name} 不存在，使用模拟数据")
                    return jsonify(get_mock_data(table_name, limit, offset))

                # 查询数据
                query = f"SELECT * FROM `{table_name}` LIMIT %s OFFSET %s"
                cursor.execute(query, (limit, offset))
                rows = cursor.fetchall()

                # 处理Decimal类型
                for row in rows:
                    for key, value in row.items():
                        if isinstance(value, Decimal):
                            row[key] = float(value)

                return jsonify({
                    'data': rows,
                    'limit': limit,
                    'offset': offset,
                    'table': table_name,
                    'count': len(rows),
                    'source': 'database'
                })
        except Exception as e:
            print(f"数据库查询失败: {e}，使用模拟数据")
            return jsonify(get_mock_data(table_name, limit, offset))

    except Exception as e:
        print(f"API错误: {e}，使用模拟数据")
        return jsonify(get_mock_data(table_name, limit, offset))
    finally:
        if connection:
            connection.close()

@app.route('/api/database/table/<table_name>/count', methods=['GET'])
def get_table_count(table_name):
    """获取表记录总数"""
    try:
        connection = get_db_connection()
        if not connection:
            print(f"数据库连接失败，使用模拟数据计数 - 表: {table_name}")
            mock_data = get_mock_data(table_name, 1000, 0)
            return jsonify({
                'table': table_name,
                'count': mock_data['total'],
                'source': 'mock'
            })

        try:
            with connection.cursor() as cursor:
                # 检查表是否存在
                cursor.execute("SHOW TABLES LIKE %s", (table_name,))
                if not cursor.fetchone():
                    print(f"表 {table_name} 不存在，使用模拟数据计数")
                    mock_data = get_mock_data(table_name, 1000, 0)
                    return jsonify({
                        'table': table_name,
                        'count': mock_data['total'],
                        'source': 'mock'
                    })

                # 查询记录总数
                cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}`")
                result = cursor.fetchone()

                return jsonify({
                    'table': table_name,
                    'count': result[0],
                    'source': 'database'
                })
        except Exception as e:
            print(f"数据库查询失败: {e}，使用模拟数据计数")
            mock_data = get_mock_data(table_name, 1000, 0)
            return jsonify({
                'table': table_name,
                'count': mock_data['total'],
                'source': 'mock'
            })

    except Exception as e:
        print(f"API错误: {e}，使用模拟数据计数")
        mock_data = get_mock_data(table_name, 1000, 0)
        return jsonify({
            'table': table_name,
            'count': mock_data['total'],
            'source': 'mock'
        })
    finally:
        if connection:
            connection.close()

@app.route('/api/database/tables', methods=['GET'])
def get_all_tables():
    """获取所有表信息"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': '数据库连接失败'}), 500
        
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # 获取所有表
            cursor.execute("SHOW TABLES")
            tables = [list(row.values())[0] for row in cursor.fetchall()]
            
            # 获取每个表的记录数
            table_info = []
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM `{table}`")
                count = cursor.fetchone()['count']
                table_info.append({
                    'name': table,
                    'count': count
                })
            
            return jsonify({
                'tables': table_info,
                'total_tables': len(tables)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if connection:
            connection.close()

@app.route('/api/database/table/<table_name>/structure', methods=['GET'])
def get_table_structure(table_name):
    """获取表结构信息"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': '数据库连接失败'}), 500
        
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # 检查表是否存在
            cursor.execute("SHOW TABLES LIKE %s", (table_name,))
            if not cursor.fetchone():
                return jsonify({'error': f'表 {table_name} 不存在'}), 404
            
            # 获取表结构
            cursor.execute(f"DESCRIBE `{table_name}`")
            columns = cursor.fetchall()
            
            # 获取索引信息
            cursor.execute(f"SHOW INDEX FROM `{table_name}`")
            indexes = cursor.fetchall()
            
            return jsonify({
                'table': table_name,
                'columns': columns,
                'indexes': indexes
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if connection:
            connection.close()

@app.route('/api/apis', methods=['GET'])
def get_all_apis():
    """获取所有API信息"""
    try:
        # 扫描API
        apis = api_analyzer.scan_apis()

        # 按类别分组
        categorized_apis = {}
        for api_id, api_info in apis.items():
            category = api_info['category']
            if category not in categorized_apis:
                categorized_apis[category] = []
            categorized_apis[category].append(api_info)

        return jsonify({
            'success': True,
            'apis': apis,
            'categorized_apis': categorized_apis,
            'total_apis': len(apis),
            'categories': list(categorized_apis.keys())
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/apis/<api_id>', methods=['GET'])
def get_api_details(api_id):
    """获取特定API的详细信息"""
    try:
        apis = api_analyzer.scan_apis()

        if api_id not in apis:
            return jsonify({
                'success': False,
                'error': f'API {api_id} 不存在'
            }), 404

        api_info = apis[api_id]

        # 添加相关表的信息
        related_tables = []
        for table_name in api_info['database_tables']:
            if table_name != 'dynamic' and table_name != 'INFORMATION_SCHEMA.TABLES':
                # 获取表的基本信息
                try:
                    connection = get_db_connection()
                    if connection:
                        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                            cursor.execute("SHOW TABLES LIKE %s", (table_name,))
                            if cursor.fetchone():
                                cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}`")
                                count_result = cursor.fetchone()
                                related_tables.append({
                                    'name': table_name,
                                    'count': count_result['count'] if count_result else 0,
                                    'exists': True
                                })
                            else:
                                related_tables.append({
                                    'name': table_name,
                                    'count': 0,
                                    'exists': False
                                })
                        connection.close()
                except Exception as e:
                    related_tables.append({
                        'name': table_name,
                        'count': 0,
                        'exists': False,
                        'error': str(e)
                    })

        api_info['related_tables_info'] = related_tables

        return jsonify({
            'success': True,
            'api': api_info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tables/<table_name>/apis', methods=['GET'])
def get_table_apis(table_name):
    """获取使用特定表的所有API"""
    try:
        apis = api_analyzer.scan_apis()
        table_apis = []

        for api_id, api_info in apis.items():
            if table_name in api_info['database_tables']:
                table_apis.append(api_info)

        return jsonify({
            'success': True,
            'table': table_name,
            'apis': table_apis,
            'total_apis': len(table_apis)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        connection = get_db_connection()
        if connection:
            connection.close()
            return jsonify({
                'status': 'healthy',
                'database': 'connected',
                'message': '数据库API服务正常运行'
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'database': 'disconnected',
                'message': '数据库连接失败'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    print("🚀 启动数据库可视化API服务器...")
    print("📊 API端点:")
    print("   - GET /api/health - 健康检查")
    print("   - GET /api/database/tables - 获取所有表")
    print("   - GET /api/database/table/<name> - 获取表数据")
    print("   - GET /api/database/table/<name>/count - 获取表记录数")
    print("   - GET /api/database/table/<name>/structure - 获取表结构")
    print("🌐 服务地址: http://localhost:5001")
    print("🔗 跨域支持: 已启用")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )
