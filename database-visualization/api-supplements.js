// API信息补充数据
// 用于补充自动扫描无法获取的详细信息

const apiSupplements = {
    // 用户认证相关API
    'routes_login': {
        description: '用户登录认证，支持学生、教师、管理员登录',
        parameters: {
            'username': { type: 'string', required: true, description: '用户名或邮箱' },
            'password': { type: 'string', required: true, description: '用户密码' },
            'device_type': { type: 'string', required: false, description: '设备类型' },
            'device_id': { type: 'string', required: false, description: '设备唯一标识' }
        },
        responses: {
            '200': {
                'success': 'boolean',
                'access_token': 'string',
                'refresh_token': 'string',
                'user': 'object',
                'expires_in': 'number'
            },
            '401': {
                'success': 'boolean',
                'message': 'string',
                'error': 'object'
            }
        },
        database_tables: ['users', 'user_sessions'],
        example_request: {
            'username': 'test_student_001',
            'password': 'student123',
            'device_type': 'web'
        },
        example_response: {
            'success': true,
            'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...',
            'user': {
                'id': 2,
                'username': 'test_student_001',
                'role': 'student',
                'real_name': '测试学生'
            }
        }
    },
    
    'routes_register': {
        description: '用户注册，创建新的学生、教师或管理员账户',
        parameters: {
            'username': { type: 'string', required: true, description: '用户名' },
            'email': { type: 'string', required: true, description: '邮箱地址' },
            'password': { type: 'string', required: true, description: '密码' },
            'real_name': { type: 'string', required: true, description: '真实姓名' },
            'role': { type: 'string', required: true, description: '用户角色：student/teacher/admin' },
            'grade': { type: 'number', required: false, description: '年级（学生必填）' },
            'school': { type: 'string', required: false, description: '学校名称' },
            'class_name': { type: 'string', required: false, description: '班级名称' }
        },
        responses: {
            '201': {
                'success': 'boolean',
                'message': 'string',
                'user_id': 'number'
            },
            '400': {
                'success': 'boolean',
                'message': 'string',
                'errors': 'object'
            }
        },
        database_tables: ['users'],
        example_request: {
            'username': 'new_student',
            'email': 'student@example.com',
            'password': 'password123',
            'real_name': '新学生',
            'role': 'student',
            'grade': 7,
            'school': '测试中学'
        }
    },
    
    // 作业管理相关API
    'routes_create_homework': {
        description: '创建新作业，教师可以创建包含多个题目的作业',
        parameters: {
            'title': { type: 'string', required: true, description: '作业标题' },
            'description': { type: 'string', required: false, description: '作业描述' },
            'subject': { type: 'string', required: true, description: '学科' },
            'grade': { type: 'number', required: true, description: '年级' },
            'difficulty_level': { type: 'number', required: true, description: '难度等级1-5' },
            'questions': { type: 'array', required: true, description: '题目列表' },
            'due_date': { type: 'string', required: false, description: '截止日期' },
            'max_score': { type: 'number', required: false, description: '总分' }
        },
        responses: {
            '201': {
                'success': 'boolean',
                'homework_id': 'number',
                'message': 'string'
            },
            '400': {
                'success': 'boolean',
                'message': 'string',
                'errors': 'array'
            }
        },
        database_tables: ['homeworks', 'questions', 'homework_questions'],
        example_request: {
            'title': '七年级数学第一章练习',
            'subject': '数学',
            'grade': 7,
            'difficulty_level': 3,
            'questions': [
                {
                    'content': '计算：2x + 3 = 7',
                    'type': 'calculation',
                    'score': 10
                }
            ]
        }
    },
    
    'routes_list_homeworks': {
        description: '获取作业列表，支持分页和筛选',
        parameters: {
            'page': { type: 'number', required: false, description: '页码，默认1' },
            'limit': { type: 'number', required: false, description: '每页数量，默认10' },
            'subject': { type: 'string', required: false, description: '学科筛选' },
            'grade': { type: 'number', required: false, description: '年级筛选' },
            'keyword': { type: 'string', required: false, description: '关键词搜索' },
            'category': { type: 'string', required: false, description: '分类筛选' }
        },
        responses: {
            '200': {
                'success': 'boolean',
                'homeworks': 'array',
                'total': 'number',
                'page': 'number',
                'total_pages': 'number'
            }
        },
        database_tables: ['homeworks', 'users'],
        example_response: {
            'success': true,
            'homeworks': [
                {
                    'id': 1,
                    'title': '七年级数学第一章练习',
                    'subject': '数学',
                    'grade': 7,
                    'difficulty_level': 3,
                    'question_count': 10,
                    'created_at': '2024-01-15T10:00:00'
                }
            ],
            'total': 25,
            'page': 1,
            'total_pages': 3
        }
    },
    
    // 推荐系统相关API
    'blueprints_recommend_knowledge_points': {
        description: '基于AI的知识点推荐，根据用户学习状态和上下文推荐相关知识点',
        parameters: {
            'question_id': { type: 'number', required: false, description: '题目ID，基于题目推荐' },
            'context': { type: 'string', required: false, description: '学习上下文内容' },
            'limit': { type: 'number', required: false, description: '推荐数量限制，默认5' }
        },
        responses: {
            '200': {
                'success': 'boolean',
                'recommendations': 'array',
                'total': 'number',
                'timestamp': 'string'
            }
        },
        database_tables: ['knowledge_points', 'knowledge_relationships', 'users'],
        example_request: {
            'context': '解一元二次方程',
            'limit': 3
        },
        example_response: {
            'success': true,
            'recommendations': [
                {
                    'id': 2,
                    'name': '代数表达式',
                    'description': '用字母和数字表示的数学表达式',
                    'grade_level': 2,
                    'difficulty_level': 2,
                    'relevance_score': 0.8,
                    'recommendation_reason': '与输入内容相关',
                    'related_points': [
                        {
                            'id': 1,
                            'name': '基本运算',
                            'relationship_type': 'prerequisite'
                        }
                    ]
                }
            ],
            'total': 1
        }
    },
    
    'blueprints_recommend_symbols': {
        description: '数学符号智能推荐，基于上下文和用户习惯推荐合适的数学符号',
        parameters: {
            'context': { type: 'string', required: true, description: '当前输入上下文' },
            'question_text': { type: 'string', required: false, description: '题目文本' },
            'limit': { type: 'number', required: false, description: '推荐数量，默认5' }
        },
        responses: {
            '200': {
                'success': 'boolean',
                'recommendations': 'array',
                'context_analysis': 'object'
            }
        },
        database_tables: ['symbol_recommendations', 'users'],
        example_request: {
            'context': '2x + 3 =',
            'question_text': '解方程：2x + 3 = 7',
            'limit': 5
        },
        example_response: {
            'success': true,
            'recommendations': [
                {
                    'symbol': '7',
                    'latex': '7',
                    'description': '数字7',
                    'category': 'number',
                    'confidence': 0.9
                },
                {
                    'symbol': 'x',
                    'latex': 'x',
                    'description': '变量x',
                    'category': 'variable',
                    'confidence': 0.8
                }
            ]
        }
    },
    
    // 评分系统相关API
    'routes_grade_submission': {
        description: '自动评分学生作业提交，支持多种题型的智能评分',
        parameters: {
            'submission_id': { type: 'number', required: true, description: '提交ID（路径参数）' }
        },
        responses: {
            '200': {
                'success': 'boolean',
                'grading_result': 'object',
                'total_score': 'number',
                'max_score': 'number'
            },
            '404': {
                'success': 'boolean',
                'message': 'string'
            }
        },
        database_tables: ['homework_submissions', 'grading_results', 'questions'],
        example_response: {
            'success': true,
            'grading_result': {
                'submission_id': 123,
                'total_score': 85,
                'max_score': 100,
                'question_results': [
                    {
                        'question_id': 1,
                        'score': 10,
                        'max_score': 10,
                        'is_correct': true
                    }
                ]
            }
        }
    },
    
    // 数据可视化相关API
    'db_viz_table_data': {
        description: '获取数据库表的实时数据，支持分页和筛选',
        parameters: {
            'table_name': { type: 'string', required: true, description: '表名（路径参数）' },
            'limit': { type: 'number', required: false, description: '查询数量限制，默认10' },
            'offset': { type: 'number', required: false, description: '偏移量，默认0' }
        },
        responses: {
            '200': {
                'data': 'array',
                'limit': 'number',
                'offset': 'number',
                'count': 'number',
                'table': 'string',
                'source': 'string'
            }
        },
        database_tables: ['dynamic'],
        example_request: {
            'limit': 10,
            'offset': 0
        },
        example_response: {
            'data': [
                {
                    'id': 1,
                    'username': 'test_student_001',
                    'role': 'student',
                    'created_at': '2024-01-15T10:00:00'
                }
            ],
            'limit': 10,
            'offset': 0,
            'count': 1,
            'table': 'users',
            'source': 'database'
        }
    }
};

// 导出补充数据
if (typeof module !== 'undefined' && module.exports) {
    module.exports = apiSupplements;
} else if (typeof window !== 'undefined') {
    window.apiSupplements = apiSupplements;
}
