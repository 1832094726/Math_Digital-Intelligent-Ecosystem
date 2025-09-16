// 数据库API接口
class DatabaseAPI {
    constructor() {
        this.baseURL = 'http://172.104.172.5:5001/api';
        this.cache = new Map();
        this.cacheTimeout = 30000; // 30秒缓存
    }

    // 获取表的实时数据
    async getTableData(tableName, limit = 10, offset = 0) {
        const cacheKey = `${tableName}_${limit}_${offset}`;
        
        // 检查缓存
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }

        try {
            const response = await fetch(`${this.baseURL}/database/table/${tableName}?limit=${limit}&offset=${offset}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // 缓存结果
            this.cache.set(cacheKey, {
                data: data,
                timestamp: Date.now()
            });

            return data;
        } catch (error) {
            console.error(`获取表 ${tableName} 数据失败:`, error);
            return this.getMockData(tableName, limit);
        }
    }

    // 获取表的记录总数
    async getTableCount(tableName) {
        const cacheKey = `${tableName}_count`;
        
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }

        try {
            const response = await fetch(`${this.baseURL}/database/table/${tableName}/count`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            this.cache.set(cacheKey, {
                data: data.count,
                timestamp: Date.now()
            });

            return data.count;
        } catch (error) {
            console.error(`获取表 ${tableName} 记录数失败:`, error);
            return this.getMockCount(tableName);
        }
    }

    // 获取认证token
    getAuthToken() {
        return localStorage.getItem('auth_token') || 'mock_token';
    }

    // 模拟数据（当API不可用时）
    getMockData(tableName, limit) {
        const mockData = {
            users: [
                { id: 1, username: 'test_student_001', email: 'student@test.com', role: 'student', real_name: '测试学生', grade: 7, school: '测试中学', class_name: '七年级1班' },
                { id: 2, username: 'test_teacher_001', email: 'teacher@test.com', role: 'teacher', real_name: '测试老师', grade: null, school: '测试中学', class_name: null },
                { id: 3, username: 'test_admin', email: 'admin@test.com', role: 'admin', real_name: '系统管理员', grade: null, school: '测试中学', class_name: null }
            ],
            homeworks: [
                { id: 1, title: '七年级数学第一章练习', subject: '数学', grade: 7, difficulty_level: 3, question_count: 10, max_score: 100, is_published: 1 },
                { id: 2, title: '代数基础练习', subject: '数学', grade: 7, difficulty_level: 2, question_count: 8, max_score: 80, is_published: 1 },
                { id: 3, title: '几何图形认识', subject: '数学', grade: 7, difficulty_level: 4, question_count: 12, max_score: 120, is_published: 0 }
            ],
            questions: [
                { id: 1, homework_id: 1, content: '计算：2x + 3 = 7，求x的值', question_type: 'calculation', score: 10.00, difficulty: 2, order_index: 1 },
                { id: 2, homework_id: 1, content: '下列哪个是质数？', question_type: 'single_choice', score: 5.00, difficulty: 1, order_index: 2 },
                { id: 3, homework_id: 2, content: '化简：3(x+2) - 2(x-1)', question_type: 'calculation', score: 15.00, difficulty: 3, order_index: 1 }
            ],
            homework_submissions: [
                { id: 1, student_id: 1, homework_id: 1, score: 85.00, max_score: 100.00, status: 'graded', time_spent: 1800, attempt_count: 1 },
                { id: 2, student_id: 1, homework_id: 2, score: 72.00, max_score: 80.00, status: 'graded', time_spent: 1200, attempt_count: 2 }
            ],
            schools: [
                { id: 1, school_name: '北京市第一中学', school_code: 'BJ001', school_type: 'middle', education_level: '初中', is_active: 1, created_at: '2024-01-15 09:00:00', updated_at: '2024-01-15 09:00:00' },
                { id: 2, school_name: '上海实验小学', school_code: 'SH002', school_type: 'primary', education_level: '小学', is_active: 1, created_at: '2024-01-16 10:30:00', updated_at: '2024-01-16 10:30:00' },
                { id: 3, school_name: '深圳科技高中', school_code: 'SZ003', school_type: 'high', education_level: '高中', is_active: 1, created_at: '2024-01-17 14:20:00', updated_at: '2024-01-17 14:20:00' },
                { id: 4, school_name: '广州外国语学校', school_code: 'GZ004', school_type: 'comprehensive', education_level: '九年一贯制', is_active: 1, created_at: '2024-01-18 08:45:00', updated_at: '2024-01-18 08:45:00' },
                { id: 5, school_name: '杭州西湖中学', school_code: 'HZ005', school_type: 'middle', education_level: '初中', is_active: 0, created_at: '2024-01-19 16:15:00', updated_at: '2024-01-19 16:15:00' }
            ],
            grades: [
                { id: 1, school_id: 1, grade_name: '七年级', grade_level: 7, academic_year: '2024-2025', is_active: 1 }
            ],
            classes: [
                { id: 1, school_id: 1, grade_id: 1, class_name: '七年级1班', class_code: 'G7C1', head_teacher_id: 2, is_active: 1 }
            ]
        };

        const data = mockData[tableName] || [];
        return {
            data: data.slice(0, limit),
            total: data.length,
            limit: limit,
            offset: 0,
            table: tableName
        };
    }

    // 模拟记录数
    getMockCount(tableName) {
        const mockCounts = {
            users: 12,
            homeworks: 6,
            questions: 42,
            homework_submissions: 19,
            homework_progress: 25,
            schools: 5,
            grades: 3,
            classes: 8,
            class_students: 35,
            subjects: 5,
            courses: 12,
            course_modules: 24,
            chapters: 48,
            lessons: 120,
            class_schedules: 15,
            homework_assignments: 18,
            homework_favorites: 8,
            homework_reminders: 12,
            learning_paths: 15,
            notifications: 45,
            user_sessions: 8,
            curriculum_standards: 2
        };
        return mockCounts[tableName] || 0;
    }

    // 清除缓存
    clearCache() {
        this.cache.clear();
    }

    // 格式化数据显示
    formatValue(value, fieldType) {
        if (value === null || value === undefined) {
            return '<span style="color: #666; font-style: italic;">NULL</span>';
        }

        if (typeof value === 'string' && value.length > 50) {
            return value.substring(0, 50) + '...';
        }

        if (fieldType && fieldType.includes('datetime')) {
            return new Date(value).toLocaleString('zh-CN');
        }

        if (fieldType && fieldType.includes('json')) {
            try {
                return JSON.stringify(JSON.parse(value), null, 2);
            } catch {
                return value;
            }
        }

        return value;
    }

    // 获取字段类型的显示颜色
    getFieldTypeColor(fieldType) {
        if (fieldType.includes('int') || fieldType.includes('bigint')) {
            return '#007bff';
        }
        if (fieldType.includes('varchar') || fieldType.includes('text')) {
            return '#28a745';
        }
        if (fieldType.includes('datetime') || fieldType.includes('timestamp')) {
            return '#ffc107';
        }
        if (fieldType.includes('decimal') || fieldType.includes('float')) {
            return '#17a2b8';
        }
        if (fieldType.includes('json')) {
            return '#6f42c1';
        }
        if (fieldType.includes('enum')) {
            return '#fd7e14';
        }
        return '#6c757d';
    }
}

// 创建全局实例
window.databaseAPI = new DatabaseAPI();
