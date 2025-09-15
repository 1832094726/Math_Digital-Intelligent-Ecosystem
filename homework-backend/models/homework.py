# -*- coding: utf-8 -*-
"""
作业模型
"""
import json
from datetime import datetime
from models.database import db

class Homework:
    def __init__(self, id=None, title=None, description=None, subject=None, grade=None,
                 difficulty_level=3, question_count=0, max_score=100, time_limit=None,
                 due_date=None, start_date=None, is_published=0, is_template=0,
                 created_by=None, category=None, tags=None, instructions=None,
                 auto_grade=1, max_attempts=1, show_answers=0, created_at=None, updated_at=None):
        self.id = id
        self.title = title
        self.description = description
        self.subject = subject
        self.grade = grade
        self.difficulty_level = difficulty_level
        self.question_count = question_count
        self.max_score = max_score
        self.time_limit = time_limit
        self.due_date = due_date
        self.start_date = start_date
        self.is_published = is_published
        self.is_template = is_template
        self.created_by = created_by
        self.category = category
        self.tags = json.dumps(tags) if isinstance(tags, (list, dict)) else tags
        self.instructions = instructions
        self.auto_grade = auto_grade
        self.max_attempts = max_attempts
        self.show_answers = show_answers
        self.created_at = created_at
        self.updated_at = updated_at

    @staticmethod
    def create(homework_data):
        """创建新作业"""
        try:
            sql = """
            INSERT INTO homeworks (title, description, subject, grade, difficulty_level, 
            question_count, max_score, time_limit, due_date, start_date, is_published, 
            is_template, created_by, category, tags, instructions, auto_grade, 
            max_attempts, show_answers)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                homework_data.get('title'),
                homework_data.get('description'),
                homework_data.get('subject'),
                homework_data.get('grade'),
                homework_data.get('difficulty_level', 3),
                homework_data.get('question_count', 0),
                homework_data.get('max_score', 100),
                homework_data.get('time_limit'),
                homework_data.get('due_date'),
                homework_data.get('start_date'),
                homework_data.get('is_published', 0),
                homework_data.get('is_template', 0),
                homework_data.get('created_by'),
                homework_data.get('category'),
                json.dumps(homework_data.get('tags', [])) if homework_data.get('tags') else None,
                homework_data.get('instructions'),
                homework_data.get('auto_grade', 1),
                homework_data.get('max_attempts', 1),
                homework_data.get('show_answers', 0)
            )
            
            homework_id = db.execute_insert(sql, params)
            return Homework.get_by_id(homework_id)
            
        except Exception as e:
            print(f"创建作业失败: {e}")
            raise

    @staticmethod
    def get_by_id(homework_id):
        """根据ID获取作业"""
        sql = "SELECT * FROM homeworks WHERE id = %s"
        result = db.execute_query_one(sql, (homework_id,))
        return Homework(**result) if result else None

    @staticmethod
    def get_by_teacher(teacher_id, page=1, limit=10):
        """获取教师创建的作业列表"""
        offset = (page - 1) * limit
        sql = """
        SELECT * FROM homeworks 
        WHERE created_by = %s 
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        results = db.execute_query(sql, (teacher_id, limit, offset))
        return [Homework(**row) for row in results]

    @staticmethod
    def get_published(grade=None, subject=None, page=1, limit=10):
        """获取已发布的作业列表"""
        offset = (page - 1) * limit
        
        where_conditions = ["is_published = 1"]
        params = []
        
        if grade:
            where_conditions.append("grade = %s")
            params.append(grade)
        
        if subject:
            where_conditions.append("subject = %s")
            params.append(subject)
        
        params.extend([limit, offset])
        
        sql = f"""
        SELECT * FROM homeworks 
        WHERE {' AND '.join(where_conditions)}
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        
        results = db.execute_query(sql, params)
        return [Homework(**row) for row in results]

    def update(self, update_data):
        """更新作业信息"""
        try:
            allowed_fields = [
                'title', 'description', 'subject', 'grade', 'difficulty_level',
                'question_count', 'max_score', 'time_limit', 'due_date', 'start_date',
                'is_published', 'is_template', 'category', 'tags', 'instructions',
                'auto_grade', 'max_attempts', 'show_answers'
            ]
            
            update_fields = []
            update_values = []
            
            for field in allowed_fields:
                if field in update_data:
                    update_fields.append(f"{field} = %s")
                    
                    # 处理JSON字段
                    if field == 'tags':
                        value = update_data[field]
                        if isinstance(value, (list, dict)):
                            update_values.append(json.dumps(value))
                        else:
                            update_values.append(value)
                    else:
                        update_values.append(update_data[field])
            
            if update_fields:
                update_fields.append("updated_at = %s")
                update_values.append(datetime.now())
                update_values.append(self.id)
                
                sql = f"UPDATE homeworks SET {', '.join(update_fields)} WHERE id = %s"
                db.execute_update(sql, update_values)
                
                # 更新实例属性
                for field in allowed_fields:
                    if field in update_data:
                        setattr(self, field, update_data[field])
                self.updated_at = datetime.now()
                
                return True
            return False
            
        except Exception as e:
            print(f"更新作业失败: {e}")
            raise

    def delete(self):
        """删除作业"""
        try:
            sql = "DELETE FROM homeworks WHERE id = %s"
            db.execute_delete(sql, (self.id,))
            return True
        except Exception as e:
            print(f"删除作业失败: {e}")
            raise

    def publish(self):
        """发布作业"""
        return self.update({'is_published': 1})

    def unpublish(self):
        """取消发布作业"""
        return self.update({'is_published': 0})

    def to_dict(self):
        """转换为字典"""
        # 动态计算题目数量
        actual_question_count = self._get_actual_question_count()

        # 基础数据
        result = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "subject": self.subject,
            "grade": self.grade,
            "difficulty_level": self.difficulty_level,
            "question_count": actual_question_count,
            "max_score": self.max_score,
            "time_limit": self.time_limit,
            "due_date": str(self.due_date) if self.due_date else None,
            "start_date": str(self.start_date) if self.start_date else None,
            "is_published": bool(self.is_published),
            "is_template": bool(self.is_template),
            "created_by": self.created_by,
            "category": self.category,
            "tags": json.loads(self.tags) if self.tags else [],
            "instructions": self.instructions,
            "auto_grade": bool(self.auto_grade),
            "max_attempts": self.max_attempts,
            "show_answers": bool(self.show_answers),
            "created_at": str(self.created_at) if self.created_at else None,
            "updated_at": str(self.updated_at) if self.updated_at else None,
        }

        # 添加前端兼容字段
        result.update({
            # 前端期望的字段名
            "deadline": str(self.due_date) if self.due_date else None,
            "difficulty": self.difficulty_level,
            "status": "not_started",  # 默认状态，实际应该从学生进度获取
            "progress": 0,  # 默认进度，实际应该从学生进度获取
            "problems": [],  # 题目列表，前端期望的字段
            "savedAnswers": {}  # 保存的答案，前端期望的字段
        })

        return result

    def _get_actual_question_count(self):
        """获取实际题目数量"""
        try:
            from models.database import db
            sql = "SELECT COUNT(*) as count FROM questions WHERE homework_id = %s"
            result = db.execute_query_one(sql, (self.id,))
            return result['count'] if result else 0
        except Exception as e:
            print(f"获取题目数量失败: {e}")
            return self.question_count or 0

    @staticmethod
    def search(keyword=None, grade=None, subject=None, category=None, created_by=None, page=1, limit=10):
        """搜索作业"""
        offset = (page - 1) * limit
        
        where_conditions = []
        params = []
        
        if keyword:
            where_conditions.append("(title LIKE %s OR description LIKE %s)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        
        if grade:
            where_conditions.append("grade = %s")
            params.append(grade)
        
        if subject:
            where_conditions.append("subject = %s")
            params.append(subject)
        
        if category:
            where_conditions.append("category = %s")
            params.append(category)
        
        if created_by:
            where_conditions.append("created_by = %s")
            params.append(created_by)
        
        params.extend([limit, offset])
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        sql = f"""
        SELECT * FROM homeworks 
        WHERE {where_clause}
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        
        results = db.execute_query(sql, params)
        return [Homework(**row) for row in results]

    @staticmethod
    def get_statistics(teacher_id=None):
        """获取作业统计信息"""
        stats = {}
        
        # 基础统计
        if teacher_id:
            where_clause = "WHERE created_by = %s"
            params = [teacher_id]
        else:
            where_clause = ""
            params = []
        
        # 总作业数
        sql = f"SELECT COUNT(*) FROM homeworks {where_clause}"
        result = db.execute_query_one(sql, params)
        stats['total_homeworks'] = result['COUNT(*)'] if result else 0
        
        # 已发布作业数
        published_where = "is_published = 1"
        if teacher_id:
            published_where += " AND created_by = %s"
            published_params = [teacher_id]
        else:
            published_params = []
        
        sql = f"SELECT COUNT(*) FROM homeworks WHERE {published_where}"
        result = execute_query(sql, published_params, fetch_one=True)
        stats['published_homeworks'] = result['COUNT(*)'] if result else 0
        
        # 草稿作业数
        stats['draft_homeworks'] = stats['total_homeworks'] - stats['published_homeworks']
        
        return stats
