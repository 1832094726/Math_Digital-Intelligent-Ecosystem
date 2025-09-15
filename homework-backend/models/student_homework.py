"""
学生作业接收与展示模型 - 负责学生端作业管理
"""
import json
from datetime import datetime, timedelta
from models.database import db
from typing import List, Dict, Any, Optional

class StudentHomework:
    """学生作业模型"""
    
    @staticmethod
    def get_student_homework_list(student_id: int, status: str = None, 
                                 search_keyword: str = None, 
                                 page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        获取学生的作业列表
        
        Args:
            student_id: 学生ID
            status: 状态筛选 (pending, in_progress, completed, overdue)
            search_keyword: 搜索关键词
            page: 页码
            page_size: 每页数量
            
        Returns:
            Dict: 作业列表和分页信息
        """
        try:
            # 构建基础查询 - 适配现有表结构
            base_query = """
            SELECT ha.id as assignment_id, ha.due_date_override as due_date, ha.notes as instructions, 
                   CASE WHEN ha.is_active = 1 THEN 'active' ELSE 'inactive' END as assignment_status,
                   h.id as homework_id, h.title, h.description, h.difficulty_level, 
                   h.time_limit as estimated_time, h.tags as knowledge_points, h.created_at as homework_created_at,
                   c.class_name,
                   hp.progress_data, hp.last_saved_at, hp.completion_rate,
                   hs.submitted_at, hs.score, hs.status as submission_status
            FROM homework_assignments ha
            JOIN homeworks h ON ha.homework_id = h.id
            JOIN classes c ON ha.assigned_to_id = c.id AND ha.assigned_to_type = 'class'
            JOIN class_students cs ON c.id = cs.class_id
            LEFT JOIN homework_progress hp ON h.id = hp.homework_id AND hp.student_id = %s
            LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id AND hs.student_id = %s
            WHERE cs.student_id = %s AND cs.is_active = 1 AND ha.is_active = 1
            """
            
            params = [student_id, student_id, student_id]
            conditions = []
            
            # 状态筛选
            if status:
                if status == 'pending':
                    conditions.append("hs.submitted_at IS NULL AND COALESCE(ha.due_date_override, h.due_date) > NOW()")
                elif status == 'in_progress':
                    conditions.append("hp.progress_data IS NOT NULL AND hs.submitted_at IS NULL")
                elif status == 'completed':
                    conditions.append("hs.submitted_at IS NOT NULL")
                elif status == 'overdue':
                    conditions.append("hs.submitted_at IS NULL AND COALESCE(ha.due_date_override, h.due_date) < NOW()")
            
            # 搜索关键词
            if search_keyword:
                conditions.append("(h.title LIKE %s OR h.description LIKE %s)")
                params.extend([f"%{search_keyword}%", f"%{search_keyword}%"])
            
            # 添加条件
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            # 排序和分页
            count_query = f"SELECT COUNT(*) as total FROM ({base_query}) as filtered_results"
            total_count = db.execute_query_one(count_query, params)['total']
            
            base_query += " ORDER BY COALESCE(ha.due_date_override, h.due_date) ASC, h.created_at DESC"
            base_query += f" LIMIT {page_size} OFFSET {(page - 1) * page_size}"
            
            homework_list = db.execute_query(base_query, params)
            
            # 处理结果数据
            for homework in homework_list:
                # 解析JSON字段
                if homework.get('knowledge_points'):
                    homework['knowledge_points'] = json.loads(homework['knowledge_points'])
                if homework.get('progress_data'):
                    homework['progress_data'] = json.loads(homework['progress_data'])
                
                # 计算作业状态
                homework['computed_status'] = StudentHomework._compute_homework_status(homework)
                
                # 计算剩余时间
                if homework['due_date']:
                    remaining_time = homework['due_date'] - datetime.now()
                    homework['remaining_days'] = remaining_time.days if remaining_time.days > 0 else 0
                    homework['is_urgent'] = remaining_time.days <= 1 and remaining_time.total_seconds() > 0
                else:
                    homework['remaining_days'] = None
                    homework['is_urgent'] = False
            
            return {
                "success": True,
                "data": homework_list,
                "pagination": {
                    "current_page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"获取作业列表失败: {str(e)}"}
    
    @staticmethod
    def _compute_homework_status(homework: Dict[str, Any]) -> str:
        """计算作业状态"""
        if homework['submitted_at']:
            return 'completed'
        elif homework['due_date'] and homework['due_date'] < datetime.now():
            return 'overdue'
        elif homework['progress_data']:
            return 'in_progress'
        else:
            return 'pending'
    
    @staticmethod
    def get_homework_detail(student_id: int, assignment_id: int) -> Dict[str, Any]:
        """
        获取作业详细信息
        
        Args:
            student_id: 学生ID
            assignment_id: 作业分发ID
            
        Returns:
            Dict: 作业详细信息
        """
        try:
            # 验证学生是否有权限访问此作业
            assignment = db.execute_query_one(
                """SELECT ha.*, h.title, h.description, h.instructions, h.knowledge_points,
                          h.difficulty_level, h.estimated_time, h.questions_data,
                          c.class_name
                   FROM homework_assignments ha
                   JOIN homeworks h ON ha.homework_id = h.id
                   JOIN classes c ON ha.class_id = c.id
                   JOIN class_students cs ON c.id = cs.class_id
                   WHERE ha.id = %s AND cs.student_id = %s AND cs.is_active = 1""",
                (assignment_id, student_id)
            )
            
            if not assignment:
                return {"success": False, "message": "作业不存在或无权限访问"}
            
            # 获取学生的进度信息
            progress = db.execute_query_one(
                """SELECT * FROM homework_progress 
                   WHERE homework_id = %s AND student_id = %s""",
                (assignment['homework_id'], student_id)
            )
            
            # 获取提交信息
            submission = db.execute_query_one(
                """SELECT * FROM homework_submissions 
                   WHERE assignment_id = %s AND student_id = %s""",
                (assignment_id, student_id)
            )
            
            # 解析JSON字段
            if assignment.get('knowledge_points'):
                assignment['knowledge_points'] = json.loads(assignment['knowledge_points'])
            if assignment.get('questions_data'):
                assignment['questions_data'] = json.loads(assignment['questions_data'])
            if progress and progress.get('progress_data'):
                progress['progress_data'] = json.loads(progress['progress_data'])
            if submission and submission.get('answers'):
                submission['answers'] = json.loads(submission['answers'])
            
            # 计算状态和时间信息
            assignment['computed_status'] = StudentHomework._compute_homework_status({
                'submitted_at': submission['submitted_at'] if submission else None,
                'due_date': assignment['due_date'],
                'progress_data': progress['progress_data'] if progress else None
            })
            
            if assignment['due_date']:
                remaining_time = assignment['due_date'] - datetime.now()
                assignment['remaining_hours'] = max(0, remaining_time.total_seconds() / 3600)
                assignment['is_urgent'] = remaining_time.days <= 1 and remaining_time.total_seconds() > 0
            
            return {
                "success": True,
                "data": {
                    "assignment": assignment,
                    "progress": progress,
                    "submission": submission
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"获取作业详情失败: {str(e)}"}
    
    @staticmethod
    def get_homework_statistics(student_id: int) -> Dict[str, Any]:
        """
        获取学生作业统计信息
        
        Args:
            student_id: 学生ID
            
        Returns:
            Dict: 统计信息
        """
        try:
            # 总作业数
            total_count = db.execute_query_one(
                """SELECT COUNT(*) as count FROM homework_assignments ha
                   JOIN class_students cs ON ha.assigned_to_id = cs.class_id
                   WHERE cs.student_id = %s AND cs.is_active = 1 AND ha.assigned_to_type = 'class' AND ha.is_active = 1""",
                (student_id,)
            )['count']
            
            # 已完成作业数
            completed_count = db.execute_query_one(
                """SELECT COUNT(*) as count FROM homework_assignments ha
                   JOIN class_students cs ON ha.assigned_to_id = cs.class_id AND ha.assigned_to_type = 'class'
                   JOIN homework_submissions hs ON ha.id = hs.assignment_id
                   WHERE cs.student_id = %s AND cs.is_active = 1 
                   AND ha.is_active = 1 AND hs.student_id = %s AND hs.submitted_at IS NOT NULL""",
                (student_id, student_id)
            )['count']
            
            # 进行中作业数
            in_progress_count = db.execute_query_one(
                """SELECT COUNT(*) as count FROM homework_assignments ha
                   JOIN class_students cs ON ha.assigned_to_id = cs.class_id AND ha.assigned_to_type = 'class'
                   LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id AND hs.student_id = %s
                   JOIN homework_progress hp ON ha.homework_id = hp.homework_id AND hp.student_id = %s
                   WHERE cs.student_id = %s AND cs.is_active = 1 
                   AND ha.is_active = 1 AND hs.submitted_at IS NULL""",
                (student_id, student_id, student_id)
            )['count']
            
            # 过期作业数
            overdue_count = db.execute_query_one(
                """SELECT COUNT(*) as count FROM homework_assignments ha
                   JOIN class_students cs ON ha.assigned_to_id = cs.class_id AND ha.assigned_to_type = 'class'
                   LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id AND hs.student_id = %s
                   WHERE cs.student_id = %s AND cs.is_active = 1 
                   AND ha.is_active = 1 AND hs.submitted_at IS NULL AND ha.due_date < NOW()""",
                (student_id, student_id)
            )['count']
            
            # 平均分数
            avg_score = db.execute_query_one(
                """SELECT AVG(hs.score) as avg_score FROM homework_submissions hs
                   JOIN homework_assignments ha ON hs.assignment_id = ha.id
                   JOIN class_students cs ON ha.assigned_to_id = cs.class_id AND ha.assigned_to_type = 'class'
                   WHERE cs.student_id = %s AND hs.student_id = %s AND hs.score IS NOT NULL""",
                (student_id, student_id)
            )['avg_score'] or 0
            
            completion_rate = (completed_count / total_count * 100) if total_count > 0 else 0
            
            return {
                "success": True,
                "data": {
                    "total_count": total_count,
                    "completed_count": completed_count,
                    "in_progress_count": in_progress_count,
                    "pending_count": total_count - completed_count - in_progress_count - overdue_count,
                    "overdue_count": overdue_count,
                    "completion_rate": round(completion_rate, 2),
                    "average_score": round(float(avg_score), 2)
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"获取统计信息失败: {str(e)}"}
    
    @staticmethod
    def mark_homework_favorite(student_id: int, assignment_id: int, is_favorite: bool) -> Dict[str, Any]:
        """
        标记/取消标记作业收藏
        
        Args:
            student_id: 学生ID
            assignment_id: 作业分发ID
            is_favorite: 是否收藏
            
        Returns:
            Dict: 操作结果
        """
        try:
            # 验证作业权限
            assignment = db.execute_query_one(
                """SELECT ha.id FROM homework_assignments ha
                   JOIN class_students cs ON ha.assigned_to_id = cs.class_id AND ha.assigned_to_type = 'class'
                   WHERE ha.id = %s AND cs.student_id = %s AND cs.is_active = 1""",
                (assignment_id, student_id)
            )
            
            if not assignment:
                return {"success": False, "message": "作业不存在或无权限访问"}
            
            if is_favorite:
                # 添加收藏
                db.execute_insert(
                    """INSERT IGNORE INTO homework_favorites 
                       (student_id, assignment_id) VALUES (%s, %s)""",
                    (student_id, assignment_id)
                )
                message = "收藏成功"
            else:
                # 取消收藏
                db.execute_delete(
                    """DELETE FROM homework_favorites 
                       WHERE student_id = %s AND assignment_id = %s""",
                    (student_id, assignment_id)
                )
                message = "取消收藏成功"
            
            return {"success": True, "message": message}
            
        except Exception as e:
            return {"success": False, "message": f"操作失败: {str(e)}"}
    
    @staticmethod
    def get_favorite_homeworks(student_id: int) -> List[Dict[str, Any]]:
        """
        获取收藏的作业列表
        
        Args:
            student_id: 学生ID
            
        Returns:
            List: 收藏的作业列表
        """
        try:
            favorites = db.execute_query(
                """SELECT ha.id as assignment_id, ha.due_date, ha.instructions,
                          h.title, h.description, h.difficulty_level,
                          c.class_name, hf.created_at as favorited_at
                   FROM homework_favorites hf
                   JOIN homework_assignments ha ON hf.assignment_id = ha.id
                   JOIN homeworks h ON ha.homework_id = h.id
                   JOIN classes c ON ha.class_id = c.id
                   WHERE hf.student_id = %s
                   ORDER BY hf.created_at DESC""",
                (student_id,)
            )
            
            return favorites
            
        except Exception as e:
            print(f"获取收藏作业失败: {e}")
            return []


class HomeworkProgress:
    """作业进度管理"""
    
    @staticmethod
    def save_progress(student_id: int, homework_id: int, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存作业进度
        
        Args:
            student_id: 学生ID
            homework_id: 作业ID
            progress_data: 进度数据
            
        Returns:
            Dict: 保存结果
        """
        try:
            # 计算完成率
            completion_rate = HomeworkProgress._calculate_completion_rate(progress_data)
            
            # 检查是否已存在进度记录
            existing = db.execute_query_one(
                "SELECT id FROM homework_progress WHERE student_id = %s AND homework_id = %s",
                (student_id, homework_id)
            )
            
            if existing:
                # 更新现有记录
                db.execute_update(
                    """UPDATE homework_progress 
                       SET progress_data = %s, completion_rate = %s, last_saved_at = NOW()
                       WHERE student_id = %s AND homework_id = %s""",
                    (json.dumps(progress_data), completion_rate, student_id, homework_id)
                )
            else:
                # 创建新记录
                db.execute_insert(
                    """INSERT INTO homework_progress 
                       (student_id, homework_id, progress_data, completion_rate)
                       VALUES (%s, %s, %s, %s)""",
                    (student_id, homework_id, json.dumps(progress_data), completion_rate)
                )
            
            return {"success": True, "message": "进度保存成功", "completion_rate": completion_rate}
            
        except Exception as e:
            return {"success": False, "message": f"保存进度失败: {str(e)}"}
    
    @staticmethod
    def _calculate_completion_rate(progress_data: Dict[str, Any]) -> float:
        """计算完成率"""
        if not progress_data or 'answers' not in progress_data:
            return 0.0
        
        answers = progress_data['answers']
        total_questions = len(answers) if answers else 0
        completed_questions = sum(1 for answer in answers.values() if answer and str(answer).strip())
        
        return (completed_questions / total_questions * 100) if total_questions > 0 else 0.0
    
    @staticmethod
    def get_progress(student_id: int, homework_id: int) -> Dict[str, Any]:
        """
        获取作业进度
        
        Args:
            student_id: 学生ID
            homework_id: 作业ID
            
        Returns:
            Dict: 进度信息
        """
        try:
            progress = db.execute_query_one(
                """SELECT * FROM homework_progress 
                   WHERE student_id = %s AND homework_id = %s""",
                (student_id, homework_id)
            )
            
            if progress and progress.get('progress_data'):
                progress['progress_data'] = json.loads(progress['progress_data'])
            
            return {"success": True, "data": progress}
            
        except Exception as e:
            return {"success": False, "message": f"获取进度失败: {str(e)}"}


class HomeworkReminder:
    """作业提醒管理"""
    
    @staticmethod
    def get_due_reminders(student_id: int, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        获取即将到期的作业提醒
        
        Args:
            student_id: 学生ID
            hours_ahead: 提前多少小时提醒
            
        Returns:
            List: 提醒列表
        """
        try:
            deadline = datetime.now() + timedelta(hours=hours_ahead)
            
            reminders = db.execute_query(
                """SELECT ha.id as assignment_id, ha.due_date,
                          h.title, h.difficulty_level,
                          c.class_name
                   FROM homework_assignments ha
                   JOIN homeworks h ON ha.homework_id = h.id
                   JOIN classes c ON ha.class_id = c.id
                   JOIN class_students cs ON c.id = cs.class_id
                   LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id AND hs.student_id = %s
                   WHERE cs.student_id = %s AND cs.is_active = 1
                   AND ha.is_active = 1 AND hs.submitted_at IS NULL
                   AND ha.due_date BETWEEN NOW() AND %s
                   ORDER BY ha.due_date ASC""",
                (student_id, student_id, deadline)
            )
            
            # 计算剩余时间
            for reminder in reminders:
                if reminder['due_date']:
                    remaining_time = reminder['due_date'] - datetime.now()
                    reminder['remaining_hours'] = max(0, remaining_time.total_seconds() / 3600)
            
            return reminders
            
        except Exception as e:
            print(f"获取提醒失败: {e}")
            return []
