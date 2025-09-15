"""
作业分发模型 - 负责作业分发和班级管理相关的数据操作
"""
import json
from datetime import datetime
from models.database import db
from typing import List, Dict, Any, Optional

class Assignment:
    """作业分发模型"""
    
    @staticmethod
    def assign_homework_to_class(homework_id: int, class_id: int, teacher_id: int, 
                               due_date: str, instructions: str = None) -> Dict[str, Any]:
        """
        将作业分发给班级
        
        Args:
            homework_id: 作业ID
            class_id: 班级ID  
            teacher_id: 教师ID
            due_date: 截止日期
            instructions: 特殊说明
            
        Returns:
            Dict: 分发结果
        """
        try:
            # 检查作业是否存在且已发布
            homework = db.execute_query_one(
                "SELECT id, title, status FROM homeworks WHERE id = %s",
                (homework_id,)
            )
            if not homework:
                return {"success": False, "message": "作业不存在"}
            
            if homework['status'] != 'published':
                return {"success": False, "message": "只能分发已发布的作业"}
            
            # 检查班级是否存在
            class_info = db.execute_query_one(
                "SELECT id, class_name FROM classes WHERE id = %s",
                (class_id,)
            )
            if not class_info:
                return {"success": False, "message": "班级不存在"}
            
            # 检查是否已经分发过
            existing = db.execute_query_one(
                "SELECT id FROM homework_assignments WHERE homework_id = %s AND class_id = %s",
                (homework_id, class_id)
            )
            if existing:
                return {"success": False, "message": "该作业已分发给此班级"}
            
            # 创建作业分发记录
            assignment_id = db.execute_insert(
                """INSERT INTO homework_assignments 
                   (homework_id, class_id, teacher_id, due_date, instructions, status)
                   VALUES (%s, %s, %s, %s, %s, 'active')""",
                (homework_id, class_id, teacher_id, due_date, instructions)
            )
            
            # 获取班级学生列表
            students = db.execute_query(
                """SELECT student_id FROM class_students 
                   WHERE class_id = %s AND status = 'active'""",
                (class_id,)
            )
            
            # 为每个学生创建通知
            for student in students:
                Notification.create_assignment_notification(
                    user_id=student['student_id'],
                    homework_id=homework_id,
                    assignment_id=assignment_id,
                    class_id=class_id
                )
            
            return {
                "success": True,
                "message": "作业分发成功",
                "assignment_id": assignment_id,
                "student_count": len(students)
            }
            
        except Exception as e:
            return {"success": False, "message": f"分发失败: {str(e)}"}
    
    @staticmethod
    def get_class_assignments(class_id: int, status: str = None) -> List[Dict[str, Any]]:
        """
        获取班级的作业分发列表
        
        Args:
            class_id: 班级ID
            status: 状态筛选
            
        Returns:
            List: 作业分发列表
        """
        try:
            base_query = """
            SELECT ha.*, h.title, h.description, h.difficulty_level, h.estimated_time,
                   c.class_name
            FROM homework_assignments ha
            JOIN homeworks h ON ha.homework_id = h.id
            JOIN classes c ON ha.class_id = c.id
            WHERE ha.class_id = %s
            """
            params = [class_id]
            
            if status:
                base_query += " AND ha.status = %s"
                params.append(status)
                
            base_query += " ORDER BY ha.created_at DESC"
            
            assignments = db.execute_query(base_query, params)
            
            # 为每个分发获取完成统计
            for assignment in assignments:
                stats = Assignment.get_assignment_statistics(assignment['id'])
                assignment.update(stats)
            
            return assignments
            
        except Exception as e:
            print(f"获取班级作业分发失败: {e}")
            return []
    
    @staticmethod
    def get_teacher_assignments(teacher_id: int, status: str = None) -> List[Dict[str, Any]]:
        """
        获取教师的作业分发列表
        
        Args:
            teacher_id: 教师ID
            status: 状态筛选
            
        Returns:
            List: 作业分发列表
        """
        try:
            base_query = """
            SELECT ha.*, h.title, h.description, h.difficulty_level, h.estimated_time,
                   c.class_name, c.id as class_id
            FROM homework_assignments ha
            JOIN homeworks h ON ha.homework_id = h.id
            JOIN classes c ON ha.class_id = c.id
            WHERE ha.teacher_id = %s
            """
            params = [teacher_id]
            
            if status:
                base_query += " AND ha.status = %s"
                params.append(status)
                
            base_query += " ORDER BY ha.created_at DESC"
            
            assignments = db.execute_query(base_query, params)
            
            # 为每个分发获取完成统计
            for assignment in assignments:
                stats = Assignment.get_assignment_statistics(assignment['id'])
                assignment.update(stats)
            
            return assignments
            
        except Exception as e:
            print(f"获取教师作业分发失败: {e}")
            return []
    
    @staticmethod
    def get_assignment_statistics(assignment_id: int) -> Dict[str, Any]:
        """
        获取作业分发的统计信息
        
        Args:
            assignment_id: 分发ID
            
        Returns:
            Dict: 统计信息
        """
        try:
            # 获取总学生数
            total_students = db.execute_query_one(
                """SELECT COUNT(*) as total FROM class_students cs
                   JOIN homework_assignments ha ON cs.class_id = ha.class_id
                   WHERE ha.id = %s AND cs.status = 'active'""",
                (assignment_id,)
            )['total']
            
            # 获取已提交数（这里假设有homework_submissions表）
            submitted_count = 0  # 暂时设为0，等实现提交功能后再更新
            
            return {
                "total_students": total_students,
                "submitted_count": submitted_count,
                "pending_count": total_students - submitted_count,
                "completion_rate": round((submitted_count / total_students * 100) if total_students > 0 else 0, 2)
            }
            
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {
                "total_students": 0,
                "submitted_count": 0,
                "pending_count": 0,
                "completion_rate": 0
            }
    
    @staticmethod
    def update_assignment_status(assignment_id: int, status: str, teacher_id: int) -> Dict[str, Any]:
        """
        更新作业分发状态
        
        Args:
            assignment_id: 分发ID
            status: 新状态 (active, paused, completed, cancelled)
            teacher_id: 教师ID
            
        Returns:
            Dict: 更新结果
        """
        try:
            # 验证教师权限
            assignment = db.execute_query_one(
                "SELECT teacher_id FROM homework_assignments WHERE id = %s",
                (assignment_id,)
            )
            if not assignment:
                return {"success": False, "message": "作业分发不存在"}
            
            if assignment['teacher_id'] != teacher_id:
                return {"success": False, "message": "无权限操作此作业分发"}
            
            # 更新状态
            db.execute_update(
                "UPDATE homework_assignments SET status = %s, updated_at = NOW() WHERE id = %s",
                (status, assignment_id)
            )
            
            return {"success": True, "message": "状态更新成功"}
            
        except Exception as e:
            return {"success": False, "message": f"状态更新失败: {str(e)}"}
    
    @staticmethod
    def get_assignment_detail(assignment_id: int) -> Dict[str, Any]:
        """
        获取作业分发详细信息
        
        Args:
            assignment_id: 分发ID
            
        Returns:
            Dict: 详细信息
        """
        try:
            assignment = db.execute_query_one(
                """SELECT ha.*, h.title, h.description, h.difficulty_level, 
                          h.estimated_time, h.knowledge_points, h.instructions as homework_instructions,
                          c.class_name, c.class_code
                   FROM homework_assignments ha
                   JOIN homeworks h ON ha.homework_id = h.id
                   JOIN classes c ON ha.class_id = c.id
                   WHERE ha.id = %s""",
                (assignment_id,)
            )
            
            if not assignment:
                return {"success": False, "message": "作业分发不存在"}
            
            # 获取统计信息
            stats = Assignment.get_assignment_statistics(assignment_id)
            assignment.update(stats)
            
            # 解析JSON字段
            if assignment.get('knowledge_points'):
                assignment['knowledge_points'] = json.loads(assignment['knowledge_points'])
            
            return {"success": True, "data": assignment}
            
        except Exception as e:
            return {"success": False, "message": f"获取详情失败: {str(e)}"}


class ClassManagement:
    """班级管理模型"""
    
    @staticmethod
    def get_teacher_classes(teacher_id: int) -> List[Dict[str, Any]]:
        """
        获取教师负责的班级列表
        
        Args:
            teacher_id: 教师ID
            
        Returns:
            List: 班级列表
        """
        try:
            classes = db.execute_query(
                """SELECT c.*, g.grade_name, s.school_name,
                          COUNT(cs.student_id) as student_count
                   FROM classes c
                   LEFT JOIN grades g ON c.grade_id = g.id
                   LEFT JOIN schools s ON c.school_id = s.id
                   LEFT JOIN class_students cs ON c.id = cs.class_id AND cs.status = 'active'
                   WHERE c.head_teacher_id = %s
                   GROUP BY c.id
                   ORDER BY g.grade_level, c.class_name""",
                (teacher_id,)
            )
            return classes
            
        except Exception as e:
            print(f"获取教师班级失败: {e}")
            return []
    
    @staticmethod
    def get_class_students(class_id: int) -> List[Dict[str, Any]]:
        """
        获取班级学生列表
        
        Args:
            class_id: 班级ID
            
        Returns:
            List: 学生列表
        """
        try:
            students = db.execute_query(
                """SELECT u.id, u.username, u.real_name, u.email,
                          cs.student_number, cs.joined_at
                   FROM class_students cs
                   JOIN users u ON cs.student_id = u.id
                   WHERE cs.class_id = %s AND cs.status = 'active'
                   ORDER BY cs.student_number""",
                (class_id,)
            )
            return students
            
        except Exception as e:
            print(f"获取班级学生失败: {e}")
            return []


class Notification:
    """通知模型"""
    
    @staticmethod
    def create_assignment_notification(user_id: int, homework_id: int, 
                                     assignment_id: int, class_id: int) -> int:
        """
        创建作业分发通知
        
        Args:
            user_id: 用户ID
            homework_id: 作业ID
            assignment_id: 分发ID
            class_id: 班级ID
            
        Returns:
            int: 通知ID
        """
        try:
            # 获取作业标题
            homework = db.execute_query_one(
                "SELECT title FROM homeworks WHERE id = %s",
                (homework_id,)
            )
            homework_title = homework['title'] if homework else "未知作业"
            
            # 获取班级名称
            class_info = db.execute_query_one(
                "SELECT class_name FROM classes WHERE id = %s",
                (class_id,)
            )
            class_name = class_info['class_name'] if class_info else "未知班级"
            
            notification_id = db.execute_insert(
                """INSERT INTO notifications 
                   (user_id, type, title, content, related_id, data)
                   VALUES (%s, 'homework_assignment', %s, %s, %s, %s)""",
                (
                    user_id,
                    "新作业通知",
                    f"您在{class_name}收到新作业：{homework_title}",
                    assignment_id,
                    json.dumps({
                        "homework_id": homework_id,
                        "assignment_id": assignment_id,
                        "class_id": class_id
                    })
                )
            )
            
            return notification_id
            
        except Exception as e:
            print(f"创建通知失败: {e}")
            return 0
    
    @staticmethod
    def get_user_notifications(user_id: int, unread_only: bool = False) -> List[Dict[str, Any]]:
        """
        获取用户通知列表
        
        Args:
            user_id: 用户ID
            unread_only: 是否只获取未读通知
            
        Returns:
            List: 通知列表
        """
        try:
            query = """
            SELECT * FROM notifications 
            WHERE user_id = %s
            """
            params = [user_id]
            
            if unread_only:
                query += " AND is_read = FALSE"
            
            query += " ORDER BY created_at DESC LIMIT 50"
            
            notifications = db.execute_query(query, params)
            
            # 解析data字段
            for notification in notifications:
                if notification.get('data'):
                    try:
                        notification['data'] = json.loads(notification['data'])
                    except:
                        notification['data'] = {}
            
            return notifications
            
        except Exception as e:
            print(f"获取通知失败: {e}")
            return []
    
    @staticmethod
    def mark_notification_read(notification_id: int, user_id: int) -> bool:
        """
        标记通知为已读
        
        Args:
            notification_id: 通知ID
            user_id: 用户ID
            
        Returns:
            bool: 是否成功
        """
        try:
            db.execute_update(
                """UPDATE notifications SET is_read = TRUE, read_at = NOW() 
                   WHERE id = %s AND user_id = %s""",
                (notification_id, user_id)
            )
            return True
            
        except Exception as e:
            print(f"标记通知已读失败: {e}")
            return False

