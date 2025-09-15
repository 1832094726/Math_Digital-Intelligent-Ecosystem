"""
创建测试作业分发数据
"""
from models.database import db

def create_test_assignment():
    """创建测试作业分发"""
    try:
        # 1. 检查是否有作业
        homework = db.execute_query_one("SELECT * FROM homeworks WHERE is_published = 1 LIMIT 1")
        if not homework:
            print("❌ 没有已发布的作业，正在创建...")
            
            # 创建一个测试作业
            homework_id = db.execute_insert(
                """INSERT INTO homeworks 
                   (title, description, subject, grade, difficulty_level, question_count, 
                    max_score, time_limit, due_date, is_published, created_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 
                           DATE_ADD(NOW(), INTERVAL 7 DAY), 1, 3)""",
                ("测试数学作业", "这是一个测试作业，包含基础数学题目", "数学", 7, "medium", 5, 
                 100.0, 60, )
            )
            print(f"✅ 测试作业创建成功，ID: {homework_id}")
        else:
            homework_id = homework['id']
            print(f"✅ 找到已发布作业，ID: {homework_id}")
        
        # 2. 检查学生信息
        student = db.execute_query_one("SELECT * FROM users WHERE username = 'test_student_001'")
        if not student:
            print("❌ 找不到测试学生")
            return False
        
        student_id = student['id']
        print(f"✅ 学生ID: {student_id}")
        
        # 3. 检查学生所在班级
        class_membership = db.execute_query_one(
            "SELECT * FROM class_students WHERE student_id = %s AND is_active = 1",
            (student_id,)
        )
        
        if not class_membership:
            print("❌ 学生不在任何班级中")
            return False
        
        class_id = class_membership['class_id']
        print(f"✅ 学生所在班级ID: {class_id}")
        
        # 4. 检查是否已有作业分发
        existing_assignment = db.execute_query_one(
            """SELECT * FROM homework_assignments 
               WHERE homework_id = %s AND assigned_to_type = 'class' AND assigned_to_id = %s""",
            (homework_id, class_id)
        )
        
        if existing_assignment:
            print(f"✅ 作业分发已存在，ID: {existing_assignment['id']}")
        else:
            # 创建作业分发
            assignment_id = db.execute_insert(
                """INSERT INTO homework_assignments 
                   (homework_id, assigned_to_type, assigned_to_id, assigned_by, due_date_override)
                   VALUES (%s, %s, %s, %s, DATE_ADD(NOW(), INTERVAL 7 DAY))""",
                (homework_id, 'class', class_id, 3)  # 3是教师ID
            )
            print(f"✅ 作业分发创建成功，ID: {assignment_id}")
        
        # 5. 验证学生能否看到作业
        print("\n=== 验证学生可见作业 ===")
        visible_assignments = db.execute_query(
            """SELECT ha.id, h.title, h.due_date, ha.due_date_override
               FROM homework_assignments ha
               JOIN homeworks h ON ha.homework_id = h.id
               JOIN class_students cs ON ha.assigned_to_id = cs.class_id
               WHERE cs.student_id = %s AND ha.assigned_to_type = 'class' 
               AND cs.is_active = 1 AND ha.is_active = 1""",
            (student_id,)
        )
        
        print(f"学生可见作业数量: {len(visible_assignments)}")
        for assignment in visible_assignments:
            print(f"  - {assignment['title']} (截止: {assignment['due_date_override'] or assignment['due_date']})")
        
        return len(visible_assignments) > 0
        
    except Exception as e:
        print(f"❌ 创建测试分发失败: {e}")
        return False

if __name__ == "__main__":
    create_test_assignment()
