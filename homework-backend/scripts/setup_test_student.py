"""
创建测试学生账号
"""
from models.database import db
from models.user import User
from services.auth_service import AuthService

def setup_test_student():
    """创建或重置测试学生账号"""
    try:
        # 检查学生账号是否存在
        existing_student = db.execute_query_one(
            "SELECT * FROM users WHERE username = %s",
            ("test_student_001",)
        )
        
        if existing_student:
            print(f"✅ 学生账号已存在: {existing_student['username']}")
            print(f"   角色: {existing_student['role']}")
            print(f"   状态: {existing_student['is_active']}")
            
            # 重置密码
            password_hash = User.hash_password("password123")
            db.execute_update(
                "UPDATE users SET password_hash = %s WHERE username = %s",
                (password_hash, "test_student_001")
            )
            print("✅ 密码已重置为: password123")
            
        else:
            print("❌ 学生账号不存在，正在创建...")
            
            # 创建学生账号
            password_hash = User.hash_password("password123")
            
            student_id = db.execute_insert(
                """INSERT INTO users (username, password_hash, email, real_name, role, is_active)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                ("test_student_001", password_hash, "student001@test.com", "测试学生001", "student", True)
            )
            
            print(f"✅ 学生账号创建成功，ID: {student_id}")
        
        # 确保学生在班级中
        student_user = db.execute_query_one(
            "SELECT id FROM users WHERE username = %s",
            ("test_student_001",)
        )
        
        if student_user:
            student_id = student_user['id']
            
            # 检查是否已经在班级中
            class_membership = db.execute_query_one(
                "SELECT * FROM class_students WHERE student_id = %s",
                (student_id,)
            )
            
            if not class_membership:
                # 获取第一个班级
                first_class = db.execute_query_one("SELECT id FROM classes LIMIT 1")
                
                if first_class:
                    # 添加到班级
                    db.execute_insert(
                        """INSERT INTO class_students (class_id, student_id, student_number, status)
                           VALUES (%s, %s, %s, %s)""",
                        (first_class['id'], student_id, "S20240001", "active")
                    )
                    print(f"✅ 学生已加入班级 ID: {first_class['id']}")
                else:
                    print("⚠️ 没有找到班级，学生可能无法看到作业")
            else:
                print(f"✅ 学生已在班级中: {class_membership['class_id']}")
        
        # 测试登录
        print("\n=== 测试学生登录 ===")
        try:
            user = User.authenticate("test_student_001", "password123")
            if user:
                print("✅ 学生登录测试成功")
                return True
            else:
                print("❌ 学生登录测试失败: 用户名或密码错误")
                return False
        except Exception as e:
            print(f"❌ 登录测试出错: {e}")
            return False
        
    except Exception as e:
        print(f"❌ 设置学生账号失败: {e}")
        return False

if __name__ == "__main__":
    setup_test_student()
