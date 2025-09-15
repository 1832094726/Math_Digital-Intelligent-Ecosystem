"""
作业分发管理API测试脚本
"""
import requests
import json
from datetime import datetime, timedelta

# 配置
BASE_URL = "http://localhost:5000/api"

# 全局变量存储token
teacher_token = None
student_token = None

def test_teacher_login():
    """测试教师登录"""
    global teacher_token
    print("\n=== 教师登录测试 ===")
    
    # 使用之前创建的teacher001账号
    login_data = {
        "username": "teacher001",
        "password": "password123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"登录状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                teacher_token = result['data']['token']
                print(f"✅ 教师登录成功")
                print(f"用户信息: {result['data']['user']['username']}")
                return True
            else:
                print(f"❌ 教师登录失败: {result}")
                return False
        else:
            print(f"❌ 教师登录失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 教师登录请求失败: {e}")
        return False

def test_student_login():
    """测试学生登录"""
    global student_token
    print("\n=== 学生登录测试 ===")
    
    # 使用之前创建的test_student_001账号
    login_data = {
        "username": "test_student_001",
        "password": "password123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"登录状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                student_token = result['data']['token']
                print(f"✅ 学生登录成功")
                print(f"用户信息: {result['data']['user']['username']}")
                return True
            else:
                print(f"❌ 学生登录失败: {result}")
                return False
        else:
            print(f"❌ 学生登录失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 学生登录请求失败: {e}")
        return False

def test_get_teacher_classes():
    """测试获取教师班级"""
    print("\n=== 获取教师班级测试 ===")
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/assignment/classes/my", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 获取班级成功")
            print(f"班级数量: {result['total']}")
            for cls in result['data']:
                print(f"  - 班级: {cls['class_name']} (ID: {cls['id']}, 学生数: {cls['student_count']})")
            return result['data']
        else:
            print(f"❌ 获取班级失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取班级请求失败: {e}")
        return []

def test_get_class_students(class_id):
    """测试获取班级学生"""
    print(f"\n=== 获取班级{class_id}学生测试 ===")
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/assignment/classes/{class_id}/students", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 获取学生成功")
            print(f"学生数量: {result['total']}")
            for student in result['data']:
                print(f"  - 学生: {student['real_name']} ({student['username']}, 学号: {student['student_number']})")
            return result['data']
        else:
            print(f"❌ 获取学生失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取学生请求失败: {e}")
        return []

def test_assign_homework(homework_id, class_id):
    """测试分发作业"""
    print(f"\n=== 分发作业测试 (作业{homework_id} -> 班级{class_id}) ===")
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    
    # 设置截止时间为7天后
    due_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    
    assignment_data = {
        "homework_id": homework_id,
        "class_id": class_id,
        "due_date": due_date,
        "instructions": "请仔细完成作业，注意书写规范。"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/assignment/assign", json=assignment_data, headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code in [200, 201]:
            result = response.json()
            if result['success']:
                print(f"✅ 作业分发成功")
                print(f"分发ID: {result['assignment_id']}")
                print(f"通知学生数: {result['student_count']}")
                return result['assignment_id']
            else:
                print(f"❌ 作业分发失败: {result['message']}")
                return None
        else:
            print(f"❌ 作业分发失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 作业分发请求失败: {e}")
        return None

def test_get_teacher_assignments():
    """测试获取教师作业分发列表"""
    print("\n=== 获取教师作业分发列表测试 ===")
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/assignment/teacher/my", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 获取分发列表成功")
            print(f"分发数量: {result['total']}")
            for assignment in result['data']:
                print(f"  - 作业: {assignment['title']} -> 班级: {assignment['class_name']}")
                print(f"    状态: {assignment['status']}, 截止: {assignment['due_date']}")
                print(f"    统计: {assignment['submitted_count']}/{assignment['total_students']} ({assignment['completion_rate']}%)")
            return result['data']
        else:
            print(f"❌ 获取分发列表失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取分发列表请求失败: {e}")
        return []

def test_get_assignment_detail(assignment_id):
    """测试获取作业分发详情"""
    print(f"\n=== 获取作业分发{assignment_id}详情测试 ===")
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/assignment/{assignment_id}", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                assignment = result['data']
                print(f"✅ 获取分发详情成功")
                print(f"作业标题: {assignment['title']}")
                print(f"班级: {assignment['class_name']}")
                print(f"截止时间: {assignment['due_date']}")
                print(f"特殊说明: {assignment.get('instructions', '无')}")
                print(f"完成统计: {assignment['submitted_count']}/{assignment['total_students']}")
                return assignment
            else:
                print(f"❌ 获取分发详情失败: {result['message']}")
                return None
        else:
            print(f"❌ 获取分发详情失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取分发详情请求失败: {e}")
        return None

def test_update_assignment_status(assignment_id, status):
    """测试更新作业分发状态"""
    print(f"\n=== 更新作业分发{assignment_id}状态测试 ({status}) ===")
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    update_data = {"status": status}
    
    try:
        response = requests.put(f"{BASE_URL}/assignment/{assignment_id}/status", 
                              json=update_data, headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 状态更新成功: {result['message']}")
                return True
            else:
                print(f"❌ 状态更新失败: {result['message']}")
                return False
        else:
            print(f"❌ 状态更新失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 状态更新请求失败: {e}")
        return False

def test_get_student_notifications():
    """测试获取学生通知"""
    print("\n=== 获取学生通知测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/assignment/notifications/my", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 获取通知成功")
            print(f"通知数量: {result['total']}")
            for notification in result['data']:
                print(f"  - {notification['title']}: {notification['content']}")
                print(f"    类型: {notification['type']}, 已读: {notification['is_read']}")
            return result['data']
        else:
            print(f"❌ 获取通知失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取通知请求失败: {e}")
        return []

def get_homework_for_assignment():
    """获取可用于分发的作业"""
    print("\n=== 获取可分发作业 ===")
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/homework/list?status=published", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success'] and result['data']:
                print(f"✅ 找到{len(result['data'])}个已发布的作业")
                for hw in result['data']:
                    print(f"  - ID: {hw['id']}, 标题: {hw['title']}, 状态: {hw['status']}")
                return result['data'][0]['id']  # 返回第一个作业的ID
            else:
                print(f"❌ 没有找到已发布的作业")
                return None
        else:
            print(f"❌ 获取作业失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取作业请求失败: {e}")
        return None

def main():
    """主测试流程"""
    print("🚀 开始作业分发管理API测试")
    
    # 1. 登录测试
    if not test_teacher_login():
        print("❌ 教师登录失败，停止测试")
        return
    
    if not test_student_login():
        print("❌ 学生登录失败，停止测试")
        return
    
    # 2. 获取教师班级
    classes = test_get_teacher_classes()
    if not classes:
        print("❌ 没有班级，停止测试")
        return
    
    class_id = classes[0]['id']  # 使用第一个班级
    
    # 3. 获取班级学生
    test_get_class_students(class_id)
    
    # 4. 获取可分发的作业
    homework_id = get_homework_for_assignment()
    if not homework_id:
        print("❌ 没有可分发的作业，停止测试")
        return
    
    # 5. 分发作业
    assignment_id = test_assign_homework(homework_id, class_id)
    if not assignment_id:
        print("❌ 作业分发失败，停止测试")
        return
    
    # 6. 获取教师分发列表
    test_get_teacher_assignments()
    
    # 7. 获取分发详情
    test_get_assignment_detail(assignment_id)
    
    # 8. 更新分发状态
    test_update_assignment_status(assignment_id, "paused")
    test_update_assignment_status(assignment_id, "active")
    
    # 9. 获取学生通知
    test_get_student_notifications()
    
    print("\n🎉 作业分发管理API测试完成！")

if __name__ == "__main__":
    main()

