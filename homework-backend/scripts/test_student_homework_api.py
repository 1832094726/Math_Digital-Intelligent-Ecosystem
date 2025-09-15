"""
学生作业接收与展示API测试脚本
"""
import requests
import json
from datetime import datetime

# 配置
BASE_URL = "http://localhost:5000/api"

# 全局变量存储token
student_token = None

def test_student_login():
    """测试学生登录"""
    global student_token
    print("\n=== 学生登录测试 ===")
    
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
                student_token = result['data']['access_token']
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

def test_get_homework_list():
    """测试获取作业列表"""
    print("\n=== 获取学生作业列表测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        # 测试获取所有作业
        response = requests.get(f"{BASE_URL}/student/homework/list", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 获取作业列表成功")
            print(f"作业总数: {result['pagination']['total_count']}")
            print(f"当前页: {result['pagination']['current_page']}")
            
            for homework in result['data']:
                print(f"  - 作业: {homework['title']}")
                print(f"    班级: {homework['class_name']}")
                print(f"    状态: {homework['computed_status']}")
                print(f"    截止时间: {homework['due_date']}")
                if homework['remaining_days'] is not None:
                    print(f"    剩余天数: {homework['remaining_days']}")
                print(f"    紧急: {'是' if homework['is_urgent'] else '否'}")
                
            return result['data']
        else:
            print(f"❌ 获取作业列表失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取作业列表请求失败: {e}")
        return []

def test_get_homework_list_with_filters():
    """测试带筛选条件的作业列表"""
    print("\n=== 测试作业列表筛选 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    # 测试不同状态筛选
    statuses = ['pending', 'in_progress', 'completed', 'overdue']
    
    for status in statuses:
        try:
            response = requests.get(f"{BASE_URL}/student/homework/list?status={status}", headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"  {status}: {result['pagination']['total_count']} 个作业")
            else:
                print(f"  {status}: 获取失败")
        except Exception as e:
            print(f"  {status}: 请求失败 - {e}")

def test_get_homework_detail(assignment_id):
    """测试获取作业详情"""
    print(f"\n=== 获取作业{assignment_id}详情测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/student/homework/{assignment_id}", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                assignment = result['data']['assignment']
                progress = result['data']['progress']
                submission = result['data']['submission']
                
                print(f"✅ 获取作业详情成功")
                print(f"作业标题: {assignment['title']}")
                print(f"作业描述: {assignment['description']}")
                print(f"班级: {assignment['class_name']}")
                print(f"难度: {assignment['difficulty_level']}")
                print(f"预计用时: {assignment['estimated_time']} 分钟")
                print(f"状态: {assignment['computed_status']}")
                
                if assignment.get('remaining_hours'):
                    print(f"剩余时间: {assignment['remaining_hours']:.2f} 小时")
                
                if progress:
                    print(f"进度: {progress.get('completion_rate', 0)}%")
                    print(f"最后保存: {progress.get('last_saved_at', '未保存')}")
                else:
                    print("进度: 未开始")
                
                if submission:
                    print(f"提交状态: {submission['status']}")
                    if submission['submitted_at']:
                        print(f"提交时间: {submission['submitted_at']}")
                    if submission['score']:
                        print(f"分数: {submission['score']}")
                else:
                    print("提交状态: 未提交")
                
                return assignment
            else:
                print(f"❌ 获取作业详情失败: {result['message']}")
                return None
        else:
            print(f"❌ 获取作业详情失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取作业详情请求失败: {e}")
        return None

def test_get_homework_statistics():
    """测试获取作业统计"""
    print("\n=== 获取作业统计测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/student/homework/statistics", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                stats = result['data']
                print(f"✅ 获取统计信息成功")
                print(f"总作业数: {stats['total_count']}")
                print(f"已完成: {stats['completed_count']}")
                print(f"进行中: {stats['in_progress_count']}")
                print(f"待完成: {stats['pending_count']}")
                print(f"已过期: {stats['overdue_count']}")
                print(f"完成率: {stats['completion_rate']}%")
                print(f"平均分: {stats['average_score']}")
                return stats
            else:
                print(f"❌ 获取统计信息失败: {result['message']}")
                return None
        else:
            print(f"❌ 获取统计信息失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取统计信息请求失败: {e}")
        return None

def test_save_homework_progress(homework_id):
    """测试保存作业进度"""
    print(f"\n=== 保存作业{homework_id}进度测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    # 模拟答题进度
    progress_data = {
        "answers": {
            "q1": "这是第一题的答案",
            "q2": "2x + 3 = 7",
            "q3": ""  # 未完成的题目
        },
        "start_time": "2024-12-12T10:00:00",
        "current_question": 2,
        "total_questions": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/student/homework/{homework_id}/progress", 
                               json={"progress_data": progress_data}, headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 保存进度成功")
                print(f"完成率: {result['completion_rate']}%")
                return True
            else:
                print(f"❌ 保存进度失败: {result['message']}")
                return False
        else:
            print(f"❌ 保存进度失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 保存进度请求失败: {e}")
        return False

def test_get_homework_progress(homework_id):
    """测试获取作业进度"""
    print(f"\n=== 获取作业{homework_id}进度测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/student/homework/{homework_id}/progress", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success'] and result['data']:
                progress = result['data']
                print(f"✅ 获取进度成功")
                print(f"完成率: {progress['completion_rate']}%")
                print(f"最后保存: {progress['last_saved_at']}")
                if progress.get('progress_data'):
                    answers = progress['progress_data'].get('answers', {})
                    completed_answers = sum(1 for answer in answers.values() if answer and str(answer).strip())
                    print(f"已答题数: {completed_answers}/{len(answers)}")
                return progress
            else:
                print(f"进度: 未开始")
                return None
        else:
            print(f"❌ 获取进度失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取进度请求失败: {e}")
        return None

def test_homework_favorite(assignment_id):
    """测试作业收藏功能"""
    print(f"\n=== 作业{assignment_id}收藏测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        # 收藏作业
        response = requests.post(f"{BASE_URL}/student/homework/{assignment_id}/favorite", 
                               json={"is_favorite": True}, headers=headers)
        print(f"收藏状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 收藏成功: {result['message']}")
            else:
                print(f"❌ 收藏失败: {result['message']}")
        
        # 获取收藏列表
        response = requests.get(f"{BASE_URL}/student/homework/favorites", headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 收藏列表: {result['total']} 个作业")
            for fav in result['data']:
                print(f"  - {fav['title']} (班级: {fav['class_name']})")
        
        # 取消收藏
        response = requests.post(f"{BASE_URL}/student/homework/{assignment_id}/favorite", 
                               json={"is_favorite": False}, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 取消收藏成功: {result['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 收藏测试请求失败: {e}")
        return False

def test_get_homework_reminders():
    """测试获取作业提醒"""
    print("\n=== 获取作业提醒测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/student/homework/reminders?hours_ahead=48", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 获取提醒成功")
            print(f"提醒数量: {result['total']}")
            
            for reminder in result['data']:
                print(f"  - 作业: {reminder['title']}")
                print(f"    班级: {reminder['class_name']}")
                print(f"    截止时间: {reminder['due_date']}")
                print(f"    剩余时间: {reminder['remaining_hours']:.2f} 小时")
                print(f"    难度: {reminder['difficulty_level']}")
            
            return result['data']
        else:
            print(f"❌ 获取提醒失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取提醒请求失败: {e}")
        return []

def test_get_homework_dashboard():
    """测试获取作业仪表板"""
    print("\n=== 获取作业仪表板测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/student/homework/dashboard", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                dashboard = result['data']
                print(f"✅ 获取仪表板数据成功")
                
                # 统计信息
                stats = dashboard['statistics']
                print(f"\n📊 统计信息:")
                print(f"  总作业: {stats['total_count']}")
                print(f"  完成率: {stats['completion_rate']}%")
                print(f"  平均分: {stats['average_score']}")
                
                # 待完成作业
                print(f"\n📝 最近待完成 ({len(dashboard['recent_pending'])}):")
                for hw in dashboard['recent_pending']:
                    print(f"  - {hw['title']} (截止: {hw['due_date']})")
                
                # 进行中作业
                print(f"\n🚀 进行中 ({len(dashboard['in_progress'])}):")
                for hw in dashboard['in_progress']:
                    print(f"  - {hw['title']} (进度: {hw.get('completion_rate', 0)}%)")
                
                # 紧急提醒
                print(f"\n⏰ 紧急提醒 ({len(dashboard['urgent_reminders'])}):")
                for reminder in dashboard['urgent_reminders']:
                    print(f"  - {reminder['title']} (剩余: {reminder['remaining_hours']:.1f}h)")
                
                return dashboard
            else:
                print(f"❌ 获取仪表板失败: {result['message']}")
                return None
        else:
            print(f"❌ 获取仪表板失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取仪表板请求失败: {e}")
        return None

def test_search_homeworks():
    """测试搜索作业"""
    print("\n=== 搜索作业测试 ===")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        # 测试关键词搜索
        response = requests.get(f"{BASE_URL}/student/homework/search?keyword=数学", headers=headers)
        print(f"搜索状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 搜索成功")
            print(f"搜索结果: {result['pagination']['total_count']} 个作业")
            
            for hw in result['data']:
                print(f"  - {hw['title']}")
        else:
            print(f"❌ 搜索失败: {response.text}")
        
        # 获取筛选选项
        response = requests.get(f"{BASE_URL}/student/homework/filters/options", headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 筛选选项获取成功")
            print(f"可用状态: {len(result['data']['statuses'])} 个")
            print(f"可用难度: {len(result['data']['difficulties'])} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ 搜索测试请求失败: {e}")
        return False

def main():
    """主测试流程"""
    print("🚀 开始学生作业接收与展示API测试")
    
    # 1. 学生登录
    if not test_student_login():
        print("❌ 学生登录失败，停止测试")
        return
    
    # 2. 获取作业列表
    homework_list = test_get_homework_list()
    
    # 3. 测试筛选
    test_get_homework_list_with_filters()
    
    # 4. 获取统计信息
    test_get_homework_statistics()
    
    if homework_list:
        # 使用第一个作业进行详细测试
        first_assignment = homework_list[0]
        assignment_id = first_assignment['assignment_id']
        homework_id = first_assignment['homework_id']
        
        # 5. 获取作业详情
        test_get_homework_detail(assignment_id)
        
        # 6. 保存和获取进度
        test_save_homework_progress(homework_id)
        test_get_homework_progress(homework_id)
        
        # 7. 测试收藏功能
        test_homework_favorite(assignment_id)
    
    # 8. 获取提醒
    test_get_homework_reminders()
    
    # 9. 获取仪表板
    test_get_homework_dashboard()
    
    # 10. 测试搜索功能
    test_search_homeworks()
    
    print("\n🎉 学生作业接收与展示API测试完成！")

if __name__ == "__main__":
    main()
