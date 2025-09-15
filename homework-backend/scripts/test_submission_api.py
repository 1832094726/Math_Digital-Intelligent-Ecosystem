"""
作业提交API测试脚本
"""
import requests
import json
from datetime import datetime

# 配置
BASE_URL = "http://localhost:5000/api"
student_token = None
assignment_id_to_submit = 1 # 假设学生有权访问ID为1的作业分发

def login_student():
    """学生登录获取token"""
    global student_token
    login_data = {"username": "test_student_001", "password": "password123"}
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        if response.status_code == 200 and response.json().get('success'):
            student_token = response.json()['data']['access_token']
            print("✅ 学生登录成功")
            return True
        print(f"❌ 学生登录失败: {response.status_code} {response.text}")
        return False
    except Exception as e:
        print(f"❌ 登录请求异常: {e}")
        return False

def test_submit_homework():
    """测试提交作业"""
    print("\n=== 测试作业提交 ===")
    if not student_token:
        print("❌ 未登录，跳过测试")
        return

    headers = {"Authorization": f"Bearer {student_token}", "Content-Type": "application/json"}
    
    # 模拟学生答案
    answers = {
        "q1": "模拟答案1",
        "q2": "模拟答案2"
    }
    
    submission_data = {
        "answers": answers,
        "time_spent": 1800  # 30分钟
    }

    try:
        response = requests.post(f"{BASE_URL}/submission/{assignment_id_to_submit}", 
                                 data=json.dumps(submission_data), headers=headers)
        
        print(f"提交作业 状态码: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            if result.get('success'):
                print(f"✅ 作业提交成功! Submission ID: {result.get('submission_id')}")
                print(f"  自动评分结果: {result.get('auto_grade_result')}")
                return result.get('submission_id')
            else:
                print(f"❌ 作业提交失败: {result.get('message')}")
        else:
            print(f"❌ 作业提交请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 提交请求异常: {e}")
    return None

def test_get_submission_result(submission_id):
    """测试获取提交结果"""
    print(f"\n=== 测试获取提交结果 (ID: {submission_id}) ===")
    if not student_token:
        print("❌ 未登录，跳过测试")
        return

    headers = {"Authorization": f"Bearer {student_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/submission/{submission_id}/result", headers=headers)
        print(f"获取结果 状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 成功获取提交结果:")
                print(json.dumps(result.get('data'), indent=2, ensure_ascii=False))
            else:
                print(f"❌ 获取结果失败: {result.get('message')}")
        else:
            print(f"❌ 获取结果请求失败: {response.text}")

    except Exception as e:
        print(f"❌ 获取结果请求异常: {e}")

def main():
    """主测试流程"""
    print("🚀 开始作业提交API测试")
    if login_student():
        submission_id = test_submit_homework()
        if submission_id:
            test_get_submission_result(submission_id)
    print("\n🎉 作业提交API测试完成!")

if __name__ == "__main__":
    main()

