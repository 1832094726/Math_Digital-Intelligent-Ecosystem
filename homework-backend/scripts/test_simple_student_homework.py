"""
简单测试学生作业功能
"""
import requests

BASE_URL = "http://localhost:5000/api"

def test_simple_api():
    """简单测试API"""
    
    # 1. 学生登录
    login_data = {
        "username": "test_student_001",
        "password": "password123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        result = response.json()
        token = result['data']['access_token']
        print(f"✅ 学生登录成功")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. 测试获取筛选选项
        response = requests.get(f"{BASE_URL}/student/homework/filters/options", headers=headers)
        if response.status_code == 200:
            print("✅ 筛选选项获取成功")
        else:
            print(f"❌ 筛选选项失败: {response.text}")
        
        # 3. 直接查询数据库看看有什么数据
        print("\n=== 检查数据库数据 ===")
        
    else:
        print(f"❌ 登录失败: {response.text}")

if __name__ == "__main__":
    test_simple_api()

