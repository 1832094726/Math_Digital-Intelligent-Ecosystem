#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API健康检查脚本
"""
import requests
import json
import time

class APIHealthChecker:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
    
    def login(self):
        """登录获取token"""
        print("🔐 执行登录...")
        
        try:
            response = self.session.post(f"{self.base_url}/api/auth/login", json={
                "username": "test_student_001",
                "password": "password"
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.token = data['data']['access_token']
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.token}'
                    })
                    print("✅ 登录成功")
                    return True
                else:
                    print(f"❌ 登录失败: {data.get('message')}")
            else:
                print(f"❌ 登录请求失败: {response.status_code}")
            
        except Exception as e:
            print(f"❌ 登录异常: {e}")
        
        return False
    
    def check_endpoint(self, method, endpoint, data=None, expected_status=200):
        """检查单个API端点"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                print(f"❌ 不支持的HTTP方法: {method}")
                return False
            
            if response.status_code == expected_status:
                print(f"✅ {method} {endpoint} - {response.status_code}")
                return True
            else:
                print(f"❌ {method} {endpoint} - 期望 {expected_status}, 实际 {response.status_code}")
                if response.text:
                    print(f"   响应: {response.text[:100]}...")
                return False
                
        except Exception as e:
            print(f"❌ {method} {endpoint} - 异常: {e}")
            return False
    
    def check_auth_apis(self):
        """检查认证相关API"""
        print("\n🔐 检查认证API...")
        
        results = []
        
        # 登录API
        results.append(self.check_endpoint('POST', '/api/auth/login', {
            "username": "test_student_001",
            "password": "password"
        }))
        
        # 用户信息API
        if self.token:
            results.append(self.check_endpoint('GET', '/api/auth/me'))
        
        return all(results)
    
    def check_homework_apis(self):
        """检查作业相关API"""
        print("\n📚 检查作业API...")
        
        results = []
        
        # 获取作业列表
        results.append(self.check_endpoint('GET', '/api/homework'))
        
        # 获取学生作业
        results.append(self.check_endpoint('GET', '/api/student-homework'))
        
        return all(results)
    
    def check_submission_apis(self):
        """检查提交相关API"""
        print("\n📤 检查提交API...")
        
        results = []
        
        # 获取提交列表
        results.append(self.check_endpoint('GET', '/api/submissions'))
        
        return all(results)
    
    def check_recommendation_apis(self):
        """检查推荐相关API"""
        print("\n🤖 检查推荐API...")
        
        results = []
        
        # 符号推荐
        results.append(self.check_endpoint('POST', '/api/enhanced-symbol/recommend', {
            "content": "计算 2 + 3 = ?",
            "context": "数学题目"
        }))
        
        # 知识点推荐
        results.append(self.check_endpoint('POST', '/api/enhanced-symbol/knowledge', {
            "content": "有理数运算",
            "grade": 7
        }))
        
        return all(results)
    
    def check_grading_apis(self):
        """检查评分相关API"""
        print("\n🎯 检查评分API...")
        
        results = []
        
        # 注意：评分API需要教师权限，这里只检查端点是否存在
        # 期望返回403（权限不足）而不是404（端点不存在）
        results.append(self.check_endpoint('GET', '/api/grading/rules/1', expected_status=403))
        
        return all(results)
    
    def check_database_connection(self):
        """检查数据库连接"""
        print("\n🗄️ 检查数据库连接...")
        
        # 通过API间接检查数据库连接
        return self.check_endpoint('GET', '/api/auth/me')
    
    def run_full_check(self):
        """运行完整的健康检查"""
        print("🏥 开始API健康检查...")
        print(f"🌐 目标服务器: {self.base_url}")
        
        # 登录
        if not self.login():
            print("❌ 无法登录，跳过需要认证的API检查")
            return False
        
        # 检查各个模块的API
        results = []
        
        results.append(self.check_auth_apis())
        results.append(self.check_homework_apis())
        results.append(self.check_submission_apis())
        results.append(self.check_recommendation_apis())
        results.append(self.check_grading_apis())
        results.append(self.check_database_connection())
        
        # 总结结果
        success_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 检查结果: {success_count}/{total_count} 模块正常")
        
        if success_count == total_count:
            print("✅ 所有API模块正常运行")
            return True
        else:
            print("⚠️ 部分API模块存在问题")
            return False
    
    def monitor_performance(self, duration=60):
        """性能监控"""
        print(f"\n⏱️ 开始性能监控 ({duration}秒)...")
        
        start_time = time.time()
        request_count = 0
        error_count = 0
        response_times = []
        
        while time.time() - start_time < duration:
            try:
                start_request = time.time()
                response = self.session.get(f"{self.base_url}/api/auth/me")
                end_request = time.time()
                
                request_count += 1
                response_time = end_request - start_request
                response_times.append(response_time)
                
                if response.status_code != 200:
                    error_count += 1
                
                time.sleep(1)  # 每秒一次请求
                
            except Exception as e:
                error_count += 1
                print(f"⚠️ 请求异常: {e}")
        
        # 计算统计信息
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            print(f"📈 性能统计:")
            print(f"  总请求数: {request_count}")
            print(f"  错误数: {error_count}")
            print(f"  错误率: {error_count/request_count*100:.1f}%")
            print(f"  平均响应时间: {avg_response_time*1000:.1f}ms")
            print(f"  最大响应时间: {max_response_time*1000:.1f}ms")
            print(f"  最小响应时间: {min_response_time*1000:.1f}ms")

def main():
    """主函数"""
    checker = APIHealthChecker()
    
    # 运行健康检查
    success = checker.run_full_check()
    
    if success:
        print("\n🎉 系统运行正常，可以进行性能监控")
        # 可选：运行性能监控
        # checker.monitor_performance(30)
    else:
        print("\n⚠️ 系统存在问题，请检查日志")

if __name__ == "__main__":
    main()
