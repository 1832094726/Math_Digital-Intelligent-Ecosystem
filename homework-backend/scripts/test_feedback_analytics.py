#!/usr/bin/env python3
"""
测试作业反馈和统计分析功能
"""

import requests
import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FeedbackAnalyticsTest:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.token = None
        self.test_homework_id = None
        
    def login(self, username='teacher1', password='password123'):
        """登录获取token"""
        print("🔐 正在登录...")
        
        response = requests.post(f'{self.base_url}/api/auth/login', json={
            'username': username,
            'password': password
        })
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                self.token = data['data']['token']
                print(f"✅ 登录成功，用户: {data['data']['user']['username']}")
                return True
            else:
                print(f"❌ 登录失败: {data.get('message')}")
                return False
        else:
            print(f"❌ 登录请求失败: {response.status_code}")
            return False
    
    def get_headers(self):
        """获取请求头"""
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def test_homework_feedback(self, homework_id=1):
        """测试作业反馈功能"""
        print(f"\n📊 测试作业反馈功能 (作业ID: {homework_id})...")
        
        # 测试获取反馈
        response = requests.get(
            f'{self.base_url}/api/feedback/homework/{homework_id}',
            headers=self.get_headers()
        )
        
        print(f"反馈API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                feedback = data['data']
                print("✅ 反馈获取成功")
                print(f"   作业标题: {feedback['homework_info']['title']}")
                print(f"   个人得分: {feedback['personal_performance']['total_score']}/{feedback['personal_performance']['max_score']}")
                print(f"   班级平均分: {feedback['class_statistics']['class_average']}")
                print(f"   班级排名: {feedback['class_statistics']['student_rank']}/{feedback['class_statistics']['total_students']}")
                print(f"   学习建议数量: {len(feedback['learning_suggestions'])}")
                print(f"   题目反馈数量: {len(feedback['question_feedback'])}")
                
                # 显示学习建议
                if feedback['learning_suggestions']:
                    print("   学习建议:")
                    for suggestion in feedback['learning_suggestions'][:2]:
                        print(f"     - {suggestion['title']}: {suggestion['content'][:50]}...")
                
                return True
            else:
                print(f"❌ 反馈获取失败: {data.get('message')}")
                return False
        else:
            print(f"❌ 反馈请求失败: {response.status_code}")
            if response.text:
                print(f"   错误信息: {response.text}")
            return False
    
    def test_feedback_sharing(self, homework_id=1):
        """测试反馈分享功能"""
        print(f"\n🔗 测试反馈分享功能 (作业ID: {homework_id})...")
        
        response = requests.post(
            f'{self.base_url}/api/feedback/homework/{homework_id}/share',
            json={'type': 'link'},
            headers=self.get_headers()
        )
        
        print(f"分享API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ 分享链接生成成功")
                print(f"   分享链接: {data.get('share_url', 'N/A')}")
                print(f"   过期时间: {data.get('expires_at', 'N/A')}")
                return True
            else:
                print(f"❌ 分享失败: {data.get('message')}")
                return False
        else:
            print(f"❌ 分享请求失败: {response.status_code}")
            return False
    
    def test_homework_analytics(self, homework_id=1):
        """测试作业统计分析功能"""
        print(f"\n📈 测试作业统计分析功能 (作业ID: {homework_id})...")
        
        response = requests.get(
            f'{self.base_url}/api/analytics/homework/{homework_id}',
            headers=self.get_headers()
        )
        
        print(f"分析API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                analytics = data['data']
                print("✅ 分析报告生成成功")
                print(f"   作业标题: {analytics['homework_info']['title']}")
                
                # 基础统计
                basic_stats = analytics['basic_statistics']
                print(f"   总分配数: {basic_stats['total_assignments']}")
                print(f"   完成数量: {basic_stats['completed_count']}")
                print(f"   完成率: {basic_stats['completion_rate']}%")
                print(f"   平均分: {basic_stats['average_score']}")
                print(f"   平均用时: {basic_stats['average_completion_time']}分钟")
                
                # 分数分布
                score_dist = analytics['score_distribution']
                print(f"   分数分布统计:")
                print(f"     平均分: {score_dist['statistics']['mean']}")
                print(f"     中位数: {score_dist['statistics']['median']}")
                print(f"     标准差: {score_dist['statistics']['std_dev']}")
                print(f"     参与人数: {score_dist['statistics']['total_students']}")
                
                # 分数段分布
                if score_dist['distribution']:
                    print("   分数段分布:")
                    for grade in score_dist['distribution']:
                        print(f"     {grade['label']}: {grade['count']}人 ({grade['percentage']}%)")
                
                # 题目分析
                question_analysis = analytics['question_analysis']
                print(f"   题目分析数量: {len(question_analysis)}")
                if question_analysis:
                    print("   题目难度分布:")
                    difficulty_count = {}
                    for q in question_analysis:
                        difficulty = q['difficulty_level']
                        difficulty_count[difficulty] = difficulty_count.get(difficulty, 0) + 1
                    for difficulty, count in difficulty_count.items():
                        print(f"     {difficulty}: {count}题")
                
                # 知识点分析
                knowledge_analysis = analytics['knowledge_analysis']
                print(f"   知识点分析数量: {len(knowledge_analysis)}")
                if knowledge_analysis:
                    weak_points = [k for k in knowledge_analysis if k['mastery_rate'] < 60]
                    if weak_points:
                        print(f"   薄弱知识点 ({len(weak_points)}个):")
                        for point in weak_points[:3]:
                            print(f"     {point['knowledge_point']}: {point['mastery_rate']}%")
                
                # 学生表现
                student_perf = analytics['student_performance']
                print(f"   需要关注的学生: {len(student_perf['struggling_students'])}人")
                print(f"   表现优秀的学生: {len(student_perf['excellent_students'])}人")
                
                # 教学建议
                suggestions = analytics['teaching_suggestions']
                print(f"   教学建议数量: {len(suggestions)}")
                if suggestions:
                    print("   主要建议:")
                    for suggestion in suggestions[:2]:
                        print(f"     [{suggestion['priority']}] {suggestion['title']}")
                        print(f"       {suggestion['content'][:60]}...")
                
                return True
            else:
                print(f"❌ 分析报告生成失败: {data.get('message')}")
                return False
        else:
            print(f"❌ 分析请求失败: {response.status_code}")
            if response.text:
                print(f"   错误信息: {response.text}")
            return False
    
    def test_analytics_export(self, homework_id=1):
        """测试分析报告导出功能"""
        print(f"\n📄 测试分析报告导出功能 (作业ID: {homework_id})...")
        
        for format_type in ['pdf', 'excel']:
            response = requests.post(
                f'{self.base_url}/api/analytics/homework/{homework_id}/export',
                json={'format': format_type},
                headers=self.get_headers()
            )
            
            print(f"{format_type.upper()}导出响应状态: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ {format_type.upper()}导出成功")
                    print(f"   下载链接: {data.get('download_url', 'N/A')}")
                else:
                    print(f"❌ {format_type.upper()}导出失败: {data.get('message')}")
            else:
                print(f"❌ {format_type.upper()}导出请求失败: {response.status_code}")
    
    def test_teacher_overview(self):
        """测试教师概览功能"""
        print(f"\n👨‍🏫 测试教师概览功能...")
        
        response = requests.get(
            f'{self.base_url}/api/analytics/overview',
            headers=self.get_headers()
        )
        
        print(f"概览API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                overview = data['data']
                print("✅ 教师概览获取成功")
                
                summary = overview['summary']
                print(f"   总作业数: {summary['total_homeworks']}")
                print(f"   已发布作业: {summary['published_homeworks']}")
                print(f"   平均分: {summary['average_score']}")
                
                recent_homeworks = overview['recent_homeworks']
                print(f"   最近作业数量: {len(recent_homeworks)}")
                if recent_homeworks:
                    print("   最近作业:")
                    for hw in recent_homeworks[:3]:
                        print(f"     {hw['title']}: {hw['submissions']}份提交, 平均{hw['average_score']}分")
                
                return True
            else:
                print(f"❌ 概览获取失败: {data.get('message')}")
                return False
        else:
            print(f"❌ 概览请求失败: {response.status_code}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始测试作业反馈和统计分析功能")
        print("=" * 60)
        
        # 登录
        if not self.login():
            print("❌ 登录失败，无法继续测试")
            return False
        
        # 测试结果统计
        test_results = []
        
        # 测试作业反馈
        test_results.append(("作业反馈", self.test_homework_feedback()))
        
        # 测试反馈分享
        test_results.append(("反馈分享", self.test_feedback_sharing()))
        
        # 测试作业统计分析
        test_results.append(("统计分析", self.test_homework_analytics()))
        
        # 测试报告导出
        test_results.append(("报告导出", self.test_analytics_export()))
        
        # 测试教师概览
        test_results.append(("教师概览", self.test_teacher_overview()))
        
        # 输出测试结果
        print("\n" + "=" * 60)
        print("📋 测试结果汇总:")
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n总计: {passed}/{total} 项测试通过")
        
        if passed == total:
            print("🎉 所有测试通过！作业反馈和统计分析功能正常工作")
            return True
        else:
            print("⚠️  部分测试失败，请检查相关功能")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试作业反馈和统计分析功能')
    parser.add_argument('--url', default='http://localhost:5000', help='API服务器地址')
    parser.add_argument('--homework-id', type=int, default=1, help='测试用的作业ID')
    parser.add_argument('--username', default='teacher1', help='登录用户名')
    parser.add_argument('--password', default='password123', help='登录密码')
    
    args = parser.parse_args()
    
    # 创建测试实例
    tester = FeedbackAnalyticsTest(args.url)
    
    # 运行测试
    success = tester.run_all_tests()
    
    # 退出
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
