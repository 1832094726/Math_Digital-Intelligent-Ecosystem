#!/usr/bin/env python3
"""
测试API可视化服务器
"""

import requests
import json

def test_api_visualization():
    try:
        # 测试API可视化服务器
        response = requests.get('http://localhost:5001/api/apis', timeout=5)
        
        if response.status_code == 200:
            print("✅ API可视化服务器正常运行")
            data = response.json()
            
            if data.get('success'):
                apis = data.get('data', {}).get('apis', {})
                print(f"📊 发现 {len(apis)} 个API端点")
                
                # 检查是否包含新的反馈和分析API
                feedback_found = False
                analytics_found = False
                
                for api_id, api_info in apis.items():
                    if 'feedback' in api_id.lower():
                        feedback_found = True
                        print(f"  ✅ 找到反馈API: {api_info.get('path', 'N/A')}")
                    elif 'analytics' in api_id.lower():
                        analytics_found = True
                        print(f"  ✅ 找到分析API: {api_info.get('path', 'N/A')}")
                
                if feedback_found and analytics_found:
                    print("🎉 新的反馈和分析API已成功添加到可视化系统！")
                else:
                    print("⚠️  部分新API可能未正确添加")
                    
            else:
                print("❌ API响应格式错误")
                print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"❌ API可视化服务器响应错误: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API可视化服务器 (http://localhost:5001)")
        print("请确保API可视化服务器正在运行")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == '__main__':
    test_api_visualization()
