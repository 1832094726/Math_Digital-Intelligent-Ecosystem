#!/usr/bin/env python3
"""
直接测试API可视化端点
"""

import requests
import json

def test_api_direct():
    try:
        print("🔍 测试API可视化端点...")
        
        # 测试健康检查
        health_response = requests.get('http://localhost:5001/api/health', timeout=5)
        print(f"健康检查: {health_response.status_code}")
        
        # 测试API列表
        apis_response = requests.get('http://localhost:5001/api/apis', timeout=10)
        print(f"API列表响应: {apis_response.status_code}")
        
        if apis_response.status_code == 200:
            data = apis_response.json()
            print(f"响应成功: {data.get('success', False)}")
            
            if data.get('success'):
                apis_data = data.get('data', {})
                apis = apis_data.get('apis', {})
                categorized = apis_data.get('categorized_apis', {})
                
                print(f"📊 总API数量: {len(apis)}")
                print(f"📊 分类数量: {len(categorized)}")
                
                # 显示前几个API
                for i, (api_id, api_info) in enumerate(list(apis.items())[:5]):
                    print(f"  {i+1}. {api_id}: {api_info.get('path', 'N/A')} - {api_info.get('description', 'N/A')[:50]}...")
                
                # 检查新的API
                feedback_apis = [api_id for api_id in apis.keys() if 'feedback' in api_id.lower()]
                analytics_apis = [api_id for api_id in apis.keys() if 'analytics' in api_id.lower()]
                
                print(f"🔍 反馈相关API: {len(feedback_apis)}")
                for api_id in feedback_apis:
                    print(f"  - {api_id}: {apis[api_id].get('path', 'N/A')}")
                
                print(f"🔍 分析相关API: {len(analytics_apis)}")
                for api_id in analytics_apis:
                    print(f"  - {api_id}: {apis[api_id].get('path', 'N/A')}")
                    
            else:
                print("❌ API响应success为false")
                print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"❌ API请求失败: {apis_response.status_code}")
            print(apis_response.text[:500])
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_api_direct()
