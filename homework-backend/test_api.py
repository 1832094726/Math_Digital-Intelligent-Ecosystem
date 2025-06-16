import requests
import json

def test_get_knowledge_by_id():
    """测试通过题目ID获取知识点"""
    response = requests.get("http://localhost:5000/api/knowledge/question", params={"questionId": 101})
    print("通过题目ID获取知识点:")
    if response.status_code == 200:
        print("状态码:", response.status_code)
        data = response.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("请求失败，状态码:", response.status_code)
        print(response.text)

def test_get_knowledge_by_text():
    """测试通过题目文本获取知识点"""
    question_text = "解方程 x² - 5x + 6 = 0"
    response = requests.get("http://localhost:5000/api/knowledge/question", params={"text": question_text})
    print("\n通过题目文本获取知识点:")
    if response.status_code == 200:
        print("状态码:", response.status_code)
        data = response.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("请求失败，状态码:", response.status_code)
        print(response.text)

def test_post_knowledge():
    """测试通过POST方法获取知识点"""
    data = {"text": "若 a + b = 5，ab = 6，则 a² + b² = ?"}
    response = requests.post("http://localhost:5000/api/knowledge/question", json=data)
    print("\n通过POST方法获取知识点:")
    if response.status_code == 200:
        print("状态码:", response.status_code)
        data = response.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("请求失败，状态码:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    print("开始测试知识点提取API...")
    try:
        test_get_knowledge_by_id()
        test_get_knowledge_by_text()
        test_post_knowledge()
        print("\n测试完成！")
    except Exception as e:
        print(f"测试过程中发生错误: {e}") 