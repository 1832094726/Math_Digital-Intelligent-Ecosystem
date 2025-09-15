import json

# 关键词分类

# 加载 Math_23k 数据集
math_23k_data_path = "../../Datasets/Math_23K.json"  # Math_23k 数据集的路径
with open(math_23k_data_path, "r", encoding="utf-8") as file:
    math_23k_data = json.load(file)

# 几何题目的关键词列表
geometry_keywords = ['梯形','三角形', '圆形', '正方形', '长方形','圆环', '平行四边形' ,'正方体', '长方体', '正方体', '圆柱', '圆锥', '水结成冰', '水结冰', '冰化成水', '冰化水', '冰融化', "结成冰"]  # 根据需要添加更多关键词

# 筛选几何题目
geometry_questions = []
for item in math_23k_data:
    text = item["original_text"]
    if any(keyword in text for keyword in geometry_keywords):
        geometry_questions.append(item)

# 将几何题目保存到新的 JSON 文件中
with open('../../Datasets/geometry_questions.json', 'w', encoding='utf-8') as f:
    json.dump(geometry_questions, f, ensure_ascii=False, indent=4)
