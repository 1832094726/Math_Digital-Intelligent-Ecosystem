import json
import pandas as pd
# 读取 questions 数据
with open("../matrix_data/情景模式库_聚类结果.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 读取 clusters 数据
with open("../matrix_data/情景模式库_符号概率.json", "r", encoding="utf-8") as f:
    clusters = json.load(f)
    
# # 合并符号及概率
# def merge_symbols(questions, clusters):
#     for question in questions:
#         cluster_id = str(question["cluster"])
#         cluster_symbols = clusters.get(cluster_id, {}).get("symbols", [])
        
#         # 将 answer_symbols 转换为带概率的格式（概率为 1）
#         answer_symbols_with_prob = [{"symbol": sym, "probability": 1.0} for sym in question["answer_symbols"]]
        
#         # 合并 cluster 符号，同时避免重复符号
#         existing_symbols = set(question["answer_symbols"])
#         for sym_data in cluster_symbols:
#             if sym_data["symbol"] not in existing_symbols:
#                 answer_symbols_with_prob.append(sym_data)
        
#         # 更新 question 中的符号和概率
#         question["answer_symbols"] = answer_symbols_with_prob

def merge_symbols(questions, clusters):
    for question in questions:
        cluster_id = str(question["cluster"])
        cluster_symbols = clusters.get(cluster_id, {}).get("symbols", [])
        
        # 直接将 cluster 符号及其概率赋值为 question["answer_symbols"]
        question["answer_symbols"] = cluster_symbols
# 执行合并
merge_symbols(questions, clusters)

# 输出结果
output_file = "./matrix_data/cluster_new.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)

print(f"数据已成功保存到 {output_file}")

with open("train_data.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

matrix_dict = {item['id']: item['answer_symbols'] for item in questions}

# 遍历 traindata 并更新 answer_symbols
for entry in train_data:
    id = entry['id']
    if id in matrix_dict:
        # 替换 traindata 中的 answer_symbols
        entry['answer_symbols'] = matrix_dict[id]
    
output_filename = "./new_data/train_data_cluster.json"

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

print(f"数据已输出到 {output_filename}")

# 初始化存储数据的列表
rating_data = []

# space_symbol.json
with open("symbol_space.json", "r", encoding="utf-8") as f:
    symbols_list = json.load(f)

with open("./new_data/train_data_cluster.json", "r", encoding="utf-8") as f:
    user = json.load(f)
    
for i, item in enumerate(user):
    user_id = i
    answer_symbols = item["answer_symbols"]

    for j, symbol in enumerate(symbols_list):
        # 找到符号对应的概率
        rating = 0  # 默认值为 0
        if isinstance(answer_symbols, list) and all(isinstance(item, dict) for item in answer_symbols):
            for answer_symbol in answer_symbols:
                if answer_symbol["symbol"] == symbol:  # 检查是否匹配
                    rating = answer_symbol["probability"]  # 设置对应的概率
                    break  # 如果找到了匹配的符号就跳出循环
        else:
            for answer_symbol in answer_symbols:
                if answer_symbol == answer_symbols:  # 检查是否存在
                    rating = 1.0  # 设置默认概率为1

        # 保存数据
        rating_data.append({
            "user_id": user_id,
            "item_id": j,  # item_id 是符号在符号列表中的索引
            "rating": rating
        })
        
# 将数据转换为 DataFrame
df = pd.DataFrame(rating_data)

# 保存为 CSV 文件
csv_filename = "train/rating_data_cluster.csv"
df.to_csv(csv_filename, index=False, header=["user_id", "item_id", "rating"], encoding="utf-8")

print(f"CSV文件已成功保存为 {csv_filename}")
