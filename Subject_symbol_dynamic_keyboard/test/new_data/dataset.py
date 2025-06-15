import pandas as pd
import json
import random

with open("symbol_space.json", "r", encoding="utf-8") as f:
    symbols = json.load(f)

# 构造 item_id 和 symbol 的对应关系
item = {'item_id': range(len(symbols)), 'symbol': symbols}

# 创建 DataFrame
item_df = pd.DataFrame(item)

# 保存为CSV文件
csv_filename = "item_data.csv"
item_df.to_csv(csv_filename, index=False, header=["item_id", "symbol"], encoding="utf-8")

print(f"CSV文件已成功保存为 {csv_filename}")
#
# # 读取所有数据的 JSON 文件
# with open("all_output.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 打乱数据顺序
# random.shuffle(data)
#
# # 按 9:1 比例划分为 train_data 和 test_data
# train_size = int(0.95 * len(data))
# train_data = data[:train_size]
# test_data = data[train_size:]
#
# # 将 train_data 写入 train_data.json 文件
# with open("train_data.json", "w", encoding="utf-8") as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=4)
#
# # 将 test_data 写入 test_data.json 文件
# with open("test_data.json", "w", encoding="utf-8") as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=4)


with open("train_data.json", "r", encoding="utf-8") as f:
    user = json.load(f)

# 从数据中提取 user_id 和 question
user_data = [{'user_id': i, 'question': item['question']} for i, item in enumerate(user)]

# 转换为 DataFrame
user_df = pd.DataFrame(user_data)

# 保存为 CSV 文件
csv_filename = "train/user_data.csv"
user_df.to_csv(csv_filename, index=False, header=["user_id", "question"], encoding="utf-8")

print(f"CSV文件已成功保存为 {csv_filename}")

# 初始化存储数据的列表
rating_data = []

# space_symbol.json
with open("symbol_space.json", "r", encoding="utf-8") as f:
    symbols_list = json.load(f)

# 遍历每个问题
for i, item in enumerate(user):
    user_id = i
    answer_symbols = item["answer_symbols"]

    # 为每个符号创建 rating
    for j, symbol in enumerate(symbols_list):
        # item_id 是符号在符号列表中的索引
        item_id = j
        # rating 是 1 或 0，表示符号是否出现在 answer_symbols 中
        rating = 1 if symbol in answer_symbols else 0

        # 如果符号存在于 answer_symbols 中，才保存数据
        # if rating == 1:
        rating_data.append({"user_id": user_id, "item_id": item_id, "rating": rating})

# 将数据转换为 DataFrame
df = pd.DataFrame(rating_data)

# 保存为 CSV 文件
csv_filename = "train/rating_data.csv"
df.to_csv(csv_filename, index=False, header=["user_id", "item_id", "rating"], encoding="utf-8")

print(f"CSV文件已成功保存为 {csv_filename}")

with open("test_data.json", "r", encoding="utf-8") as f:
    user = json.load(f)

# 从数据中提取 user_id 和 question
user_data = [{'user_id': i, 'question': item['question']} for i, item in enumerate(user)]

# 转换为 DataFrame
user_df = pd.DataFrame(user_data)

# 保存为 CSV 文件
csv_filename = "test/user_data.csv"
user_df.to_csv(csv_filename, index=False, header=["user_id", "question"], encoding="utf-8")

print(f"CSV文件已成功保存为 {csv_filename}")

# 初始化存储数据的列表
rating_data = []

# space_symbol.json
with open("symbol_space.json", "r", encoding="utf-8") as f:
    symbols_list = json.load(f)

# 遍历每个问题
for i, item in enumerate(user):
    user_id = i
    answer_symbols = item["answer_symbols"]

    # 为每个符号创建 rating
    for j, symbol in enumerate(symbols_list):
        # item_id 是符号在符号列表中的索引
        item_id = j
        # rating 是 1 或 0，表示符号是否出现在 answer_symbols 中
        rating = 1 if symbol in answer_symbols else 0

        # 如果符号存在于 answer_symbols 中，才保存数据
        rating_data.append({"user_id": user_id, "item_id": item_id, "rating": rating})

# 将数据转换为 DataFrame
df = pd.DataFrame(rating_data)

# 保存为 CSV 文件
csv_filename = "test/rating_data.csv"
df.to_csv(csv_filename, index=False, header=["user_id", "item_id", "rating"], encoding="utf-8")

print(f"CSV文件已成功保存为 {csv_filename}")



# # 读取 CSV 文件
# df = pd.read_csv("new_data/train/rating_data.csv")

# # 选择所有 rating == 1 的数据
# positive_samples = df[df['rating'] == 1]

# # 选择所有 rating == 0 的数据
# negative_samples = df[df['rating'] == 0]

# # 随机选择与 rating == 1 数量相同的 rating == 0 样本
# num_positive = len(positive_samples)
# selected_negative_samples = negative_samples.sample(n=num_positive, random_state=42)

# # 合并正样本和负样本
# final_samples = pd.concat([positive_samples, selected_negative_samples])

# # 打乱合并后的样本顺序
# final_samples = final_samples.sample(frac=1, random_state=42).reset_index(drop=True)

# # 保存最终的 CSV 文件
# final_samples.to_csv("new_data/train/rating_data_final.csv", index=False)

# print("新的 CSV 文件已保存为 'new_data/rating_data_final.csv'")
