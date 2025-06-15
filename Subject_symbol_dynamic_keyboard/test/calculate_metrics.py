import json
import pandas as pd
import numpy as np

def evaluate_symbol_predictions(data):
    total_correct = 0  # 正确预测的符号总数
    total_predicted = 0  # 预测的符号总数
    total_standard = 0  # 标准符号的总数

    dcg = 0  # 折扣累积增益
    ideal_dcg = 0  # 理想折扣累积增益
    total_outlier_rate = 0  # 用于汇总离群率
    total_reciprocal_rank = 0  # 用于计算平均交互排名 (MRR)
    total_mrr_count = 0  # 计算 MRR 的总样本数

    for entry in data:
        standard_symbols = set(entry["answer_symbols"])  # 标准符号集合
        predicted_symbols = [pred["symbol"] for pred in entry["recommended_symbols"]]  # 预测符号列表

        # 计算当前问题的正确预测符号
        correct_symbols = set(predicted_symbols) & standard_symbols
        total_correct += len(correct_symbols)
        total_standard += len(standard_symbols)
        total_predicted += len(predicted_symbols)

        # 计算DCG和理想DCG
        for i, symbol in enumerate(predicted_symbols):
            if symbol in standard_symbols:
                dcg += 1 / np.log2(i + 2)  # 加1以避免对数为零
        ideal_dcg += sum(1 / np.log2(i + 2) for i in range(len(standard_symbols)))

        # 计算离群率
        rankings = []
        for i, symbol in enumerate(predicted_symbols):
            if symbol in standard_symbols:
                rankings.append(i + 1)  # 加1因为排名从1开始

        if rankings:
            max_rank = max(rankings)  # 找到最大排名
            outlier_rate = len(standard_symbols) / max_rank  # 使用最大排名计算离群率
        else:
            outlier_rate = 0  # 如果没有找到匹配的符号

        total_outlier_rate += outlier_rate  # 汇总离群率

        # 计算 MRR
        if correct_symbols:  # 如果有正确的符号
            first_relevant_rank = min([i + 1 for i, symbol in enumerate(predicted_symbols) if symbol in correct_symbols])
            total_reciprocal_rank += 1 / first_relevant_rank
            total_mrr_count += 1


    # 准确率 （准确率是正确预测的总数与标准符号总数的比率）
    accuracy = total_correct / total_standard if total_standard > 0 else 0

    # 计算精确率（在所有预测的符号中，有多少是实际标准符号）
    precision = total_correct / total_predicted if total_predicted > 0 else 0

    # 在符号推荐中准确率和召回率的计算方法相同
    # 召回率（在所有标准符号中，有多少被成功推荐）
    recall = accuracy

    # F1 值（准确率和召回率的调和平均数，可以综合评估推荐的性能）
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # NDCG（评估推荐结果的排名质量，考虑到推荐的符号的顺序）
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

    # 平均离群率(标准答案符号个数与预测符号中包含标准符号所需的排名名次之比)
    average_outlier_rate = total_outlier_rate / len(data) if len(data) > 0 else 0

    # MRR
    mrr = total_reciprocal_rank / total_mrr_count if total_mrr_count > 0 else 0


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "ndcg": ndcg,
        "average_outlier_rate": average_outlier_rate,
        "mrr": mrr
    }


def evaluate_per_symbol_predictions(data):
    total_correct = 0  # 正确预测符号的总数
    total_standard_symbols = 0  # 标准符号的总数
    accuracies = []  # 存储每一条数据的准确率

    for entry in data:
        standard_symbols = set(entry["answer_symbols"])  # 标准符号集合
        # 预测符号列表
        predicted_symbols = [pred["symbol"] for pred in entry["recommended_symbols"]]

        # 计算当前问题的正确预测符号
        correct_symbols = set(predicted_symbols) & standard_symbols
        total_correct += len(correct_symbols)
        total_standard_symbols += len(standard_symbols)

        # 计算准确率
        accuracy = len(correct_symbols) / len(standard_symbols) if len(standard_symbols) > 0 else 0
        accuracies.append({"id": entry["id"], "accuracy": accuracy})

    # 总体准确率
    overall_accuracy = total_correct / total_standard_symbols if total_standard_symbols > 0 else 0

    return accuracies, overall_accuracy

def evaluate_gpt_predictions(data):
    total_correct = 0  # 正确预测符号的总数
    total_standard_symbols = 0  # 标准符号的总数
    total_gpt_predicted_symbols = 0  # GPT 预测符号的总数

    dcg = 0  # 折扣累积增益
    ideal_dcg = 0  # 理想折扣累积增益
    total_outlier_rate = 0  # 用于汇总离群率
    total_reciprocal_rank = 0  # 用于计算平均交互排名 (MRR)
    total_mrr_count = 0  # 计算 MRR 的总样本数

    for entry in data:
        standard_symbols = set(entry["answer_symbols"])  # 标准符号集合
        gpt_predicted_symbols = set(entry["ui_predicted"])   # GPT 预测的符号列表

        # 计算当前问题的正确预测符号
        correct_symbols = gpt_predicted_symbols & standard_symbols
        total_correct += len(correct_symbols)
        total_standard_symbols += len(standard_symbols)
        total_gpt_predicted_symbols += len(gpt_predicted_symbols)

        # 计算DCG和理想DCG
        for i, symbol in enumerate(gpt_predicted_symbols):
            if symbol in standard_symbols:
                dcg += 1 / np.log2(i + 2)  # 加1以避免对数为零
        ideal_dcg += sum(1 / np.log2(i + 2) for i in range(len(standard_symbols)))

        # 计算离群率
        rankings = []
        for i, symbol in enumerate(gpt_predicted_symbols):
            if symbol in standard_symbols:
                rankings.append(i + 1)  # 加1因为排名从1开始

        if rankings:
            max_rank = max(rankings)  # 找到最大排名
            outlier_rate = len(standard_symbols) / max_rank  # 使用最大排名计算离群率
        else:
            outlier_rate = 0  # 如果没有找到匹配的符号

        total_outlier_rate += outlier_rate  # 汇总离群率

        # 计算 MRR
        if correct_symbols:  # 如果有正确的符号
            first_relevant_rank = min([i + 1 for i, symbol in enumerate(gpt_predicted_symbols) if symbol in correct_symbols])
            total_reciprocal_rank += 1 / first_relevant_rank
            total_mrr_count += 1

    # 准确率
    accuracy = total_correct / total_standard_symbols if total_standard_symbols > 0 else 0

    # 计算精确率
    precision = total_correct / total_gpt_predicted_symbols if total_gpt_predicted_symbols > 0 else 0

    # 召回率（在所有标准符号中，有多少被成功推荐）
    recall = accuracy

    # F1 值
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # NDCG
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

    # 平均离群率
    average_outlier_rate = total_outlier_rate / len(data) if len(data) > 0 else 0

    # MRR
    mrr = total_reciprocal_rank / total_mrr_count if total_mrr_count > 0 else 0


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "ndcg": ndcg,
        "average_outlier_rate": average_outlier_rate,
        "mrr": mrr
    }

def evaluate_per_gpt_predictions(data):
    total_correct = 0  # 正确预测符号的总数
    total_standard_symbols = 0  # 标准符号的总数
    accuracies = []  # 存储每一条数据的准确率

    for entry in data:
        standard_symbols = set(entry["answer_symbols"])  # 标准符号集合
        gpt_predicted_symbols = set(entry["ui_predicted"])  # GPT 预测的符号集合

        # 计算当前问题的正确预测符号
        correct_symbols = gpt_predicted_symbols & standard_symbols
        total_correct += len(correct_symbols)
        total_standard_symbols += len(standard_symbols)

        # 计算准确率
        accuracy = len(correct_symbols) / len(standard_symbols) if len(standard_symbols) > 0 else 0
        accuracies.append({"id": entry["id"], "accuracy": accuracy})

    # 总体准确率
    overall_accuracy = total_correct / total_standard_symbols if total_standard_symbols > 0 else 0

    return accuracies, overall_accuracy


# # 加载Excel文件
# excel_file_path = "./gpt/few-cot.xlsx"
# df= pd.read_excel(excel_file_path, header=None)
#
# # 生成Excel数据映射
# excel_data = {}
# for index, row in df.iterrows():
#     row_id = int(row[0])  # 第一列作为序号
#     symbols = row[2].strip("[]").replace("'", "").split(",")
#     symbols = [symbol.strip() for symbol in symbols]
#     excel_data[row_id] = symbols
#
# #
# # 加载数据
# with open('./new_data/test_data.json', 'r', encoding='utf-8') as f:
#     json_data = json.load(f)
#
# # 将Excel数据添加到JSON中的GPT_prediction_data字段
# for entry in json_data:
#     entry_id = int(entry["id"])
#     if excel_data.get(entry_id):
#         entry["GPT_prediction_data"] = excel_data[entry_id]
#     else:
#         entry["GPT_prediction_data"] = []  # 如果Excel中没有相关数据，则设置为空列表
#
# # 将最终结果写入新的JSON文件
# output_file_path = "./gpt/calculate_metrics_data-few-cot.json"  # 替换为实际的输出文件路径
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     json.dump(json_data, f, ensure_ascii=False, indent=4)
# #
# print(f"JSON数据已成功写入到 {output_file_path}")

with open('./updated_data_test_origin_2.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)
data = json_data

for item in data:
    if len(item['answer_symbols']) == 0:
        # 删除这个item
        data.remove(item)

# 调用评估函数
result = evaluate_symbol_predictions(data)
print("OURS Prediction Data Metrics:")
# 输出评估结果
print("Accuracy:", result["accuracy"])
print("Precision:", result["precision"])
print("Recall:", result["recall"])
print("F1 Score:", result["f1_score"])
print("NDCG:", result["ndcg"])
print("Average Outlier Rate:", result["average_outlier_rate"])
print("MRR:", result["mrr"])


# # 调用评估函数，计算 GPT_prediction_data 的评估结果
# gpt_result = evaluate_gpt_predictions(data)
# # 输出 GPT_prediction_data 的评估结果
# print("协同过滤模型(IBCF单独) Prediction Data Metrics:")
# print("Accuracy:", gpt_result ["accuracy"])
# print("Precision:", gpt_result ["precision"])
# print("Recall:", gpt_result ["recall"])
# print("F1 Score:", gpt_result ["f1_score"])
# print("NDCG:",gpt_result["ndcg"])
# print("Average Outlier Rate:", gpt_result["average_outlier_rate"])
# print("MRR:", gpt_result["mrr"])


# # 调用评估函数，计算 predicted_symbols 的评估结果
symbol_accuracies, overall_symbol_accuracy = evaluate_per_symbol_predictions(data)
print("Predicted Symbols Accuracy per Entry:")
for accuracy in symbol_accuracies:
    print(f"ID: {accuracy['id']}, Accuracy: {accuracy['accuracy']:.2f}")

print(f"Overall Predicted Symbols Accuracy: {overall_symbol_accuracy:.2f}")


# 调用评估函数，计算 GPT_prediction_data 的评估结果
# gpt_accuracies, overall_gpt_accuracy = evaluate_per_gpt_predictions(data)
# print("\n协同过滤模型(IBCF单独) Prediction Data Accuracy per Entry:")
# for accuracy in gpt_accuracies:
#     print(f"ID: {accuracy['id']}, Accuracy: {accuracy['accuracy']:.2f}")

# print(f"Overall GPT Prediction Data Accuracy: {overall_gpt_accuracy:.2f}")
