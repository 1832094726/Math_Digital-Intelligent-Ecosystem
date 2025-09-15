import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# 向量相似度


# 加载数据集
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 加载数据集
geometry_data = load_dataset('../../Datasets/change_add_train_layer_1_layer_2_layer_3.json')
math_23k_data = load_dataset('../../Datasets/Math_23K.json')

# 提取 original_text
geometry_texts = [item['original_text'] for item in geometry_data]
math_23k_texts = [item['original_text'] for item in math_23k_data]

# 特征提取
tfidf_vectorizer = TfidfVectorizer()
tfidf_geometry = tfidf_vectorizer.fit_transform(geometry_texts)
tfidf_math_23k = tfidf_vectorizer.transform(math_23k_texts)

# 计算相似度
similarity_matrix = cosine_similarity(tfidf_math_23k, tfidf_geometry)

# 选择相似度高的题目
threshold = 0.5  # 可以根据需要调整阈值
similar_questions_indices = (similarity_matrix.max(axis=1) > threshold).nonzero()[0]
similar_questions = [math_23k_data[i] for i in similar_questions_indices]

# 保存到新的数据集
with open('../../Datasets/similar_geometry_questions.json', 'w', encoding='utf-8') as f:
    json.dump(similar_questions, f, ensure_ascii=False, indent=4)
