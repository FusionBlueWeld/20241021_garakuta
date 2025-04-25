import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_vectors(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def vectorize(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_similarities(input_vector, vectors_data):
    similarities = []
    for vector_str, data in vectors_data.items():
        vector = np.array(eval(vector_str))
        similarity = cosine_similarity([input_vector], [vector])[0][0]
        similarities.append((data, similarity, vector_str))
    return similarities

def main():
    # モデルとトークナイザーの初期化
    model_name = 'intfloat/multilingual-e5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # JSONファイルの読み込み
    json_file_path = "output_vectors_and_data.json"
    vectors_data = load_vectors(json_file_path)

    # 入力文章
    input_text = input("類似度を計算したい文章を入力してください: ")

    # 入力文章のベクトル化
    input_vector = vectorize(input_text, tokenizer, model)

    # 類似度の計算
    similarities = calculate_similarities(input_vector, vectors_data)

    # 結果の表示（類似度でソート）
    print("\n類似度の高い順に結果を表示します：")
    for data, similarity, vector_str in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]:
        print(f"データ: {data}")
        print(f"類似度: {similarity:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()