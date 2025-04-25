import json
from transformers import AutoTokenizer, AutoModel
import torch

def vectorize(text, tokenizer, model):
    """
    入力された文章をベクトル化する関数
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

def process_csv_to_json(input_file_path, output_file_path):
    # モデルとトークナイザーを初期化
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

    result_dict = {}

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        next(input_file)  # ヘッダー行をスキップ

        for line in input_file:
            row = line.strip().split(',')
            if len(row) >= 14:
                text = row[13]  # 14列目のデータを取得
                vector = vectorize(text, tokenizer, model)
                
                # 1列目から13列目までのデータを格納（空白セルはNoneに置換）
                row_data = [cell if cell != '' else None for cell in row[:13]]
                
                # ベクトルをキーとして使用するために文字列に変換
                vector_key = str(vector)
                result_dict[vector_key] = row_data

    # 結果をJSONファイルに書き込む
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(result_dict, output_file, ensure_ascii=False, indent=2)

    print(f"処理が完了し、結果が {output_file_path} に保存されました。")

# 入力CSVファイルと出力JSONファイルのパスを指定して処理を実行
input_csv_file_path = "法令シート.csv"
output_json_file_path = "output_vectors_and_data.json"
process_csv_to_json(input_csv_file_path, output_json_file_path)