import csv
from transformers import AutoTokenizer

def count_tokens(text):
    """
    入力された文章のトークン数を計算し、512トークンを超えるかどうかを判定する関数
    """
    # モデル名を指定してトークナイザーを初期化
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    
    # 文章をトークン化
    tokens = tokenizer.encode(text)
    
    # トークン数を取得
    token_count = len(tokens)
    
    return token_count

# CSVファイルのパス
csv_path = r"C:\Users\tsuts\Desktop\Python\20240711_E5test\法令シート.csv"

# CSVファイルを開いて処理
with open(csv_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    
    # ヘッダーをスキップ
    next(csv_reader)
    
    # 各行を処理
    for row_number, row in enumerate(csv_reader, start=2):  # 2から始めるのは、1行目がヘッダーだから
        if len(row) >= 14:  # 14列目が存在することを確認
            text = row[13]  # 14列目（インデックスは13）のテキストを取得
            token_count = count_tokens(text)
            
            print(f"行 {row_number}: トークン数 = {token_count}")
            if token_count > 512:
                print("  この文章は512トークンを超えています。")
            else:
                print("  この文章は512トークン以下です。")
            print()  # 空行を挿入して見やすくする
        else:
            print(f"行 {row_number}: データが不足しています。")