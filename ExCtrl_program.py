import pandas as pd
import csv

'''環境省からダウンロードした輸出外為法のExcelファイルを読み込み、特定のセルの内容をCSVに書き出すプログラム'''

# 1. Excelファイルを読み込む
excel_path = "data/gattai_matrix_20231201.xlsx"
xl = pd.ExcelFile(excel_path)

# 2. スキャン開始行番号の抽出
start_rows = {}
for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    row_num = None
    for i, cell in enumerate(df.iloc[:200, 0]):
        if pd.notnull(cell) and "輸出令" in str(cell):
            row_num = i
            break
    start_rows[sheet_name] = row_num

# 3. 結合セルのスキャン
with open('output.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)

    for sheet_name, start_row in start_rows.items():
        print(f"\n{sheet_name}")

        if start_row is None:
            continue

        df = xl.parse(sheet_name)
        start_scan_row = start_row + 2

        # 輸出管理令に関する処理
        export_law_rows = [i for i, cell in enumerate(df.iloc[start_scan_row:, 0], start=start_scan_row) if pd.notnull(cell) and "輸出令" in str(cell)]

        for j in range(len(export_law_rows)):
            merged_start = export_law_rows[j]

            if j + 1 < len(export_law_rows):
                merged_end = export_law_rows[j + 1] - 1
            else:
                merged_end = len(df) - 1

            content = df.iloc[merged_start, 0].replace('\n', '')
            b_texts = df.iloc[merged_start:merged_end + 1, 1].dropna().map(lambda x: str(x).replace('\n', '')).tolist()

            for n in b_texts:
                writer.writerow([sheet_name, content, "", "", n])  # b_textsの内容をCSVに書き出す

            # 貨物等省令に関する処理
            cargo_law_rows = [i for i in range(merged_start, merged_end + 1) if pd.notnull(df.iloc[i, 2]) and '貨物等省令' in str(df.iloc[i, 2])]

            # 貨物等省令の項目があるとき
            if cargo_law_rows:

                for k in range(len(cargo_law_rows)):
                    sub_start = cargo_law_rows[k]
                    if k + 1 < len(cargo_law_rows):
                        sub_end = cargo_law_rows[k + 1] - 1
                    else:
                        sub_end = merged_end

                    sub_content = df.iloc[sub_start, 2].replace('\n', '')
                    sub_texts = df.iloc[sub_start:sub_end + 1, 3].dropna().map(lambda x: str(x).replace('\n', '')).tolist()

                    for n in sub_texts:
                        writer.writerow([sheet_name, content, sub_content, "", n])  # sub_textsの内容をCSVに書き出す

                    add_rows = [i for i in range(sub_start, sub_end + 1) if pd.notnull(df.iloc[i, 4])]

                    sub_additions = []
                    for l in range(len(add_rows)):
                        add_start = add_rows[l]
                        if l + 1 < len(add_rows):
                            add_end = add_rows[l + 1] - 1
                        else:
                            add_end = sub_end

                        add_content = df.iloc[add_start, 4].replace('\n', '')
                        add_texts = df.iloc[add_start:add_end + 1, [5, 6]].dropna(how='all').stack().map(lambda x: str(x).replace('\n', '')).tolist()

                        for n in add_texts:
                            writer.writerow([sheet_name, content, sub_content, add_content, n])  # add_textsの内容をCSVに書き出す

            # 貨物等省令の項目が無いとき
            else:
                add_rows = [i for i in range(merged_start, merged_end + 1) if pd.notnull(df.iloc[i, 4])]

                sub_additions = []
                for l in range(len(add_rows)):
                    add_start = add_rows[l]
                    if l + 1 < len(add_rows):
                        add_end = add_rows[l + 1] - 1
                    else:
                        add_end = merged_end

                    add_content = df.iloc[add_start, 4].replace('\n', '')
                    add_texts = df.iloc[add_start:add_end + 1, [5, 6]].dropna(how='all').stack().map(lambda x: str(x).replace('\n', '')).tolist()

                    for n in add_texts:
                        writer.writerow([sheet_name, content, "", add_content, n])  # add_textsの内容をCSVに書き出す

