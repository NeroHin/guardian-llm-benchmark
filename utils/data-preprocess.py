import pandas as pd
import json
import opencc
'''
1. 讀取 dataset/data_person_1000_zh.json
2. 將 dataset/data_person_1000_zh.json 轉換為 pandas DataFrame，目標欄位是 `naturalParagraph`
3. 使用 opencc 將 DataFrame 中的 `naturalParagraph` 欄位中的簡體中文轉換為繁體中文
4. 轉換後只保留 `naturalParagraph` 欄位並存為 csv 檔案，檔案名稱為 `dataset/data_person_1000_target.csv`
'''

converter = opencc.OpenCC('s2twp.json')

def convert_to_traditional(text):
    return converter.convert(text)

def main():
    df = pd.read_json('dataset/data_person_1000_zh.json')
    df['naturalParagraph'] = df['naturalParagraph'].apply(lambda x: convert_to_traditional(x))
    df.to_csv('dataset/data_person_1000_target.csv', index=False, columns=['naturalParagraph'])

if __name__ == '__main__':
    main()