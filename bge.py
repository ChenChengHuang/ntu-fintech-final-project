import os
import json
import argparse
from tqdm import tqdm
import pdfplumber
import torch
import numpy as np
from FlagEmbedding import FlagReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 確定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path, cache_file):
    # 檢查是否存在已保存的數據文件
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            corpus_dict = json.load(f)
            corpus_dict = {int(key): value for key, value in corpus_dict.items()}
        print(f"Loaded cached data from {cache_file}")
    else:
        # 如果沒有已保存的數據，讀取PDF文件
        masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
        corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}
        
        # 保存到本地文件中
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_dict, f, ensure_ascii=False, indent=4)
        print(f"Saved parsed data to {cache_file}")
    
    return corpus_dict

# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件
    return pdf_text  # 返回萃取出的文本


# 使用 reranker 的 compute_score 方法進行重排序
def retrieve(qs, source, corpus_dict, reranker):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False,
    )

    chunked_corpus = []
    chunk_to_file_mapping = []  # 新增一個mapping，用來記錄每個 chunk 來自於哪個文件

    for file in source:
        doc = corpus_dict[int(file)]
        # 將文檔分割為 chunks
        chunks = text_splitter.split_text(doc)
        chunked_corpus.extend(chunks)
        chunk_to_file_mapping.extend([file] * len(chunks))  # 每個 chunk 對應的文件名
    
    # 將初步檢索的文檔對和查詢對組合在一起
    query_document_pairs = [[qs, doc] for doc in chunked_corpus]
    
    # 使用 reranker 的 compute_score 方法計算查詢和文檔的分數
    scores = reranker.compute_score(query_document_pairs)

    # 找到最高分的索引
    best_score_index = np.argmax(scores)

    # 使用 chunk_to_file_mapping 找到對應的文件名
    best_match_filename = chunk_to_file_mapping[best_score_index]
    
    return best_match_filename


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    args = parser.parse_args()  # 解析參數
    answer_dict = {"answers": []}  # 初始化字典

    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案
    
    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance, "i.json")
    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance, "f.json")

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 進行檢索
            retrieved = retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance, reranker)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        elif q_dict['category'] == 'insurance':
            retrieved = retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance, reranker)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq, reranker)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
