# 【AI CUP 2024玉山人工智慧公開挑戰賽】初賽程式碼
## 執行方式
```
 python bge.py \
    --question_path ../dataset/preliminary/questions_example.json \
    --source_path ../reference \
    --output_path ../dataset/preliminary/pred_retrieve.json
```
預處理和檢索都由一個.py檔案完成，故不包含 Preprocess 及 (Retrieval) Model 兩個資料夾  
程式預設預處理完成的json會存在./insurance.json和./finance.json中。
若以上兩個檔案不存在則會用pdfplumber來處理並儲存。  
程式會下載並使用此模型: 
https://huggingface.co/BAAI/bge-reranker-v2-m3
執行GPU為一張RTX3090
