## 執行方式
```
 python gemma.py \
    --question_path ../dataset/preliminary/questions_example.json \
    --source_path ../reference \
    --output_path ../dataset/preliminary/pred_retrieve.json
```
程式預設預處理完成的json會存在./insurance.json和./finance.json中。  
若以上兩個檔案不存在則會用pdfplumber來處理。  
程式會下載並使用此模型:  
https://huggingface.co/BAAI/bge-reranker-v2-gemma  
執行GPU為一張RTX3090
