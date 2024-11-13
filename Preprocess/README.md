## 執行方式
```
python preprocess.py Llamaparse_api_key\
/path/to/document/directory \
/path/to/output/json
```
Llamaparse_api_key: Llamaparse api服務的金鑰  
/path/to/document/directory: pdf資料的目錄  
/path/to/output/json: 希望輸出的json檔  
## 說明
我們使用Llamaparse來作為OCR工具。  
在完成OCR後會把結果中的一些標點符號、url、數字等去除。
