import nest_asyncio
import sys
nest_asyncio.apply()

from llama_parse import LlamaParse 
from llama_index.core import SimpleDirectoryReader
import json
import os
import re


def clean_text(text):
    # Preprocess document by removing redundant text

    text = re.sub(r"http\S+", "",text) # Removing URLs 

    punctuations = '@#!?+&*[],:/();$=><|{}^' + "'`" + '_' + '\n' + ' ' + '「」【】：※' 
    for p in punctuations:
        text = text.replace(p,'') # Removing punctuations

    number_value = re.compile(r'\d{5,}')
    text = number_value.sub(r'', text) # Removing number values
    page_number = re.compile(r'~\d+~|-\d+-')
    text = page_number.sub(r'', text) # Removing page numbers
    return text
def main(api_key, doc_dir, output_path):
    # Parse documents through LlamaParse api and save json file to output_path
    parser = LlamaParse(
        api_key=api_key,  
        result_type="text", 
        verbose=True,
        language="ch_tra",
    )

    document_dict = {}

    for file in os.listdir(doc_dir):
        idx = file.replace('.pdf', '')
        if idx not in document_dict:
            print(idx)
            file_path = os.path.join("./reference/finance", file)
            documents = parser.load_data(file_path)
            text = ""
            for document in documents:
                if text == "":
                    text = document.text
                else:
                    text += document.text
            document_dict[idx] = text
    
    document_dict = dict(sorted(document_dict.items(), key = lambda x: int(x[0])))

    document_dict_clean = {k:clean_text(v) for k, v in document_dict.items()}

    with open(output_path, "w", encoding='utf8') as f:
        json.dump(document_dict_clean, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    api_key = sys.argv[1]
    doc_dir = sys.argv[2]
    output_path = sys.argv[3]
    main(api_key=api_key, doc_dir=doc_dir, output_path=output_path)
