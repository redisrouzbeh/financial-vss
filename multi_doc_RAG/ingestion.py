import os
import warnings
import json
import json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.partition.pdf import partition_pdf
import nltk


warnings.filterwarnings("ignore")
dir_path = os.getcwd()
parent_directory = os.path.dirname(dir_path)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ROOT_DIR"] = parent_directory
print(dir_path)
print(parent_directory)

data_path = f"{parent_directory}/resources/filings"
dirs = os.listdir(data_path)


def get_sec_data():
    sec_data = {}
    for dir in dirs:
        dir_path = os.path.join(data_path, dir)
        ticker = str(dir).split("/")[len(dir.split("/")) - 1]
        sec_data[ticker] = {
            "10K_files": [],
            "metadata_file": [],
            "transcript_files": [],
        }
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            full_path = os.path.join(dir_path, file)
            if '10K' in str(file):
                sec_data[ticker]["10K_files"] = sec_data[ticker]["10K_files"] + [full_path]
            elif 'metadata' in str(file):
                sec_data[ticker]["metadata_file"] = sec_data[ticker]["metadata_file"] + [full_path]
            elif 'transcript' in str(file):
                sec_data[ticker]["transcript_files"] = sec_data[ticker]["transcript_files"] + [full_path]

    print(f" ✅ Loaded doc info for  {len(sec_data.keys())} tickers...")
    return sec_data


def load_json_metadata(path):
    obj = {}
    with open(path, 'r') as json_file:
        meta_dict = json.load(json_file)
        obj['ticker'] = meta_dict['Ticker']
        obj['company_name'] = meta_dict['Name']
        obj['sector'] = meta_dict['Sector']
        obj['asset_class'] = meta_dict['Asset Class']
        obj['market_value'] = float(str(meta_dict['Market Value']).replace(",", ""))
        obj['weight'] = meta_dict['Weight (%)']
        obj['notional_value'] = float(str(meta_dict['Notional Value']).replace(",", ""))
        obj['shares'] = float(str(meta_dict['Shares']).replace(",", ""))
        obj['location'] = meta_dict['Location']
        obj['price'] = float(str(meta_dict['Price']).replace(",", ""))
        obj['exchange'] = meta_dict['Exchange']
        obj['currency'] = meta_dict['Currency']
        obj['fx_rate'] = meta_dict['FX Rate']
        obj['market_currency'] = meta_dict['Market Currency']
        obj['accrual_date'] = meta_dict['Accrual Date']

    return obj

def redis_bulk_upload(data_dict, index, embeddings):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    for ticker in list(data_dict.keys()):
        if len(data_dict[ticker]["metadata_file"]) > 0:
            shared_metadata = load_json_metadata(data_dict[ticker]["metadata_file"][0])

        for filing_file in data_dict[ticker]["10K_files"]:

            if not os.path.isfile(filing_file):
                continue

            filing_file_filename = str(filing_file).split("/")[len(str(filing_file).split("/")) - 1]
            chunks = []
            try:
                loader = UnstructuredFileLoader(
                    filing_file, mode="single", strategy="fast"
                )
                chunks = loader.load_and_split(text_splitter)
            except Exception as e:
                print(f"Error chunking {filing_file} skipping")
                continue
            chunk_objs_to_load = []
            for i, chunk in enumerate(chunks):
                content = str(chunk.page_content)
                source_doc_full_path = str(chunk.metadata['source'])
                source_doc = str(source_doc_full_path).split("/")[len(str(source_doc_full_path).split("/")) - 1]
                assert source_doc == filing_file_filename
                obj_to_load = shared_metadata.copy()
                obj_to_load['chunk_id'] = f"{filing_file_filename}-{i}"
                obj_to_load['source_doc'] = f"{source_doc}"
                obj_to_load['content'] = content
                emb = embeddings.embed_query(content)
                obj_to_load['text_embedding'] = np.array(emb).astype(np.float32).tobytes()
                chunk_objs_to_load.append(obj_to_load)
            keys = index.load(chunk_objs_to_load, id_field="chunk_id")
            print(f"✅ Loaded {len(keys)} chunks for ticker={ticker} from {source_doc}")

