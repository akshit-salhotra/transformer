from .tokenizer import bpe_df_corpus
import pandas as pd
import os 
import json
import regex as re

data_dir="data/train"
start_idx=256
vocab_size=None
pattern= re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}{1,4}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",re.IGNORECASE)#for forced splits
save_dir="vocab"



os.makedirs(save_dir,exist_ok=True)
parquet_list=sorted(os.listdir(data_dir))
dfs=[pd.read_parquet(p) for p in parquet_list]
merged_df=pd.concat(dfs,ignore_index=True)

merged_dict=bpe_df_corpus(merged_df['translation'],['en','de'],pattern,vocab_size-start_idx,start_idx)

with open(f'{save_dir+os.sep}merges.json','w') as file:
    json.dump(merged_dict,file)