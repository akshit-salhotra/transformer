import fireducks as pd #fireducks has more optimised implementation
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm

save_plots_dir=""
parquet_list=[]

os.makedirs(save_plots_dir,exist_ok=True)
dfs=[pd.read_parquet(p) for p in parquet_list]
merged_df=pd.concat(dfs,ignore_index=True)

max_seq_en=0
max_seq_de=0
largest_en=None
largest_de=None
seq_en={}
seq_de={}
for idx,row in tqdm(merged_df.iterrows()):
    l_en=len(row['translation.en'])
    l_de=len(row['translation.de'])
    
    seq_en[l_en]=seq_en.get(l_en,0)
    seq_de[l_de]=seq_de.get(l_de,0)
    
    if l_en>max_seq_en:
        max_seq_en=len(row['translation.en'])
        largest_en=row['translation.en']
        
    if l_de>max_seq_de:
        max_seq_de=len(row['translation.de'])
        largest_de=row['translation.de']
        
    
plt.subplot(1,2,1)
plt.scatter(seq_en.keys(),seq_en.values())
plt.title("English")
plt.xlabel("number of characters")
plt.ylabel("frequency")

plt.subplot(1,2,2)
plt.scatter(seq_de.keys(),seq_de.values())
plt.title("German")
plt.xlabel("number of characters")
plt.ylabel("frequency")

plt.savefig(f"{save_plots_dir+os.sep}frequency_plots.png")

print("the largest english sequence is :",)
print("the largest german sequence is:",)