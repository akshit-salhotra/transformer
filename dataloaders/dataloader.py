from torch.utils.data import Dataset
import pandas as pd
from itertools import accumulate
from tokenizer.tokenizer import encode
import json


def collate_train(batch):
    en,de=zip(*batch)
    
    

class En_De(Dataset):
    """"
    Dataloader for english to german translation
    """
    def __init__(self,parquet_list:list,merges_json:str):
        super().__init__() 
        self.dfs=[pd.read_parquet(parquet) for parquet in parquet_list]
        self.final_df=pd.concat(self.dfs,ignore_index=True)
        # self.lengths=[len(df) for df in self.dfs]
        # self.lengths=list(accumulate(self.lengths))
        with open(merges_json,'r') as file:
            self.merges=json.load(file)

    def __getitem__(self,index:int):
        
        # df=None
        # for i in range(len(self.lengths)):
        #     if index<i:
        #         df=self.dfs[i]
        #         if i>0:
        #             index=index-self.lengths[i-1]
        
        en=encode(self.final_df.iloc[index]['translation.en'],self.merges)
        de=encode(self.final_df.iloc[index]['translation.de'],self.merges)
        
        return en,de

    def __len__(self):
        return self.lengths[-1]
