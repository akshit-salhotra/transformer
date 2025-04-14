from model.transformer import Transformer
from dataloaders.dataloader import En_De
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm 
import os 

batch=None
shuffle=None
timeout=None
workers=None
n_dim=None
vocab_size=None
lr=None
epochs=None
val_freq=None
save_freq=None
save_dir=None

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=Transformer(n_dim,vocab_size)
train_dataset=En_De()
val_dataset=En_De()
train_loader=DataLoader(train_dataset,batch,shuffle,timeout=timeout,num_workers=workers)
val_loader=DataLoader(val_dataset,batch,shuffle,timeout=timeout,num_workers=workers)
criteron=nn.CrossEntropyLoss()
optimiser=optim.Adam(model.parameters(),lr)
pbar=tqdm(range(epochs))

for i in pbar:
    model.train()
    epoch_loss=0
    for data,label in train_loader:
        data=[d.to(device) for d in data]
        label=label.to(device)
        
        logits=model(*data)
        loss=criteron(logits,label)
        
        loss.backwards()
        optimiser.step()
        optimiser.zero_grad()
        epoch_loss+=loss
    epoch_loss/=len(train_dataset)
    pbar.set_postfix({"average loss":epoch_loss})
    
    if i%val_freq==0 or i+1==epochs:
        model.eval()
        with torch.no_grad():
            for data,label in val_loader:
                data=[d.to(device) for d in data]
                label=label.to(device)
                #to be completed 
                logits
    
    if i%save_freq+1==0 or i+1==epochs:
        torch.save(f'{save_dir+os.sep}epoch{i}')
        
    
        