import torch.nn as nn
from helper import Encoder,Decoder
class Transformer(nn.Module):
    def __init__(self,features,vocab_size,mlp_factor=4,num_heads=8,num_blocks=6):
        super(Transformer,self).__init__()
        self.num_blocks=num_blocks
        self.encoder=nn.ModuleList([Encoder(features,mlp_factor=mlp_factor,num_heads=num_heads) for i in range(num_blocks)])
        self.decoder=nn.ModuleList([Decoder(features,mlp_factor=mlp_factor,num_heads=num_heads) for i in range(num_blocks)])
        self.project=nn.Linear(features,vocab_size)
        
    def forward(self,x_encoder,x_decoder):
        for encoder in self.encoder:
            x_encoder=encoder(x_encoder)
        for decoder in self.decoder:
            x_decoder=decoder(x_decoder,x_encoder)
        out=self.project(x_decoder)
        return out
if __name__=="__main__":
    from torchsummary import summary
    model=Transformer(512,50000)
    summary(model,[(10,512),(10,512)],2)
    
        
            
        

            
        
        
        