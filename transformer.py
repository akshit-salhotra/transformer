import torch.nn as nn
from helper import Encoder,Decoder
class Transformer(nn.Module):
    def __init__(self,features,vocab_size,mlp_factor=4,num_heads=8,num_blocks=6):
        super(Transformer,self).__init__()
        self.num_blocks=num_blocks
        self.encoder=Encoder(features,mlp_factor=mlp_factor,num_heads=num_heads)
        self.decoder=Decoder(features,mlp_factor=mlp_factor,num_heads=num_heads)
        self.project=nn.Linear(features,vocab_size)
        
    def forward(self,x_encoder,x_decoder):
        for i in range(self.num_blocks):
            x_encoder=self.encoder(x_encoder)
        for i in range(self.num_blocks):
            x_decoder=self.decoder(x_decoder,x_encoder)
        out=self.project(x_decoder)
        return out
    
if __name__=="__main__":
    from torchsummary import summary
    model=Transformer(512,50000)
    summary(model,[(10,512),(10,512)],2)
    
        
            
        

            
        
        
        