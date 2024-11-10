import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self,features,num_head=8,cross=False,mask=False):
        super(MHA,self).__init__()
        self.cross=cross
        self.num_head=num_head
        self.mask=mask
        
        assert features%num_head==0,"number of attention heads must be divisible by d_model"
        
        self.query=nn.Linear(features,features)
        self.key=nn.Linear(features,features)
        self.value=nn.Linear(features,features)

    def forward(self,x,x_encoder=None):
        if self.cross:
            q=self.query(x)
            k=self.key(x_encoder)
            v=self.value(x_encoder)              
        else:
            q=self.query(x)
            k=self.key(x)
            v=self.value(x)
        batch,seq,d_model=q.shape
        
        #head are divided by reshaping
        q=q.view(batch,seq,self.num_head,-1)
        k=k.view(batch,seq,self.num_head,-1)
        v=v.view(batch,seq,self.num_head,-1)
        
        q=torch.permute(q,(0,2,1,3))
        k=torch.permute(k,(0,2,1,3))
        v=torch.permute(v,(0,2,1,3))
        alignment=torch.matmul(q,torch.transpose(k,2,3))/(math.sqrt(d_model/self.num_head))
        if self.mask:
            mask=torch.zeros_like(alignment[0,0,:,:])
            i,j=torch.meshgrid(torch.arange(seq),torch.arange(seq))
            mask[i>j]=1
            alignment=torch.masked_fill(alignment,mask==1,1e-9)
            
        new_x=(nn.functional.softmax(alignment,dim=-1)@v)
        out=torch.permute(new_x,(0,2,1,3)).contiguous().view(batch,seq,-1)
        out=out+x
        out=nn.LayerNorm(x.shape)(out)           

        return out

class MLP(nn.Module):
    def __init__(self,features,mlp_factor=4):
        super(MLP,self).__init__()
        
        if not isinstance(features, int) or not isinstance(mlp_factor, int):
            raise ValueError("Both 'features' and 'mlp_factor' should be integers.")
        
        self.mlp1=nn.Linear(features,mlp_factor*features)
        self.mlp2=nn.Linear(mlp_factor*features,features)
        
    def forward(self,x):
        batch,seq,d_model=x.shape
        out_=x.view(-1,d_model)
        out1=self.mlp1(out_)
        out2=self.mlp2(out1)
        out2=out2.view(batch,seq,d_model)
        final_out=nn.LayerNorm(x.shape)(out2+x)
        return final_out
    
class Decoder(nn.Module):
    def __init__(self,features,mlp_factor=4,num_heads=8):
        super(Decoder,self).__init__()
        self.self_attention=MHA(512,mask=True,num_head=num_heads)
        self.cross_attention=MHA(512,cross=True,num_head=num_heads)
        self.mlp=MLP(features,mlp_factor=mlp_factor)
         
    def forward(self,x,x_encoder):
        self_attend_out=self.self_attention(x)
        cross_attend_out=self.cross_attention(self_attend_out,x_encoder)
        out=self.mlp(cross_attend_out)
        return out
    
class Encoder(nn.Module):
    def __init__(self,features,mlp_factor=4,num_heads=4):
        super(Encoder,self).__init__()
        self.attention=MHA(features,num_heads)
        self.MLP=MLP(features,mlp_factor)
    def forward(self,x):
        x=self.attention(x)
        out=self.MLP(x)
        return out
    
if __name__=="__main__":
    from torchsummary import summary
    model=MHA(512,mask=True)
    input=torch.randn((5,15,512))
    model(input)
    summary(model,(15,512),5)