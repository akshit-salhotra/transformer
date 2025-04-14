import torch
import torch.nn as nn
import math
import transformers

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
        
        self.project=nn.Linear(features,features)

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
            alignment=torch.masked_fill(alignment,mask==1,float('-inf'))
            
        new_x=(nn.functional.softmax(alignment,dim=-1)@v)
        out=torch.permute(new_x,(0,2,1,3)).contiguous().view(batch,seq,-1)
        out=self.project(out)
        out=out+x
        out=nn.LayerNorm(x.shape[-1])(out)           

        return out

class MLP(nn.Module):
    
    def __init__(self,features,mlp_factor=4):
        super(MLP,self).__init__()
        
        if not isinstance(features, int) or not isinstance(mlp_factor, int):
            raise ValueError("Both 'features' and 'mlp_factor' should be integers.")
        
        self.mlp1=nn.Linear(features,mlp_factor*features)
        self.mlp2=nn.Linear(mlp_factor*features,features)
        self.ReLU=nn.ReLU()
        
    def forward(self,x):
        batch,seq,d_model=x.shape
        out_=x.view(-1,d_model)
        out1=self.mlp1(out_)
        out1=self.ReLU(out1)
        out2=self.mlp2(out1)
        out2=out2.view(batch,seq,d_model)
        final_out=nn.LayerNorm(d_model)(out2+x)
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
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model=transformers.BertModel.from_pretrained('bert-base-uncased')
    def forward(self,x):
        x=self.attention(x)
        out=self.MLP(x)
        return out
    def encode_input(self,x):
        '''
        returns the embedding vector and inputs of sequence after appending [CLS] and [SEP] token and the start and end of the sequence.
        inputs:(dict) containing input_ids,attention_mask,token_type_ids
        outputs.last_hidden_state: a tensor of shape (batch,n_seq,latent_dim)
        '''
        inputs=self.tokenizer(x,return_tensors='pt',padding=True)
        print(inputs)
        # input=torch.tensor([token_id])
        # print(input)
        with torch.no_grad():
            outputs=self.model(inputs['input_ids'],inputs['attention_mask'])
        #encode() maps the words to unique ids ,note that the tokenizer appends [CLS] token in the beginning and [SEP] token at the end        
        return outputs.last_hidden_state,inputs
    
    def encode_with_pos(self,x):
        '''
        adds positional embeddings to the embeddings returned by encode_input()
        '''
        out_encode,token_id=self.encode_input(x)
        # print(out_encode.shape)
        pos_encode=torch.zeros(out_encode.shape[1],out_encode.shape[2])
        # i,j=torch.meshgrid(torch.arange(out_encode.shape[1]),torch.arange(out_encode.shape[2]))
        j=torch.arange(out_encode.shape[2])
        # print(torch.arange(out_encode.shape[1]).view(-1,1).shape,torch.arange(start=0,end=out_encode.shape[2],step=2).view(1,-1).shape)
        pos_encode[:,j%2==0]=torch.sin(torch.arange(out_encode.shape[1]).view(-1,1)/(10000)**(2*torch.arange(start=0,end=out_encode.shape[2],step=2).view(1,-1)/out_encode.shape[0]))
        pos_encode[:,j%2!=0]=torch.sin(torch.arange(out_encode.shape[1]).view(-1,1)/(10000)**(2*torch.arange(start=1,end=out_encode.shape[2],step=2).view(1,-1)/out_encode.shape[0]))
        return pos_encode+out_encode,token_id
    

if __name__=="__main__":
    from torchsummary import summary
    # model=MHA(512,mask=True)
    # input=torch.randn((5,15,512))
    # model(input)
    # summary(model,(15,512),5)
    model=Encoder(512)
    print(model.encode_with_pos(['Hi, how are you today?','hi'])[0].shape)
    # print(model.encode_with_pos('Hi, how are you today?')[0].shape,model.encode_with_pos('Hi, how are you today?')[0])
    