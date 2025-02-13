import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict 

class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0<=p<=1
        self.p = p

    def forward(self, x):
        if self.training and self.p>0.001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0)/(1-self.p)
        return x
class CrossAttention(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_head = conf.num_head
        self.hidden_size = conf.hidden_size
        assert self.hidden_size%self.num_head == 0,  "隐层大小必须等于头的整数倍 "

        self.Q = nn.Linear( self.hidden_size, self.hidden_size)
        self.K = nn.Linear( self.hidden_size, self.hidden_size)
        self.V = nn.Linear( self.hidden_size, self.hidden_size)
        
        self.KR = nn.Linear( self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(1, self.num_head, 1, self.hidden_size//self.num_head))
        self.v = nn.Parameter(torch.Tensor(1, self.num_head, 1, 1, self.hidden_size//self.num_head))
        
        self.attn_drop = MyDropout(conf.drop_attn)
        self.ff_final =  nn.Linear(self.hidden_size, self.hidden_size)
        self.drop_ff = MyDropout(conf.drop_ff)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        self.ff1 = nn.Linear(self.hidden_size,3*self.hidden_size)
        self.act = nn.LeakyReLU()
        self.drop_ff1 = MyDropout(conf.drop_ff1)

        self.ff2 = nn.Linear(3*self.hidden_size,self.hidden_size)
        self.drop_ff2 = MyDropout(conf.drop_ff2)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        
        
    def transform_head(self,x):
        B,L,H = x.size()
        x = x.view(B,L,self.num_head,H//self.num_head).permute(0,2,1,3)
        return x	

    def transform_hidden(self,x):
        B,head,L,H = x.size()
        x = x.permute(0,2,1,3).contiguous().view(B,L,-1)
        return x

    def forward(self,inp):
        """
            x -> B L H,
            Mask -> B L 
        """
        x,y,mask,pos_emb = inp
        q = self.transform_head(self.Q(x)) # B head C H
        k = self.transform_head(y)#self.K(y))
        v = self.transform_head(self.V(y))
       
        rk = self.KR(pos_emb) # B C L H
        B,C,L,H = rk.size()
        rk = rk.view(B,C,L,self.num_head,H//self.num_head).permute(0, 3, 1, 4, 2) # rk B C L head h -> B head C h L
        attn = ( (q + self.u) @ k.transpose(-2,-1) + ((q.unsqueeze(-2) + self.v ) @ rk).squeeze(-2) )  #/ math.sqrt(self.hidden_size//self.num_head)
       
        attn = attn.masked_fill(~mask,-1e6)
            
        attn = self.attn_drop(F.softmax(attn,-1))
        output = attn @ v
        output = self.transform_hidden( output)
        output = self.drop_ff(self.act(self.ff_final(output))) + x
        output = self.norm1(output)
        x_ = output
        output = self.ff2(self.drop_ff1(self.act(self.ff1(output))))
        output = self.drop_ff2(output) + x_
        output = self.norm2(output)
        
        return (output,y,mask,pos_emb)


class AttentionNoRelativeKey(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_head = conf.num_head
        self.hidden_size = conf.hidden_size
        assert  self.hidden_size % self.num_head == 0,  "隐层大小必须等于头的整数倍 "
        self.Q = nn.Linear( self.hidden_size, self.hidden_size)
        self.K = nn.Linear( self.hidden_size, self.hidden_size)
        self.V = nn.Linear( self.hidden_size, self.hidden_size)
        
        self.KR = nn.Linear( self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(1, self.num_head, 1, self.hidden_size//self.num_head))
        self.v = nn.Parameter(torch.Tensor(1, self.num_head, 1, 1, self.hidden_size//self.num_head))
        
        self.attn_drop = MyDropout(conf.drop_attn)
        self.ff_final =  nn.Linear(self.hidden_size, self.hidden_size)
        self.drop_ff = MyDropout(conf.drop_ff)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        self.ff1 = nn.Linear(self.hidden_size,3*self.hidden_size)
        self.act = nn.LeakyReLU()
        self.drop_ff1 = MyDropout(conf.drop_ff1)

        self.ff2 = nn.Linear(3*self.hidden_size,self.hidden_size)
        self.drop_ff2 = MyDropout(conf.drop_ff2)
        self.norm2 = nn.LayerNorm(self.hidden_size)


    def transform_head(self,x):
        B,L,H = x.size()
        x = x.view(B,L,self.num_head,H//self.num_head).permute(0,2,1,3)
        return x	

    def transform_hidden(self,x):
        B,head,L,H = x.size()
        x = x.permute(0,2,1,3).contiguous().view(B,L,-1)
        return x

    def forward(self,inp):
        """
            x -> B L H,
            Mask -> B L 
        """
        x, mask = inp
        q = self.transform_head(self.Q(x))
        k = self.transform_head(x)#self.K(x))
        v = self.transform_head(self.V(x))
       
        attn = q @ k.transpose(-2,-1) #/ math.sqrt(self.hidden_size//self.num_head)

        attn = attn.masked_fill(~mask,-1e6)
            
        attn = self.attn_drop(F.softmax(attn,-1))
        output = attn @ v
        output = self.transform_hidden( output)
        output = self.drop_ff(self.act(self.ff_final(output))) + x
        output = self.norm1(output)
        x_ = output
        output = self.ff2(self.drop_ff1(self.act(self.ff1(output))))
        output = self.drop_ff2(output) + x_
        output = self.norm2(output)
        return  (output, mask)

class AttentionWithRelativeKey(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_head = conf.num_head
        self.hidden_size = conf.hidden_size
        assert  self.hidden_size % self.num_head == 0,  "隐层大小必须等于头的整数倍 "

        self.Q = nn.Linear( self.hidden_size, self.hidden_size)
        self.K = nn.Linear( self.hidden_size, self.hidden_size)
        self.V = nn.Linear( self.hidden_size, self.hidden_size)
        
        self.KR = nn.Linear( self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(1, self.num_head, 1, self.hidden_size//self.num_head))
        self.v = nn.Parameter(torch.Tensor(1, self.num_head, 1, 1, self.hidden_size//self.num_head))
        
        self.attn_drop = MyDropout(conf.drop_attn)
        self.ff_final =  nn.Linear(self.hidden_size, self.hidden_size)
        self.drop_ff = MyDropout(conf.drop_ff)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        self.ff1 = nn.Linear(self.hidden_size,3*self.hidden_size)
        self.act = nn.LeakyReLU()
        self.drop_ff1 = MyDropout(conf.drop_ff1)

        self.ff2 = nn.Linear(3*self.hidden_size,self.hidden_size)
        self.drop_ff2 = MyDropout(conf.drop_ff2)
        self.norm2 = nn.LayerNorm(self.hidden_size)

    def transform_head(self,x):
        B,L,H = x.size()
        x = x.view(B,L,self.num_head,H//self.num_head).permute(0,2,1,3)
        return x	

    def transform_hidden(self,x):
        B,head,L,H = x.size()
        x = x.permute(0,2,1,3).contiguous().view(B,L,-1)
        return x

    def forward(self,inp):
        """
            x -> B L H,
            Mask -> B L 
        """
        x, mask, pos_emb = inp
        q = self.transform_head(self.Q(x))
        k = self.transform_head(self.K(x))
        v = self.transform_head(self.V(x))
        rk = self.KR(pos_emb) # B C L H
        B,C,L,H = rk.size()
        rk = rk.view(B,C,L,self.num_head,H//self.num_head).permute(0, 3, 1, 4, 2) # rk B C L head h -> B head C h L
        attn =  ( (q + self.u) @ k.transpose(-2,-1) + ((q.unsqueeze(-2) + self.v ) @ rk).squeeze(-2) )  #/ math.sqrt(self.hidden_size//self.num_head)

        attn = attn.masked_fill(~mask,-1e6)
            
        attn = self.attn_drop(F.softmax(attn,-1))
        output = attn @ v
        output = self.transform_hidden( output)
        output = self.drop_ff(self.act(self.ff_final(output))) + x
        output = self.norm1(output)
        x_ = output
        output = self.ff2(self.drop_ff1(self.act(self.ff1(output))))
        output = self.drop_ff2(output) + x_
        output = self.norm2(output)
        
        return (output, mask, pos_emb)
    
class CrossAttention_Block(nn.Module):
    def __init__(self,conf, layer):
        super().__init__()
        self.conf = conf
        self.attn = None
        if layer > 0:
            self.attn = nn.Sequential( OrderedDict([( "cross_attention{:d}".format(i), CrossAttention(conf)) for i in range(layer)] ) )
                                    
    def forward(self,x,y,mask,pos_emb):
        if self.attn != None:
            x = self.attn((x, y, mask, pos_emb))[0]
        return x

class AttentionNoRelativeKey_Block(nn.Module):
    def __init__(self,conf, layer):
        super().__init__()
        self.conf = conf
        self.attn = None
        if layer>0:
            self.attn = nn.Sequential( OrderedDict([( "attention_norel{:d}".format(i), AttentionNoRelativeKey(conf)) for i in range(layer)] ) )
                
    def forward(self,char, mask):
        if self.attn != None:
            char = self.attn((char, mask))[0]
        return char

class AttentionWithRelativeKey_Block(nn.Module):
    def __init__(self,conf, layer):
        super().__init__()
        self.conf = conf
        self.attn = None
        if layer>0:
            self.attn = nn.Sequential( OrderedDict([( "attnetion_rel{:d}".format(i), AttentionWithRelativeKey(conf)) for i in range(layer)] ) )
                
    def forward(self, lattice, mask, pos_emb):
        if self.attn != None:
            lattice = self.attn((lattice, mask,pos_emb))[0]
        return lattice
    
def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb

class Four_Pos_Fusion_Embedding(nn.Module):
    def __init__(self,conf,mode):
        super().__init__()
        self.hidden_size = conf.hidden_size
        self.max_seq_len=conf.max_seq_len
        self.pe = nn.Parameter(get_embedding( self.max_seq_len,self.hidden_size,rel_pos_init=1),requires_grad=False)
        self.mode = mode
        self.pe_ss = self.pe 
        self.pe_se = self.pe
        self.pe_es = self.pe
        self.pe_ee = self.pe
        if mode == "0":
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),
                                                    nn.LeakyReLU(inplace=True))
        if mode == "2":
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),
                                                    nn.LeakyReLU(inplace=True))
        if mode == "4":
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.LeakyReLU(inplace=True))
    def forward(self,pos_s,pos_e, lenth = None):
        batch = pos_s.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了
        if self.mode == "0":
            ps = lenth.unsqueeze(-1)- pos_s.unsqueeze(-2)
            pe = lenth.unsqueeze(-1)- pos_e.unsqueeze(-2)
            max_char_len = lenth.size(1)
            max_lat_len = pos_s.size(1)
            pe_ss = self.pe_ss[(ps).view(-1)+self.max_seq_len].view(size=[batch,max_char_len,max_lat_len,-1])
            pe_ee = self.pe_ee[(pe).view(-1)+self.max_seq_len].view(size=[batch,max_char_len,max_lat_len,-1])

            pe_0 = torch.cat([pe_ss, pe_ee],dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_0)
            return rel_pos_embedding

        else:  
            pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
            pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
            pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
            pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)
            max_seq_len = pos_s.size(1)

            if self.mode ==  "4":
                pe_ss = self.pe_ss[(pos_ss).view(-1) + self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
                pe_se = self.pe_se[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
                pe_es = self.pe_es[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
                pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
                pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
                rel_pos_embedding = self.pos_fusion_forward(pe_4)
                return rel_pos_embedding

            if self.mode == "2":
        
                pe_ss = self.pe_ss[(pos_ss).view(-1)+self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
                #pe_se = self.pe_se[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
                #pe_es = self.pe_es[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
                pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
                pe_2 = torch.cat([pe_ss,pe_ee],dim=-1)

                rel_pos_embedding = self.pos_fusion_forward(pe_2)
                return rel_pos_embedding
