import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules import *
import copy
from crf import CRF


class Block(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.conf = conf
        self.char_block =  AttentionNoRelativeKey_Block(conf,conf.char_layer)#AttentionNoRelativeKey_Block(conf)
        #self.lattice_block = AttentionWithRelativeKey_Block(conf,conf.lattice_layer)
        self.char_lattice = CrossAttention_Block(conf,conf.char_lattice_layer)
        #self.char_lattice1 = CrossAttention_Block(conf,conf.char_lattice_layer)
        self.char_block1 =  AttentionNoRelativeKey_Block(conf,conf.char_layer)
        #self.lattice_char = CrossAttention_Block(conf,conf.lattice_char_layer)
        #self.ff = nn.Linear(2*conf.hidden_size, conf.hidden_size)
        # self.param = nn.Parameter(torch.Tensor(1,1,conf.hidden_size))
        self.norm = nn.LayerNorm(conf.hidden_size)

        # self.ff1 = nn.Linear(conf.hidden_size,3*conf.hidden_size)
        # self.act = nn.LeakyReLU()
        # self.drop_ff1 = MyDropout(conf.drop_ff1)

        # self.ff2 = nn.Linear(3*conf.hidden_size,conf.hidden_size)
        # self.drop_ff2 = MyDropout(conf.drop_ff2)
        # self.norm2 = nn.LayerNorm(conf.hidden_size)
        

        
    def forward(self,char,lattice, char_mask, lattice_mask, lattice_pos_emb, char_lattice_pos_emb, lattice_char_pos_emb):
#         lattice_out = self.lattice_block(lattice, lattice_mask, lattice_pos_emb)
#         lattice_char_out = self.lattice_char(lattice_out, char, char_mask,  lattice_char_pos_emb)
#         char_lattice_out = self.char_lattice(char, lattice_char_out, lattice_mask, char_lattice_pos_emb)
#         char_out = self.char_block(char_lattice_out,char_mask)
#         char_lattice_out = self.char_lattice(char, lattice, lattice_mask, char_lattice_pos_emb)
        char = self.char_block(char,char_mask)
        char = self.char_lattice(char, lattice, lattice_mask, char_lattice_pos_emb)
        char = self.char_block1(char,char_mask)
        # out =  F.sigmoid(self.param)*char_out + (1-F.sigmoid(self.param))*char_lattice_out #self.ff(torch.cat([char_out , char_lattice_out],-1))
        # out =  self.norm(out)
        # output = self.ff2(self.drop_ff1(self.act(self.ff1(out))))
        # output = self.drop_ff2(output) + out
        # output = self.norm2(output)
        return char
    
class ModelV1(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.block = Block(conf)

    def forward(self,char,lattice, char_mask, lattice_mask,  lattice_pos_emb, char_lattice_pos_emb):
        char_mask = char_mask.unsqueeze(1).unsqueeze(1)
        lattice_mask = lattice_mask.unsqueeze(1).unsqueeze(1)
        lattice_char_pos_emb = char_lattice_pos_emb.transpose(-3,-2)
        char_lattice_out = self.block(char, lattice, char_mask, lattice_mask,  lattice_pos_emb,char_lattice_pos_emb, lattice_char_pos_emb)
        return char_lattice_out

class BaseModel(nn.Module):
    def __init__(self,conf,lattice_embedding,bigram_embedding,label_size):
        super(BaseModel, self).__init__()
        self.conf = conf
        self.char_proj = nn.Linear(2*conf.emb_size, conf.hidden_size)
        self.lattice_proj = nn.Linear(conf.emb_size, conf.hidden_size)
        
        if conf.abs_pos:
            self.abs_pos = nn.Parameter(get_embedding( conf.max_seq_len,conf.hidden_size,rel_pos_init=0),requires_grad=False)
            #self.abs_proj = nn.Sequential(nn.Linear(conf.hidden_size, conf.hidden_size), nn.LeakyReLU(inplace=True))
            
        if conf.rel_pos:
            self.pos_emb_0 =  Four_Pos_Fusion_Embedding(self.conf,"0")
            self.pos_emb_4 = Four_Pos_Fusion_Embedding(self.conf,"4")
            
        self.drop_char_emb = MyDropout(conf.drop_emb)
        self.drop_lat_emb = MyDropout(conf.drop_emb)

        self.backbone = ModelV1(conf)
        self.pred = nn.Linear(conf.hidden_size,label_size)
        
        self.crf = CRF(label_size)
        self.crf.trans_m = nn.Parameter(torch.zeros([label_size,label_size]),requires_grad=True)
        
        self.lattice_embedding = nn.Embedding(lattice_embedding.size()[0],lattice_embedding.size()[1], padding_idx=0, _weight = lattice_embedding)
        self.bigram_embedding = nn.Embedding(bigram_embedding.size()[0],bigram_embedding.size()[1],  padding_idx=0, _weight = bigram_embedding)
        
        self.drop_output = MyDropout(conf.drop_output)
        
    def get_mask(self,seq_len, max_len=None):
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
        return mask
    
    def forward(self,x, is_train=True):
        char, lattice, bigram, ps,pe, label, char_len, lattice_len = x
        max_char_len = char.size(1)
        max_lattice_len = lattice.size(1)
        char_mask = self.get_mask(char_len).bool()
        lattice_mask = self.get_mask(lattice_len).bool()

        if is_train:
            # 1 is unk , drop words and char randomly
            randn_mask_char = (torch.rand_like(char, dtype=torch.float) <= 0.01) & char_mask
            randn_mask_bigram = (torch.rand_like(bigram , dtype=torch.float) <= 0.01) & char_mask
            randn_mask_lattice =  (torch.rand_like(lattice, dtype=torch.float) <= 0.01) & lattice_mask
            char.masked_fill_(randn_mask_char, 1)
            bigram.masked_fill_(randn_mask_bigram, 1)
            lattice.masked_fill_(randn_mask_lattice, 1)
            
        bigram_emb = self.bigram_embedding(bigram)
        char_emb = self.lattice_embedding(char)
        ch_bi_emb = torch.cat([char_emb,bigram_emb],-1)
        
        lattice_emb = self.lattice_embedding(lattice)
        
        ch_bi_emb = self.drop_char_emb(ch_bi_emb)
        lattice_emb = self.drop_lat_emb(lattice_emb)
        
        ch_bi_emb = self.char_proj(ch_bi_emb)
        lattice_emb = self.lattice_proj(lattice_emb)
        
        if self.conf.abs_pos:
            ch_bi_emb = ch_bi_emb + self.abs_pos[torch.arange(max_char_len, device= char.device)].unsqueeze(0)

        if self.conf.rel_pos:
            lattice_pos_emb = self.pos_emb_4(ps,pe)
            char_lattice_pos_emb = self.pos_emb_0(ps,pe,torch.arange(max_char_len, device= char.device).unsqueeze(0))

        encoded = self.backbone(ch_bi_emb, lattice_emb, char_mask, lattice_mask, lattice_pos_emb, char_lattice_pos_emb)
        encoded = self.drop_output(encoded)
        pred = self.pred(encoded)
        mask = self.get_mask(char_len)
        if is_train:
            crf_loss = self.crf(pred, label, mask).mean(dim=0)
            return {'loss': crf_loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}
            return result
        

class FlatLattice(nn.Module):
    def __init__(self,conf,lattice_embedding,bigram_embedding,label_size):
        super(FlatLattice, self).__init__()
        self.conf = conf
        self.char_proj =  nn.Linear(2*conf.emb_size, conf.hidden_size)
        self.lattice_proj = nn.Linear(conf.emb_size, conf.hidden_size)
        
        self.drop_char_emb = MyDropout(conf.drop_emb)
        self.drop_lat_emb = MyDropout(conf.drop_emb)        
        
        self.backbone = AttentionWithRelativeKey(conf)
        self.output = nn.Linear( conf.hidden_size, label_size)
        
        self.crf = CRF(label_size)
        self.crf.trans_m = nn.Parameter(torch.zeros([label_size,label_size]),requires_grad=True)
        
        self.lattice_embedding = nn.Embedding(lattice_embedding.size()[0],lattice_embedding.size()[1], _weight = lattice_embedding)
        self.bigram_embedding = nn.Embedding(bigram_embedding.size()[0],bigram_embedding.size()[1], _weight = bigram_embedding)
        
        self.pos_emb = Four_Pos_Fusion_Embedding(self.conf,"2")
        self.drop_output = MyDropout(conf.drop_output)

    
    def get_mask(self,seq_len, max_len=None):
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
        return mask

    def forward(self,x, is_train):
        
        char_and_lattice, lattice, bigram, ps,pe, label, char_len, lattice_len = x
        batch = char_and_lattice.size(0)
        max_len = char_and_lattice.size(1)
        max_seq_len = torch.max(char_len).item()

        char_mask = self.get_mask(char_len,max_len).bool()
        bigram_mask = self.get_mask(char_len).bool()
        lattice_mask = self.get_mask(char_len+lattice_len).bool() ^ char_mask.bool()

        if is_train:
            # 1 is unk , drop words and char randomly
            randn_mask_char = (torch.rand_like(char_and_lattice, dtype=torch.float)<= 0.015) & (char_mask | lattice_mask) 
            randn_mask_bigram = (torch.rand_like(bigram, dtype=torch.float) <= 0.015) & bigram_mask
            char_and_lattice.masked_fill_(randn_mask_char, 1)
            bigram.masked_fill_(randn_mask_bigram, 1)
        

        
        raw_emb = self.lattice_embedding(char_and_lattice) 
        bigram_emb = self.bigram_embedding(bigram)
        bigram_emb = torch.cat([bigram_emb, torch.zeros(size=[batch,max_len-max_seq_len,bigram_emb.size(-1)]).to(bigram_emb)],dim=1)
        concate = torch.cat([raw_emb, bigram_emb],-1)
        
        concate = self.drop_char_emb(concate)
        raw_emb = self.drop_lat_emb(raw_emb)
        
        ch_bi_emb = self.char_proj(concate)
        ch_bi_emb.masked_fill_(~(char_mask.unsqueeze(-1)), 0)
        
        l_emb = self.lattice_proj(raw_emb)
        l_emb.masked_fill_(~(lattice_mask.unsqueeze(-1)), 0)
        
        # ch_bi_emb = self.drop_char_emb(ch_bi_emb)
        # l_emb = self.drop_lat_emb(l_emb)
        
        emb = ch_bi_emb + l_emb
        pos_emb = self.pos_emb(ps, pe)
        
        encoded = self.backbone((emb, self.get_mask(char_len+lattice_len).bool().unsqueeze(1).unsqueeze(1),pos_emb))[0]
        encoded = self.drop_output(encoded)
        pred = self.output(encoded[:,:max_seq_len,:])
        seq_mask = self.get_mask(char_len).bool()
        if is_train:
            crf_loss = self.crf(pred, label, seq_mask).mean(dim=0)
            return {'loss': crf_loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, seq_mask)
            result = {'pred': pred}
            return result

class BiLSTM(nn.Module):
    def __init__(self,conf,lattice_embedding,bigram_embedding,crf):
        super(BiLSTM, self).__init__()
        self.conf = conf
        self.char_proj = nn.Sequential(  nn.Linear(conf.emb_size, conf.hidden_size)
                                      )
        self.drop_emb = MyDropout(conf.drop_ff)

        self.backbone = nn.LSTM(input_size = conf.hidden_size, hidden_size = conf.hidden_size, num_layers = 1, bidirectional= True)
        self.output = nn.Linear( 2*conf.hidden_size, crf.num_tags)
        self.crf = crf
        self.weight_norm(lattice_embedding,bigram_embedding)

    def weight_norm(self,lattice_embedding,bigram_embedding):
        # with torch.no_grad():        
        #     lattice_embedding = lattice_embedding / (torch.norm(lattice_embedding, dim=1, keepdim=True) + 1e-6)
        #     bigram_embedding  = bigram_embedding / (torch.norm(bigram_embedding, dim=1, keepdim=True) + 1e-6)
        self.lattice_embedding = lattice_embedding#nn.Embedding(lattice_embedding.size()[0],lattice_embedding.size()[1], _weight = lattice_embedding)
        self.bigram_embedding = bigram_embedding#nn.Embedding(bigram_embedding.size()[0],bigram_embedding.size()[1], _weight = bigram_embedding)
    def _genmask(self,lenth):
        with torch.no_grad():
            max_len = torch.max(lenth)
            mask = torch.zeros([lenth.size()[0],max_len], device = lenth.device)
            for b in range(lenth.size()[0]):
                mask[b,0:lenth[b]] = 1
        return mask.bool(), max_len

    def forward(self,x, is_train):
        char, lattice, bigram, ps,pe, label, char_len, lattice_len = x
        # print(x)
        # exit()
        char_mask, max_len = self._genmask(char_len)

        char_emb = self.lattice_embedding(char)
        char_emb = self.char_proj(char_emb).masked_fill(~char_mask.unsqueeze(-1),0)
        char_emb = self.drop_emb(char_emb)
        char_emb, (_,_ ) = self.backbone(char_emb.transpose(0,1))
        pred = self.output(char_emb.transpose(0,1))
        if is_train:
            crf_loss = self.crf(pred, label, char_mask).mean(dim=0)
            return {'loss': crf_loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, char_mask)
            result = {'pred': pred}
            return result

class Transformer(nn.Module):
    def __init__(self,conf,lattice_embedding,bigram_embedding,crf):
        super(Transformer, self).__init__()
        self.conf = conf
        self.char_proj = nn.Sequential(  nn.Linear(conf.emb_size, conf.hidden_size)
                                      )
        self.drop_emb = MyDropout(conf.drop_ff)

        self.backbone = AttentionNoRelativeKey_Block(conf,conf.Transformer)
        self.output = nn.Linear( conf.hidden_size, crf.num_tags)
        self.crf = crf
        self.weight_norm(lattice_embedding,bigram_embedding)
        self.pos_init()
        
    def weight_norm(self,lattice_embedding,bigram_embedding):
        with torch.no_grad():        
            lattice_embedding = lattice_embedding / (torch.norm(lattice_embedding, dim=1, keepdim=True) + 1e-6)
            bigram_embedding  = bigram_embedding / (torch.norm(bigram_embedding, dim=1, keepdim=True) + 1e-6)
        self.lattice_embedding = nn.Embedding(lattice_embedding.size()[0],lattice_embedding.size()[1], padding_idx= 0 , _weight = lattice_embedding)
        self.bigram_embedding = nn.Embedding(bigram_embedding.size()[0],bigram_embedding.size()[1], padding_idx= 0 , _weight = bigram_embedding)
    
    def pos_init(self):
        with torch.no_grad():
            max_len = self.conf.max_seq_len
            hidden_size = self.conf.hidden_size
            pe = torch.zeros(2*max_len, hidden_size)
            position = torch.arange(-max_len, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_size, 2) *
                                -(math.log(10000.0) / hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        
            self.char_pos =  nn.Parameter(pe,requires_grad=False)

    
    def _genmask(self,lenth):
        with torch.no_grad():
            max_len = torch.max(lenth)
            mask = torch.zeros([lenth.size()[0],max_len], device = lenth.device)
            for b in range(lenth.size()[0]):
                mask[b,0:lenth[b]] = 1
        return mask.bool(), max_len

    def forward(self,x, is_train):
        char, lattice, bigram, ps,pe, label, char_len, lattice_len = x
        # print(x)
        # exit()
        char_mask, max_len = self._genmask(char_len)

        char_emb = self.lattice_embedding(char)
        char_emb = self.char_proj(char_emb).masked_fill(~char_mask.unsqueeze(-1),0)
        char_emb = self.drop_emb(char_emb)

        char_emb = char_emb + self.char_pos[torch.arange(max_len, device= char.device)].unsqueeze(0)

        char_emb = self.backbone(char_emb, char_mask.unsqueeze(1).unsqueeze(1))
        pred = self.output(char_emb)
        if is_train:
            crf_loss = self.crf(pred, label, char_mask).mean(dim=0)
            return {'loss': crf_loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, char_mask)
            result = {'pred': pred}
            return result