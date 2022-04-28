import copy,math
import torch
import torch.nn as nn
import torch.nn.functional as F
PAD_NUM = 0

def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query,key,value,mask=None,dropout=None):
    """

    :param query: (batch_size,h,seq_len,embedding)
    :param key:
    :param value:
    :param mask: (batch_size,1,1,seq_len)
    :param dropout:
    :return: (batch_size,h,seq_len,embedding)
    """
    d_k = query.size(-1)
    score = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == PAD_NUM,-1e9)
    p_atten = F.softmax(score,dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten,value),p_atten


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = nn.LayerNorm(layer.layerNormSize)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, layerNormSize, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.layerNormSize = layerNormSize
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(layerNormSize, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self,layerNormSize,p):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(layerNormSize)
        self.dropout = nn.Dropout(p)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model
        self.atten = None
        self.dropout = nn.Dropout(dropout)
        self.linears = clone(nn.Linear(d_model,d_model),4)

    def forward(self,query,key,value,mask=None):
        batch_size = query.size(0)
        query,key,value = [l(x).view(batch_size,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size,1,dk) => (batch_size,1,1,seq_len)
        x,self.atten = attention(query,key,value,mask,self.dropout)
        return self.linears[-1](x.transpose(1,2).contiguous().view(batch_size,-1,self.d_model))