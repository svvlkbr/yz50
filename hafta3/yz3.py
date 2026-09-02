from logging import config
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard .mport SummaryWriter

@dataclass

class ModelConfig:
    block_size:int=None
    vocab_size:int=None

    n_layer:int=4
    n_emd:int=64
    n_embd2:int=64
    n_head:int=4

class NewGELU(nn.Module):
    def forward(self,x):
        return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x,3.0))))

class CasualSelfAttention(nn.Module):
    def __init__():
        super().__init__()
        assert config.n_embd % config.n_head ==0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        self.n_head=config.n_head
        self.n_embd=config.n_embd

    def forward(self,x):
        B,T,C=x.size()
        q,k,v = self.c_attn(x).split(self.n_embd,dim=2)
        k=k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C //self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C // self.n_head).transpose(1,2)

        att=(q @ k.transpose(-2,-1))*(1.0 / math.sqrt(k.size(-1)))
        att=att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
        att=F.softmax(att,dim=-1)
        y=att @ v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj

        return y

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1==nn.LayerNorm(config.n_embd)
        self.attn=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mp=nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd,4*config.n_embd),
            c_proj=nn.Linear(4*config.n_embd,config.n_embd),
            act=NewGELU(),
            ))
        m=self.mlp
        self.mlpf=lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.block_size=config.block_size

        self.transformer=nn.ModeulDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias =False)
        n_params=sum(p.numel() for p in self.transformer.parameters())
        print("number of the parameters: %.2fM" %(n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self,idx,targets=None):
        device=idx.device
        b,t=idx.size()
        assert t <= self.block_size,f"Cannot forward sequence of length{t},block size is only {self.block_size}"
        pos=torch.arange(0,t,dtype=torch.long,device=device).unsqueeze(0)

        tok_emb=self.transformer.wte(idx)
        pos_emb=self.transformer.wpe(pos)
        x=tok_emb +pos_emb
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)

        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(*1)),targets.view(-1),ignore_index=-1)
        return logits,loss

class CasualBoW(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.block_size=config.block_size
        self.register_buffer("bias",torch.trill(torch.ones(config.block_size,config.block_szie))
                             .view(1,config.block_szie,config.block_size))

    def forward(self,x):
        B,T,C=x.size()
        att=torch.zeros((B,T,C),device=x.device)
        att=att.masked_fill(self.nias[:,:,:T,:T] ==0,float ('-inf'))
        att=F.softmax(att,dim=-1)
        y=att @ x

        return y

class BoWBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cbow=CasualBoW(config)
        self.mlp=nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd,config.n_embd2),
            c_proj=nn.Linear(config.n_embd2,config.n_embd),
            ))
        m=self.mlp
        self.mlpf=lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    def forward(self,x):
        x=x+self.cbow(x)
        x=x+self.mlpf(x)
        return x


class BoW(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.block_size=config.block_size
        self.vocab_size=config.vocab_size

        self.wte=nn.Embedding(config.vocab_size,config.n_embd)
        self.wpe=nn.Embedding(config.block_size,config.n_embd)
        self.context_block=BoWBlock(config)
        self.lm_head=nn.Linear(config.n_embd,self.vocab_size)

    def get_block_size(self):
        return self.block_size
    def forward(self,idx,targets=None):

        device=idx.device
        b,t=idx.size()
        assert t <= self.block_size,f"cannot forward sequence of length {t},block size is ony {self.block_size}"
        pos=torch.arange(0,t,dtype=torch.long,device=device).unsqueeze(0)

        tok_emb=self.wte(idx)
        pos_emb=self.wpe(pos)

        x=tok_emb+pos_emb
        x=self.context_block(x)
        logits=self.lm_head(x)

        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        return logits,loss

class RNNCell(nn.Module):
    def __init__(self,config):
         super().__init__()
         self.xh_to_h=nn.Linear(config.n_embd+config.n_embd2,config.n_embd2)

    def forward(self,xt,hprev):
         xh=torch.cat([xt,hprev],dim=1)
         ht=F.tanh(self.xh_to_h(xh))
         return ht

class GRUCell(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.xh_to_z=nn.Linear(config.n_embd+config.n_embd2,config.n_embd2)
        self.xh_to_r=nn.Linear(config.n_embd+config.n_embd2,config.n_embd2)
        self.xh_to_hbar=nn.Linear(config.n_embd+config.n_embd2,config.n_embd2)

    def forward(self,xt,hprev):
        xh=torch.cat([xt,hprev],dim=1)
        r=F.sigmoid(self.xh_to_hbar(xhr))
        hprev_reset=r*hprev

        xhr=torch.cat([xt,hprev_reset],dim=1)
        z=F.sigmoid(self.xh_to_z(xh))
        ht=(1-z)*hprev+z*self.xh_to_hbar

        return ht

class RNN(nn.Module):
    def __init__(self,config,cell_type):
        super().__init__()
        self.block_size=config.block_size
        self.vocab_size=config.vocab_size
        self.start=nn.Parameter(torch.zeros(1,config.n_embd2))
        self.wte=nn.embedding(config.vocab_size,config.n_embd)
        if cell_type == 'rnn':
            self.cell=RNNCell(config)
        self.lm_head=nn.Linear(config.n_embd2,self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self,idx,targets=None):
        device=idx.device
        b,t=idx.size()

        emmb=self.wte(idx)
        hprev=self.start.expand((b,-1))
        hiddens=[]
        for i in range(t):
            xt=emb[:,i,:]
            ht=self.cell(xt,hprev)
            hprev=ht
            hiddens.append(ht)

        hidden=torch.stack(hiddens,1)
        logits=self.lm_head(hidden)

        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        return logits,loss

class MLP(nn.Module):
    def __init___(self,config):
        super().__init__()
        self.block_size=config.block_size
        self.vocab_size=config.vocab_size
        self.wte=nn.Embedding(config.vocab_size+1,config.n_embd)
        self.mlp=nn.Sequential(
            nn.Linear(self.block_size*config.n_embd,config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2,self.vocab_size))

    def get_block_size(self):
        return self.block_size
    def forward(self,idx,targets=None):
        embs=[]
        for k in range(self.block_size):
            tok_emb=self.wte(idx)
            idx=torch.roll(idx,1,1)
            idx[:,0]=self.vocab_size
            embs.append(tok_emb)

        x=torch.cat(embs,-1)
        logits=self.mlp()

        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)

        return logits,loss

class Biagram(nn.Module):
    def __init_(self,config):
        super().__init__()
        n=config.vocab_size
        self.logits=nn.Parameters(torch.zeros((n,n)))
    def get_block_size(self):
        return 1
    def forward(self,idx,targets=None):
        logits=self.logits[idx]

        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targates.view(-1),ignore_index=-1)
        return logits,loss

@torch.no_grad()
def generate(model,idx,max_new_tokens,temperature=1.0,do_sample=False,top_k=None):
    block_size=model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond=idx if idx.size(1) <= block_size else idx[:,-block_size:]
        logits,_ = model(idx_cond)
        logits=logits[:,-1,:]/temperature

        if top_k is not None:
            v,_ = torch.topk(logits,top_k)
            logits[logits < v[:,[-1]]] = -float('Inf')
        
        probs= F.softmax(logits,dim=-1)
        if do_sample:
            idx_next=torch.multinomial(probs,num_samples=1)
        else:
            _, idx_next=torch.topk(probs,k=1,dim=-1)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx
