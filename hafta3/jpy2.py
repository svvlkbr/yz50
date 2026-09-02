import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
"""
words = open('names (1).txt', 'r', encoding='utf-8').read().splitlines()
print(words[:8])


chars = sorted(list(set(''.join(words))))

stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(itos)
print(itos)
print(vocab_size)

block_size=3

def build_dataset(words):
    X,Y=[],[]
    for w in words:
        context=[0]*block_size
        for ch in w+'.':
            ix=stoi[ch]
            X.append(context)
            Y.append(ix)
            context=context[1:]+[ix]

    X=torch.tensor(X)
    Y=torch.tensor(Y)

    print(X.shape,Y.shape)
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1=int(0.8*len(words))
n2=int(0.9*len(words))

Xtr,Ytr=build_dataset(words[:n1])
Xdev,Ydev=build_dataset(words[n1:n2])
Xte,Yte=build_dataset(words[n2:])

n_embd=10
n_hidden=200

g=torch.Generator().manual_seed(2147483647)
C=torch.randn((vocab_size,n_embd),generator=g)
W1=torch.randn((n_embd*block_size,n_hidden),generator=g)*(5/3)/((n_embd*block_size)**0.5)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

bngain=torch.ones((1,n_hidden))
bnbias=torch.zeros((1,n_hidden))
bnmean_running=torch.zeros((1,n_hidden))
bnstd_running=torch.ones((1,n_hidden))

parameters=[C,W1,W2,b2,bngain,bnbias]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad=True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix=torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
    Xb,Yb=Xtr[ix],Ytr[ix]

    emb=C[Xb]
    embcat=emb.view(emb.shape[0],-1)

    hpreact=embcat @ W1

    bnmeani=hpreact.mean(0,keepdim=True)
    bnstdi=hpreact.std(0,keepdim=True)
    hpreact=bngain*(hpreact-bnmeani) / bnstdi+bnbias
    with torch.no_grad():
        bnmean_running=0.999*bnmean_running+0.001*bnmeani
        bnstd_running=0.999*bnstd_running+0.001*bnstdi

    h=torch.tanh(hpreact)
    logits=h @ W2+b2
    loss=F.cross_entropy(logits,Yb)

    for p in parameters:
        p.grad=None
    loss.backward()

    lr=0.1 if i <100000 else 0.01
    for p in parameters:
        p.data += -lr* p.grad

    if i % 10000 ==0:
        print(f'{i:7d}/{max_steps:7d}:{loss.item():.4f}')
    lossi.append(loss.log10().item())

plt.plot(lossi)
plt.show()

with torch.no_grad():
    emb=C[Xtr]
    embcat=emb.view(emb.shape[0],-1)
    hpreact=embcat @ W1
    bnmean=hpreact.mean(0,keepdim=True)
    bnstd=hpreact.std(0,keepdim=True)

@torch.no_grad()
def split_loss(split):
    x,y={
        'train':(Xtr,Ytr),
        'val':(Xdev,Ydev),
        'test':(Xte,Yte),}[split]
    emb=C[x]
    embcat=emb.view(emb.shape[0],-1)
    hpreact=embcat @ W1
    hpreact=bngain*(hpreact-bnmean_running)
    h=torch.tanh(hpreact)
    logits=h @ W2+b2
    loss=F.cross_entropy(logits,y)
    print(split,loss.item())
split_loss('train')
split_loss('val')
"""


"""
class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight=torch.randn((fan_in,fan_out),generator=g)/ fan_in **0.5
        self.bias=torch.zeros(fan_out) if bias else None
    def __call__(self,x):
        self.out=x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] +([] if self.bias is None else [self.bias])
class BatchNorm1d:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        self.eps=eps
        self.momentum=momentum
        self.training=True
        self.gamma=torch.ones(dim)
        self.beta=torch.zeros(dim)
        self.running_mean=torch.zeros(dim)
        self.running_var=torch.ones(dim)
    def __call__(self,x):
        if self.training:
            xmean=xmean(0,keepdim=True)
            xvar=xvar(0,keepdim=True)
        else:
            xmean=self.running_mean
            xvar=self.running_var
        xhat=(x-xmean) / torch.sqrt(xvar+self.eps)
        self.out=self.gamma*xhat+self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*xmean
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*xvar
        return self.out
    def parameters(self):
        return [self.gamma,self.beta]

class Tanh:
    def __call__(self,x):
        self.out=torch.tanh(x)
        return self.out
    def parameters(self):
        return []
n_embd=10
n_hidden=100
g=torch.Generator().manual_seed(2147483647)

block_size=3
words = open('names (1).txt', 'r', encoding='utf-8').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(itos)

C = torch.randn((vocab_size, n_embd),            generator=g)
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]

with torch.no_grad():
  layers[-1].gamma *= 0.1
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.0 

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) 
for p in parameters:
  p.requires_grad = True
 """


from ipywidgets import interact,fixed,interact_manual
import ipywidgets as widgets
import scipy.stats as stats
import numpy as np

def normshow(x0):

    g=torch.Generator().manual_seed(2147483647+1)
    x=torch.randn(5,generator=g)*5
    x[0]=x0
    sig=x.std
    y=(x-mu)/sig

    plt.figure(figsize=(10,5))
    plt.pot([-6,6],[0,0],'k')

    xx=np.linspace(-6,6,100)
    plt.plot(xx,stats.norm.pdf(xx,mu,sig),'b')

    xx=np.linspace(-6,6,100)
    plt.plot(xx,stats.norm.pdf(xx,0,1),'r')

    for i in range(len(x)):
        plt.plot([x[i],y[i]],[1,0],'k',alpha=0.2)
    plt.scatter(x.data,torch.ones_like(x).data,c='b',s=100)
    plt.scatter(y.data,torch.zeros_likeke(y).data,c='r', s=100)
    plt.xlim(-6,6)
    plt.title('input mu %.2f std %.2f' % (mu, sig))
    plt.show()
interact(normshow,x0=(-30,30,0.5));
