from tkinter import N
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

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
n=batch_size
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]
lossi = []

"""
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

    logit_maxes=logits.max(1,keepdim=True).values
    norm_logits=logits-logit_maxes
    counts=norm_logits.exp()
    counts_sum=counts.sum(1,keepdim=True)
    counts_sum_inv=counts_sum ** -1
    probs =counts * counts_sum_inv
    logprobs=probs.log()

    loss= -logprobs[range(n),Yb].mean()

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
def cmp(s,dt,t):
    ex=torch.all(dt==t.grad).item()
    app=torch.allclose(dt,t.grad)
    maxdiff=(dt-t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

"""
emb = C[Xb] 
embcat = emb.view(emb.shape[0], -1) 
hprebn = embcat @ W1 + n1 
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) 
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact) 
logits = h @ W2 + b2 
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes 
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

"""
emb = C[Xb]
emb.retain_grad()
embcat = emb.view(emb.shape[0], -1)
embcat.retain_grad()
    
hprebn = embcat @ W1
hprebn.retain_grad()
    
bnmeani = hprebn.mean(0, keepdim=True)
bnmeani.retain_grad()
    
bndiff = hprebn - bnmeani
bndiff.retain_grad()
    
bndiff2 = bndiff ** 2
bndiff2.retain_grad()
    
bnvar = bndiff2.mean(0, keepdim=True)
bnvar.retain_grad()
    
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnvar_inv.retain_grad()
    
bnraw = bndiff * bnvar_inv
bnraw.retain_grad()
    
bngain_reshaped = bngain # boyut uyumu için
hpreact = bngain * bnraw + bnbias
hpreact.retain_grad()
    
h = torch.tanh(hpreact)
h.retain_grad()
    
logits = h @ W2 + b2
logits.retain_grad()
    
logit_maxes = logits.max(1, keepdim=True).values
logit_maxes.retain_grad()
    
norm_logits = logits - logit_maxes
norm_logits.retain_grad()
    
counts = norm_logits.exp()
counts.retain_grad()
    
counts_sum = counts.sum(1, keepdim=True)
counts_sum.retain_grad()
    
counts_sum_inv = counts_sum ** -1
counts_sum_inv.retain_grad()
    
probs = counts * counts_sum_inv
probs.retain_grad()
    
logprobs = probs.log()
logprobs.retain_grad()
    
loss = -logprobs[range(n), Yb].mean()
loss.backward()

dlogprobs=torch.zeros_like(logprobs)
dlogprobs[range(n),Yb]=-1.0/n
dprobs=(1.0/probs) * dlogprobs
dcounts_sum_inv=(counts*dprobs).sum(1,keepdim=True)
dcounts=counts_sum_inv*dprobs
dcounts_sum=(-counts_sum **-2)*dcounts_sum_inv
dcounts += torch.ones_like(counts)*dcounts_sum
dnorm_logits=counts*dcounts
dlogits=dnorm_logits.clone()
dlogits_maxes=(-dnorm_logits).sum(1,keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1])*dlogits_maxes
dh=dlogits @ W2.T
dW2=h.T @ dlogits
db2=dlogits.sum(0)
dhpreact=(1.0-h **2)*dh
dbngain=(bnraw**dhpreact).sum(0,keepdim=True)
dbnraw=bngain*dhpreact
dnbias=dhpreact.sum(0,keepdim=True)
dbndiff = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
dbndiff += (2*bndiff) * dbndiff2
dhprebn = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0)
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
  for j in range(Xb.shape[1]):
    ix = Xb[k,j]
    dC[ix] += demb[k,j]

cmp('logprobs',dlogprobs,logprobs)
