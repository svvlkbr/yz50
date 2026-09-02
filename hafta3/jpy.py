words = open('names (1).txt', 'r', encoding='utf-8').read().splitlines()
print(words[:10])
# print(len(words))
# print(min(len(w) for w in words))
b={}
for w in words:
    chs=['<S>']+ list(w)+['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        biagram=(ch1,ch2)
        b[biagram]=b.get(biagram,0)+1
print(sorted(b.items(), key=lambda kv: -kv[1]))

import torch

N=torch.zeros((27,27), dtype=torch.int32)

chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}

for w in words:
    chs=['.']+ list(w) +['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        N[ix1,ix2] += 1

import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(N,cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr=itos[i]+itos[j]
        plt.text(j,i,chstr,ha="center",va="bottom",color='gray')
        plt.text(j,i,N[i,j].item(),ha="center",va="top",color='gray')
plt.axis('off');
plt.show()
"""
g=torch.Generator().manual_seed(2147483647)
ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
itos[ix]
"""

g=torch.Generator().manual_seed(2147483647)
p=torch.rand(3,generator=g)
p=p/p.sum()
p
torch.multinomial(p,num_samples=100, replacement=True,generator=g)

for i in range(5):
    out=[]
    ix=0
    while True:
        p=P[ix]
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix ==0:
            break
print(''.join(out))

log_likelihood=0.0
n=0

for w in words:
    chs=['.']+list(w)+['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        prob=P[ix1,ix2]
        logprob=torch.log(prob)
        log_likelihood += logprob
        n +=1
print(f'{log_likelihood=}')
nll= -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')


xs,ys=[],[]
for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)
xs=torch.tensor(xs)
ys=torch.tensor(ys)

import torch.nn.functional as F
xenc=F.one_hot(xs,num_classes=27).float()
xenc.shape
plt.imshow(xenc)
xenc.dtype
W=torch.randn((27,1))
xenc @ W

logits = xenc @ W
counts= logits.exp()
probs=counts/ counts.sum(1,keepdims = True)
probs