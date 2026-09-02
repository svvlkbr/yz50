import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words= open('turkce_names.txt', 'r', encoding='utf-8').read().splitlines()
print(words[:5])

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)


N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


P = (N + 1).float()
P /= P.sum(1, keepdims=True)


g = torch.Generator().manual_seed(2147483647)
print("\n--- Türkçe Bigram Modeli Üretim Örnekleri ---")
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))


xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()


xenc = F.one_hot(xs, num_classes=vocab_size).float()


g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)


print("\n--- Tek Katmanlı Sinir Ağı Eğitimi ---")
for k in range(100):
    logits = xenc @ W 
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()

    W.grad = None
    loss.backward()
    
    W.data += -50.0 * W.grad
    
    if k % 10 == 0:
        print(f"Adım {k:2d} | Loss: {loss.item():.4f}")

print(f"Son Eğitim Loss Değeri: {loss.item():.4f}")