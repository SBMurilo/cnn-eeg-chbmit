# %% [markdown]
# # PASSO 2 — Normalização por janela e por canal (única mudança)
# Cole DEPOIS da célula do Passo 1. Retreina do zero com os mesmos dados e a mesma
# arquitetura; muda SÓ a normalização. Compare com AUC 0.707 / sensib. 38% do Passo 1.

# %%
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def norm_janela(x):
    # x: (N, canais, tempo) -> cada janela/canal com média 0 e desvio 1
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)

X_all = norm_janela(janelas_completas)
y_all = torch.as_tensor(np.asarray(rotulos_completos), dtype=torch.long)
idx_te = np.asarray(indices_teste)
idx_tr = np.setdiff1d(np.arange(len(y_all)), idx_te)

# mesma arquitetura, pesos reiniciados (assume que o modelo é nn.Module comum)
modelo2 = copy.deepcopy(modelo)
for m in modelo2.modules():
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

# pesos de classe para o desbalanceamento do treino
cont = torch.bincount(y_all[idx_tr]).float()
crit = nn.CrossEntropyLoss(weight=(cont.sum() / cont))  # ponytail: peso simples, sem tuning
opt = torch.optim.Adam(modelo2.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCAS, LOTE = 15, 64
Xtr, ytr = X_all[idx_tr], y_all[idx_tr]
Xte, yte = X_all[idx_te], y_all[idx_te]

def avaliar():
    modelo2.eval()
    with torch.no_grad():
        p = torch.softmax(modelo2(Xte), 1)[:, 1].numpy()
    y = yte.numpy()
    pred = (p >= 0.5).astype(int)
    sens, espec = (pred[y == 1] == 1).mean(), (pred[y == 0] == 0).mean()
    return roc_auc_score(y, p), sens, espec

for ep in range(EPOCAS):
    modelo2.train()
    perm = torch.randperm(len(ytr))
    for i in range(0, len(perm), LOTE):
        b = perm[i:i + LOTE]
        opt.zero_grad()
        crit(modelo2(Xtr[b]), ytr[b]).backward()
        opt.step()
    auc, sens, espec = avaliar()
    print(f"ep {ep+1:2d} | AUC {auc:.3f} | sensib. {sens:.2%} | especif. {espec:.2%}")
# Obs.: acompanhar o teste por época é só para DIAGNÓSTICO; não escolha a época
# olhando o chb08 (isso vira o Passo 4: validação por paciente + early stopping).
