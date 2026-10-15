# %% [markdown]
# # PASSO 3 — Global Average Pooling (única mudança) com controle justo
# Cole DEPOIS do Passo 2. Volta à normalização global original (AUC 0.707) e treina
# DUAS arquiteturas no MESMO laço: (A) a original, (B) com GAP no lugar de flatten+FC.

# %%
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

y_all = torch.as_tensor(np.asarray(rotulos_completos), dtype=torch.long)
X_raw = torch.as_tensor(np.asarray(janelas_completas), dtype=torch.float32)
idx_te = np.asarray(indices_teste)
idx_tr = np.setdiff1d(np.arange(len(y_all)), idx_te)
mu, sd = X_raw[idx_tr].mean(), X_raw[idx_tr].std()      # global, só do treino
X_all = (X_raw - mu) / sd
Xtr, ytr, Xte, yte = X_all[idx_tr], y_all[idx_tr], X_all[idx_te], y_all[idx_te]

def modelo_gap(c_in):
    def bloco(i, o, k):
        return [nn.Conv1d(i, o, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2)]
    return nn.Sequential(*bloco(c_in, 16, 21), *bloco(16, 32, 11), *bloco(32, 64, 7),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 2))

def original_resetado():
    m = copy.deepcopy(modelo)
    for l in m.modules():
        if hasattr(l, "reset_parameters"):
            l.reset_parameters()
    return m

def treinar(m, epocas=15, lote=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epocas):
        m.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), lote):
            b = perm[i:i + lote]
            opt.zero_grad(); crit(m(Xtr[b]), ytr[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(Xte), 1)[:, 1].numpy()
        y = yte.numpy(); pred = (p >= 0.5)
        hist.append((roc_auc_score(y, p), pred[y == 1].mean(), (~pred[y == 0]).mean()))
    return hist

A, B = original_resetado(), modelo_gap(Xtr.shape[1])
print("params A (original):", sum(p.numel() for p in A.parameters()),
      "| B (GAP):", sum(p.numel() for p in B.parameters()))
hA, hB = treinar(A), treinar(B)
print(f"\n{'ep':>3} | {'A: AUC':>7} {'sens':>6} {'espec':>6} | {'B: AUC':>7} {'sens':>6} {'espec':>6}")
for i, (a, b) in enumerate(zip(hA, hB), 1):
    print(f"{i:>3} | {a[0]:>7.3f} {a[1]:>6.1%} {a[2]:>6.1%} | {b[0]:>7.3f} {b[1]:>6.1%} {b[2]:>6.1%}")
