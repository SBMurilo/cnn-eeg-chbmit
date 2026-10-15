# %% [markdown]
# # PASSO 4 — Avaliação honesta: leave-one-patient-out + early stopping por paciente
# Cole DEPOIS do Passo 3 (usa modelo_gap e original_resetado definidos lá).
# Cada paciente com crise vira teste uma vez; outro paciente com crise vira validação
# (escolhe a época e para o treino). O teste nunca decide nada.

# %%
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

y_all = torch.as_tensor(np.asarray(rotulos_completos), dtype=torch.long)
X_raw = torch.as_tensor(np.asarray(janelas_completas), dtype=torch.float32)
pac = np.array([origem_completa[i]["arquivo"][:5] for i in range(len(y_all))])
com_crise = [p for p in np.unique(pac) if (y_all[pac == p] == 1).any()]
print("pacientes com crise:", com_crise)

def metr(m, X, y):
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(X), 1)[:, 1].numpy()
    y = y.numpy(); pred = p >= 0.5
    return roc_auc_score(y, p), pred[y == 1].mean(), (~pred[y == 0]).mean()

def fold(fabrica, teste, val, max_ep=20, paciencia=4, lote=64):
    tr = ~np.isin(pac, [teste, val])
    mu, sd = X_raw[tr].mean(), X_raw[tr].std()
    X = (X_raw - mu) / sd
    Xtr, ytr = X[tr], y_all[tr]
    Xv, yv = X[pac == val], y_all[pac == val]
    Xt, yt = X[pac == teste], y_all[pac == teste]
    torch.manual_seed(0)
    m = fabrica()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    melhor, best_sd, sem_melhora = -1, None, 0
    for _ in range(max_ep):
        m.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), lote):
            b = perm[i:i + lote]
            opt.zero_grad(); nn.functional.cross_entropy(m(Xtr[b]), ytr[b]).backward(); opt.step()
        auc_v = metr(m, Xv, yv)[0]
        if auc_v > melhor:
            melhor, best_sd, sem_melhora = auc_v, copy.deepcopy(m.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    m.load_state_dict(best_sd)
    return melhor, metr(m, Xt, yt)

def lopo(fabrica, nome):
    res = []
    print(f"\n== {nome} ==\n{'teste':>6} {'val':>6} {'AUC val':>8} | {'AUC':>6} {'sens':>6} {'espec':>6}")
    for k, t in enumerate(com_crise):
        v = com_crise[(k + 1) % len(com_crise)]
        av, (a, s, e) = fold(fabrica, t, v)
        res.append((a, s, e)); print(f"{t:>6} {v:>6} {av:>8.3f} | {a:>6.3f} {s:>6.1%} {e:>6.1%}")
    r = np.array(res)
    print(f"{'MÉDIA':>6} {'':>6} {'':>8} | {r[:,0].mean():>6.3f} {r[:,1].mean():>6.1%} {r[:,2].mean():>6.1%}")
    print(f"{'DESVIO':>6} {'':>6} {'':>8} | {r[:,0].std():>6.3f} {r[:,1].std():>6.1%} {r[:,2].std():>6.1%}")

lopo(lambda: modelo_gap(X_raw.shape[1]), "B: GAP (28k params)")
lopo(original_resetado, "A: original (290k params)")
