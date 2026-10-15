# %% [markdown]
# # PASSO 10 — Quanto vem de amplitude e quanto de frequência? (sem rede neural)
# Cole DEPOIS do Passo 9. Classificador linear em 3 conjuntos de características, no mesmo
# leave-one-patient-out (treina com todos os outros pacientes). Responde:
#  - amplitude sozinha já explica o AUC?
#  - o espectro relativo (sem amplitude) separa crise nos pacientes difíceis (chb06, chb08)?

# %%
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

Xn = X_orig.numpy() if "X_orig" in globals() else X_raw.numpy()      # janelas sem filtro
f, P = welch(Xn, fs=256, nperseg=256, axis=-1)                       # (N, canais, freqs)
bandas = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BP = np.stack([P[..., (f >= a) & (f < b)].sum(-1) for a, b in bandas], -1)   # (N, canais, 5)

amp = np.log(Xn.std(-1) + 1e-12)                                     # (N, canais)
rel = (BP / BP.sum(-1, keepdims=True)).reshape(len(Xn), -1)          # espectro relativo
absb = np.log(BP + 1e-18).reshape(len(Xn), -1)                       # potência absoluta por banda
conjuntos = {"amplitude": amp, "espectro relativo": rel, "abs+rel": np.hstack([absb, rel])}

y = y_all.numpy()
print(f"{'teste':>6} | " + " ".join(f"{k:>17}" for k in conjuntos))
medias = {k: [] for k in conjuntos}
for t in com_crise:
    te = pac == t
    linha = []
    for nome, F in conjuntos.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        clf.fit(F[~te], y[~te])
        a = roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]); medias[nome].append(a); linha.append(a)
    print(f"{t:>6} | " + " ".join(f"{a:>17.3f}" for a in linha))
print(f"{'MÉDIA':>6} | " + " ".join(f"{np.mean(v):>17.3f}" for v in medias.values()))
