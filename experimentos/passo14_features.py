# %% [markdown]
# # PASSO 14 — Características clássicas no domínio do tempo (baratas na FPGA)
# Cole DEPOIS do Passo 13. Mesmo leave-one-patient-out da regressão logística do Passo 10
# (referência: só amplitude = 0.875; chb06 0.360, chb08 0.925). Pergunta: algo além da amplitude
# generaliza entre pacientes? Todas as características são somas/diferenças por canal.

# %%
from scipy.stats import kurtosis

Xn = X_raw.numpy()                                   # janelas sem filtro (N, canais, tempo)
d = np.diff(Xn, axis=-1); dd = np.diff(d, axis=-1)
s0, s1, s2 = Xn.std(-1) + 1e-12, d.std(-1) + 1e-12, dd.std(-1) + 1e-12
amp = np.log(s0)                                     # amplitude
ll = np.log(np.abs(d).mean(-1) + 1e-12)              # line length (soma de |diferenças|)
mob = s1 / s0                                        # Hjorth: mobilidade (~frequência média)
comp = (s2 / s1) / mob                               # Hjorth: complexidade
kur = np.nan_to_num(kurtosis(Xn, axis=-1))           # curtose (picos)

conjuntos = {"amp": amp,
             "amp+linelength": np.hstack([amp, ll]),
             "amp+hjorth": np.hstack([amp, mob, comp]),
             "amp+ll+hjorth+kurt": np.hstack([amp, ll, mob, comp, kur])}
print(f"{'teste':>6} | " + " ".join(f"{k:>19}" for k in conjuntos))
medias = {k: [] for k in conjuntos}
for t in com_crise:
    te = pac == t
    linha = []
    for nome, F in conjuntos.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        clf.fit(F[~te], y[~te])
        a = roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]); medias[nome].append(a); linha.append(a)
    print(f"{t:>6} | " + " ".join(f"{a:>19.3f}" for a in linha))
print(f"{'MÉDIA':>6} | " + " ".join(f"{np.mean(v):>19.3f}" for v in medias.values()))
