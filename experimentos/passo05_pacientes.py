# %% [markdown]
# # PASSO 5 — Por que chb06 e chb08 são difíceis? (sem treinar)
# Cole DEPOIS do Passo 4. Compara características simples do sinal por paciente.

# %%
import numpy as np

X = np.asarray(janelas_completas); y = np.asarray(rotulos_completos)
pac = np.array([origem_completa[i]["arquivo"][:5] for i in range(len(y))])
arq = np.array([origem_completa[i]["arquivo"] for i in range(len(y))])
sd = X.std(-1)                                   # (N, canais): desvio de cada janela/canal
print(f"{'pac':>6} {'arq.crise':>9} {'jan.crise':>9} {'jan.norm':>9} | "
      f"{'std normal':>10} {'std crise':>10} {'razão':>6} | {'canais mortos':>13}")
for p in np.unique(pac):
    m = pac == p
    n_arq = len(np.unique(arq[m & (y == 1)]))
    sn = np.median(sd[m & (y == 0)]); sc = np.median(sd[m & (y == 1)]) if (m & (y == 1)).any() else np.nan
    mortos = int((np.median(sd[m], 0) < 1e-6).sum())
    print(f"{p:>6} {n_arq:>9} {int((m & (y==1)).sum()):>9} {int((m & (y==0)).sum()):>9} | "
          f"{sn:>10.1f} {sc:>10.1f} {sc/sn:>6.2f} | {mortos:>13}")
