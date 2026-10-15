# %% [markdown]
# # PASSO 16 — O que há de diferente no chb06? (amplitude por canal e por arquivo)
# Cole DEPOIS do Passo 15. d = log-amplitude mediana (crise) - (normal), por canal.
# d > 0: crise com mais amplitude. Pergunta: no chb06 d é negativo em todos os canais
# (padrão invertido) ou só em alguns? E isso vale para todos os arquivos dele?

# %%
amp = conjuntos["amp"]                                   # (N, canais) log do desvio-padrão
arq = np.array([origem_completa[i]["arquivo"] for i in range(len(y))])

D = {p: np.median(amp[(pac == p) & (y == 1)], 0) - np.median(amp[(pac == p) & (y == 0)], 0) for p in com_crise}
print(f"{'pac':>6} {'d médio':>8} {'canais d>0':>11} {'corr c/ média dos outros':>25}")
for p in com_crise:
    outros = np.mean([D[q] for q in com_crise if q != p], 0)
    print(f"{p:>6} {D[p].mean():>8.2f} {int((D[p] > 0).sum()):>8}/{len(D[p])} {np.corrcoef(D[p], outros)[0, 1]:>25.2f}")

m = pac == "chb06"
base = np.median(amp[m & (y == 0)], 0)
print(f"\nchb06 por arquivo (d vs. normais do paciente)")
print(f"{'arquivo':<14} {'jan.crise':>9} {'d médio':>8} {'canais d>0':>11}")
for a in np.unique(arq[m & (y == 1)]):
    d = np.median(amp[m & (y == 1) & (arq == a)], 0) - base
    print(f"{a:<14} {int((m & (y == 1) & (arq == a)).sum()):>9} {d.mean():>8.2f} {int((d > 0).sum()):>8}/{len(d)}")
