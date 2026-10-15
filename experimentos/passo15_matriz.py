# %% [markdown]
# # PASSO 15 — Treina em UM paciente, testa em OUTRO (matriz 7x7)
# Cole DEPOIS do Passo 14 (usa conjuntos, y, pac, com_crise). Regressão logística só com amplitude.
# Linha = paciente de treino, coluna = paciente de teste. Mostra quanto o resultado depende de
# QUEM treina e QUEM testa, e se juntar vários pacientes no treino (LOPO) realmente ajuda.

# %%
F = conjuntos["amp"]
pts = list(com_crise)
M = np.full((len(pts), len(pts)), np.nan)
for i, a in enumerate(pts):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    clf.fit(F[pac == a], y[pac == a])
    for j, b in enumerate(pts):
        if a != b:
            M[i, j] = roc_auc_score(y[pac == b], clf.predict_proba(F[pac == b])[:, 1])

print("treina \\ testa | " + " ".join(f"{p:>6}" for p in pts) + " |  média")
for i, a in enumerate(pts):
    print(f"{a:>14} | " + " ".join(f"{'  —' if i == j else f'{M[i, j]:.3f}':>6}" for j in range(len(pts)))
          + f" | {np.nanmean(M[i]):.3f}")
print(f"{'média coluna':>14} | " + " ".join(f"{np.nanmean(M[:, j]):>6.3f}" for j in range(len(pts))))
print("Referência (treina nos outros 6, Passo 10): 0.875 de média")
