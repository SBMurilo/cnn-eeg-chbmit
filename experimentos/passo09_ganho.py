# %% [markdown]
# # PASSO 9 — Ganho aleatório no treino (única mudança em relação ao Passo 7)
# Cole DEPOIS do Passo 7. A cada lote, multiplica cada janela por um ganho aleatório
# (log-uniforme, 0.25x a 4x). O modelo não pode mais confiar só na amplitude absoluta
# e precisa usar a FORMA do sinal. Diferente do Passo 2, a amplitude relativa entre canais
# (onde a crise aparece) é preservada. Compare com o Passo 7: AUC médio 0.863, chb06 0.531, chb08 0.689.

# %%
def fold(fabrica, teste, vals, max_ep=20, paciencia=4, lote=64):
    tr = ~np.isin(pac, [teste, *vals])
    mu, sd = X_raw[tr].mean(), X_raw[tr].std()
    X = (X_raw - mu) / sd
    Xtr, ytr = X[tr], y_all[tr]
    Xt, yt = X[pac == teste], y_all[pac == teste]
    val = [(X[pac == v], y_all[pac == v]) for v in vals]
    torch.manual_seed(0)
    m = fabrica()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    melhor, best_sd, sem_melhora = -1, None, 0
    for _ in range(max_ep):
        m.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), lote):
            b = perm[i:i + lote]
            ganho = torch.exp(torch.empty(len(b), 1, 1).uniform_(np.log(0.25), np.log(4.0)))
            opt.zero_grad(); nn.functional.cross_entropy(m(Xtr[b] * ganho), ytr[b]).backward(); opt.step()
        auc_v = np.mean([metr(m, Xv, yv)[0] for Xv, yv in val])
        if auc_v > melhor:
            melhor, best_sd, sem_melhora = auc_v, copy.deepcopy(m.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    m.load_state_dict(best_sd)
    return melhor, metr(m, Xt, yt)

lopo(lambda: modelo_gap(X_raw.shape[1]), "B: GAP + ganho aleatório (validação com 2 pacientes)")
