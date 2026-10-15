# %% [markdown]
# # PASSO 7 — Validação com 2 pacientes (única mudança) + LOPO, sem filtro
# Cole DEPOIS do Passo 6. Redefine fold/lopo do Passo 4: o early stopping passa a olhar
# o AUC médio de 2 pacientes de validação, para um paciente ruim não decidir sozinho.

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
            opt.zero_grad(); nn.functional.cross_entropy(m(Xtr[b]), ytr[b]).backward(); opt.step()
        auc_v = np.mean([metr(m, Xv, yv)[0] for Xv, yv in val])
        if auc_v > melhor:
            melhor, best_sd, sem_melhora = auc_v, copy.deepcopy(m.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    m.load_state_dict(best_sd)
    return melhor, metr(m, Xt, yt)

def lopo(fabrica, nome):
    res, n = [], len(com_crise)
    print(f"\n== {nome} ==\n{'teste':>6} {'val':>13} {'AUC val':>8} | {'AUC':>6} {'sens':>6} {'espec':>6}")
    for k, t in enumerate(com_crise):
        vs = [com_crise[(k + 1) % n], com_crise[(k + 2) % n]]
        av, (a, s, e) = fold(fabrica, t, vs)
        res.append((a, s, e)); print(f"{t:>6} {'+'.join(vs):>13} {av:>8.3f} | {a:>6.3f} {s:>6.1%} {e:>6.1%}")
    r = np.array(res)
    print(f"{'MÉDIA':>6} {'':>13} {'':>8} | {r[:,0].mean():>6.3f} {r[:,1].mean():>6.1%} {r[:,2].mean():>6.1%}")
    print(f"{'DESVIO':>6} {'':>13} {'':>8} | {r[:,0].std():>6.3f} {r[:,1].std():>6.1%} {r[:,2].std():>6.1%}")

lopo(lambda: modelo_gap(X_raw.shape[1]), "B: GAP, validação com 2 pacientes")
