# %% [markdown]
# # PASSO 11 — CNN + amplitude por canal (modelo híbrido)
# Cole DEPOIS do Passo 10. Só uma mudança em relação ao GAP do Passo 7: o log do desvio-padrão
# de cada canal (23 números) entra junto com as 64 características da CNN, antes da camada final.
# Redefine `fold` SEM o ganho aleatório do Passo 9 (volta ao treino do Passo 7).
# Referências (LOPO): CNN GAP 0.863 | regressão logística só com amplitude 0.875 (chb08: 0.925).

# %%
class Hibrido(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        def bloco(i, o, k):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2)]
        self.conv = nn.Sequential(*bloco(c_in, 16, 21), *bloco(16, 32, 11), *bloco(32, 64, 7),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.amp = nn.BatchNorm1d(c_in)                 # padroniza o log-amplitude por canal
        self.drop, self.fc = nn.Dropout(0.5), nn.Linear(64 + c_in, 2)

    def forward(self, x):
        a = self.amp(torch.log(x.std(-1) + 1e-6))
        return self.fc(self.drop(torch.cat([self.conv(x), a], 1)))

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

lopo(lambda: Hibrido(X_raw.shape[1]), "Híbrido: CNN GAP + log-amplitude por canal")
