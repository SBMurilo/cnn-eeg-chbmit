# %% [markdown]
# # PASSO 13 — CNN como correção sobre o baseline de amplitude
# Cole DEPOIS do Passo 12. O Passo 12 mostrou: a regressão logística (23 amplitudes) dá 0.875 de AUC
# mesmo com os mesmos pacientes de treino da CNN -> o problema é a CNN estragar o que a amplitude já sabe.
# Aqui: logit = regressão logística CONGELADA (amplitude) + saída da CNN (camada final iniciada em zero,
# AdamW com decaimento forte). A época 0 é a própria regressão; a CNN só fica se melhorar a validação.
# A coluna 'ep' mostra a época escolhida (0 = a CNN não ajudou).

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class ResidualCNN(nn.Module):
    def __init__(self, c_in, w, b):
        super().__init__()
        def bloco(i, o, k):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2)]
        self.conv = nn.Sequential(*bloco(c_in, 16, 21), *bloco(16, 32, 11), *bloco(32, 64, 7),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5))
        self.fc = nn.Linear(64, 2)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)       # começa = só regressão
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(b), dtype=torch.float32))

    def forward(self, x):
        z = (torch.log(x.std(-1) + 1e-6) * self.w).sum(1) + self.b          # logit da regressão
        base = torch.stack([torch.zeros_like(z), z], 1)
        return base + self.fc(self.conv(x))

def fold13(teste, vals, max_ep=20, paciencia=4, lote=64):
    tr = ~np.isin(pac, [teste, *vals])
    mu, sd = X_raw[tr].mean(), X_raw[tr].std()
    X = (X_raw - mu) / sd
    F = torch.log(X.std(-1) + 1e-6).numpy()
    sc = StandardScaler().fit(F[tr])
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(F[tr]), y_all[tr].numpy())
    w = lr.coef_[0] / sc.scale_
    b = lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / sc.scale_).sum()
    Xtr, ytr = X[tr], y_all[tr]
    Xt, yt = X[pac == teste], y_all[pac == teste]
    val = [(X[pac == v], y_all[pac == v]) for v in vals]
    torch.manual_seed(0)
    m = ResidualCNN(X.shape[1], w, b)
    cont = torch.bincount(ytr).float()
    crit = nn.CrossEntropyLoss(weight=cont.sum() / (2 * cont))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    av = lambda: np.mean([metr(m, Xv, yv)[0] for Xv, yv in val])
    melhor, ep_b, best_sd, sem = av(), 0, copy.deepcopy(m.state_dict()), 0   # época 0 = só regressão
    for ep in range(1, max_ep + 1):
        m.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), lote):
            bi = perm[i:i + lote]
            opt.zero_grad(); crit(m(Xtr[bi]), ytr[bi]).backward(); opt.step()
        a = av()
        if a > melhor:
            melhor, ep_b, best_sd, sem = a, ep, copy.deepcopy(m.state_dict()), 0
        else:
            sem += 1
            if sem >= paciencia:
                break
    m.load_state_dict(best_sd)
    return melhor, ep_b, metr(m, Xt, yt)

res, n = [], len(com_crise)
print(f"{'teste':>6} {'val':>13} {'AUC val':>8} {'ep':>3} | {'AUC':>6} {'sens':>6} {'espec':>6}")
for k, t in enumerate(com_crise):
    vs = [com_crise[(k + 1) % n], com_crise[(k + 2) % n]]
    av_, ep_, (a, s, e) = fold13(t, vs)
    res.append((a, s, e)); print(f"{t:>6} {'+'.join(vs):>13} {av_:>8.3f} {ep_:>3} | {a:>6.3f} {s:>6.1%} {e:>6.1%}")
r = np.array(res)
print(f"{'MÉDIA':>6} {'':>13} {'':>8} {'':>3} | {r[:,0].mean():>6.3f} {r[:,1].mean():>6.1%} {r[:,2].mean():>6.1%}")
print("Referências: regressão pura 0.875 | CNN GAP 0.863 | híbrido 0.859")
