# %% [markdown]
# # PASSO 8 — Calibração por paciente (limiar a partir dos próprios sinais normais)
# Cole DEPOIS do Passo 7. Mesmo treino do Passo 7; muda só COMO o limiar é escolhido.
# Para cada paciente de teste: limiar = percentil 90 das probabilidades das SUAS janelas
# normais (especificidade ≈ 90% por construção) -> o que interessa é a sensibilidade.

# %%
if "_metr_orig" not in globals():
    _metr_orig = metr
def metr(m, X, y):                      # guarda as probabilidades da última avaliação
    global ULT
    m.eval()
    with torch.no_grad():
        ULT = (torch.softmax(m(X), 1)[:, 1].numpy(), y.numpy())
    return _metr_orig(m, X, y)

print(f"{'teste':>6} | {'AUC':>6} | {'fixo 0.5: sens':>14} {'espec':>6} | {'limiar p90':>10} {'sens':>6}")
res = []
for k, t in enumerate(com_crise):
    n = len(com_crise)
    _, (auc, s05, e05) = fold(lambda: modelo_gap(X_raw.shape[1]), t,
                              [com_crise[(k + 1) % n], com_crise[(k + 2) % n]])
    p, y = ULT
    thr = np.percentile(p[y == 0], 90)
    sens = (p[y == 1] > thr).mean()
    res.append((auc, s05, e05, sens))
    print(f"{t:>6} | {auc:>6.3f} | {s05:>14.1%} {e05:>6.1%} | {thr:>10.4f} {sens:>6.1%}")
r = np.array(res)
print(f"{'MÉDIA':>6} | {r[:,0].mean():>6.3f} | {r[:,1].mean():>14.1%} {r[:,2].mean():>6.1%} | {'':>10} {r[:,3].mean():>6.1%}")
# ponytail: percentil 90 nas janelas normais amostradas (50/arquivo), não em gravação contínua;
# na versão final, calibrar com uma gravação-base (sem crise) do paciente e medir falsos alarmes/h.
