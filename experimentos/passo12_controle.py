# %% [markdown]
# # PASSO 12 — Controle justo: regressão logística com os MESMOS pacientes de treino da CNN
# Cole DEPOIS do Passo 11 (usa amp, y, pac, com_crise do Passo 10).
# No Passo 10 a regressão treinou com todos os outros pacientes; a CNN treina sem os 2
# pacientes de validação. Aqui a regressão também exclui esses 2. Se o AUC do chb08 continuar
# alto (~0.9), o problema é a CNN; se cair para ~0.7, a causa é quais pacientes entram no treino.

# %%
def lr_auc(F, t, excluir):
    tr = ~np.isin(pac, [t, *excluir])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    clf.fit(F[tr], y[tr])
    return roc_auc_score(y[pac == t], clf.predict_proba(F[pac == t])[:, 1])

n = len(com_crise)
print(f"{'teste':>6} {'val (excluídos)':>16} | {'LR todos':>9} {'LR = CNN':>9}")
r = []
for k, t in enumerate(com_crise):
    vs = [com_crise[(k + 1) % n], com_crise[(k + 2) % n]]
    a, b = lr_auc(amp, t, []), lr_auc(amp, t, vs)
    r.append((a, b)); print(f"{t:>6} {'+'.join(vs):>16} | {a:>9.3f} {b:>9.3f}")
r = np.array(r)
print(f"{'MÉDIA':>6} {'':>16} | {r[:,0].mean():>9.3f} {r[:,1].mean():>9.3f}")
