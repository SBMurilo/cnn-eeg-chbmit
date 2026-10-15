# %% [markdown]
# # PASSO 1 — Diagnóstico sem retreinar
# Cole esta célula DEPOIS da última célula do notebook (matriz de confusão).
# Usa o `modelo` já treinado e as variáveis X_teste, y_teste, indices_teste e
# origem_completa, que já existem no notebook.
#
# Pergunta que ela responde: o modelo NÃO reconhece as crises do chb08, ou ele
# reconhece mas com probabilidade baixa demais (problema de calibração/limiar)?

# %%
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

modelo.eval()
with torch.no_grad():
    logits = modelo(X_teste)
    prob_crise = torch.softmax(logits, dim=1)[:, 1].numpy()

# Obs.: usamos y_teste (e não `todos_rotulos`, que a célula da matriz de confusão
# sobrescreveu com uma lista).
y_true = y_teste.numpy()

# --- 1) AUC: a capacidade de SEPARAR as classes, independente do limiar --------
auc = roc_auc_score(y_true, prob_crise)
print(f"AUC no chb08: {auc:.3f}   (0.5 = chute, 1.0 = separação perfeita)\n")

# --- 2) Quão confiante o modelo é nas janelas de crise? ------------------------
p10, p50, p90 = np.percentile(prob_crise[y_true == 1], [10, 50, 90])
print("Prob. de crise atribuída às janelas que SÃO crise:")
print(f"  percentil 10: {p10:.3f} | mediana: {p50:.3f} | percentil 90: {p90:.3f}\n")

# --- 3) Varredura de limiar (APENAS diagnóstico, não é resultado válido) --------
# Escolher o limiar olhando o paciente de teste seria "colar" no gabarito.
print(f"{'limiar':>7} {'sensib.':>8} {'especif.':>9} {'acur.bal.':>10}")
for lim in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
    pred = (prob_crise >= lim).astype(int)
    sens = (pred[y_true == 1] == 1).mean()
    espec = (pred[y_true == 0] == 0).mean()
    print(f"{lim:>7.2f} {sens:>8.2%} {espec:>9.2%} {(sens + espec) / 2:>10.2%}")

# --- 4) Onde o modelo erra: sensibilidade por arquivo ---------------------------
arquivos_teste = np.array([origem_completa[i]["arquivo"] for i in indices_teste])
pred_05 = (prob_crise >= 0.5).astype(int)
print(f"\n{'arquivo':<14} {'janelas crise':>14} {'janelas normais':>16} {'sensib.':>8} {'especif.':>9}")
for arq in np.unique(arquivos_teste):
    m = arquivos_teste == arq
    n_crise = int((y_true[m] == 1).sum())
    n_norm = int((y_true[m] == 0).sum())
    sens = f"{(pred_05[m & (y_true == 1)] == 1).mean():.2%}" if n_crise else "—"
    espec = f"{(pred_05[m & (y_true == 0)] == 0).mean():.2%}" if n_norm else "—"
    print(f"{arq:<14} {n_crise:>14} {n_norm:>16} {sens:>8} {espec:>9}")
