# %% [markdown]
# # PASSO 6 — Filtro passa-faixa 0.5–40 Hz (única mudança) + LOPO
# Cole DEPOIS do Passo 4 (usa lopo, modelo_gap e X_raw definidos lá).
# Compare a MÉDIA e, principalmente, as linhas chb06 e chb08 com o Passo 4 (B: 0.867).

# %%
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

X_orig = X_raw.clone()                                   # guarda os dados do Passo 4
sos = butter(4, [0.5, 40], btype="band", fs=256, output="sos")
# ponytail: filtra cada janela de 2 s (efeito de borda nas pontas); o ideal seria filtrar
# o sinal contínuo antes de janelar — refazer o pré-processamento se o filtro ajudar.
X_raw = torch.as_tensor(sosfiltfilt(sos, X_orig.numpy(), axis=-1).copy(), dtype=torch.float32)

lopo(lambda: modelo_gap(X_raw.shape[1]), "B: GAP + passa-faixa 0.5–40 Hz")

X_raw = X_orig                                           # restaura para os próximos passos
