# Diagnóstico e baseline: passos 1 a 16 (CHB-MIT)

Experimentos feitos sobre o notebook `02_explorar_chbmit.ipynb`. Cada passo muda um fator por vez. Os trechos de código estão em [`experimentos/`](../experimentos) e como células no fim do notebook; eles dependem das variáveis do notebook (`modelo`, `janelas_completas`, `rotulos_completos`, `origem_completa`, `indices_teste`, ...) e devem ser executados em ordem, na mesma sessão.

Avaliação: leave-one-patient-out (LOPO) entre os pacientes com crise (chb01, 02, 03, 05, 06, 07, 08), métrica principal AUC por paciente. O chb04 entra só com janelas normais (os arquivos de crise não foram baixados).

## Dados (janelas por paciente)

| Paciente | Normais | Crises | Total |
|---|---|---|---|
| chb01 | 500 | 219 | 719 |
| chb02 | 250 | 85 | 335 |
| chb03 | 450 | 199 | 649 |
| chb04 | 50 | 0 | 50 |
| chb05 | 350 | 278 | 628 |
| chb06 | 450 | 75 | 525 |
| chb07 | 250 | 162 | 412 |
| chb08 | 350 | 459 | 809 |

Janelas normais amostradas aleatoriamente (até 50 por arquivo); janelas de crise sem sobreposição.

## Ponto de partida

Treino chb01–07, teste chb08: acurácia 63,7%, sensibilidade 38%, AUC 0,707. Acurácia de treino ~99,9% (sobreajuste ao paciente).

## Experimentos

| Passo | Script | Mudança | Resultado |
|---|---|---|---|
| 1 | `passo01_diagnostico.py` | Diagnóstico sem retreinar | AUC 0,707; probabilidades bimodais (mediana 0,02, p90 = 1,0): não é só limiar |
| 2 | `passo02_norm_janela.py` | Normalização por janela/canal | Piorou (AUC 0,72 → 0,55 ao longo das épocas); apaga a amplitude |
| 3 | `passo03_gap.py` | Global average pooling vs flatten+FC | Empate: 28k vs 290k parâmetros, AUC ~0,69 vs ~0,74 (ruído entre épocas) |
| 4 | `passo04_lopo.py` | LOPO + early stopping por paciente | AUC médio 0,867 (GAP) vs 0,869 (original); desvio ~0,18–0,20 |
| 5 | `passo05_pacientes.py` | Razão de amplitude crise/normal por paciente | chb06 = 0,88; demais 2,6–6,5 |
| 6 | `passo06_bandpass.py` | Passa-faixa 0,5–40 Hz | Sem ganho (AUC médio 0,855) |
| 7 | `passo07_val2.py` | Validação com 2 pacientes | AUC médio 0,863; limiar 0,5 instável entre pacientes |
| 8 | `passo08_calibracao.py` | Limiar por paciente (p90 dos normais dele) | Sensibilidade 84,7–99,1% em 5 pacientes; média 74,0% a especificidade ≈ 90% |
| 9 | `passo09_ganho.py` | Ganho aleatório 0,25×–4× no treino | Piorou: AUC médio 0,786 (chb08 0,48, chb06 0,44) |
| 10 | `passo10_bandas.py` | Baseline linear (regressão logística), LOPO | Só log-amplitude por canal: AUC médio 0,875 (chb08 0,925). Espectro relativo: 0,477. Abs+rel: 0,881 |
| 11 | `passo11_hibrido.py` | CNN GAP + log-amplitude por canal | AUC médio 0,859; chb08 0,725 |
| 12 | `passo12_controle.py` | Regressão com os mesmos pacientes de treino da CNN | AUC médio 0,875; chb08 0,910 |
| 13 | `passo13_residual.py` | CNN residual sobre a regressão | AUC médio 0,861; chb08 0,779 |
| 14 | `passo14_features.py` | Características clássicas (line length, Hjorth, curtose) | amp 0,875 · +line length 0,889 · +Hjorth 0,855 · todas 0,901 (chb06 0,542) |
| 15 | `passo15_matriz.py` | Treina em 1 paciente, testa em outro | 1 professor (exceto chb06) já dá AUC médio 0,85–0,87; chb06 como professor: 0,315 |
| 16 | `passo16_chb06.py` | Amplitude por canal e por arquivo | Nos 6 pacientes bons a amplitude da crise sobe em 23/23 canais (~2,6× a 6×); no chb06 só 4/23 canais |

## Conclusões

- A CNN generaliza bem para 5 de 7 pacientes (AUC 0,94–0,99). Falha no chb06 e é fraca no chb08 (~0,7), onde a regressão de amplitude vai bem (0,91–0,95).
- Uma regressão logística com 23 números (log-amplitude por canal) iguala ou supera a CNN (0,875 vs 0,863). Usar como baseline.
- Sete variações da CNN (arquitetura, normalização, filtro, aumento de dados, validação, híbrido, residual) ficaram entre 0,79 e 0,87; nenhuma superou o baseline de amplitude.
- Remover a amplitude (normalização por janela, ganho aleatório) piora. O espectro relativo isolado não discrimina.
- Características clássicas elevam o baseline para ~0,90, ganho pequeno (+0,03) e escolhido entre 4 conjuntos nos mesmos pacientes de teste (otimista).
- Quantidade de pacientes de treino não é o gargalo do baseline de amplitude (1 paciente já dá ~0,86; 6 dão 0,875). Para a CNN ainda não foi testado.
- O chb06 é um regime diferente: crises curtas (6–21 janelas por arquivo) sem aumento de amplitude. Nenhum método testado resolve. Hipóteses não verificadas: rótulos deslocados no tempo ou tipo de crise diferente.
- A escala das probabilidades muda por paciente (limiar calibrado de 0,0025 a 0,97): limiar fixo em 0,5 não serve; é preciso calibração por paciente.
- GAP (28k parâmetros) empata com a arquitetura original (290k).

## Limitações

- Sete pacientes; o AUC médio esconde desvio de ~0,2 e é puxado pelo chb06.
- Especificidade medida em janelas normais amostradas (até 50 por arquivo), não em gravação contínua. Falta medir falsos alarmes por hora.
- A dependência da amplitude expõe o detector a artefatos de movimento.
- Os resultados dos passos 7 a 16 estão neste documento; as células correspondentes no notebook foram acrescentadas sem saídas.

## Próximos passos possíveis

1. Quantização INT8 do modelo escolhido (GAP) e medir a queda de AUC.
2. Falsos alarmes por hora em gravação contínua (exige janelas contíguas).
3. Mais pacientes (chb09 em diante) e os arquivos de crise do chb04.
4. Investigar o sinal e as anotações do chb06.
