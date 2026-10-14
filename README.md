# CNN 1D multicanal para detecção de crises epilépticas (CHB-MIT)

Exploração do [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet; Shoeb, 2009) com uma **CNN 1D multicanal em PyTorch**, avaliada em um **paciente nunca visto** durante o treino.

A base reúne registros contínuos de EEG de pacientes pediátricos, em formato EDF, com 23 canais a 256 Hz e marcação do início e do fim de cada crise.

## Dados

Os dados **não estão no repositório**. Baixe os pacientes na [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) e coloque as pastas `chbXX` (com os `.edf` e o `chbXX-summary.txt`) na mesma pasta dos notebooks.

## Pipeline

1. Leitura dos arquivos EDF com MNE e das marcações de crise nos arquivos `-summary.txt`.
2. Extração de janelas de 2 s (512 amostras × 23 canais): todas as janelas de crise e até 50 janelas normais por arquivo, com margem de 10 s ao redor de cada crise para evitar contaminação.
3. **Divisão por paciente**: treino com vários pacientes e teste em um paciente que a rede nunca viu (`chb08`). A escolha do paciente de teste é uma única variável (`pacientes_teste`).
4. CNN 1D multicanal: 3 camadas convolucionais (kernels 21, 11 e 7; 16, 32 e 64 filtros) com max pooling, seguidas de 2 camadas lineares. Total de 290.146 parâmetros, 50 épocas.

## Resultados

Teste no paciente `chb08`: 809 janelas (350 normais e 459 de crise).

| Notebook | Treino | Variação | Acurácia | Recall de crise | Precisão de crise |
|----------|--------|----------|---------:|----------------:|------------------:|
| `01_explorar_chbmit` | chb01–07 (3.318 janelas) | dropout 0,2 | 61,68% | 35% | 94% |
| `02_explorar_chbmit` | chb01–03, 05–07 (3.268 janelas) | data augmentation + dropout 0,5 | 57,23% | 27% | 93% |

Matriz de confusão do notebook 01 (linhas = real, colunas = previsto):

|  | Prev. normal | Prev. crise |
|--|-------------:|------------:|
| **Normal** | 339 | 11 |
| **Crise** | 299 | 160 |

Nos dois casos a acurácia de treino chega a ~100%, mas no paciente novo a rede acerta quase todas as janelas normais (recall de 97%) e **perde a maior parte das crises**. Há forte sobreajuste ao treino e grande variabilidade entre pacientes, o que mostra por que a divisão por paciente é a avaliação mais realista.

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook
```

## Limitações

- Foram usados apenas 7 a 8 pacientes (`chb01` a `chb08`) e um único paciente de teste, então os números têm alta incerteza.
- Poucas janelas normais por arquivo (até 50) e classes desbalanceadas entre pacientes.
- Resultados de exploração, sem validação cruzada entre pacientes.

## Projeto relacionado

[`cnn-eeg-epilepsia`](https://github.com/SBMurilo/cnn-eeg-epilepsia): CNN 1D no dataset de Bonn (canal único), com quantização INT8 e exportação para FPGA. Os números de Bonn (97% a 99%) usam divisão aleatória de janelas e não são comparáveis aos daqui.

## Tecnologias

Python, PyTorch, MNE, NumPy, scikit-learn, Matplotlib, Jupyter.
