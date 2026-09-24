# ecg-ml — Classificação de batimentos cardíacos (MIT-BIH)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ljb6/ecg-ml/blob/main/notebooks/ecg_heartbeat.ipynb)

Trabalho Final da disciplina **M10 — Deep Learning**, Problema 6.

Uma clínica usa monitores Holter (mais de 100 mil batimentos por exame de 24 h). O sistema **marca os batimentos suspeitos** para o técnico revisar, em vez de ele olhar o traçado inteiro. É uma triagem, não um diagnóstico.

Comparamos dois modelos em Keras sobre os mesmos dados:

- **MLP** (linha de base): trata os 187 pontos do batimento como colunas independentes.
- **CNN 1D**: filtros convolucionais que aproveitam a estrutura temporal do sinal.

## Como rodar

### Opção 1 — Google Colab (recomendada)

1. Clique no botão **Open in Colab** acima.
2. Menu *Ambiente de execução → Executar tudo*.

O notebook detecta que está no Colab, clona este repositório em `/content/ecg-ml` e baixa o dataset com `kagglehub`. Não é preciso instalar nada. A GPU é opcional: na CPU o treino leva poucos minutos.

### Opção 2 — Máquina local (plano B)

Requer **Python 3.13** (o TensorFlow 2.21 ainda não suporta Python 3.14).

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/ecg_heartbeat.ipynb
```

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/ecg_heartbeat.ipynb
```

Depois, use *Run → Run All Cells*.

### Opções úteis do notebook (célula `CONFIG`)

| Opção | Para que serve |
| --- | --- |
| `TREINAR = False` | Não treina: carrega os modelos já salvos em `results/models/` e só avalia |
| `MODO_RAPIDO = True` | Só 3 épocas (~1 min), para demonstrações ao vivo |
| `PESO_CLASSES` | Intensidade da compensação do desbalanceamento: `"raiz"` (padrão), `"balanceado"` ou `"nenhum"` |
| `REVOCACAO_ALVO` | Fração mínima dos batimentos anormais que a triagem deve marcar |

## Dados

Dataset [ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (Kachuee et al., 2018), derivado do [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/). Usamos só `mitbih_train.csv` (87.554 batimentos) e `mitbih_test.csv` (21.892). Cada linha tem 187 amostras a 125 Hz e o rótulo: N (normal), S (supraventricular), V (ventricular), F (fusão) e Q (desconhecido).

Os dados ficam em `data/heartbeat/`, que não vai para o Git. O notebook baixa automaticamente. Se o download falhar, baixe os dois CSVs no Kaggle e coloque nessa pasta.

## Estrutura

```
ecg-ml/
├── README.md
├── USO_DE_IA.md              ferramentas de IA usadas e em quais partes
├── requirements.txt          dependências para rodar localmente
├── data/heartbeat/           dataset (baixado pelo notebook, fora do Git)
├── notebooks/
│   ├── ecg_heartbeat.ipynb   pipeline completo: dados, EDA, modelos, treino, avaliação
│   └── test.ipynb            teste inicial de download do dataset
└── results/
    ├── figuras/              curvas de treino, matrizes de confusão, triagem, vazamento
    ├── models/               modelos treinados (.keras) e históricos de treino
    └── metricas.json         métricas finais no conjunto de teste
```

## Principais pontos do trabalho

- **Desbalanceamento:** cerca de 83% dos batimentos são normais. Usamos pesos de classe na perda (raiz quadrada do peso balanceado, escolhida por experimento) e avaliamos com macro-F1, revocação por classe e matriz de confusão.
- **Triagem:** convertemos o modelo em "normal × suspeito" e escolhemos o limiar na validação para marcar pelo menos 99% dos anormais, medindo quanto o técnico precisa revisar.
- **Vazamento:** a divisão treino/teste foi feita por batimento, não por paciente. O notebook mede esse efeito com um classificador de vizinho mais próximo, e os resultados devem ser lidos como otimistas para pacientes novos.
