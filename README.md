# Predição de Falhas em SSDs — Projeto Final de Aprendizagem de Máquina

Para cada disco SSD e cada dia, prever se ele **falhará nos próximos N dias** a
partir de 24 atributos SMART ao longo de até 360 dias, comparando quatro
modelos: **LSTM, Regressão Logística, MLP e Random Forest**.

**Configuração oficial:** `WINDOW = 90` (janela das features tabulares,
escolhida em `experimentos/AMA_experimento_janela.ipynb`) e
`contamination_level = 7` (horizonte de predição). Há uma **segunda bateria
completa, paralela, para o horizonte de 30 dias** (modelos re-treinados do zero)
— **não comparável em números** com a de 7 dias, porque a prevalência de
positivos é diferente (~9% vs ~2%).

> Para a apresentação, veja **`APRESENTACAO.md`** (roteiro, resultados e
> dicionário das features).

## Estrutura de pastas

```
comum/             ssd_utils.py, data.pickle, mask.pickle   (código + dados compartilhados)
horizonte_7dias/   bateria oficial de 7 dias (4 modelos + comparação) e seus artefatos
horizonte_30dias/  bateria de 30 dias (re-treino do zero) e seus artefatos
experimentos/      varredura da janela e robustez (GroupKFold)
arquivo_morto/     material que NÃO faz parte do deliverable (ver abaixo)
```

Os notebooks **resolvem os caminhos sozinhos**: um cabeçalho no topo de cada um
sobe na árvore de diretórios até achar `comum/ssd_utils.py` e define
`COMUM_DIR`, `H7_DIR`, `H30_DIR`, `EXP_DIR`, `DATA_DIR` e `OUTPUT_DIR` — então
funcionam de qualquer diretório de trabalho, sem caminhos absolutos fixos.

### `comum/`
| Arquivo | Função |
|---|---|
| `ssd_utils.py` | Funções compartilhadas: carregamento, rotulagem (`create_class_labels`), splits por disco, features de janela deslizante (`build_windowed_features`, 24×5 = 120), limiar ótimo por F1, métricas, salvamento |
| `data.pickle` | 5343 discos × 360 dias × 24 atributos SMART |
| `mask.pickle` | Máscara de dias válidos (1 = dia válido; 0 = após a falha/padding) |

### `horizonte_7dias/` (bateria oficial — `contamination_level = 7`)
| Arquivo | Função |
|---|---|
| `AMA_projeto_LSTM.ipynb` | LSTM (PyTorch) sobre a sequência 3D → `lstm_7_dias.pth` |
| `AMA_projeto_LogisticRegression.ipynb` | Baseline linear tabular |
| `AMA_projeto_MLP.ipynb` | Rede neural tabular (não-linear, sem memória temporal) |
| `AMA_projeto_RandomForest.ipynb` | Ensemble de árvores tabular |
| `AMA_comparacao_modelos.ipynb` | Comparação dos 4 modelos + baseline trivial: tabelas, ROC/PR, confusão, tempos, curvas de aprendizagem, análise por disco |
| `resultados_*.pkl` | Métricas + predições de cada modelo (consumidos pela comparação e pelos experimentos) |
| `lstm_7_dias.pth` | Checkpoint do LSTM (consumido pela comparação) |
| `comp_*.png`, `*_learning_curve.png/.pkl`, `lstm_*.png`, `rf_feature_importance.png` | Figuras e dados das curvas |

### `horizonte_30dias/` (bateria de 30 dias — `contamination_level = 30`)
Mesma estrutura, com sufixo `_30d` (notebooks `*_30d.ipynb`, `resultados_*_30d.pkl`,
`lstm_30_dias.pth`, `comp_*_30d.png`). A comparação faz a inferência do LSTM-30d
a partir de `lstm_30_dias.pth`.

### `experimentos/`
| Arquivo | Função |
|---|---|
| `AMA_experimento_janela.ipynb` | Varredura `window ∈ {7,…,90}` (RF, na validação); embasou `WINDOW = 90` |
| `AMA_robustez_groupkfold.ipynb` | Variabilidade (AP/AUC, média ± dp) dos tabulares com `GroupKFold(5)` por disco |

### `arquivo_morto/` (não faz parte do deliverable)
- `relatorios_processo/` — relatórios internos do desenvolvimento (auditorias/revisões); não são necessários para usar o projeto.
- `modelos_treinados/` — os `modelo_*.pkl` (modelos treinados salvos). São **saídas regeneráveis** e **nenhum notebook os lê de volta** (o RF chega a ~170–240 MB cada) — ficam aqui para não pesar o compartilhamento.
- `logs_execucao/` — logs da última execução em lote.
- `AMA_experimento_horizonte.ipynb` (+ `.pkl`/`.png`) — versão antiga do experimento de horizonte, **substituída** pela bateria completa de 30 dias.
- `gerar_notebook_comparacao.py`, `plano.md` — artefatos antigos de desenvolvimento.

### Raiz
`README.md`, `APRESENTACAO.md` (roteiro da apresentação), `COMO_EXECUTAR.md`
(guia do executor em lote), `executar_tudo.py` / `executar_tudo.bat` (executor),
`ama_projeto.pdf` (enunciado).

## Ordem de execução

**Bateria de 7 dias (oficial):**
1. `horizonte_7dias/AMA_projeto_LSTM.ipynb` (PyTorch; idealmente em GPU) → `lstm_7_dias.pth`.
2. `horizonte_7dias/AMA_projeto_LogisticRegression / RandomForest / MLP.ipynb`
   (qualquer ordem) → `resultados_*.pkl`.
3. `experimentos/AMA_experimento_janela.ipynb` e `AMA_robustez_groupkfold.ipynb`
   (leem os `resultados_*.pkl` de 7 dias).
4. `horizonte_7dias/AMA_comparacao_modelos.ipynb`.

**Bateria de 30 dias:**
1. `horizonte_30dias/AMA_projeto_LogisticRegression_30d / RandomForest_30d / MLP_30d.ipynb`
   → `resultados_*_30d.pkl`.
2. `horizonte_30dias/AMA_projeto_LSTM_30d.ipynb` → `lstm_30_dias.pth`.
3. `horizonte_30dias/AMA_comparacao_modelos_30d.ipynb`.

As duas baterias são independentes; compare modelos **dentro** de cada horizonte,
nunca os números de 7 dias contra os de 30 dias.

## Execução automática (opcional)

Para rodar **todos os notebooks de uma vez** (exceto o treino do LSTM), em
processos/kernels isolados:

```
python executar_tudo.py
```

Detalhes, dependências e solução de problemas em `COMO_EXECUTAR.md`.

## Sobre o repositório no GitHub (o que está e o que não está)

Por limite de tamanho do GitHub (100 MB por arquivo) e por serem **regeneráveis**,
ficaram **fora do versionamento** (ver `.gitignore`):

- **Dados:** `comum/data.pickle` (~352 MB) e `comum/mask.pickle`.
- **Modelos e resultados:** todos os `*.pkl` (`resultados_*`, `modelo_*`,
  `*_learning_curve`) e os checkpoints `*.pth` do LSTM.
- **Atributo SMART (nota):** no dataset os atributos são nomeados
  `smart_00…smart_23` apenas pela **ordem das colunas** — **não** são os IDs
  SMART oficiais; o significado real de cada um ainda está sendo levantado.
- **Pastas auxiliares:** `projeto_colab/`, `arquivo_morto/`, `.venv/`,
  `logs_execucao/`.

O que **está** no repositório: o código (`ssd_utils.py`, `executar_tudo.py`),
**todos os notebooks com as saídas/figuras embutidas** (os resultados aparecem
direto no GitHub) e as figuras `.png`.

**Para re-executar a partir de um clone:**
1. Coloque `data.pickle` e `mask.pickle` em `comum/` (compartilhados à parte,
   ex.: Google Drive — não vão no git por serem grandes demais).
2. Para rodar as comparações sem retreinar tudo, coloque os checkpoints
   `lstm_7_dias.pth` / `lstm_30_dias.pth` em `horizonte_7dias/` /
   `horizonte_30dias/`.
3. Siga a *Ordem de execução* acima. (Os `*.pkl`/`*.png` são recriados ao rodar
   os notebooks.)
