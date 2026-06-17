# Guia de Apresentação — Predição de Falhas em SSDs

Roteiro de tópicos para a apresentação (30 min, 4 integrantes). Ordenado do
contexto até a discussão final, com os pontos a destacar e os **números reais**
do projeto. Ao final há (a) uma **seção à parte explicando cada feature** e as
agregações de janela, e (b) uma **sugestão de divisão entre os 4 apresentadores**.

---

## Mensagem central (a "história" que amarra tudo)

> O mesmo dado SMART é dado a modelos de famílias diferentes, organizados em
> **dois eixos**: **linear × não-linear** e **tabular (vetor fixo) × sequencial
> (memória temporal)**. A comparação isola cada ganho:
> **LR → (MLP, RF)** mede o ganho de **não-linearidade**;
> **(MLP, RF) → LSTM** mede o ganho de **modelagem sequencial**.
> Resultado: a não-linearidade ajuda **pouco**; a modelagem sequencial é
> **dominante** (o LSTM dispara na frente).

Tudo na apresentação deve voltar a essa frase.

---

## 1. Contexto do problema

- **Objetivo:** prever, para cada disco SSD e cada dia, se ele vai **falhar nos
  próximos N dias** (manutenção preditiva). Antecipar a falha permite trocar/
  fazer backup antes da perda de dados.
- **Por que é difícil:** falha é evento **raro** (forte desbalanceamento), e o
  sinal de degradação é sutil e se desenrola ao longo do tempo.
- **Por que importa (contexto da disciplina):** é um problema real de
  classificação binária sobre séries temporais, com desbalanceamento extremo —
  ótimo para discutir **representação de dados**, **escolha de métricas** e
  **famílias de modelos**.

## 2. O dataset e sua organização

- **Formato:** `data.pickle` = tensor **(5343 discos × 360 dias × 24 atributos
  SMART)**; `mask.pickle` = **(5343 × 360)** marcando os dias válidos.
- **Atributos SMART:** 24 indicadores de saúde do disco (ver *Dicionário de
  features*), medidos por dia.
- **Máscara / tempo de vida:** cada disco tem um número de dias válidos
  (`mask == 1`) seguido de preenchimento (`mask == 0`). **Todos os discos do
  dataset falham** — a série termina na falha (truncada em no máximo 360 dias).
  Disco que sobrevive aos 360 dias "falha" no último dia observado.
- Destacar que **não existe disco "saudável para sempre"** no dado: o que muda é
  *quando* ele falha. (Esse foi um ponto de interpretação que a equipe alinhou.)

## 3. Formulação como aprendizado supervisionado

- **Rótulo (`create_class_labels`):** marca como **1** os últimos
  `contamination_level` dias válidos antes da falha; o resto é **0**.
  Configuração oficial: **`contamination_level = 7`** ("vai falhar em ≤ 7 dias").
- **Desbalanceamento:** com 7 dias, só **~2,1%** das amostras são positivas
  (no conjunto de teste). Isso guia toda a escolha de métricas (Tópico 7).
- **Divisão treino/teste/validação por DISCO** (não por linha): discos
  `0–3739` treino, `3740–4540` teste, `4541–5342` validação. Crucial: separar
  **discos inteiros** evita vazamento (dias do mesmo disco não aparecem em
  treino e teste ao mesmo tempo). O **mesmo split** é usado por todos os
  modelos, para comparação justa.

## 4. Como cada modelo "enxerga" os dados (o coração do trabalho)

Mesmo dado, **duas representações**:

- **Entrada 3D (sequencial) — LSTM:** recebe a sequência inteira do disco,
  `(dias × 24 atributos)`. O modelo processa dia a dia e **carrega memória** do
  passado — enxerga tendência e ordem temporal nativamente.
- **Entrada 2D (tabular) — LR, MLP, RF:** cada amostra é um par `(disco, dia)`
  resumido num **vetor fixo**. Como esses modelos não veem sequência, damos a
  eles um resumo dos últimos **`WINDOW = 90`** dias por **janela deslizante**:
  para cada atributo, 5 estatísticas — **valor atual, média, desvio-padrão (std),
  mínimo, máximo** → **24 × 5 = 120 features**.
  - A janela é **causal** (só usa passado, nunca o futuro) e **não cruza
    discos**.
  - `WINDOW = 90` foi a janela escolhida num experimento de varredura (testando
    7, 14, 30, 60, 90) — janelas maiores aproximam a estatística de "desde o
    início", diluindo o contraste local.
- **Mensagem:** a janela deslizante é a "ponte" que dá contexto temporal a
  modelos tabulares — mas é um resumo, não a sequência; o LSTM não precisa dela.

## 5. Os modelos e o que cada um representa

| Modelo | Eixo | Papel na comparação |
|---|---|---|
| **Regressão Logística (LR)** | linear, tabular | baseline; fronteira linear sobre as 120 features |
| **MLP** | não-linear, tabular | mesma entrada da LR, mas com não-linearidade (rede densa) |
| **Random Forest (RF)** | não-linear, tabular | ensemble de árvores; captura interações; dá importância de features |
| **LSTM** | não-linear, sequencial | rede recorrente sobre a sequência 3D; é o "algoritmo externo à disciplina" |

- LR e MLP exigem **padronização** das features (StandardScaler ajustado **só no
  treino**); RF é invariante a escala (não precisa).
- LSTM: 3 camadas LSTM (24→256→128→64) + camadas densas (64→32→16→1), perda
  **BCE mascarada** (ignora o padding), treino por épocas com *early stopping*.

## 6. Busca de hiperparâmetros e seleção do limiar

- **Busca:** `RandomizedSearchCV` com `scoring='average_precision'` (mesma
  métrica da comparação — evita o viés do corte fixo de 0,5 do F1). Feita numa
  **subamostra estratificada (~300 mil linhas)** por custo, e o **ajuste final é
  no treino completo**.
- **Seleção de limiar:** o limiar de decisão é escolhido **na validação**
  (maximizando F1 na curva precisão-recall) e só então aplicado ao teste —
  nunca se "espia" o teste para calibrar.
- **Ponto didático forte (mostrar):** o RF teve **AP≈0,59 na validação cruzada
  da busca**, mas **AP≈0,17 no teste**. Por quê? A CV interna embaralha linhas
  (dias do mesmo disco podem cair em treino e validação) → otimista; o teste
  segura **discos inteiros** → honesto. Isso **valida a escolha do split por
  disco**.

## 7. Avaliação: por que não usar acurácia

- Com ~2% de positivos, **prever "nunca falha" dá ~98% de acurácia** e **0 de
  recall** — inútil. Mostrar a linha **Baseline (tudo negativo)**: acurácia
  0,979 e **F1 = 0**.
- Métricas usadas:
  - **AP (Average Precision)** — área sob a curva precisão-recall; baseline
    aleatório = prevalência (~0,02). **Métrica principal**, ideal para
    desbalanceamento.
  - **AUC-ROC** — capacidade de ranqueamento; baseline aleatório = 0,5.
  - **F1 / precisão / recall** — no limiar escolhido (recall = "quantas falhas
    eu pego"; precisão = "quantos alarmes são verdadeiros").
- AP e AUC **não dependem do limiar** (avaliam o ranking das probabilidades) —
  por isso são a base da comparação entre modelos.

## 8. Resultados (números reais — conjunto de TESTE, 7 dias)

| Modelo | AP | AUC | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| **LSTM** | **0,910** | **0,993** | **0,911** | 0,996 | 0,840 |
| Random Forest | 0,174 | 0,787 | 0,237 | 0,281 | 0,205 |
| MLP | 0,168 | 0,782 | 0,238 | 0,250 | 0,227 |
| Regressão Logística | 0,087 | 0,723 | 0,132 | 0,181 | 0,103 |
| Baseline (tudo negativo) | ~0,021 | 0,500 | 0,000 | 0,000 | 0,000 |

Leitura (amarrar na mensagem central):
- **LSTM domina** (AP 0,91 vs ≤0,17 dos tabulares): o ganho de **modelagem
  sequencial** é enorme.
- Entre tabulares: **RF ≈ MLP > LR**. A **não-linearidade** (LR→MLP/RF) ajuda,
  mas **pouco** (AP de ~0,09 para ~0,17). RF e MLP praticamente empatam.
- Todos **muito acima do baseline** em AP/AUC — então aprenderam sinal real,
  mesmo os tabulares; só que o teto da representação tabular é baixo.

## 9. Importância das features (interpretação — RF)

- O RF expõe quais features mais pesam. **Domínio claro do atributo `smart_05`**:
  suas 5 agregações (std, max, atual, média, min) aparecem todas no topo, com o
  **`smart_05_std` em 1º** (≈0,046). Depois vêm `smart_19`, `smart_23`,
  `smart_22`, `smart_00`.
- **Interpretação:** que o **desvio-padrão** (variabilidade recente) de um
  atributo seja a feature nº1 faz sentido — degradação aparece como **mudança/
  instabilidade** dos valores, não só pelo valor absoluto.
- **Importante (a levantar):** `smart_05` é o atributo na **6ª posição** do
  dataset (índice 5, contando do 0) — **não é o "SMART ID 5"**. Os nomes
  `smart_00…smart_23` refletem apenas a **ordem das colunas** no dataset, não o
  número oficial do atributo SMART. A equipe **ainda está levantando** a que
  atributo real cada coluna corresponde — **não afirmem o significado sem
  confirmar** (ver *Dicionário de features*).

## 10. Análise operacional por disco (visão prática)

- As métricas por amostra subestimam a utilidade real: na prática, um disco é
  "salvo" com **um único alarme a tempo** dentro da janela de risco.
- Reportamos, por disco: **taxa de detecção** (% de discos com ≥1 alarme certo),
  **falsos alarmes por disco** e **antecedência** (dias entre o 1º alarme e a
  falha). Ex. (7 dias): o LSTM detecta ~84% dos discos com **antecedência média
  ~6 dias** e quase zero falso alarme.
- Mensagem: a leitura **operacional** muda a conversa de "F1 baixo" para "quantos
  discos eu realmente salvo e com quanta antecedência".

## 11. Horizonte de 30 dias (análise de sensibilidade)

- Refizemos tudo com `contamination_level = 30` ("falha em ≤ 30 dias").
- **Resultados (teste, 30 dias):** LSTM AP **0,962**; RF **0,308** > MLP 0,235 >
  LR 0,201. O ranking se mantém (LSTM ≫ RF ≥ MLP > LR).
- **Caveat obrigatório:** **não comparar números entre horizontes** — a
  prevalência sobe de ~2% (7d) para ~9% (30d), o que **infla** F1/AP
  mecanicamente. Comparar modelos **dentro** de cada horizonte.
- Utilidade: 30 dias dá **mais antecedência** para agir, ao custo de uma
  fronteira de rótulo mais "longe" da falha.

## 12. Conclusões, limitações e trabalhos futuros

- **Conclusões:** (1) modelagem **sequencial** (LSTM) é o fator decisivo; (2)
  não-linearidade dá ganho real mas pequeno entre os tabulares; (3) acurácia é
  enganosa — AP/AUC + baseline contam a história certa; (4) split por disco e
  limiar na validação foram essenciais para números honestos.
- **Limitações:** janela tabular é um resumo (perde ordem fina); todos os discos
  do dataset falham (não há classe "saudável" verdadeira); custo de treino do
  LSTM é alto.
- **Futuro:** outras arquiteturas sequenciais (GRU/Transformer), features de
  tendência (inclinação), calibração de probabilidade, e validar em discos de
  outros fabricantes.

---

# Seção à parte — Dicionário de features

## O que é SMART
**SMART** (*Self-Monitoring, Analysis and Reporting Technology*) é o sistema
embutido no disco que reporta **indicadores de saúde** (contadores de erro,
desgaste, temperatura, etc.). Cada atributo SMART tem um **ID** e um valor por
dia. O dataset traz **24 atributos** por disco por dia, nomeados `smart_00` …
`smart_23`.

> **Atenção — os nomes NÃO são os IDs SMART:** `smart_XX` indica apenas a
> **posição/ordem da coluna no dataset** (`smart_05` = 6ª coluna, índice 5), e
> **não** o número oficial do atributo SMART. **O significado de cada coluna
> ainda está sendo levantado pela equipe** — confirmem antes de nomear qualquer
> feature na apresentação. A tabela abaixo é só uma **referência** dos atributos
> SMART mais associados a falha em SSDs, para ajudar nesse mapeamento.

| ID SMART | Nome | O que indica (relação com falha) |
|---|---|---|
| 5 | Reallocated Sectors Count | setores defeituosos remapeados — **forte preditor** |
| 9 | Power-On Hours | horas ligado (idade/uso) |
| 12 | Power Cycle Count | nº de ciclos liga/desliga |
| 171 | Program Fail Count | falhas de gravação (NAND) |
| 172 | Erase Fail Count | falhas de apagamento (NAND) |
| 173 | Wear Leveling Count | desgaste médio das células (ciclos P/E) |
| 174 | Unexpected Power Loss | quedas de energia inesperadas |
| 177 | Wear Range Delta | desigualdade de desgaste entre blocos |
| 187 | Reported Uncorrectable Errors | erros não corrigíveis — **forte preditor** |
| 194 | Temperature | temperatura |
| 197 | Current Pending Sector Count | setores instáveis aguardando remapeamento — **forte** |
| 198 | Offline Uncorrectable | setores irrecuperáveis — **forte** |
| 199 | UDMA CRC Error Count | erros de transmissão na interface |
| 202 / 231 / 233 | Media Wearout / SSD Life Left | vida útil restante da NAND |
| 241 / 242 | Total LBAs Written / Read | volume total escrito/lido |

(Lista de **referência**, não o mapeamento do dataset: os 24 atributos
`smart_00…smart_23` ainda precisam ser mapeados para os atributos reais — tarefa
em aberto, pois os nomes são só a ordem das colunas.)

## O que significam os sufixos (agregações da janela)
Cada atributo SMART vira **5 features**, resumindo os últimos `WINDOW = 90` dias
daquele disco até o dia atual:

- **`_atual`** — o **valor do dia** (a leitura SMART no instante da amostra).
- **`_media`** — **média** dos últimos 90 dias (nível típico recente).
- **`_std`** — **desvio-padrão** dos últimos 90 dias: **o quanto o valor variou/
  oscilou** na janela. Alto = instabilidade (sinal de degradação).
- **`_min`** — **menor** valor na janela.
- **`_max`** — **maior** valor na janela (picos recentes).

Exemplo de leitura: `smart_05_std` alto = o atributo da **6ª coluna** (qual seja
— a confirmar) **andou variando** nas últimas semanas → indício de degradação.
Por isso ele é a feature mais importante para o RF, mesmo sem sabermos ainda o
nome oficial dele.

---

# Sugestão de divisão (4 apresentadores, ~30 min)

Cada bloco ~7 min + ~2 min de transição/perguntas no fim. Ajustem ao estilo de
vocês; o importante é que **cada um amarre seu trecho na mensagem central**.

**Apresentador 1 — Problema e dados (Tópicos 1–3) [~7 min]**
- Contexto e motivação (manutenção preditiva), por que é difícil (raro/temporal).
- Organização do dataset (5343 × 360 × 24, máscara, "todos falham").
- Formulação: rótulo "falha em N dias", desbalanceamento ~2%, split **por disco**.
- *Gancho:* "como mostrar esse mesmo dado para modelos tão diferentes?" → passa pro 2.

**Apresentador 2 — Representação e modelos tabulares (Tópicos 4–6) [~8 min]**
- Os dois eixos (linear×não-linear, tabular×sequencial) e a entrada **2D vs 3D**.
- **Janela deslizante**: 24×5 = 120 features, causal, `WINDOW = 90`.
- LR, MLP, RF (o que cada um traz; padronização só p/ LR e MLP).
- Busca de hiperparâmetros (AP, subamostra + fit completo) e **limiar na
  validação**; o caso CV 0,59 vs teste 0,17 como lição de vazamento.

**Apresentador 3 — LSTM, avaliação e resultados (Tópico 5-LSTM, 7, 8) [~8 min]**
- LSTM: entrada 3D, arquitetura, BCE mascarada, treino por épocas (curva de
  perda completa, melhor época).
- **Por que AP/AUC e não acurácia** + baseline "tudo negativo".
- **Tabela de resultados (7 dias)** e a leitura: LSTM domina; não-linearidade
  ajuda pouco. (Mostrar ROC, PR e barras com baseline.)

**Apresentador 4 — Interpretação e discussão (Tópicos 9–12) [~7 min]**
- **Importância das features** (smart_05 e o porquê do `_std`) + remeter ao
  *Dicionário de features*.
- **Análise por disco** (detecção, antecedência) — visão operacional.
- **Horizonte de 30 dias** (sensibilidade + caveat de prevalência).
- **Conclusões, limitações e futuro.** Fechar repetindo a mensagem central.

> Figuras de apoio (nas pastas `horizonte_7dias/` e `horizonte_30dias/`):
> `comp_metricas_barras*` (com baseline), `comp_roc*`, `comp_pr*`,
> `comp_confusao*`, `comp_tempo_treino*`, `comp_learning_curves*`,
> `comp_rf_feature_importance*` (importância de features),
> `lstm_loss_history*` / `lstm_train_loss*` (curva de perda do LSTM).
