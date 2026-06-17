# Como executar tudo em uma única vez (sem supervisão)

Guia completo para rodar **todos os notebooks do projeto em sequência**, numa
máquina só, deixando ligado (ex.: a noite toda), sem precisar mexer. Escrito para
o cenário "**tenho uma chance**": siga o **checklist pré-voo** antes de disparar.

`executar_tudo.py` executa os 10 notebooks na ordem correta de dependências.
**Não** treina o LSTM (os checkpoints `lstm_7_dias.pth` / `lstm_30_dias.pth` já
existem); as duas comparações entram, pois só fazem **inferência** com esses
checkpoints.

---

## TL;DR (receita rápida)

```bash
# 1. (recomendado) ambiente isolado, a partir da RAIZ do projeto
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 2. dependências
pip install --upgrade pip
pip install numpy pandas "scikit-learn>=1.5" matplotlib torch nbconvert ipykernel

# 3. registrar este ambiente como o kernel "python3" (que os notebooks usam)
python -m ipykernel install --user --name python3 --display-name "Python 3 (SSD)"

# 4. PRÉ-VOO: checa nbconvert, arquivos e kernels SEM rodar nada pesado
python executar_tudo.py --from 99

# 5. rodar tudo
python executar_tudo.py
```

No fim: notebooks já executados, figuras/`.pkl` salvos, um log por notebook em
`logs_execucao/` e um `RESUMO_EXECUCAO.txt` com OK/FALHOU/tempo de cada um.

---

## 1. Como funciona a orquestração de kernels

O `executar_tudo.py` **não executa o código dos notebooks dentro dele mesmo**.
Para cada notebook ele chama:

```
python -m jupyter nbconvert --to notebook --execute --inplace <notebook>
```

O `nbconvert` **sobe um kernel novo**, roda o notebook célula a célula nesse
kernel, salva o resultado e **encerra o kernel**. Só então o próximo notebook
começa. Consequências (todas desejáveis):

- **Sem acúmulo de memória** entre notebooks (dados ~370 MB + modelo RF ~500 MB +
  features ~1,9 M × 120 são liberados a cada troca) — era isso que travava o
  VSCode rodando manualmente.
- **Isolamento**: um erro/estado de um notebook não contamina o próximo.
- **Timeout por célula** (padrão 3 h): se uma célula travar, aquele notebook é
  abortado e o runner segue para o próximo.

**Qual kernel o nbconvert usa?** O definido nos metadados de cada notebook, que é
`python3`. Ou seja: o nbconvert procura, no registro de kernels do Jupyter, um
kernel chamado `python3` e usa o Python **dele** para executar. Por isso o passo
de **registrar o seu ambiente como `python3`** (TL;DR passo 3) é o que garante
que tudo rode no ambiente certo (com `torch`, `sklearn` etc.).

> O Python que **lança** o `executar_tudo.py` só precisa do `nbconvert`
> (+ `jupyter`/`ipykernel`). Quem realmente roda os modelos é o kernel `python3`.
> Mantê-los no **mesmo ambiente** (como na receita) é o jeito mais simples e à
> prova de erro.

---

## 2. Dependências

Instale **tudo no mesmo ambiente** que vai lançar o runner e servir de kernel:

```bash
pip install numpy pandas "scikit-learn>=1.5" matplotlib torch nbconvert ipykernel
```

| Pacote | Usado para | Quem precisa |
|---|---|---|
| `numpy` | arrays, dados | todos |
| `pandas` | features de janela (`ssd_utils`), importâncias | treino + comparação |
| `scikit-learn` (**≥1.5**) | LR/RF/MLP, busca, métricas, curva de aprendizagem | treino + experimentos + comparação |
| `matplotlib` | todas as figuras | todos |
| `torch` | inferência do LSTM | só as 2 comparações |
| `nbconvert` | orquestrar (rodar cada notebook) | o runner |
| `ipykernel` | fornecer/registrar o kernel `python3` | o kernel |

Observações:
- **`scikit-learn` recente** importa: os notebooks usam `penalty='elasticnet'` +
  `l1_ratio` e `precision_recall_curve`. Versão antiga pode quebrar. Use **≥1.5**.
- **`torch` em CPU basta** — as comparações só fazem *inferência* do LSTM (não
  treinam). **Não precisa de GPU.** (Instalar a versão CPU já resolve.)
- Se você **esquecer o `nbconvert`**, o próprio runner tenta instalá-lo
  automaticamente no início; mas é mais seguro já incluí-lo na lista acima.

---

## 3. Configurar o(s) kernel(s)

Os notebooks têm `kernelspec = "python3"`. Você precisa que esse nome aponte para
o ambiente onde instalou as dependências. Há duas formas:

**Forma A (recomendada) — registrar seu ambiente como `python3`:**
```bash
python -m ipykernel install --user --name python3 --display-name "Python 3 (SSD)"
```
Isso faz o kernel `python3` apontar para o Python atual (o do venv). Sobrescreve
um `python3` anterior, o que é o que queremos.

**Forma B — registrar com outro nome e dizer ao runner para usá-lo:**
```bash
python -m ipykernel install --user --name ssd --display-name "SSD"
python executar_tudo.py --kernel ssd
```

**Conferir se ficou certo:**
```bash
python -m jupyter kernelspec list
```
Deve listar `python3` (ou `ssd`) apontando para a pasta do seu ambiente. O
**pré-voo** (`--from 99`) também imprime essa lista para você verificar.

---

## 4. Arquivos que precisam estar no lugar

A estrutura do projeto já vem com eles; confirme antes de rodar:

- `comum/data.pickle`, `comum/mask.pickle` — dados de entrada.
- `horizonte_7dias/lstm_7_dias.pth` — checkpoint do LSTM de 7 dias.
- `horizonte_30dias/lstm_30_dias.pth` — checkpoint do LSTM de 30 dias.

Se algum `.pth` faltar, **só a comparação daquele horizonte falha** (as demais
rodam). O pré-voo avisa se um `.pth` estiver ausente.

---

## 5. Pré-voo (faça SEMPRE antes de deixar rodando)

```bash
python executar_tudo.py --from 99
```

Como `--from 99` pula todos os notebooks, isso **não roda nada pesado**, mas
executa todas as verificações iniciais:
- garante/instala o `nbconvert`;
- confirma que os 10 notebooks existem;
- avisa se algum `.pth` está faltando;
- imprime os kernels disponíveis (confira que `python3` aponta para o seu
  ambiente).

Se o pré-voo passar limpo, você está pronto para o disparo de verdade.

> Quer uma checagem ainda mais forte? Rode **um** notebook leve de ponta a ponta
> primeiro: `python executar_tudo.py --from 1` e interrompa após o 1º terminar
> (LR de 7 dias costuma ser rápido). Se ele gerar `resultados_logistic_regression.pkl`
> sem erro, o ambiente está 100%.

---

## 6. Disparar a execução completa

A partir da **raiz do projeto**, com o ambiente ativo:

```bash
python executar_tudo.py
```

No Windows também dá para **duplo-clique em `executar_tudo.bat`** — mas só
funciona se o `python` do PATH for o ambiente com as dependências. Rodar pelo
terminal com o venv ativado é mais garantido.

### Para uma noite realmente sem supervisão
- **Impeça a máquina de dormir** (senão o processo congela):
  - Windows: *Configurações → Sistema → Energia* → "Suspensão: Nunca" (ao menos
    ligada na tomada). Via terminal (admin): `powercfg /change standby-timeout-ac 0`.
  - Linux: desative suspensão automática / `systemd-inhibit`.
- **Mantenha o terminal aberto** (o processo morre se fechar a janela). Em
  Linux/Mac dá para usar `nohup python executar_tudo.py &` e acompanhar por
  `tail -f RESUMO_EXECUCAO.txt` / `logs_execucao/`.
- Garanta **espaço em disco** (alguns GB; os notebooks regravam `.pkl`/`.png`) e
  que o **carregador** esteja conectado.

---

## 7. O que acontece / saídas

- Cada notebook é salvo **já executado** (`--inplace`): de manhã, abra e os
  resultados/figuras estarão embutidos.
- Os artefatos normais são gravados nas pastas de cada notebook
  (`resultados_*.pkl`, `modelo_*.pkl`, `comp_*.png`, `*_learning_curve*.png/.pkl`
  etc.).
- `logs_execucao/NN_<notebook>.log` — saída completa de cada notebook (com o
  traceback, se falhar).
- `RESUMO_EXECUCAO.txt` — tabela final: OK / FALHOU / PULADO + tempo de cada um.

---

## 8. Ordem executada e dependências

1. `horizonte_7dias/` — LR → RF → MLP  (geram os `resultados_*.pkl` de 7d)
2. `experimentos/` — janela → robustez  (leem os `resultados_*.pkl` de 7d)
3. `horizonte_7dias/AMA_comparacao_modelos` — inferência LSTM-7d + figuras
4. `horizonte_30dias/` — LR-30d → RF-30d → MLP-30d  (geram os `*_30d.pkl`)
5. `horizonte_30dias/AMA_comparacao_modelos_30d` — inferência LSTM-30d + figuras

A ordem respeita as dependências: cada comparação roda **depois** dos treinos do
seu horizonte (ela lê os `resultados_*.pkl` e os `*_learning_curve*.pkl` que os
treinos geram). Se um notebook de treino falhar, os que dependem dele falham
rápido (por falta do `.pkl`); o runner **continua** e registra tudo. Para
retomar após corrigir, use `--from N`.

---

## 9. Tempo esperado

Roda em algumas horas (depende da máquina). Pontos mais pesados:
- **RF** (7d e 30d): busca em subamostra + fit final em ~1,2 M linhas.
- **MLP** (7d e 30d): idem + a curva de aprendizagem **re-treina até 100% do
  treino** (~2× um treino).
- As comparações são rápidas (inferência + figuras).

O timeout padrão por célula é 3 h — folgado para todos esses passos. Aumente com
`--timeout` se a sua máquina for muito lenta.

---

## 10. Opções

| Opção | Efeito |
|---|---|
| `--from N` | Começa no notebook N (1–10), pulando os anteriores. `--from 99` = só pré-voo. |
| `--timeout SEG` | Timeout por célula em segundos (padrão 10800 = 3 h). |
| `--stop-on-error` | Para na primeira falha (padrão: continua e registra). |
| `--kernel NOME` | Força um kernel específico (padrão: o `python3` do notebook). |

---

## 11. Solução de problemas

| Sintoma no log | Causa provável | Correção |
|---|---|---|
| `No module named 'torch'` nas comparações | kernel sem PyTorch | `pip install torch` no ambiente do kernel; reregistrar com `ipykernel`. |
| Erro em `elasticnet`/`l1_ratio`/`precision_recall_curve` | scikit-learn antigo | `pip install -U "scikit-learn>=1.5"`. |
| `No such kernel named python3` | kernel não registrado | rode o `ipykernel install` (seção 3) ou use `--kernel <nome>`. |
| Roda no Python errado (sem pacotes) | `python3` aponta para outro ambiente | reregistre o `python3` no ambiente certo (seção 3) e confira com `kernelspec list`. |
| `data.pickle`/`.pth` não encontrado | arquivo ausente | confirme a seção 4; o pré-voo avisa. |
| Tudo "congela" de madrugada | máquina dormiu | desative suspensão (seção 6). |
| Travou no RandomForest sem imprimir `[CV] END` | (já corrigido) oversubscription de CPU | os notebooks já usam busca em subamostra + `n_jobs` em um nível só. |

> Importante: este runner roda a **árvore local** (raiz do projeto), não a cópia
> `projeto_colab/`. No Google Colab a recomendação é outra (um notebook por vez)
> — veja `projeto_colab/README_COLAB.md`.
