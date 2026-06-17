"""
Funções compartilhadas pelos notebooks de Regressão Logística e Random Forest
do projeto de predição de falhas em SSDs.

Reproduz, de forma idêntica, a formulação de problema usada no notebook
AMA_projeto_LSTM.ipynb (split de índices, rótulos de "falha em N dias" e
métricas de avaliação), para garantir que a comparação entre os três modelos
(LSTM, Regressão Logística, Random Forest) seja justa.

IMPORTANTE: a função `create_class_labels` é uma cópia literal da função
definida na célula 8 de AMA_projeto_LSTM.ipynb. Mantenha-a sincronizada caso
aquele notebook seja alterado.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve,
)

N_TIMESTEPS = 360

# Mesma divisão de índices usada no notebook do LSTM
# (AMA_projeto_LSTM.ipynb, células 5-7: train_data = all_data[0:3740], etc.)
TRAIN_RANGE = (0, 3740)
TEST_RANGE = (3740, 4541)
VALIDATION_START = 4541


def load_dataset(data_dir='.'):
    """ Carrega data.pickle e mask.pickle do diretório informado. """
    with open(os.path.join(data_dir, 'data.pickle'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(data_dir, 'mask.pickle'), 'rb') as f:
        mask = pickle.load(f)
    return data, mask


def split_indices(n_discos):
    """ Índices de disco de cada partição, idênticos aos do notebook do LSTM. """
    return {
        'train': np.arange(TRAIN_RANGE[0], TRAIN_RANGE[1]),
        'test': np.arange(TEST_RANGE[0], TEST_RANGE[1]),
        'validation': np.arange(VALIDATION_START, n_discos),
    }


def create_class_labels(data, mask, contamination_level):
    """
    Cópia literal da função da célula 8 de AMA_projeto_LSTM.ipynb.

    Rotula como 1 os últimos `contamination_level` dias válidos antes da
    falha (ou todos os dias válidos, se o disco viveu menos que isso) e como
    0 o restante. "Falha" = primeiro timestep com mask == 0; discos que
    viveram os 360 dias falham no último timestep.
    """
    class_labels = []

    for i in range(len(data)):
        zero_list = np.zeros(N_TIMESTEPS)

        try:
            first_zero = np.where(mask[i] == 0)[0][0]
        except IndexError:
            first_zero = N_TIMESTEPS

        try:
            zero_list[first_zero - contamination_level:first_zero] = [1] * contamination_level
        except ValueError:
            # SSDs com tempo de vida menor que o nível de contaminação
            zero_list[first_zero - first_zero:first_zero] = [1] * (first_zero)

        class_labels.append(zero_list)

    return np.asarray(class_labels)


def build_windowed_features(data, mask, window=14):
    """
    Modelos tabulares (LR, RF) não enxergam a sequência inteira do disco como
    o LSTM faz — então cada amostra (disco, dia) recebe um resumo dos últimos
    `window` dias (ou menos, perto do início da série, quando ainda não há
    `window` dias de histórico):

        [valor atual, média, desvio padrão, mínimo, máximo] por atributo SMART

    O tamanho da janela é tratado como hiperparâmetro (ver
    AMA_experimento_janela.ipynb). Todos os blocos são causais: nenhum usa
    informação de timesteps futuros. A máscara é monotônica (prefixo de 1s
    seguido de sufixo de 0s), então a janela de um timestep válido nunca
    inclui padding.

    Retorna:
        X            : ndarray (n_discos*360, 24*5) float32, ordem (disco, timestep)
        disco_id     : ndarray (n_discos*360,) int
        timestep     : ndarray (n_discos*360,) int
        feature_cols : list[str] com o nome de cada coluna de X
    """
    n_discos, n_timesteps, n_features = data.shape
    feat_names = [f'smart_{j:02d}' for j in range(n_features)]

    disco_id = np.repeat(np.arange(n_discos), n_timesteps)
    timestep = np.tile(np.arange(n_timesteps), n_discos)

    df = pd.DataFrame(data.reshape(-1, n_features), columns=feat_names)
    df.insert(0, 'timestep', timestep)
    df.insert(0, 'disco_id', disco_id)

    grouped = df.groupby('disco_id', sort=False)[feat_names]
    roll = grouped.rolling(window=window, min_periods=1)

    blocks = {
        'atual': df[feat_names],
        'media': roll.mean().reset_index(level=0, drop=True),
        'std': roll.std().reset_index(level=0, drop=True).fillna(0.0),
        'min': roll.min().reset_index(level=0, drop=True),
        'max': roll.max().reset_index(level=0, drop=True),
    }

    feature_cols = []
    arrays = []
    for suffix, block in blocks.items():
        feature_cols.extend(f'{c}_{suffix}' for c in feat_names)
        arrays.append(block.to_numpy(dtype=np.float32))

    X = np.concatenate(arrays, axis=1)
    return X, disco_id, timestep, feature_cols


def best_threshold_f1(y_true, y_proba):
    """Limiar que maximiza F1 (curva PR). Escolha na VALIDACAO e aplique ao TESTE."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def select_split(X, disco_id, timestep, y_flat, mask_flat, split_disco_ids):
    """
    Filtra as linhas pertencentes aos discos de `split_disco_ids` e descarta
    o padding (mask == 0) — mesmo comportamento do `batch_evaluation` do
    notebook LSTM, que só calcula métricas em `mask == 1`.
    """
    in_split = np.isin(disco_id, split_disco_ids)
    valid = mask_flat.astype(bool)
    keep = in_split & valid
    return {
        'X': X[keep],
        'y': y_flat[keep],
        'disco_id': disco_id[keep],
        'timestep': timestep[keep],
    }


def evaluate_predictions(y_true, y_pred):
    """ Mesmo conjunto de métricas reportado no notebook do LSTM. """
    accuracy = float((y_true == y_pred).mean())
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
    }


def print_metrics(nome, metrics):
    print(f'--- {nome.upper()} ---')
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print('Matriz de confusão ([[TN, FP], [FN, TP]]):')
    print(metrics['confusion_matrix'])


def save_results(path, **kwargs):
    """
    Salva um dicionário de resultados em pickle — formato comum para os três
    modelos, usado depois para gerar gráficos comparativos (ROC, PR, barras
    de métricas) na apresentação.
    """
    kwargs['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'wb') as f:
        pickle.dump(kwargs, f)
    print(f'Resultados salvos em: {path}')
