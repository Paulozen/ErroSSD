#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Executa, em sequencia e de forma NAO-INTERATIVA, todos os notebooks do projeto
(exceto os de treino do LSTM), na ordem correta de dependencias.

Por que isto resolve o travamento entre notebooks:
cada notebook e executado num PROCESSO e KERNEL NOVOS (via `jupyter nbconvert
--execute`). Quando um notebook termina, o kernel e encerrado e toda a memoria
(dados ~370 MB, modelo do RF ~500 MB, features ~1,9 M x 120) e liberada antes do
proximo comecar. Alem disso, ha um timeout por celula: se alguma travar, aquele
notebook e abortado e o runner segue para o proximo, em vez de pendurar tudo.

Como usar (na maquina de destino):
    1. Abra um terminal NO MESMO ambiente/python que voce usa para rodar os
       notebooks no VSCode (o que tem torch, sklearn, etc.).
    2. Rode:
           python executar_tudo.py
       (ou clique duas vezes em executar_tudo.bat)

Os notebooks usam o kernel "python3" (o seu kernel com as dependencias) - este
script so precisa do pacote `nbconvert`, que ele instala automaticamente se
faltar. Cada notebook e salvo ja executado (--inplace): de manha, abra-os e os
resultados/figuras estarao la. Ha um log por notebook na pasta logs_execucao/ e
um resumo final em RESUMO_EXECUCAO.txt.

Opcoes:
    --timeout SEGUNDOS   Timeout por celula (padrao: 10800 = 3 h).
    --kernel NOME        Forca o nome do kernel (padrao: o do proprio notebook).
    --from N             Comeca a partir do notebook N (1-based) - util para
                         retomar apos uma falha sem refazer os anteriores.
    --stop-on-error      Para no primeiro notebook que falhar (padrao: continua).
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.abspath(__file__))

# Ordem de execucao (respeita as dependencias entre notebooks):
#  - LR/RF/MLP de 7d geram os resultados_*.pkl
#  - janela le o RF; robustez le os tres; comparacao 7d le os tres + lstm_7_dias.pth
#  - LR/RF/MLP de 30d geram os resultados_*_30d.pkl
#  - comparacao 30d le os tres _30d + lstm_30_dias.pth
NOTEBOOKS = [
    'horizonte_7dias/AMA_projeto_LogisticRegression.ipynb',
    'horizonte_7dias/AMA_projeto_RandomForest.ipynb',
    'horizonte_7dias/AMA_projeto_MLP.ipynb',
    'experimentos/AMA_experimento_janela.ipynb',
    'experimentos/AMA_robustez_groupkfold.ipynb',
    'horizonte_7dias/AMA_comparacao_modelos.ipynb',
    'horizonte_30dias/AMA_projeto_LogisticRegression_30d.ipynb',
    'horizonte_30dias/AMA_projeto_RandomForest_30d.ipynb',
    'horizonte_30dias/AMA_projeto_MLP_30d.ipynb',
    'horizonte_30dias/AMA_comparacao_modelos_30d.ipynb',
]

# Notebooks que fazem inferencia do LSTM (precisam de torch + do checkpoint .pth)
PRECISA_PTH = {
    'horizonte_7dias/AMA_comparacao_modelos.ipynb': 'horizonte_7dias/lstm_7_dias.pth',
    'horizonte_30dias/AMA_comparacao_modelos_30d.ipynb': 'horizonte_30dias/lstm_30_dias.pth',
}

LOG_DIR = os.path.join(RAIZ, 'logs_execucao')
RESUMO = os.path.join(RAIZ, 'RESUMO_EXECUCAO.txt')


def agora():
    return dt.datetime.now().strftime('%H:%M:%S')


def garantir_nbconvert():
    """Garante que o `nbconvert` esteja disponivel no python que roda este script."""
    try:
        import nbconvert  # noqa: F401
        return True
    except ImportError:
        print(f'[{agora()}] nbconvert nao encontrado - instalando '
              f'(python: {sys.executable})...', flush=True)
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'nbconvert'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print('ERRO ao instalar nbconvert. Saida do pip:\n' + r.stdout + r.stderr)
            print('\nInstale manualmente e rode de novo:\n'
                  f'    {sys.executable} -m pip install nbconvert')
            return False
        try:
            import importlib
            importlib.invalidate_caches()
            import nbconvert  # noqa: F401
            print(f'[{agora()}] nbconvert instalado com sucesso.', flush=True)
            return True
        except ImportError:
            print('nbconvert ainda nao importavel apos a instalacao. Verifique o ambiente.')
            return False


def listar_kernels():
    try:
        r = subprocess.run([sys.executable, '-m', 'jupyter', 'kernelspec', 'list'],
                           capture_output=True, text=True)
        return (r.stdout or '') + (r.stderr or '')
    except Exception as e:
        return f'(nao foi possivel listar kernels: {e})'


def executar_notebook(nb_rel, timeout, kernel, log_path):
    """Roda um notebook em processo/kernel novo. Retorna (ok, segundos)."""
    cmd = [sys.executable, '-m', 'jupyter', 'nbconvert',
           '--to', 'notebook', '--execute', '--inplace',
           f'--ExecutePreprocessor.timeout={timeout}',
           '--ExecutePreprocessor.startup_timeout=120']
    if kernel:
        cmd.append(f'--ExecutePreprocessor.kernel_name={kernel}')
    cmd.append(os.path.join(RAIZ, nb_rel))

    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write(f'$ {" ".join(cmd)}\n\n')
        log.flush()
        # cwd = RAIZ; o cabecalho dos notebooks resolve os caminhos sozinho
        proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace', bufsize=1)
        for linha in proc.stdout:
            log.write(linha)
            log.flush()
        proc.wait()
    return proc.returncode == 0, time.time() - t0


def fmt_dur(seg):
    m, s = divmod(int(seg), 60)
    h, m = divmod(m, 60)
    return f'{h:d}h{m:02d}m{s:02d}s' if h else f'{m:d}m{s:02d}s'


def main():
    ap = argparse.ArgumentParser(description='Executa todos os notebooks em sequencia.')
    ap.add_argument('--timeout', type=int, default=10800, help='Timeout por celula (s). Padrao: 10800.')
    ap.add_argument('--kernel', default=None, help='Nome do kernel a forcar. Padrao: o do notebook.')
    ap.add_argument('--from', dest='inicio', type=int, default=1, help='Comecar do notebook N (1-based).')
    ap.add_argument('--stop-on-error', action='store_true', help='Parar na primeira falha.')
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    print('=' * 70)
    print('  EXECUCAO EM SEQUENCIA DOS NOTEBOOKS (exceto treino do LSTM)')
    print('=' * 70)
    print(f'Python orquestrador : {sys.executable}')
    print(f'Raiz do projeto     : {RAIZ}')
    print(f'Timeout por celula  : {args.timeout}s ({fmt_dur(args.timeout)})')
    print(f'Kernel              : {args.kernel or "(o definido em cada notebook: python3)"}')
    print()

    # Preflight
    if not garantir_nbconvert():
        sys.exit(2)

    faltando = [nb for nb in NOTEBOOKS if not os.path.isfile(os.path.join(RAIZ, nb))]
    if faltando:
        print('ERRO: notebooks nao encontrados:')
        for nb in faltando:
            print('   -', nb)
        sys.exit(2)

    for nb, pth in PRECISA_PTH.items():
        if not os.path.isfile(os.path.join(RAIZ, pth)):
            print(f'AVISO: {pth} nao existe - "{nb}" provavelmente vai falhar '
                  f'na inferencia do LSTM (coloque o checkpoint no lugar).')

    print('\nKernels disponiveis nesta maquina:')
    print(listar_kernels())

    total = len(NOTEBOOKS)
    resultados = []
    t_inicio_geral = time.time()

    for i, nb in enumerate(NOTEBOOKS, start=1):
        if i < args.inicio:
            print(f'[{agora()}] ({i}/{total}) PULADO (--from {args.inicio}): {nb}')
            resultados.append((i, nb, 'PULADO', 0.0))
            continue

        nome_log = f'{i:02d}_' + nb.replace('/', '__').replace('\\', '__') + '.log'
        log_path = os.path.join(LOG_DIR, nome_log)
        print(f'[{agora()}] ({i}/{total}) >>> {nb}')
        print(f'            log: {os.path.relpath(log_path, RAIZ)}', flush=True)

        ok, seg = executar_notebook(nb, args.timeout, args.kernel, log_path)
        status = 'OK' if ok else 'FALHOU'
        print(f'[{agora()}] ({i}/{total}) {status} em {fmt_dur(seg)}: {nb}\n', flush=True)
        resultados.append((i, nb, status, seg))

        if not ok and args.stop_on_error:
            print('Parando por causa de --stop-on-error.')
            break

    # Resumo
    dur_total = time.time() - t_inicio_geral
    linhas = []
    linhas.append('=' * 70)
    linhas.append('RESUMO DA EXECUCAO  -  ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    linhas.append('=' * 70)
    for i, nb, status, seg in resultados:
        linhas.append(f'  [{status:>6}] ({i:02d}/{total}) {fmt_dur(seg):>9}  {nb}')
    n_ok = sum(1 for _, _, s, _ in resultados if s == 'OK')
    n_fail = sum(1 for _, _, s, _ in resultados if s == 'FALHOU')
    n_skip = sum(1 for _, _, s, _ in resultados if s == 'PULADO')
    linhas.append('-' * 70)
    linhas.append(f'  OK: {n_ok}   FALHOU: {n_fail}   PULADO: {n_skip}   '
                  f'Tempo total: {fmt_dur(dur_total)}')
    if n_fail:
        linhas.append('  Veja os logs em logs_execucao\\ para o traceback de cada falha.')
    linhas.append('=' * 70)
    texto = '\n'.join(linhas)
    print('\n' + texto)
    with open(RESUMO, 'w', encoding='utf-8') as f:
        f.write(texto + '\n')
    print(f'\nResumo salvo em {os.path.relpath(RESUMO, RAIZ)}')

    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
