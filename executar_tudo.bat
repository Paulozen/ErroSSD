@echo off
REM Executa todos os notebooks em sequencia (exceto treino do LSTM).
REM Rode a partir do ambiente/python que voce usa para os notebooks no VSCode.
cd /d "%~dp0"
echo Iniciando execucao em sequencia... (isso pode levar varias horas)
echo.
python executar_tudo.py %*
echo.
echo ===== Execucao finalizada. Veja RESUMO_EXECUCAO.txt e a pasta logs_execucao\ =====
pause
