@echo off
chcp 65001 > nul
set PYTHONUTF8=1
c:\Python311\python.exe -m cli.main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [translate_report] 正在翻譯報告...
    c:\Python311\python.exe translate_report.py
)
