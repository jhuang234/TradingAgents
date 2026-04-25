@echo off
chcp 65001 > nul
set PYTHONUTF8=1

if "%~1"=="" (
    echo Usage: %~nx0 TICKER
    exit /b 1
)

c:\Python311\python.exe -m cli.main --ticker %~1 --auto-defaults --llm-provider google --quick-model gemini-3.1-flash-lite-preview --deep-model gemini-3.1-flash-lite-preview --google-thinking-level high
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [translate_report] Translating report...
    c:\Python311\python.exe translate_report.py
)
