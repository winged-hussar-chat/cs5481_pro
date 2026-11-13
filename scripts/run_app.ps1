param([string]$Python = 'D:/PythonEnvs/cs5481_pro/Scripts/python.exe')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python)) { throw 'Provide the project Python path with -Python.' }
& $Python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
