$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Environnement absent. Suivre la section Installation du README.md.'
}
& $projectPython -m streamlit run app.py --server.address 127.0.0.1
