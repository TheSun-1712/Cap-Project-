# PowerShell helper to run inspection with the project's virtual environment
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment not found at $PythonExe"
    exit 1
}

& $PythonExe (Join-Path $ScriptDir "scripts\inspect_part.py") @args
