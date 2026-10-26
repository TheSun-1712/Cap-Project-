# PowerShell Launcher for 3D CAD Metrology & Defect Inspection
param (
    [string]$ref = "0101BA302380N_010 XYZ.stp",
    [string]$test = "samples_3d/part_with_dent.stl",
    [double]$tolerance = 0.20,
    [string]$outJson = "results/3d_inspection_report.json",
    [string]$outHtml = "results/3d_inspection_view.html"
)

$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at $python"
    exit 1
}

& $python scripts\inspect_3d.py --ref $ref --test $test --tolerance $tolerance --out-json $outJson --out-html $outHtml
