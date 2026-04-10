#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Activate uqala conda environment and run Python scripts

.DESCRIPTION
    This script activates the conda environment and runs any Python
    script with the correct environment variables and PATH.

.PARAMETER Script
    Path to the Python script to run

.PARAMETER Arguments
    Arguments to pass to the script

.EXAMPLE
    .\run_in_venv.ps1 pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Script,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Set up environment
$env:CONDA_DEFAULT_ENV = "uqala"
$env:CONDA_PREFIX = "C:\Users\augus\.conda\envs\uqala"
$env:PATH = "C:\Users\augus\.conda\envs\uqala;C:\Users\augus\.conda\envs\uqala\Library\mingw-w64\bin;C:\Users\augus\.conda\envs\uqala\Library\usr\bin;C:\Users\augus\.conda\envs\uqala\Library\bin;C:\Users\augus\.conda\envs\uqala\Scripts;C:\Users\augus\.conda\envs\uqala\bin;$env:PATH"

Write-Host "Activating conda environment: uqala" -ForegroundColor Green
Write-Host "Python: $env:CONDA_PREFIX\python.exe" -ForegroundColor Cyan
Write-Host "Script: $Script" -ForegroundColor Cyan

if ($Arguments) {
    Write-Host "Arguments: $($Arguments -join ' ')" -ForegroundColor Cyan
}

Write-Host ""

# Run the script
& "$env:CONDA_PREFIX\python.exe" $Script @Arguments
