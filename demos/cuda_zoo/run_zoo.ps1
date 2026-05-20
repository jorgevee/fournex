#Requires -Version 5.1
<#
.SYNOPSIS
    Compile and analyse all CUDA antipattern zoo pairs with frx compare.
.DESCRIPTION
    For each bad/good pair: compiles both with nvcc, runs `frx compare bad.cu good.cu`,
    and prints a summary header. Does not require a GPU to run static analysis.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$zoo_root = $PSScriptRoot
$pairs = @(
    @{ dir = "01_uncoalesced";     label = "Uncoalesced memory access" },
    @{ dir = "02_matmul_notiled";  label = "Naive GEMM (no tiling)" },
    @{ dir = "03_excess_sync";     label = "Excess __syncthreads()" },
    @{ dir = "04_register_pressure"; label = "High register pressure" }
)

# Check prerequisites
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Warning "nvcc not found on PATH — skipping compilation step."
    $skip_compile = $true
} else {
    $skip_compile = $false
}

if (-not (Get-Command frx -ErrorAction SilentlyContinue)) {
    Write-Error "frx not found. Install with: pip install -e backend/python"
}

foreach ($pair in $pairs) {
    $dir   = Join-Path $zoo_root $pair.dir
    $label = $pair.label
    $bad   = Join-Path $dir "bad.cu"
    $good  = Join-Path $dir "good.cu"

    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host "  $label"
    Write-Host ("=" * 60)

    if (-not $skip_compile) {
        Write-Host "  Compiling bad.cu ..."
        nvcc -O2 -o (Join-Path $dir "bad_bin")  $bad  2>&1 | Out-Null
        Write-Host "  Compiling good.cu ..."
        nvcc -O2 -o (Join-Path $dir "good_bin") $good 2>&1 | Out-Null
    }

    Write-Host ""
    Write-Host "  frx compare $($pair.dir)/bad.cu  $($pair.dir)/good.cu"
    Write-Host ""
    & frx compare $bad $good
}

Write-Host ""
Write-Host "Done. Run 'frx profile --ncu <report.csv>' after collecting NCU data for runtime confirmation."
