param(
    [string]$Exe = ".\build\Release\ttm_bench.exe",
    [string]$Output = "kernel_sweep.csv",
    [string]$Dims = "128,128,128",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$ranks = "8,16,32,64,128,256"
$modes = "0,1,2"
$nnzs = @(100000, 500000, 1000000)
$threads = @(1, 4, 8, 16)
$repeats = 5

if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Executable not found: $Exe"
}

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("ttm_sweep_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

$wroteHeader = $false
try {
    foreach ($nnz in $nnzs) {
        foreach ($threadCount in $threads) {
            $part = Join-Path $tempDir "part_nnz${nnz}_t${threadCount}.csv"
            & $Exe --output $part --dims $Dims --nnz $nnz --ranks $ranks --modes $modes --threads $threadCount --repeats $repeats --seed $Seed
            if ($LASTEXITCODE -ne 0) {
                throw "Benchmark failed for nnz=$nnz threads=$threadCount"
            }

            $lines = Get-Content -LiteralPath $part
            if (-not $wroteHeader) {
                Set-Content -LiteralPath $Output -Value $lines
                $wroteHeader = $true
            } else {
                Add-Content -LiteralPath $Output -Value ($lines | Select-Object -Skip 1)
            }
        }
    }
} finally {
    if (Test-Path -LiteralPath $tempDir) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force
    }
}

Write-Host "wrote $Output"
