param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$OutSnapshot = "scripts\latest_bioos_snapshot.json"
)

$ErrorActionPreference = "Stop"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SrcDir = Join-Path $ProjectRoot "src"

if (Test-Path -LiteralPath $VenvPython) {
    $PythonExe = $VenvPython
} else {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCmd) {
        throw "Virtual environment Python not found at '$VenvPython' and 'python' was not found on PATH."
    }
    $PythonExe = $PythonCmd.Source
}

if (-not (Test-Path -LiteralPath $SrcDir)) {
    throw "Source directory not found at: $SrcDir"
}

Push-Location $ProjectRoot
try {
    $env:PYTHONPATH = $SrcDir

    Write-Host "=== 1. Filtered Assertions (tissue=Blood, eco_levels=A) ===" -ForegroundColor Cyan
    $assertionsJson = & $PythonExe -m bioos.cli --base-url $BaseUrl assertions --tissue Blood --eco-levels A
    $assertionsObj = $assertionsJson | ConvertFrom-Json
    $assertionsCount = @($assertionsObj).Count
    Write-Host "Matched assertions count: $assertionsCount"
    $assertionsObj | ConvertTo-Json -Depth 5

    Write-Host "=== 2. Filtered Graph Traversal (start=TNF, max_hops=2, min_eco=A, tissue=Liver) ===" -ForegroundColor Cyan
    $graphJson = & $PythonExe -m bioos.cli --base-url $BaseUrl graph --start TNF --max-hops 2 --min-eco A --tissue Liver
    $graphObj = $graphJson | ConvertFrom-Json
    $edgeCount = @($graphObj.edges).Count
    Write-Host "Graph nodes: $($graphObj.nodes -join ', ') | Edges: $edgeCount"
    $graphObj | ConvertTo-Json -Depth 5

    Write-Host "=== 3. Dataset Evaluation (pbmc_viral_pilot) ===" -ForegroundColor Cyan
    $datasetJson = & $PythonExe -m bioos.cli --base-url $BaseUrl dataset `
        --name "pbmc_viral_pilot" `
        --modality "RNA-seq" `
        --tissue "Blood" `
        --cell-type "Monocyte" `
        --signals "TNF up, IL6 up, IFNG up, CXCL10 up"
    $datasetObj = $datasetJson | ConvertFrom-Json
    $activeStateNames = @($datasetObj.active_states | ForEach-Object { $_.name })
    Write-Host "Active states: $($activeStateNames -join ', ')"
    $datasetObj | ConvertTo-Json -Depth 6

    Write-Host "=== 4. Reproducibility Snapshot + SHA-256 Verification ===" -ForegroundColor Cyan
    $resolvedOut = if ([System.IO.Path]::IsPathRooted($OutSnapshot)) {
        $OutSnapshot
    } else {
        Join-Path $ProjectRoot $OutSnapshot
    }
    $outParent = Split-Path -Parent $resolvedOut
    if ($outParent -and -not (Test-Path $outParent)) {
        New-Item -ItemType Directory -Path $outParent -Force | Out-Null
    }

    $snapArgs = @("-m", "bioos.cli", "--base-url", $BaseUrl, "snapshot", "--title", "BIOOS Automated Check Snapshot", "--signals", "TNF up, IL6 up, IFNG up, CXCL10 up", "--start-node", "TNF", "--max-hops", "2", "--actor", "BIOOS-Check-Runner", "--tissue", "Blood", "--eco-levels", "A", "--out", $resolvedOut)
    $snapOutputJson = & $PythonExe @snapArgs

    $snapWrapper = $snapOutputJson | ConvertFrom-Json
    $verifySavedJson = & $PythonExe -m bioos.cli --base-url $BaseUrl verify-snapshot --file $resolvedOut
    $verifySavedObj = $verifySavedJson | ConvertFrom-Json

    Write-Host "Snapshot ID: $($snapWrapper.snapshot.snapshot_id)"
    Write-Host "Verification valid: $($verifySavedObj.valid)"
    if (-not $verifySavedObj.valid) {
        throw "Snapshot SHA-256 / PROV-O verification failed!"
    }
    $snapWrapper | ConvertTo-Json -Depth 6
}
finally {
    if (Test-Path Env:PYTHONPATH) {
        Remove-Item Env:PYTHONPATH
    }
    Pop-Location
}

