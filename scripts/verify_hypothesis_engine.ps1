param(
    [int]$MinCoverage = 80,
    [switch]$InstallCoverageIfMissing
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $scriptDir) {
    $scriptDir = (Get-Location).Path
}
$repoRoot = if ((Split-Path -Leaf $scriptDir) -eq "scripts") {
    Split-Path -Parent $scriptDir
}
else {
    $scriptDir
}

Set-Location -LiteralPath $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

$targetDir = $null
$candidates = Get-ChildItem -Path $repoRoot -Directory -Filter "hypothesis_engine_v0_1" -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "\\\.venv\\|\\__pycache__\\|\\.pytest_cache\\" }

if ($candidates) {
    $targetDir = ($candidates | Select-Object -First 1).FullName
}
else {
    Write-Error "Directory 'hypothesis_engine_v0_1' was not found under '$repoRoot'."
    exit 1
}

& $pythonExe -c "import coverage" 2>$null
if ($LASTEXITCODE -ne 0) {
    if ($InstallCoverageIfMissing) {
        Write-Output "Installing coverage..."
        & $pythonExe -m pip install coverage
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    else {
        Write-Error "Module 'coverage' is not installed in '$pythonExe'. Run with -InstallCoverageIfMissing or install coverage first."
        exit 1
    }
}

& $pythonExe -m coverage run --source=hypothesis_engine_v0_1 -m pytest -v -ra --import-mode=importlib "$targetDir"
$pytestExit = $LASTEXITCODE
if ($pytestExit -ne 0) {
    exit $pytestExit
}

& $pythonExe -m coverage report --show-missing --fail-under=$MinCoverage
exit $LASTEXITCODE