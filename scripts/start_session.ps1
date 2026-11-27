param(
    [string[]]$CoverageTargets = @()
)

Set-Location "C:\Users\Yakov Krelin\Documents\Immune Regulatory Atlas"
$python = ".venv\Scripts\python.exe"

Write-Output "=== 1. Git Status & Last Commit ==="
git status -s
git log -n 1 --stat

Write-Output "=== 2. Run Pytest ==="
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    if (Test-Path ".coverage") { Remove-Item ".coverage" -Force }
    return
}

if ($CoverageTargets.Count -gt 0) {
    Write-Output "=== 3. Coverage Check (100% required) ==="
    $includeFilter = $CoverageTargets -join ","
    & $python -m coverage run --source=hypothesis_engine_v0_1,scripts,tests -m pytest -q
    if ($LASTEXITCODE -eq 0) {
        & $python -m coverage report -m --include="$includeFilter" --fail-under=100
    }
}

Write-Output "=== 4. Clean .coverage ==="
if (Test-Path ".coverage") {
    Remove-Item ".coverage" -Force
}
Write-Output "Coverage file exists: $(Test-Path '.coverage')"
git status -s
