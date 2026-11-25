param(
    [int]$MinCoverage = 90,
    [switch]$InstallCoverageIfMissing,
    [switch]$InstallPackageIfMissing
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

function Resolve-PythonExe {
    param(
        [string]$RootDir
    )

    if (-not [string]::IsNullOrWhiteSpace($RootDir)) {
        $candidateVenv = Join-Path $RootDir ".venv\Scripts\python.exe"
        if (Test-Path -LiteralPath $candidateVenv) {
            return $candidateVenv
        }
    }

    return "python"
}

function Remove-CoverageArtifact {
    param(
        [string]$Path
    )

    if (-not [string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path)) {
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    }
}

$pythonExe = Resolve-PythonExe -RootDir $repoRoot
$coverageFile = Join-Path $repoRoot ".coverage"

$candidates = Get-ChildItem -Path $repoRoot -Directory -Filter "hypothesis_engine_v0_1" -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "\\\.venv\\|\\__pycache__\\|\\.pytest_cache\\" }

if (-not $candidates) {
    Remove-CoverageArtifact -Path $coverageFile
    Write-Error "Directory 'hypothesis_engine_v0_1' was not found under '$repoRoot'."
    exit 1
}
$targetDir = ($candidates | Select-Object -First 1).FullName

& $pythonExe -c "import hypothesis_engine_v0_1" 2>$null
if ($LASTEXITCODE -ne 0) {
    if ($InstallPackageIfMissing) {
        $packageRoot = Split-Path -Parent $targetDir
        Write-Output "Installing package from '$packageRoot'..."
        & $pythonExe -m pip install --no-build-isolation -e "$packageRoot"
        if ($LASTEXITCODE -ne 0) {
            Remove-CoverageArtifact -Path $coverageFile
            Write-Error "Failed to install package from '$packageRoot' (exit code $LASTEXITCODE)."
            exit $LASTEXITCODE
        }
    }
    else {
        Remove-CoverageArtifact -Path $coverageFile
        Write-Error "Package 'hypothesis_engine_v0_1' is not installed in '$pythonExe'. Run with -InstallPackageIfMissing or install the package first."
        exit 1
    }
}

$venvEntrypoint = Join-Path (Split-Path -Parent $pythonExe) "hypothesis-engine.exe"
if (Test-Path -LiteralPath $venvEntrypoint) {
    Write-Output "Checking entrypoint '$venvEntrypoint --help'..."
    & $venvEntrypoint --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Remove-CoverageArtifact -Path $coverageFile
        Write-Error "Entrypoint '$venvEntrypoint --help' failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}
elseif (Get-Command "hypothesis-engine" -ErrorAction SilentlyContinue) {
    Write-Output "Checking entrypoint 'hypothesis-engine --help'..."
    & hypothesis-engine --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Remove-CoverageArtifact -Path $coverageFile
        Write-Error "Entrypoint 'hypothesis-engine --help' failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}
else {
    Remove-CoverageArtifact -Path $coverageFile
    Write-Error "Entrypoint 'hypothesis-engine' was not found next to '$pythonExe' or in PATH."
    exit 1
}

& $pythonExe -c "import coverage" 2>$null
if ($LASTEXITCODE -ne 0) {
    if ($InstallCoverageIfMissing) {
        Write-Output "Installing coverage..."
        & $pythonExe -m pip install coverage
        if ($LASTEXITCODE -ne 0) {
            Remove-CoverageArtifact -Path $coverageFile
            Write-Error "Failed to install 'coverage' (exit code $LASTEXITCODE)."
            exit $LASTEXITCODE
        }
    }
    else {
        Remove-CoverageArtifact -Path $coverageFile
        Write-Error "Module 'coverage' is not installed in '$pythonExe'. Run with -InstallCoverageIfMissing or install coverage first."
        exit 1
    }
}

$finalExit = 0
try {
    Remove-CoverageArtifact -Path $coverageFile

    & $pythonExe -m coverage run --source=hypothesis_engine_v0_1 -m pytest -v -ra --import-mode=importlib "$targetDir"
    if ($LASTEXITCODE -ne 0) {
        $finalExit = $LASTEXITCODE
        Write-Error "pytest failed with exit code $finalExit."
    }
    else {
        & $pythonExe -m coverage report --show-missing --fail-under=$MinCoverage
        if ($LASTEXITCODE -ne 0) {
            $finalExit = $LASTEXITCODE
            Write-Error "Coverage check failed (threshold: $MinCoverage%, exit code $finalExit)."
        }
    }
}
finally {
    Remove-CoverageArtifact -Path $coverageFile
}

exit $finalExit