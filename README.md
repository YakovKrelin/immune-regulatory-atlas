# Immune Regulatory Atlas (BIOOS)

[![BIOOS CI](https://github.com/YakovKrelin/immune-regulatory-atlas/actions/workflows/bioos-ci.yml/badge.svg)](https://github.com/YakovKrelin/immune-regulatory-atlas/actions/workflows/bioos-ci.yml)

Multi-level human immune regulatory atlas, causal traversal engine, dataset evaluation service, and SHA-256 / PROV-O snapshot verification API.

## Quickstart

### 1. Start the API Server

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn bioos.main:app --host 127.0.0.1 --port 8000
```

### 2. Run the Test Suite

```powershell
$env:PYTHONPATH = "src"
python -m pytest -v
```

### 3. Run End-to-End Verification Checks

```powershell
.\scripts\run_bioos_checks.ps1 -BaseUrl "[http://127.0.0.1:8000](http://127.0.0.1:8000)"
```
