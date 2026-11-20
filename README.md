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






# BIOOS (`v0.1`)

BIOOS (Human Regulatory Atlas) is a software architecture for modeling multi-level biological systems, regulatory circuits, causal inference, and hypothesis construction.

Release **v0.1** establishes a stable baseline data-validation and API layer on the `main` branch with zero warnings during test execution.

---

## Architecture and Data Models (Pydantic v2)

The project uses strict validation built on **Pydantic v2**:
- Explicit resolution of forward references and deterministic model definition/rebuild ordering (`model_rebuild()` where needed) to prevent unresolved type-annotation warnings.
- Strict enforcement of API input and output contracts prior to executing domain logic.

### Core Request and Response Schemas
- **`HypothesisRequest`** — request schema for generating or evaluating a biological hypothesis (input entities, regulatory graph or pathway context, and analysis parameters).
- **`SnapshotCreateRequest`** — request schema for creating a state snapshot (`POST /v1/snapshots`), with forward references to nested structures resolved cleanly without Pydantic v2 validation warnings.

> Note: When adding new fields or nested models in `v0.2`, preserve backward compatibility with `v0.1` contracts and validate all models with warnings treated as errors.

---

## Quick Start and Environment

Recommended development setup on Windows (PowerShell / VS Code) using the local `.venv` environment in the repository root:

```powershell
Set-Location "C:\Users\Yakov Krelin\Documents\Immune Regulatory Atlas"

```
