## [0.3.0] - 2026-11-24

### Added
- Expanded regression tests in `tests/test_v03_bifurcation_pilot.py`, including `test_snapshot_adapter_attaches_sha256_and_paths_per_branch`, with all 74 tests passing under `pytest --import-mode=importlib`.

### Changed
- Standardized all user-facing strings, conflict descriptions, hypothesis labels, logic messages, and follow-up test prompts in `src/bioos/experimental/bifurcation_pilot.py` to English, and restored complete `EvidenceItem` initialization in `__main__`.
- Preserved conflict detection and inference mode adjustments in `src/bioos/experimental/snapshot_adapter.py` while assigning `related_path_refs` per branch.
- Normalized line endings in `tests/test_v03_bifurcation_pilot.py`.

### Fixed
- Resolved premature `None` returns and `NoneType` attribute errors in `evaluate_pilot_scenario` (`src/bioos/experimental/bifurcation_pilot.py`) by flattening nested control-flow guards and enforcing 4-space indentation so valid paths return a `ReasoningReport`.
## [0.3.0] - 2026-11-24

### Added
- Expanded regression tests in `tests/test_v03_bifurcation_pilot.py`, including `test_snapshot_adapter_attaches_sha256_and_paths_per_branch`, with all 74 tests passing under `pytest --import-mode=importlib`.

### Changed
- Standardized all user-facing strings, conflict descriptions, hypothesis labels, logic messages, and follow-up test prompts in `src/bioos/experimental/bifurcation_pilot.py` to English, and restored complete `EvidenceItem` initialization in `__main__`.
- Preserved conflict detection and inference mode adjustments in `src/bioos/experimental/snapshot_adapter.py` while assigning `related_path_refs` per branch.
- Normalized line endings in `tests/test_v03_bifurcation_pilot.py`.

### Fixed
- Resolved premature `None` returns and `NoneType` attribute errors in `evaluate_pilot_scenario` (`src/bioos/experimental/bifurcation_pilot.py`) by flattening nested control-flow guards and enforcing 4-space indentation so valid paths return a `ReasoningReport`.

