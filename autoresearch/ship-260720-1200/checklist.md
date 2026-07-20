# Ship Checklist — 1a: schedule.py (#2)

## Phase 1: Identify
- Ship type: code-pr
- Target: src/ald_sc/schedule.py + tests/test_schedule.py + notebooks/01_noise_schedule.ipynb
- Issue: #2

## Phase 3: Checklist
- [x] Tests pass: 17 passed (3 dit + 14 schedule)
- [x] Lint clean: ruff check — all checks passed
- [x] Format clean: ruff format --check — 5 files already formatted
- [x] No secrets in diff
- [x] Export public symbols in __init__.py (CosineSchedule, LinearSchedule)

## Phase 6: Ship
- Commit: 997bde5
- Message: feat: add cosine and linear noise schedules with v-prediction support (#2)
- Pushed to origin/main

## Phase 7: Verify
- Push confirmed: ba80f85..997bde5 main -> main
- All 17 tests pass on pushed code
