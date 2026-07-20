# Ship Summary — 1a: schedule.py (#2)

## What was shipped
- `src/ald_sc/schedule.py`: CosineSchedule + LinearSchedule with add_noise, v_target, sample_batch, sample_sigmas
- `tests/test_schedule.py`: 14 tests (shapes, monotonicity, round-trip, interface)
- `notebooks/01_noise_schedule.ipynb`: alpha_bar plot, corruption grid, v-prediction round-trip, sigma subsampling
- `src/ald_sc/__init__.py`: exports CosineSchedule, LinearSchedule

## Verification
- 17/17 tests pass
- ruff check: clean
- ruff format: clean
- Pushed to origin/main at 997bde5

## TDD cycle
- RED: ModuleNotFoundError for ald_sc.schedule
- GREEN: Implemented CosineSchedule + LinearSchedule
- Fixed test batch-size mismatch (t had 3 elements, z0 had batch 2)
- All 14 schedule tests + 3 existing dit tests pass
