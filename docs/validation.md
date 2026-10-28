# Local validation report

Tested October 27, 2026 (America/New_York), Python 3.13.3 on macOS arm64.

| Check | Outcome |
| --- | --- |
| Automated tests | 29 passed, 1 PostgreSQL test skipped |
| Ruff lint and formatting | Passed |
| Timed synthetic benchmark | 3 × 1,000 valid proposals; no replay violations |
| Tax-lot coverage | 60,000 input lots per timed batch |
| Infeasibility detection | 50 variants of one forced-exit / zero-gain-budget conflict detected |
| SQL persistence | 1,000 complete proposal round trips, on-disk SQLite |
| Concurrency | 16 duplicate requests from 8 threads persisted one proposal |
| Idempotency conflict | Reusing a key for changed inputs returned HTTP 409 |
| Atomicity | Duplicate-key failure rolled back the new snapshot and associated writes |
| Dashboard response | HTML served successfully through the API test client |
| Browser visual check | Not performed; browser access declined |
| PostgreSQL and Docker | Configuration and CI test provided; not locally executed |

The installed Starlette test client emits a deprecation warning for its current
HTTPX integration. This warning did not affect the 29 passing tests. The exact
environment is pinned in `requirements.lock` for Python 3.13.

## What the checks establish

Replay independently recomputes lot balances, final holdings, available cash,
fees, realized gains, investment exclusions, and position limits. The benchmark
fails if a nominally feasible fixture produces no proposal or if replay finds a
violation. All repeated runs include first-solve overhead. Only first-run
per-account rows are written to the CSV; timing for all three runs is retained in
the summary. SQL timing is separate and includes both transaction commit and
proposal retrieval.

These are simulator correctness and performance measurements, not evidence of
live trade execution, market outperformance, or tax savings. Replay uses the
numerical tolerance documented in the README. No UI screenshot or PostgreSQL
success is claimed without an executed check.

## Run the optional PostgreSQL test

Provide an isolated empty database. The test intentionally checks row counts and
rollback, so it should not be pointed at a populated application database.

```bash
TEST_DATABASE_URL='postgresql+psycopg://user:password@localhost/lotwise_test' \
  pytest -q tests/test_api_database.py::test_atomic_rollback_postgres
```

The GitHub Actions workflow provisions its own empty PostgreSQL 17 database and
sets this variable automatically.

