# LotWise — Tax-Aware Portfolio Rebalancing Engine

Python, CVXPY, FastAPI, SQLAlchemy, PostgreSQL-compatible SQL schema

- Engineered a Python portfolio rebalancer processing 1,000 simulated accounts and 60K tax lots in 2.1s through cached convex optimization
- Reduced positive realized gains 78% versus FIFO for identical sell quantities by optimizing tax-lot selection across synthetic portfolios
- Cut mean allocation error 55% across 1,000 simulated accounts by enforcing cash, position, and investment constraints through quadratic programming

Alternative third bullet emphasizing backend engineering:

- Built a FastAPI and SQLAlchemy audit pipeline round-tripping 1,000 proposals in 4.3s, with idempotent requests and transactional persistence

## Evidence and interview notes

All figures are from `benchmarks/results/summary.json`, seed 42. Runtime is the
median of three local batches; 2.1s excludes database I/O. Database timing is a
separate on-disk SQLite test. PostgreSQL configuration and integration tests are
provided, but PostgreSQL was not run locally in the initial validation.

The 78% figure measures positive realized lot gains, not taxes or returns. FIFO
uses exactly the same security-level sell quantities and older lots first. The
synthetic dataset intentionally includes appreciated and loss lots; it is not a
market backtest. The 55% value is rounded from 54.7689% and measures the reduction
in mean cross-sectional allocation RMSE relative to starting holdings. All 3,000
proposals across the timed repetitions passed independent replay. Fifty fixture
variants of one deliberately infeasible economic scenario emitted no trades.

Explain why a position restriction can worsen model distance, why lot selection
changes gains without changing shares sold, and why a solver success flag is not
sufficient to release trades. These tradeoffs are stronger interview material
than presenting the simulator as a production investment system.
