# LotWise — Tax-Aware Portfolio Rebalancing Engine

Python, CVXPY, scikit-learn, FastAPI, SQLAlchemy, PostgreSQL-compatible SQL schema

- Engineered a Python portfolio rebalancer processing 1,000 simulated accounts and 60K tax lots in 2.1s through cached convex optimization
- Reduced positive realized gains 78% versus FIFO for identical sell quantities by optimizing tax-lot selection across synthetic portfolios
- Predicted portfolio constraint conflicts at 90% accuracy on 500 held-out synthetic accounts by training a logistic classifier on 1,500 solver-labeled portfolios

Alternative third bullet emphasizing backend engineering:

- Built a FastAPI and SQLAlchemy audit pipeline round-tripping 1,000 proposals in 4.3s, with idempotent requests and transactional persistence

Alternative emphasizing portfolio optimization:

- Cut mean allocation error 55% across 1,000 simulated accounts by enforcing cash, position, and investment constraints through quadratic programming

## Evidence and interview notes

The optimization figures are from `benchmarks/results/summary.json`, seed 42. Runtime is the
median of three local batches; 2.1s excludes database I/O. Database timing is a
separate on-disk SQLite test. PostgreSQL configuration and integration tests are
provided, but PostgreSQL was not run locally in the initial validation.

The ML figure is from `benchmarks/ai-results/evaluation.json`: 1,500 training
accounts, 500 separate test accounts, 90% accuracy versus a 69% majority baseline.
It missed 33 of 155 conflicts, so the classifier never bypasses solver validation.
This is same-distribution synthetic evaluation, not demonstrated real-client
generalization. Optional LLM question answering is implemented and tested with
mocked provider responses; no live generation-quality number is claimed.

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
