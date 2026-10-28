"""Convex, lot-level rebalancing with independent output validation."""

import time

import cvxpy as cp
import numpy as np

from lotwise.models import Account, Proposal, Trade
from lotwise.validation import fingerprint, replay


def arrays(account: Account):
    symbols = [s.symbol for s in account.securities]
    index = {s: i for i, s in enumerate(symbols)}
    prices = np.array([s.price for s in account.securities])
    # One dummy zero-capacity lot lets cash-only accounts use the same formulation.
    incidence = np.zeros((len(symbols), max(1, len(account.lots))))
    values = np.zeros(incidence.shape[1])
    gains = np.zeros_like(values)
    for j, lot in enumerate(account.lots):
        i = index[lot.symbol]
        incidence[i, j] = 1
        values[j] = lot.shares * prices[i]
        gains[j] = 1 - lot.cost_basis_per_share / prices[i]
    nav = account.cash + values.sum()
    return symbols, prices, incidence, values / nav, gains, float(nav)


class Rebalancer:
    """Reuse a DPP problem per account shape; one instance per worker/thread."""

    def __init__(self, turnover_penalty: float = 0.001, gain_penalty: float = 0.002):
        self.turnover_penalty = turnover_penalty
        self.gain_penalty = gain_penalty
        self.cache = {}

    def _template(self, n, m):
        key = (n, m)
        if key in self.cache:
            return self.cache[key]
        p = {
            "a": cp.Parameter((n, m)),
            "current": cp.Parameter(n),
            "target": cp.Parameter(n),
            "sell_limit": cp.Parameter(m, nonneg=True),
            "buy_limit": cp.Parameter(n, nonneg=True),
            "cap": cp.Parameter(n, nonneg=True),
            "gain": cp.Parameter(m),
            "gain_cost": cp.Parameter(m, nonneg=True),
            "budget": cp.Parameter(nonneg=True),
            "reserve": cp.Parameter(nonneg=True),
            "fee": cp.Parameter(nonneg=True),
        }
        buy, sell, final = (
            cp.Variable(n, nonneg=True),
            cp.Variable(m, nonneg=True),
            cp.Variable(n, nonneg=True),
        )
        cash = cp.Variable(nonneg=True)
        gross = cp.sum(buy) + cp.sum(sell)
        constraints = [
            final == p["current"] + buy - p["a"] @ sell,
            sell <= p["sell_limit"],
            buy <= p["buy_limit"],
            final <= p["cap"],
            cp.sum(final) + cash + p["fee"] * gross == 1,
            cash >= p["reserve"],
            p["gain"] @ sell <= p["budget"],
        ]
        objective = cp.sum_squares(final - p["target"]) + self.turnover_penalty * gross
        objective += self.gain_penalty * (p["gain_cost"] @ sell)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        assert problem.is_dpp()
        self.cache[key] = p, buy, sell, problem
        return self.cache[key]

    def propose(self, account: Account) -> Proposal:
        started = time.perf_counter()
        symbols, prices, incidence, values, gains, nav = arrays(account)
        current = incidence @ values
        target = np.array([s.target_weight for s in account.securities])
        caps = np.full(len(symbols), account.max_position_weight)
        for i, symbol in enumerate(symbols):
            if symbol in account.restricted_symbols:
                caps[i] = 0
        # Fixed directions prevent same-name buy/sell round trips. This conservative
        # rule deliberately narrows the feasible set; it is not a global MIP optimum.
        sellable = current > np.minimum(target, caps) + 1e-10
        buy_limit = np.where(sellable, 0, caps)
        sell_limit = values * (incidence.T @ sellable.astype(float))
        p, buy, sell, problem = self._template(len(symbols), len(values))
        for key, value in {
            "a": incidence,
            "current": current,
            "target": target,
            "sell_limit": sell_limit,
            "buy_limit": buy_limit,
            "cap": caps,
            "gain": gains,
            "gain_cost": np.maximum(gains, 0),
            "budget": account.realized_gain_budget / nav,
            "reserve": account.min_cash_weight,
            "fee": account.transaction_cost_bps / 10000,
        }.items():
            p[key].value = value
        proposal = Proposal(
            account_id=account.account_id,
            status="infeasible",
            method="lotwise",
            input_hash=fingerprint(account),
        )
        try:
            problem.solve(
                solver="CLARABEL",
                warm_start=True,
                tol_gap_abs=1e-9,
                tol_feas=1e-9,
                tol_gap_rel=1e-9,
            )
        except cp.SolverError as exc:
            proposal.status = "rejected"
            proposal.exceptions = [f"solver failure: {type(exc).__name__}"]
            return proposal
        proposal.solver_status = problem.status
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            proposal.exceptions = [
                (
                    "No feasible proposal under the account constraints and fixed trade directions; "
                    "review restricted holdings, position caps, cash reserve, and gain budget."
                )
            ]
            proposal.metrics = {"runtime_ms": (time.perf_counter() - started) * 1000}
            return proposal
        trades = []
        # Retain economically tiny trades down to 1e-9 of NAV; never round a
        # validated proposal into whole shares after solving.
        for j, lot in enumerate(account.lots):
            value = max(0, float(sell.value[j])) * nav
            if value > nav * 1e-9:
                price = account.securities[symbols.index(lot.symbol)].price
                trades.append(
                    Trade(
                        symbol=lot.symbol,
                        side="SELL",
                        shares=value / price,
                        price=price,
                        lot_id=lot.lot_id,
                        realized_gain=value * gains[j],
                        reason="Restricted holding exit"
                        if lot.symbol in account.restricted_symbols
                        else "Rebalance overweight position; optimize sale at tax-lot level",
                    )
                )
        for i, symbol in enumerate(symbols):
            value = max(0, float(buy.value[i])) * nav
            if value > nav * 1e-9:
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="BUY",
                        shares=value / prices[i],
                        price=prices[i],
                        reason="Move toward model allocation within account limits",
                    )
                )
        metrics, errors = replay(account, trades)
        metrics["runtime_ms"] = (time.perf_counter() - started) * 1000
        proposal.status = "rejected" if errors else "optimal"
        proposal.metrics, proposal.exceptions = metrics, errors
        # Failed validation must never expose actionable trades.
        proposal.trades = [] if errors else trades
        return proposal
