"""Seeded, explicitly fictional fixtures; no market-data or client-data claims."""

import numpy as np

from lotwise.models import Account, Security, TaxLot


def generate_accounts(count: int = 1000, seed: int = 42, securities: int = 20) -> list[Account]:
    rng = np.random.default_rng(seed)
    accounts = []
    for k in range(count):
        nav = float(rng.uniform(100_000, 2_000_000))
        prices = rng.uniform(25, 250, securities)
        target = rng.dirichlet(np.full(securities, 20.0))
        weights = rng.dirichlet(target * 100) * 0.96
        symbols = [f"SIM{i:03}" for i in range(securities)]
        lots = []
        for i, symbol in enumerate(symbols):
            # Oldest to newest: appreciated, modest gain, then unrealized loss.
            parts = rng.dirichlet([3, 3, 3])
            for j, factor in enumerate((0.65, 0.90, 1.15)):
                lots.append(
                    TaxLot(
                        lot_id=f"{k}-{i}-{j}",
                        symbol=symbol,
                        shares=float(nav * weights[i] * parts[j] / prices[i]),
                        cost_basis_per_share=float(prices[i] * factor),
                    )
                )
        restricted = list(rng.choice(symbols, size=k % 3, replace=False))
        accounts.append(
            Account(
                account_id=f"SIM-{seed}-{k:05}",
                cash=nav * 0.04,
                securities=[
                    Security(symbol=s, price=float(p), target_weight=float(t))
                    for s, p, t in zip(symbols, prices, target)
                ],
                lots=lots,
                restricted_symbols=restricted,
                max_position_weight=0.20,
                realized_gain_budget=nav * float(rng.uniform(0.015, 0.025)),
                min_cash_weight=0.01,
                transaction_cost_bps=5,
            )
        )
    return accounts


def impossible_account(k: int = 0) -> Account:
    """A mandatory full exit realizes $50,000, exceeding the zero gain budget."""
    return Account(
        account_id=f"INFEASIBLE-{k}",
        cash=1000,
        securities=[
            Security(symbol="LOCKED", price=100, target_weight=0.5),
            Security(symbol="OTHER", price=100, target_weight=0.5),
        ],
        lots=[TaxLot(lot_id="appreciated", symbol="LOCKED", shares=1000, cost_basis_per_share=50)],
        restricted_symbols=["LOCKED"],
        max_position_weight=1,
        realized_gain_budget=0,
    )
