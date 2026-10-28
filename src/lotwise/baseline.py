"""Counterfactual FIFO lot selection with identical security-level sell quantities.

This isolates lot selection; FIFO is not assumed to satisfy the gain budget and
is not an alternative optimized portfolio. Input list order defines FIFO.
"""

from collections import defaultdict

from lotwise.models import Account, Proposal


def fifo_gains(account: Account, proposal: Proposal) -> dict[str, float]:
    quantities = defaultdict(float)
    for trade in proposal.trades:
        if trade.side == "SELL":
            quantities[trade.symbol] += trade.shares
    prices = {s.symbol: s.price for s in account.securities}
    positive, net = 0.0, 0.0
    for lot in account.lots:
        sold = min(lot.shares, quantities[lot.symbol])
        quantities[lot.symbol] -= sold
        gain = sold * (prices[lot.symbol] - lot.cost_basis_per_share)
        net += gain
        positive += max(gain, 0)
    return {"positive_realized_gain": positive, "net_realized_gain": net}


def optimized_positive_gains(proposal: Proposal) -> float:
    return sum(max(t.realized_gain, 0) for t in proposal.trades if t.side == "SELL")
