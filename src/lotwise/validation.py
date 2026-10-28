"""Independent ledger replay: trust executable trades, not solver status."""

import hashlib

import numpy as np

from lotwise.models import Account, Trade


def fingerprint(account: Account) -> str:
    return hashlib.sha256(account.model_dump_json().encode()).hexdigest()


def replay(account: Account, trades: list[Trade]) -> tuple[dict[str, float], list[str]]:
    securities = {s.symbol: s for s in account.securities}
    lots = {lot.lot_id: lot for lot in account.lots}
    remaining = {lot.lot_id: lot.shares for lot in account.lots}
    holdings = {s: 0.0 for s in securities}
    for lot in account.lots:
        holdings[lot.symbol] += lot.shares * securities[lot.symbol].price
    nav = account.cash + sum(holdings.values())
    initial = np.array(list(holdings.values())) / nav
    cash, gains, gross, fees = account.cash, 0.0, 0.0, 0.0
    errors = []
    buys, sells = set(), set()
    # Absolute dollar tolerance scales with NAV to accommodate floating point solves.
    tol = max(1e-5, nav * 1e-7)
    for trade in trades:
        if trade.symbol not in securities:
            errors.append("unknown trade security")
            continue
        if abs(trade.price - securities[trade.symbol].price) > 1e-8:
            errors.append("trade uses a different price snapshot")
        value = trade.shares * trade.price
        fee = value * account.transaction_cost_bps / 10000
        gross += value
        fees += fee
        if trade.side == "SELL":
            sells.add(trade.symbol)
            lot = lots.get(trade.lot_id)
            if lot is None or lot.symbol != trade.symbol:
                errors.append("sell references an invalid tax lot")
                continue
            remaining[lot.lot_id] -= trade.shares
            if remaining[lot.lot_id] * trade.price < -tol:
                errors.append("tax lot oversold")
            gain = trade.shares * (trade.price - lot.cost_basis_per_share)
            if abs(gain - trade.realized_gain) > tol:
                errors.append("incorrect realized gain")
            gains += gain
            holdings[trade.symbol] -= value
            cash += value - fee
        else:
            buys.add(trade.symbol)
            if trade.lot_id is not None or abs(trade.realized_gain) > tol:
                errors.append("buy contains invalid tax-lot metadata")
            holdings[trade.symbol] += value
            cash -= value + fee
    if buys & sells:
        errors.append("same security bought and sold in one proposal")
    if cash < account.min_cash_weight * nav - tol:
        errors.append("cash reserve violated")
    if gains > account.realized_gain_budget + tol:
        errors.append("realized-gain budget exceeded")
    for symbol, value in holdings.items():
        if value < -tol:
            errors.append(f"negative holding: {symbol}")
        if value > account.max_position_weight * nav + tol:
            errors.append(f"position cap exceeded: {symbol}")
        if symbol in account.restricted_symbols and value > tol:
            errors.append(f"restricted security retained: {symbol}")
    if abs(cash + sum(holdings.values()) + fees - nav) > tol:
        errors.append("ledger does not conserve value")
    weights = np.array(list(holdings.values())) / nav
    target = np.array([s.target_weight for s in account.securities])
    metrics = {
        "nav": nav,
        "ending_cash": cash,
        "transaction_cost": fees,
        "realized_gain": gains,
        "turnover": gross / (2 * nav),
        "weight_rmse_bps": float(np.sqrt(np.mean((weights - target) ** 2)) * 10000),
        "initial_weight_rmse_bps": float(np.sqrt(np.mean((initial - target) ** 2)) * 10000),
        "gross_traded": gross,
        "trade_count": float(len(trades)),
        "constraint_violation_count": float(len(set(errors))),
    }
    return metrics, sorted(set(errors))
