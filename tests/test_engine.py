"""Economic invariants and adversarial cases, beyond solver-success assertions."""

import pytest
from pydantic import ValidationError

from lotwise.baseline import fifo_gains, optimized_positive_gains
from lotwise.models import Account, Security, TaxLot, Trade
from lotwise.optimizer import Rebalancer
from lotwise.synthetic import generate_accounts, impossible_account
from lotwise.validation import replay


def small_account(**changes):
    data = {
        "account_id": "test",
        "cash": 1000,
        "securities": [
            Security(symbol="AAA", price=100, target_weight=0.5),
            Security(symbol="BBB", price=100, target_weight=0.5),
        ],
        "lots": [
            TaxLot(lot_id="old", symbol="AAA", shares=60, cost_basis_per_share=50),
            TaxLot(lot_id="new", symbol="AAA", shares=30, cost_basis_per_share=120),
        ],
        "max_position_weight": 1,
        "realized_gain_budget": 10000,
    }
    data.update(changes)
    return Account(**data)


@pytest.mark.parametrize("seed", [7, 42, 2026])
def test_seeded_accounts_replay_all_constraints(seed):
    engine = Rebalancer()
    for account in generate_accounts(30, seed=seed):
        proposal = engine.propose(account)
        assert proposal.status == "optimal", proposal.exceptions
        metrics, errors = replay(account, proposal.trades)
        assert not errors
        # Mandatory exclusions can increase drift; only unrestricted accounts
        # have a feasible no-trade comparison to the original model.
        if not account.restricted_symbols:
            assert metrics["weight_rmse_bps"] < metrics["initial_weight_rmse_bps"]
        assert metrics["ending_cash"] >= account.min_cash_weight * metrics["nav"] - 0.01


def test_high_basis_lot_preferred_to_fifo():
    account = small_account()
    proposal = Rebalancer().propose(account)
    assert proposal.status == "optimal"
    new_sale = sum(t.shares for t in proposal.trades if t.lot_id == "new")
    assert new_sale > 29.99
    assert (
        optimized_positive_gains(proposal) < fifo_gains(account, proposal)["positive_realized_gain"]
    )


def test_zero_gain_budget_enforced():
    account = small_account(realized_gain_budget=0)
    proposal = Rebalancer().propose(account)
    assert proposal.status == "optimal"
    assert proposal.metrics["realized_gain"] <= 0.001


def test_infeasibility_has_no_trades():
    proposal = Rebalancer().propose(impossible_account())
    assert proposal.status == "infeasible"
    assert proposal.exceptions
    assert proposal.trades == []


def test_cash_only_account():
    proposal = Rebalancer().propose(small_account(cash=10000, lots=[]))
    assert proposal.status == "optimal"
    assert all(t.side == "BUY" for t in proposal.trades)
    assert proposal.metrics["ending_cash"] >= 100 - 0.01


def test_all_restricted_stays_in_cash():
    proposal = Rebalancer().propose(small_account(restricted_symbols=["AAA", "BBB"]))
    assert proposal.status == "optimal"
    assert not any(t.side == "BUY" for t in proposal.trades)
    assert proposal.metrics["ending_cash"] > 9900


def test_position_cap_enforced():
    proposal = Rebalancer().propose(small_account(max_position_weight=0.35))
    assert proposal.status == "optimal"
    assert not replay(small_account(max_position_weight=0.35), proposal.trades)[1]


def test_zero_fees_conserve_cash():
    proposal = Rebalancer().propose(small_account(transaction_cost_bps=0))
    assert proposal.status == "optimal"
    assert proposal.metrics["transaction_cost"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"cash": float("nan")},
        {"cash": -1},
        {"cash": float("inf")},
        {"max_position_weight": 0},
        {"realized_gain_budget": -1},
        {"restricted_symbols": ["UNKNOWN"]},
    ],
)
def test_reject_invalid_inputs(change):
    with pytest.raises(ValidationError):
        small_account(**change)


def test_duplicate_lot_ids_rejected():
    account = small_account()
    with pytest.raises(ValidationError):
        small_account(lots=[account.lots[0], account.lots[0]])


def test_bad_target_weights_rejected():
    with pytest.raises(ValidationError):
        small_account(securities=[Security(symbol="AAA", price=100, target_weight=0.1)])


def test_unpriced_lot_rejected():
    with pytest.raises(ValidationError):
        small_account(lots=[TaxLot(lot_id="bad", symbol="ZZZ", shares=1, cost_basis_per_share=1)])


def test_oversell_and_gain_tampering_detected():
    trades = [
        Trade(
            symbol="AAA",
            side="SELL",
            shares=100,
            price=100,
            lot_id="old",
            realized_gain=0,
            reason="adversarial",
        )
    ]
    errors = replay(small_account(), trades)[1]
    assert "tax lot oversold" in errors
    assert "incorrect realized gain" in errors


def test_cash_overdraft_detected():
    errors = replay(
        small_account(),
        [Trade(symbol="BBB", side="BUY", shares=200, price=100, reason="adversarial")],
    )[1]
    assert "cash reserve violated" in errors


def test_restricted_purchase_detected():
    account = small_account(restricted_symbols=["BBB"])
    errors = replay(
        account, [Trade(symbol="BBB", side="BUY", shares=1, price=100, reason="adversarial")]
    )[1]
    assert "restricted security retained: BBB" in errors


def test_repeated_solves_are_reproducible():
    account = small_account()
    engine = Rebalancer()
    a, b = engine.propose(account), engine.propose(account)
    assert a.input_hash == b.input_hash
    assert a.metrics["gross_traded"] == pytest.approx(b.metrics["gross_traded"], abs=0.01)
