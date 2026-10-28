"""Validated, versioned inputs and immutable proposal payloads."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Security(StrictModel):
    symbol: str = Field(min_length=1, max_length=24, pattern=r"^[A-Z0-9.-]+$")
    price: float = Field(gt=0)
    target_weight: float = Field(ge=0, le=1)


class TaxLot(StrictModel):
    lot_id: str = Field(min_length=1, max_length=100)
    symbol: str
    shares: float = Field(gt=0)
    cost_basis_per_share: float = Field(gt=0)


class Account(StrictModel):
    account_id: str = Field(min_length=1, max_length=80)
    cash: float = Field(ge=0)
    securities: list[Security] = Field(min_length=1, max_length=500)
    lots: list[TaxLot] = Field(default_factory=list, max_length=5000)
    restricted_symbols: list[str] = Field(default_factory=list)
    max_position_weight: float = Field(default=0.20, gt=0, le=1)
    realized_gain_budget: float = Field(default=1000, ge=0)
    min_cash_weight: float = Field(default=0.01, ge=0, lt=1)
    transaction_cost_bps: float = Field(default=5, ge=0, le=100)

    @model_validator(mode="after")
    def consistency(self):
        symbols = [s.symbol for s in self.securities]
        if len(symbols) != len(set(symbols)):
            raise ValueError("security symbols must be unique")
        ids = [lot.lot_id for lot in self.lots]
        if len(ids) != len(set(ids)):
            raise ValueError("lot IDs must be unique")
        if any(lot.symbol not in symbols for lot in self.lots):
            raise ValueError("every tax lot requires a security price")
        if not set(self.restricted_symbols).issubset(symbols):
            raise ValueError("unknown restricted security")
        if not math.isclose(sum(s.target_weight for s in self.securities), 1, abs_tol=1e-8):
            raise ValueError("target weights must sum to 1")
        prices = {s.symbol: s.price for s in self.securities}
        nav = self.cash + sum(lot.shares * prices[lot.symbol] for lot in self.lots)
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError("account must have positive market value")
        return self


class Trade(StrictModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    shares: float = Field(gt=0)
    price: float = Field(gt=0)
    lot_id: str | None = None
    realized_gain: float = 0
    reason: str


class Proposal(StrictModel):
    account_id: str
    status: Literal["optimal", "infeasible", "rejected"]
    method: str
    trades: list[Trade] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    exceptions: list[str] = Field(default_factory=list)
    solver_status: str = ""
    input_hash: str = ""
