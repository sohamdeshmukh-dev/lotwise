# Optimization model

All values are normalized by starting net asset value, V. Let h be starting
security weights, t target weights, b security-level buy values, s lot-level sell
values, A the security/lot incidence matrix, x final security weights, and c cash.
The lot gain coefficient g is `(price - cost_basis) / price`.

Minimize:

```text
sum_i (x_i - t_i)^2
  + 0.001 × (sum_i b_i + sum_j s_j)
  + 0.002 × sum_j max(g_j, 0) × s_j
```

Subject to:

```text
x = h + b - A s
sum(x) + c + fee_rate × (sum(b) + sum(s)) = 1
c >= minimum_cash_weight
g · s <= realized_gain_budget / V
0 <= x <= position_caps       (cap = 0 for excluded securities)
0 <= s <= eligible_lot_values
0 <= b <= eligible_buy_limits
```

Sell eligibility is fixed from starting holdings relative to `min(target, cap)`;
buy eligibility is the complementary set. A security cannot be both bought and
sold in the resulting proposal. This is a documented heuristic around a convex
subproblem, rather than a claim to a globally optimal mixed-integer trading policy.

Positive gain penalties break the economic tie between selling lots at the same
price. For an appreciated overweight position, higher-basis lots generally permit
similar rebalancing with fewer positive gains. A tight net-gain constraint can
limit rebalancing; a forced exit can make the problem infeasible.

Cash remaining after fees is endogenous. The model does not assume that all
proceeds can immediately be reinvested without transaction costs. The initial
NAV denominator is shared by objective, position caps, cash rules, and replay.

CLARABEL solves a CVXPY disciplined parameterized program. The implementation
reuses canonicalization by matrix shape, assigns fresh account parameters, and
keeps each mutable solver instance behind a lock in the API. The independent
replay does not inspect solver variables; it uses the emitted trades and original
input snapshot. Rejected results carry no actionable trade list.

