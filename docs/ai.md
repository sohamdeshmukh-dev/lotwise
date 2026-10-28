# AI review and conflict triage

LotWise now has two independent AI capabilities. Neither changes proposals,
approves trades, bypasses the optimizer, or predicts investment returns.

## Local learned conflict triage

A logistic regression estimates whether account rules will conflict under the
existing optimizer formulation. Features come only from pre-solve inputs:
restricted holding weights, required gains from full exits, gain budget, optional
loss-lot capacity, cash reserve needs, overweight holdings, fees, and position caps.
Labels come from actual solver outcomes, not manually assigned categories.

The model is trained on 1,500 synthetic accounts (seed 2027) and evaluated once
on 500 untouched accounts generated from seed 2028, under the same stress-fixture
distribution. No holdout tuning or seed selection was performed. The new fixtures
vary restrictions, gain budgets, cash needs, caps, and cost bases independently.

| Holdout measure | Result |
| --- | ---: |
| Accuracy | 90.0% |
| Majority-class baseline accuracy | 69.0% |
| Conflict F1 | 0.830 |
| ROC AUC | 0.964 |
| Brier score | 0.071 |
| True conflicts detected | 122 of 155 |
| Conflicts missed | 33 |
| False warnings | 17 |
| Solver-rejected fixtures excluded | 0 |

The model is useful as a demonstration of learned triage, not as an authorization
gate. Its mistakes are why every account is still solved and independently replayed.
This same-distribution synthetic test does not establish generalization to real
accounts. Accounts with different security/lot counts are explicitly flagged as
outside the training shape. Other distribution shifts are possible even within
the training shape. The displayed score is not an investment risk estimate.

The dashboard shows the three largest signed standardized-feature contributions
to the logistic score. These are associations relative to training averages, not
causal explanations. Coefficients, feature order, scaler values, and evaluation
metadata are shipped as JSON; serving uses NumPy without executable pickle files.

Reproduce training and evaluation:

```bash
pip install -e '.[training]'
python benchmarks/train_triage.py --train 1500 --test 500
```

The model artifact is `src/lotwise/artifacts/triage_model.json`. Full evaluation
and held-out predictions are in `benchmarks/ai-results/`. The original portfolio
benchmark and its measured results remain separate from these ML measurements.

## Evidence report and optional LLM questions

Every persisted proposal has a local report with citations to its actual status,
constraint limits, metrics, and exceptions. This report uses templates and is
explicitly labeled **Local evidence report**, not presented as generated AI text.

When `OPENAI_API_KEY` is set in the server environment, the question field enables
an optional OpenAI integration. `OPENAI_MODEL` chooses the model (default
`gpt-6-astra`). No API key is ever sent to the browser. Restart the server after
changing its environment; a copied `.env` file alone is not automatically loaded
by the Python server. Docker Compose reads the variables from its environment or
local `.env` file.

The server sends the question and a selected evidence catalogue, including a
sample of up to eight large trades. It does not send the full account snapshot
or grant SQL, file, brokerage, or trading tools. Responses use the OpenAI
[structured outputs API](https://developers.openai.com/api/docs/guides/structured-outputs).
The request sets `store=False`; this setting is not a claim of zero provider
retention. The UI states that optional AI questions send selected facts to OpenAI.

Each generated finding must cite an ID from that proposal's evidence catalogue.
Unknown citations, missing structured answers, and digits in generated prose are
rejected. Numeric values are rendered directly from evidence rather than invented
in prose. These checks constrain the output but do not prove that every qualitative
interpretation is correct; the UI asks users to verify cited facts. Questions and
evidence are explicitly treated as untrusted data in the system prompt.

Valid AI answers, their question, provider/model, and citations are persisted in
`ai_reviews`, with an event added to the proposal audit trail. Existing proposals
and their trades remain unchanged. Provider failures return a clear error and
leave the local report available.

The OpenAI integration is tested with mocked provider responses for successful
answers, invalid citations, invented numbers, refusals/incomplete answers,
provider failures, audit persistence, and proposal immutability. **No live LLM
request was run**, because an API key was not configured. The local ML model was
trained, evaluated, and run normally.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/ai/status` | Local ML and optional provider availability |
| POST | `/ai/triage` | Pre-solve conflict score for an account |
| GET | `/proposals/{id}/review` | Local report, cited facts, and ML triage |
| POST | `/proposals/{id}/ask` | Optional structured LLM answer and audit event |
