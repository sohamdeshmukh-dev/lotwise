import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from lotwise.api import create_app
from lotwise.models import Account
from lotwise.optimizer import Rebalancer
from lotwise.review import Answer, Finding, InvalidReview, ReviewUnavailable, ask, local_report
from lotwise.synthetic import generate_accounts, impossible_account
from lotwise.triage import FEATURES, features, load_model, predict


@pytest.fixture
def account():
    return generate_accounts(1)[0]


def client_for(answer):
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(output_parsed=answer)
    return client


def good_answer():
    return Answer(
        findings=[
            Finding(
                explanation="The released proposal passed the modeled checks.",
                evidence_ids=["proposal.status"],
            )
        ],
        limitations=[],
    )


def test_triage_is_local_finite_and_does_not_mutate(account):
    before = account.model_dump_json()
    result = predict(account)
    assert 0 <= result["conflict_score"] <= 1
    assert len(result["drivers"]) == 3
    assert all(driver["feature"] in FEATURES for driver in result["drivers"])
    assert account.model_dump_json() == before
    assert predict(account) == result


def test_features_exclude_ids_and_post_solve_results(account):
    data = account.model_dump()
    data["account_id"] = "a completely different identity"
    for i, lot in enumerate(data["lots"]):
        lot["lot_id"] = f"different-{i}"
    np.testing.assert_array_equal(features(account), features(Account.model_validate(data)))


def test_out_of_training_shape_is_labeled():
    assert predict(impossible_account())["outside_training_shape"]


def test_exported_evaluation_matches_holdout_predictions():
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root / "benchmarks/ai-results/holdout.csv").open()))
    metadata = load_model()["evaluation"]
    actual = [int(row["actual_conflict"]) for row in rows]
    predicted = [int(row["predicted_conflict"]) for row in rows]
    assert len(rows) == metadata["test_accounts"]
    assert sum(a == b for a, b in zip(actual, predicted)) / len(rows) == metadata["accuracy"]
    assert metadata["training_seed"] != metadata["test_seed"]
    assert metadata["accuracy"] > metadata["majority_baseline_accuracy"]
    assert len({row["input_hash"] for row in rows}) == len(rows)


@pytest.mark.parametrize("fixture", [lambda: generate_accounts(1)[0], impossible_account])
def test_local_reports_have_real_evidence(fixture):
    account = fixture()
    proposal = Rebalancer().propose(account)
    report = local_report(account, proposal)
    assert report["provider"] == "local"
    ids = {fact["id"] for fact in report["evidence"]}
    assert all(set(f["evidence_ids"]).issubset(ids) for f in report["findings"])
    assert "not an LLM" in report["limitations"][0]


def test_no_key_requires_no_external_request(account, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ReviewUnavailable, match="OPENAI_API_KEY"):
        ask(account, Rebalancer().propose(account), "Explain the proposal")


def test_structured_llm_request_is_evidence_only_and_read_only(account):
    proposal = Rebalancer().propose(account)
    before = proposal.model_dump_json()
    client = client_for(good_answer())
    answer = ask(account, proposal, "Why these trades?", client=client)
    kwargs = client.responses.parse.call_args.kwargs
    sent = json.loads(kwargs["input"][1]["content"])
    assert set(sent) == {"question", "evidence"}
    assert "tools" not in kwargs
    assert kwargs["store"] is False
    assert answer["provider"] == "openai"
    assert proposal.model_dump_json() == before


@pytest.mark.parametrize(
    "answer",
    [
        Answer(
            findings=[Finding(explanation="Invented source.", evidence_ids=["other.account"])],
            limitations=[],
        ),
        Answer(
            findings=[
                Finding(explanation="This saves 999 dollars.", evidence_ids=["proposal.status"])
            ],
            limitations=[],
        ),
        None,
        {"findings": [], "limitations": []},
    ],
)
def test_reject_unsupported_or_incomplete_llm_answers(account, answer):
    with pytest.raises(InvalidReview):
        ask(account, Rebalancer().propose(account), "Explain", client=client_for(answer))


def test_provider_failure_preserves_local_review(account):
    client = Mock()
    client.responses.parse.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    proposal = Rebalancer().propose(account)
    with pytest.raises(ReviewUnavailable):
        ask(account, proposal, "Explain", client=client)
    assert local_report(account, proposal)["findings"]


def test_review_api_and_ai_audit(account, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(create_app("sqlite:///:memory:")) as client:
        assert client.get("/ai/status").json()["local_ml"]
        assert not client.get("/ai/status").json()["llm_configured"]
        assert client.post("/ai/triage", json=account.model_dump()).status_code == 200
        original = client.post("/proposals", json=account.model_dump()).json()
        proposal_id = original["proposal_id"]
        assert client.get(f"/proposals/{proposal_id}/review").json()["provider"] == "local"
        assert (
            client.post(
                f"/proposals/{proposal_id}/ask", json={"question": "Why these trades?"}
            ).status_code
            == 503
        )
        client.app.state.review_client = client_for(good_answer())
        response = client.post(
            f"/proposals/{proposal_id}/ask", json={"question": "Why these trades?"}
        )
        assert response.status_code == 200
        assert response.json()["review_id"]
        assert client.get(f"/proposals/{proposal_id}").json() == original
        audit = client.get(f"/proposals/{proposal_id}/audit").json()
        assert audit[-1]["event"] == "ai_review"
        assert client.get("/proposals/missing/review").status_code == 404
        assert (
            client.post("/proposals/missing/ask", json={"question": "Explain"}).status_code == 404
        )
        assert (
            client.post(f"/proposals/{proposal_id}/ask", json={"question": "x" * 1501}).status_code
            == 422
        )
