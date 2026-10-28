import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from lotwise.api import create_app
from lotwise.database import AuditRow, LotRow, ProposalRow, Repository, Snapshot, TradeRow
from lotwise.optimizer import Rebalancer
from lotwise.synthetic import generate_accounts, impossible_account


@pytest.fixture
def client():
    with TestClient(create_app("sqlite:///:memory:")) as c:
        yield c


def test_create_retrieve_audit(client):
    account = generate_accounts(1)[0].model_dump()
    response = client.post("/proposals", json=account)
    assert response.status_code == 201
    proposal = response.json()
    assert proposal["status"] == "optimal"
    assert client.get(f"/proposals/{proposal['proposal_id']}").json() == proposal
    audit = client.get(f"/proposals/{proposal['proposal_id']}/audit").json()
    assert len(audit) == 1
    assert audit[0]["input_hash"] == proposal["input_hash"]


def test_idempotency_and_conflict(client):
    account = generate_accounts(1)[0].model_dump()
    headers = {"Idempotency-Key": "test-key"}
    first = client.post("/proposals", json=account, headers=headers)
    repeat = client.post("/proposals", json=account, headers=headers)
    assert first.json() == repeat.json()
    account["cash"] += 1
    assert client.post("/proposals", json=account, headers=headers).status_code == 409


def test_concurrent_duplicate_requests(client):
    account = generate_accounts(1)[0].model_dump()

    def post(_):
        return client.post("/proposals", json=account, headers={"Idempotency-Key": "concurrent"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(post, range(16)))
    assert all(r.status_code == 201 for r in responses)
    assert len({r.json()["proposal_id"] for r in responses}) == 1


def test_validation_and_missing(client):
    assert client.post("/proposals", json={"cash": -1}).status_code == 422
    assert client.get("/proposals/missing").status_code == 404
    assert client.get("/proposals/missing/audit").status_code == 404
    assert client.get("/health").json()["mode"] == "simulation"
    assert client.get("/demo/account").status_code == 200
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Trade proposal" in dashboard.text


def test_persist_infeasible_exception(client):
    response = client.post("/proposals", json=impossible_account().model_dump()).json()
    assert response["status"] == "infeasible"
    assert response["trades"] == []
    assert client.get(f"/proposals/{response['proposal_id']}/audit").json()[0]["exceptions"]


def exercise_database(url):
    repo = Repository(url)
    account = generate_accounts(1, seed=123)[0]
    proposal = Rebalancer().propose(account)
    saved = repo.save(account, proposal, "rollback-test")
    assert repo.get(saved["proposal_id"]) == saved
    other = generate_accounts(1, seed=124)[0]
    with pytest.raises(IntegrityError):
        repo.save(other, Rebalancer().propose(other), "rollback-test")
    with Session(repo.engine) as session:
        assert session.scalar(select(func.count()).select_from(Snapshot)) == 1
        assert session.scalar(select(func.count()).select_from(LotRow)) == 60
        assert session.scalar(select(func.count()).select_from(ProposalRow)) == 1
        assert session.scalar(select(func.count()).select_from(AuditRow)) == 1
        assert session.scalar(select(func.count()).select_from(TradeRow)) == len(proposal.trades)
    repo.close()


def test_atomic_rollback_sqlite():
    exercise_database("sqlite:///:memory:")


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="requires isolated PostgreSQL database"
)
def test_atomic_rollback_postgres():
    exercise_database(os.environ["TEST_DATABASE_URL"])
