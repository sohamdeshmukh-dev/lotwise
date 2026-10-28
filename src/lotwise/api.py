"""Local review API. No broker connection or live execution endpoints."""

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError

from lotwise.database import IdempotencyConflict, Repository
from lotwise.models import Account
from lotwise.optimizer import Rebalancer
from lotwise.synthetic import generate_accounts, impossible_account


def create_app(database_url: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.repository = Repository(
            database_url or os.getenv("DATABASE_URL", "sqlite:///lotwise.db")
        )
        app.state.rebalancer = Rebalancer()
        app.state.lock = threading.Lock()
        yield
        app.state.repository.close()

    app = FastAPI(
        title="LotWise",
        version="0.1.0",
        lifespan=lifespan,
        description="Auditable SMA rebalancing simulator. Synthetic data; proposal review only.",
    )

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "simulation"}

    @app.get("/demo/account")
    def demo_account(scenario: Literal["balanced", "restricted", "infeasible"] = "restricted"):
        if scenario == "infeasible":
            return impossible_account()
        return generate_accounts(3)[0 if scenario == "balanced" else 2]

    @app.post("/proposals", status_code=201)
    def propose(
        account: Account, idempotency_key: str | None = Header(default=None, max_length=100)
    ):
        # Serialize cached solver use and in-process idempotency checks. Database
        # uniqueness is the backstop across processes, not a Python lock alone.
        with app.state.lock:
            repo = app.state.repository
            try:
                existing = repo.by_key(idempotency_key, account)
                if existing:
                    return existing
                proposal = app.state.rebalancer.propose(account)
                try:
                    return repo.save(account, proposal, idempotency_key)
                except IntegrityError:
                    # A concurrent worker may have persisted this same key or snapshot.
                    existing = repo.by_key(idempotency_key, account)
                    if existing:
                        return existing
                    return repo.save(account, proposal, idempotency_key)
            except IdempotencyConflict as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.get("/proposals/{proposal_id}")
    def get_proposal(proposal_id: str):
        proposal = app.state.repository.get(proposal_id)
        if proposal is None:
            raise HTTPException(404, "Proposal not found")
        return proposal

    @app.get("/proposals/{proposal_id}/audit")
    def audit(proposal_id: str):
        if app.state.repository.get(proposal_id) is None:
            raise HTTPException(404, "Proposal not found")
        return app.state.repository.audit(proposal_id)

    return app


app = create_app()
