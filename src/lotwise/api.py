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
from lotwise.models import Account, Proposal
from lotwise.optimizer import Rebalancer
from lotwise.review import InvalidReview, Question, ReviewUnavailable, ask, local_report
from lotwise.synthetic import generate_accounts, impossible_account
from lotwise.triage import predict


def create_app(database_url: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.repository = Repository(
            database_url or os.getenv("DATABASE_URL", "sqlite:///lotwise.db")
        )
        app.state.rebalancer = Rebalancer()
        app.state.lock = threading.Lock()
        app.state.review_client = None
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

    @app.get("/ai/status")
    def ai_status():
        return {
            "local_ml": True,
            "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
            "model": os.getenv("OPENAI_MODEL", "gpt-6-astra"),
            "data_note": "Local ML stays on this server. AI questions send the question and selected proposal facts to OpenAI.",
        }

    @app.post("/ai/triage")
    def triage(account: Account):
        return predict(account)

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

    def review_inputs(proposal_id):
        repo = app.state.repository
        payload = repo.get(proposal_id)
        if payload is None:
            raise HTTPException(404, "Proposal not found")
        account = repo.account_for(proposal_id)
        return account, Proposal.model_validate(
            {k: v for k, v in payload.items() if k != "proposal_id"}
        )

    @app.get("/proposals/{proposal_id}/review")
    def review(proposal_id: str):
        account, proposal = review_inputs(proposal_id)
        return local_report(account, proposal)

    @app.post("/proposals/{proposal_id}/ask")
    def ask_question(proposal_id: str, body: Question):
        account, proposal = review_inputs(proposal_id)
        try:
            answer = ask(account, proposal, body.question, client=app.state.review_client)
        except ReviewUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except InvalidReview as exc:
            raise HTTPException(502, str(exc)) from exc
        return app.state.repository.save_review(proposal_id, body.question, answer)

    return app


app = create_app()
