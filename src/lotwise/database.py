"""Transactional SQL persistence of snapshots, proposals, lots, and audit events."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Float, ForeignKey, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from lotwise.models import Account, Proposal
from lotwise.validation import fingerprint


class Base(DeclarativeBase):
    pass


class Snapshot(Base):
    __tablename__ = "account_snapshots"
    input_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(80), index=True)
    input_payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))


class SecurityRow(Base):
    __tablename__ = "securities"
    input_hash: Mapped[str] = mapped_column(
        ForeignKey("account_snapshots.input_hash"), primary_key=True
    )
    symbol: Mapped[str] = mapped_column(String(24), primary_key=True)
    price: Mapped[float] = mapped_column(Float)
    target_weight: Mapped[float] = mapped_column(Float)


class LotRow(Base):
    __tablename__ = "tax_lots"
    input_hash: Mapped[str] = mapped_column(
        ForeignKey("account_snapshots.input_hash"), primary_key=True
    )
    lot_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(24))
    shares: Mapped[float] = mapped_column(Float)
    cost_basis_per_share: Mapped[float] = mapped_column(Float)


class ProposalRow(Base):
    __tablename__ = "proposals"
    proposal_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    input_hash: Mapped[str] = mapped_column(ForeignKey("account_snapshots.input_hash"), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))


class TradeRow(Base):
    __tablename__ = "trades"
    proposal_id: Mapped[str] = mapped_column(ForeignKey("proposals.proposal_id"), primary_key=True)
    sequence: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    side: Mapped[str] = mapped_column(String(4))
    lot_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    realized_gain: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)


class AuditRow(Base):
    __tablename__ = "audit_events"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("proposals.proposal_id"), index=True)
    created_at: Mapped[str] = mapped_column(String(40))
    detail: Mapped[dict] = mapped_column(JSON)


class IdempotencyConflict(ValueError):
    pass


class Repository:
    def __init__(self, url="sqlite:///lotwise.db"):
        options = {}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                options["poolclass"] = StaticPool
        self.engine = create_engine(url, **options)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def foreign_keys(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self.engine)

    def by_key(self, key: str | None, account: Account):
        if key is None:
            return None
        with Session(self.engine) as session:
            row = session.scalar(select(ProposalRow).where(ProposalRow.idempotency_key == key))
            if row is None:
                return None
            if row.input_hash != fingerprint(account):
                raise IdempotencyConflict("Idempotency key already belongs to different inputs")
            return {"proposal_id": row.proposal_id, **row.payload}

    def save(self, account: Account, proposal: Proposal, key: str | None = None):
        digest = fingerprint(account)
        if proposal.input_hash != digest or proposal.account_id != account.account_id:
            raise ValueError("proposal does not match input snapshot")
        proposal_id, now = str(uuid.uuid4()), datetime.now(UTC).isoformat()
        with Session(self.engine) as session, session.begin():
            if session.get(Snapshot, digest) is None:
                session.add(
                    Snapshot(
                        input_hash=digest,
                        account_id=account.account_id,
                        input_payload=account.model_dump(),
                        created_at=now,
                    )
                )
                session.flush()
                session.add_all(
                    SecurityRow(input_hash=digest, **s.model_dump()) for s in account.securities
                )
                session.add_all(
                    LotRow(input_hash=digest, **lot.model_dump()) for lot in account.lots
                )
            session.add(
                ProposalRow(
                    proposal_id=proposal_id,
                    input_hash=digest,
                    idempotency_key=key,
                    status=proposal.status,
                    payload=proposal.model_dump(),
                    created_at=now,
                )
            )
            session.flush()
            session.add_all(
                TradeRow(proposal_id=proposal_id, sequence=i, **trade.model_dump())
                for i, trade in enumerate(proposal.trades)
            )
            session.add(
                AuditRow(
                    event_id=str(uuid.uuid4()),
                    proposal_id=proposal_id,
                    created_at=now,
                    detail={
                        "input_hash": digest,
                        "solver_status": proposal.solver_status,
                        "status": proposal.status,
                        "exceptions": proposal.exceptions,
                        "engine_version": "0.1.0",
                    },
                )
            )
        return {"proposal_id": proposal_id, **proposal.model_dump()}

    def get(self, proposal_id):
        with Session(self.engine) as session:
            row = session.get(ProposalRow, proposal_id)
            return {"proposal_id": row.proposal_id, **row.payload} if row else None

    def audit(self, proposal_id):
        with Session(self.engine) as session:
            rows = session.scalars(select(AuditRow).where(AuditRow.proposal_id == proposal_id))
            return [{"created_at": row.created_at, **row.detail} for row in rows]

    def close(self):
        self.engine.dispose()
