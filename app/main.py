"""FastAPI 入口：行情/公告导入、折算预览、锁定送审、凭号重开。"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.db import db_cursor, init_db
from app.models import (
    AdjustedPoint,
    AnnouncementIn,
    AnnouncementResult,
    LockInRequest,
    LockInResult,
    PreviewResult,
    PriceImportResult,
    PriceIn,
    RebaseResult,
    ReopenResult,
)
from app.services import (
    basis_hash_of,
    body_hash_of,
    build_series,
    prices_hash_of,
    relevant_events,
    series_factors,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="拆并股收盘价折算", version="1.0.0", lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ---------------------------------------------------------------- 行情导入

class PriceBatch(BaseModel):
    prices: list[PriceIn] = Field(min_length=1)


@app.post("/api/prices/import", response_model=PriceImportResult)
def import_prices(batch: PriceBatch):
    """一次导入单股日收盘价（分）。

    同日期同价视为重复跳过；同日期不同价属冲突，整批退回（409），
    底表保持只增不改不删，送审与更正共用此底表。
    """
    received = len(batch.prices)
    seen: dict[str, int] = {}
    for item in batch.prices:
        if item.trade_date in seen and seen[item.trade_date] != item.close_cents:
            raise HTTPException(
                status_code=422,
                detail=f"批次内同一交易日 {item.trade_date} 出现不同收盘价",
            )
        seen[item.trade_date] = item.close_cents

    conflicts: list[dict] = []
    with db_cursor() as conn:
        for trade_date, close_cents in seen.items():
            row = conn.execute(
                "SELECT close_cents FROM prices WHERE trade_date=?", (trade_date,)
            ).fetchone()
            if row is not None and row["close_cents"] != close_cents:
                conflicts.append({
                    "trade_date": trade_date,
                    "existing_cents": row["close_cents"],
                    "submitted_cents": close_cents,
                })
        if conflicts:
            raise HTTPException(
                status_code=409,
                detail={"message": "底表已存在同日期不同收盘价，整批未写入",
                        "conflicts": conflicts},
            )
        inserted = 0
        for trade_date, close_cents in seen.items():
            cur = conn.execute(
                "INSERT OR IGNORE INTO prices(trade_date, close_cents) VALUES (?,?)",
                (trade_date, close_cents),
            )
            inserted += cur.rowcount
    return PriceImportResult(
        received=received,
        inserted=inserted,
        skipped_identical=received - inserted,
        conflicts=conflicts,
    )


@app.get("/api/prices")
def list_prices(start_date: str | None = None, end_date: str | None = None):
    sql = "SELECT trade_date, close_cents FROM prices"
    where, params = [], []
    if start_date:
        where.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("trade_date <= ?")
        params.append(end_date)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY trade_date"
    with db_cursor() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {"prices": [dict(r) for r in rows]}


# ---------------------------------------------------------------- 公告导入

def _current_announcements(conn) -> list[dict]:
    """每号取最高修订（ann_no 为主键，表中即现行版本）。"""
    rows = conn.execute(
        "SELECT ann_no, revision, effective_date, old_shares, new_shares, body_hash "
        "FROM announcements ORDER BY ann_no"
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/announcements/import", response_model=AnnouncementResult)
def import_announcement(ann: AnnouncementIn):
    """公告收发规则：

    - 同号同版原样重送（正文哈希一致）→ 沿用原公告，记 reused；
    - 同号同版异文重送 → 退回 409，记 rejected_diff；
    - 更高修订 → 更正现行版本，记 updated（旧版仅留流水，现行表只留最高修订）；
    - 修订号低于现行 → 退回 422，记 rejected_old。
    """
    payload = ann.model_dump()
    digest = body_hash_of(payload)
    with db_cursor() as conn:
        row = conn.execute(
            "SELECT revision, body_hash FROM announcements WHERE ann_no=?",
            (ann.ann_no,),
        ).fetchone()

        if row is not None and ann.revision == row["revision"]:
            if digest == row["body_hash"]:
                conn.execute(
                    "INSERT INTO announcements_log(ann_no, revision, effective_date,"
                    " old_shares, new_shares, body_hash, outcome, note)"
                    " VALUES (?,?,?,?,?,?, 'reused', '同号同版原样重送，沿用原公告')",
                    (ann.ann_no, ann.revision, ann.effective_date, ann.old_shares,
                     ann.new_shares, digest),
                )
                return AnnouncementResult(**payload, outcome="reused",
                                          note="同号同版原样重送，沿用原公告")
            conn.execute(
                "INSERT INTO announcements_log(ann_no, revision, effective_date,"
                " old_shares, new_shares, body_hash, outcome, note)"
                " VALUES (?,?,?,?,?,?, 'rejected_diff', '同号同版异文，退回')",
                (ann.ann_no, ann.revision, ann.effective_date, ann.old_shares,
                 ann.new_shares, digest),
            )
            raise HTTPException(
                status_code=409,
                detail=f"公告 {ann.ann_no} 第 {ann.revision} 版已存在且正文不同，异文退回",
            )

        if row is not None and ann.revision < row["revision"]:
            conn.execute(
                "INSERT INTO announcements_log(ann_no, revision, effective_date,"
                " old_shares, new_shares, body_hash, outcome, note)"
                " VALUES (?,?,?,?,?,?, 'rejected_old', '修订号低于现行最高修订，退回')",
                (ann.ann_no, ann.revision, ann.effective_date, ann.old_shares,
                 ann.new_shares, digest),
            )
            raise HTTPException(
                status_code=422,
                detail=f"公告 {ann.ann_no} 现行修订为第 {row['revision']} 版，"
                       f"收到第 {ann.revision} 版（更低），退回",
            )

        outcome = "inserted" if row is None else "updated"
        note = "首次导入" if row is None else f"更高修订，替代第 {row['revision']} 版"
        conn.execute(
            "INSERT INTO announcements(ann_no, revision, effective_date, old_shares,"
            " new_shares, body_hash) VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(ann_no) DO UPDATE SET revision=excluded.revision,"
            " effective_date=excluded.effective_date, old_shares=excluded.old_shares,"
            " new_shares=excluded.new_shares, body_hash=excluded.body_hash",
            (ann.ann_no, ann.revision, ann.effective_date, ann.old_shares,
             ann.new_shares, digest),
        )
        conn.execute(
            "INSERT INTO announcements_log(ann_no, revision, effective_date,"
            " old_shares, new_shares, body_hash, outcome, note)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (ann.ann_no, ann.revision, ann.effective_date, ann.old_shares,
             ann.new_shares, digest, outcome, note),
        )
        return AnnouncementResult(**payload, outcome=outcome, note=note)


@app.get("/api/announcements")
def list_announcements():
    with db_cursor() as conn:
        anns = _current_announcements(conn)
    return {"announcements": anns}


# ---------------------------------------------------------------- 预览/送审

def _load_prices(conn, start_date: str, end_date: str) -> list[dict]:
    rows = conn.execute(
        "SELECT trade_date, close_cents FROM prices"
        " WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (start_date, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


def _validate_window(conn, start_date: str, end_date: str, base_date: str) -> None:
    if not (start_date <= end_date):
        raise HTTPException(status_code=422, detail="start_date 不能晚于 end_date")
    if not (start_date <= base_date <= end_date):
        raise HTTPException(status_code=422,
                            detail="基准交易日必须落在所选区间内")
    row = conn.execute(
        "SELECT 1 FROM prices WHERE trade_date=?", (base_date,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=422,
                            detail=f"基准交易日 {base_date} 在行情底表中不存在")
    cnt = conn.execute(
        "SELECT COUNT(*) AS c FROM prices WHERE trade_date BETWEEN ? AND ?",
        (start_date, end_date),
    ).fetchone()["c"]
    if cnt == 0:
        raise HTTPException(status_code=422,
                            detail="所选区间内没有任何行情数据")


def _stale_note(basis_changed: bool, prices_changed: bool) -> str:
    parts = []
    if prices_changed:
        parts.append("区间内行情有补录")
    if basis_changed:
        parts.append("现行公告已变更")
    return "；".join(parts) + "，须刷新后再送审"


def _preview(conn, start_date: str, end_date: str, base_date: str,
             client_basis_hash: str | None,
             client_prices_hash: str | None = None) -> PreviewResult:
    # 显式事务：公告与行情取自同一已提交状态（快照一致），
    # 送审路径的后续写入也加入同一事务，校验与落库原子完成。
    conn.execute("BEGIN")
    _validate_window(conn, start_date, end_date, base_date)
    anns = _current_announcements(conn)
    current_hash = basis_hash_of(anns)
    events = relevant_events(start_date, end_date, base_date, anns)
    prices = _load_prices(conn, start_date, end_date)
    current_prices_hash = prices_hash_of(prices)
    points = build_series(prices, base_date, events)
    basis_changed = client_basis_hash is not None and client_basis_hash != current_hash
    prices_changed = (client_prices_hash is not None
                      and client_prices_hash != current_prices_hash)
    stale = basis_changed or prices_changed
    return PreviewResult(
        start_date=start_date,
        end_date=end_date,
        base_date=base_date,
        basis_hash=current_hash,
        prices_hash=current_prices_hash,
        basis=events,
        points=points,
        stale=stale,
        note=_stale_note(basis_changed, prices_changed) if stale else "",
    )


@app.get("/api/adjustments/preview", response_model=PreviewResult)
def preview(start_date: str, end_date: str, base_date: str,
            basis_hash: str | None = None, prices_hash: str | None = None):
    """区间折算预览。可带上次拿到的 basis_hash / prices_hash，检测公告或行情是否已变。"""
    with db_cursor() as conn:
        return _preview(conn, start_date, end_date, base_date,
                        basis_hash, prices_hash)


@app.get("/api/adjustments/rebase", response_model=RebaseResult)
def rebase(start_date: str, end_date: str, base_date: str,
           basis_hash: str | None = None, prices_hash: str | None = None):
    """换对照基准日：只回核验信息（双指纹 + stale）与逐日换算比例。

    不回行情与折算结果——网页复用已载入的原价与精确因子自行折算，
    最后才四舍五入到分。新基准日同样须是区间内有收盘价的日期。
    """
    with db_cursor() as conn:
        conn.execute("BEGIN")  # 与首次预览同一快照口径
        _validate_window(conn, start_date, end_date, base_date)
        anns = _current_announcements(conn)
        current_hash = basis_hash_of(anns)
        prices = _load_prices(conn, start_date, end_date)
        current_prices_hash = prices_hash_of(prices)
        events = relevant_events(start_date, end_date, base_date, anns)
        basis_changed = basis_hash is not None and basis_hash != current_hash
        prices_changed = (prices_hash is not None
                          and prices_hash != current_prices_hash)
        stale = basis_changed or prices_changed
        return RebaseResult(
            start_date=start_date,
            end_date=end_date,
            base_date=base_date,
            basis_hash=current_hash,
            prices_hash=current_prices_hash,
            stale=stale,
            note=_stale_note(basis_changed, prices_changed) if stale else "",
            factors=series_factors(prices, base_date, events),
        )


@app.post("/api/submissions", response_model=LockInResult)
def lock_submission(req: LockInRequest):
    """锁定预览并取得送审号。

    依据指纹与现行公告不一致、或区间行情指纹与底表不一致（区间内补录）
    时 409 拒绝，须先刷新。两列基准日共用本入口，返回序列按所送基准日
    重算，与所选列一致。
    """
    with db_cursor() as conn:
        preview_data = _preview(conn, req.start_date, req.end_date,
                                req.base_date, req.basis_hash, req.prices_hash)
        if preview_data.stale:
            raise HTTPException(
                status_code=409,
                detail={"message": preview_data.note,
                        "current_basis_hash": preview_data.basis_hash,
                        "current_prices_hash": preview_data.prices_hash},
            )
        submission_id = "SB-" + uuid.uuid4().hex[:16].upper()
        conn.execute(
            "INSERT INTO submissions(submission_id, start_date, end_date,"
            " base_date, basis_hash) VALUES (?,?,?,?,?)",
            (submission_id, req.start_date, req.end_date, req.base_date,
             preview_data.basis_hash),
        )
        # 只留重算所需的依据，不复制行情底表
        for e in preview_data.basis:
            conn.execute(
                "INSERT INTO submission_basis(submission_id, ann_no, revision,"
                " effective_date, old_shares, new_shares) VALUES (?,?,?,?,?,?)",
                (submission_id, e["ann_no"], e["revision"], e["effective_date"],
                 e["old_shares"], e["new_shares"]),
            )
        return LockInResult(submission_id=submission_id,
                            basis_hash=preview_data.basis_hash,
                            prices_hash=preview_data.prices_hash,
                            points=preview_data.points)


@app.get("/api/submissions/{submission_id}", response_model=ReopenResult)
def reopen_submission(submission_id: str):
    """凭送审号重开：按快照依据 + 同一行情底表重现原序列。

    即使公告随后被更正、或进程重启，序列仍与锁定时一致；
    同时返回现行依据指纹与 stale 标记供研究人员比对。
    """
    with db_cursor() as conn:
        sub = conn.execute(
            "SELECT * FROM submissions WHERE submission_id=?", (submission_id,)
        ).fetchone()
        if sub is None:
            raise HTTPException(status_code=404,
                                detail=f"送审号 {submission_id} 不存在")
        basis_rows = conn.execute(
            "SELECT ann_no, revision, effective_date, old_shares, new_shares"
            " FROM submission_basis WHERE submission_id=? ORDER BY effective_date, ann_no",
            (submission_id,),
        ).fetchall()
        basis = [dict(r) for r in basis_rows]
        prices = _load_prices(conn, sub["start_date"], sub["end_date"])
        points = build_series(prices, sub["base_date"], basis)

        current_hash = basis_hash_of(_current_announcements(conn))
        stale = current_hash != sub["basis_hash"]
        note = ("公告已更正：以下为按送审时依据快照重现的原序列"
                if stale else "现行公告与送审依据一致")
        return ReopenResult(
            submission_id=submission_id,
            start_date=sub["start_date"],
            end_date=sub["end_date"],
            base_date=sub["base_date"],
            basis_hash=sub["basis_hash"],
            current_basis_hash=current_hash,
            stale=stale,
            note=note,
            points=points,
        )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
