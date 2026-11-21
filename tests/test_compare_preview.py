"""双基准日对照：换日轻量通道、快照一致、补录/公告变更检测、双列送审。"""
from __future__ import annotations

import importlib
from fractions import Fraction

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    import app.db as db
    importlib.reload(db)
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c, db_path


def _import_prices(c, rows):
    resp = c.post("/api/prices/import", json={
        "prices": [{"trade_date": d, "close_cents": p} for d, p in rows]
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def _ann(c, **kw):
    return c.post("/api/announcements/import", json=kw)


def _preview(c, base_date, **extra):
    params = {"start_date": "2026-01-05", "end_date": "2026-01-25",
              "base_date": base_date}
    params.update(extra)
    resp = c.get("/api/adjustments/preview", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _rebase(c, base_date, **extra):
    params = {"start_date": "2026-01-05", "end_date": "2026-01-25",
              "base_date": base_date}
    params.update(extra)
    return c.get("/api/adjustments/rebase", params=params)


def _lock(c, base_date, basis_hash, prices_hash=None):
    body = {"start_date": "2026-01-05", "end_date": "2026-01-25",
            "base_date": base_date, "basis_hash": basis_hash}
    if prices_hash is not None:
        body["prices_hash"] = prices_hash
    return c.post("/api/submissions", json=body)


def _seed_chain(c):
    """区间内两次事件：1:2 拆股（01-10 生效）+ 4:1 并股（01-20 生效）。"""
    _import_prices(c, [("2026-01-05", 1000), ("2026-01-15", 1000),
                       ("2026-01-25", 1000)])
    _ann(c, ann_no="A", revision=1, effective_date="2026-01-10",
         old_shares=1, new_shares=2)
    _ann(c, ann_no="B", revision=1, effective_date="2026-01-20",
         old_shares=4, new_shares=1)


# ---------------------------------------------------------- 换日轻量通道

def test_rebase_returns_only_verification_and_factors(client):
    """换日响应只含核验信息与换算比例：不回原价、不回折算结果。"""
    c, _ = client
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    resp = _rebase(c, "2026-01-05", basis_hash=pv["basis_hash"],
                   prices_hash=pv["prices_hash"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["stale"] is False
    assert set(data) == {"start_date", "end_date", "base_date", "basis_hash",
                         "prices_hash", "stale", "note", "factors"}
    assert set(data["factors"][0]) == {"trade_date", "factor"}  # 无行情字段
    assert data["basis_hash"] == pv["basis_hash"]
    assert data["prices_hash"] == pv["prices_hash"]


def test_rebase_factors_across_split_and_merge(client):
    """跨拆并股换日：换到事件另一侧的基准日，逐日因子与服务端口径一致。"""
    c, _ = client
    from app.services import factor_for_date, relevant_events
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    anns = c.get("/api/announcements").json()["announcements"]
    events = relevant_events("2026-01-05", "2026-01-25", "2026-01-05", anns)

    for new_base in ("2026-01-05", "2026-01-15", "2026-01-25"):
        data = _rebase(c, new_base, basis_hash=pv["basis_hash"],
                       prices_hash=pv["prices_hash"]).json()
        got = {f["trade_date"]: f["factor"] for f in data["factors"]}
        for d in ("2026-01-05", "2026-01-15", "2026-01-25"):
            assert got[d] == str(factor_for_date(d, new_base, events))
    # 换到 01-05（旧口径）：01-25 对两事件均处新口径，因子 2 * 1/4 = 1/2
    data = _rebase(c, "2026-01-05").json()
    got = {f["trade_date"]: f["factor"] for f in data["factors"]}
    assert got["2026-01-25"] == str(Fraction(1, 2))
    assert got["2026-01-15"] == "2"   # 仅拆股事件处新口径：×2
    assert got["2026-01-05"] == "1"


def test_rebase_validates_base_date(client):
    """两列基准日都须是区间内有收盘价的日期。"""
    c, _ = client
    _seed_chain(c)
    # 区间内但无收盘价
    assert _rebase(c, "2026-01-06").status_code == 422
    # 越出区间
    assert _rebase(c, "2026-01-26").status_code == 422
    # 非法格式
    assert _rebase(c, "2026/01/05").status_code == 422


def test_preview_exposes_prices_hash_and_is_stable(client):
    """首次预览带回行情指纹；同区间重取指纹稳定（同一已提交状态可复现）。"""
    c, _ = client
    _seed_chain(c)
    from app.services import prices_hash_of
    pv1 = _preview(c, "2026-01-25")
    pv2 = _preview(c, "2026-01-05")
    expect = prices_hash_of([{"trade_date": d, "close_cents": p} for d, p in [
        ("2026-01-05", 1000), ("2026-01-15", 1000), ("2026-01-25", 1000)]])
    assert pv1["prices_hash"] == expect == pv2["prices_hash"]
    # 带正确指纹预览：非 stale
    pv3 = _preview(c, "2026-01-25", basis_hash=pv1["basis_hash"],
                   prices_hash=pv1["prices_hash"])
    assert pv3["stale"] is False


# ---------------------------------------------------------- 补录/变更检测

def test_rebase_detects_price_backfill(client):
    """区间内补录行情后换日：stale 并提示，指纹已变。"""
    c, _ = client
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    _import_prices(c, [("2026-01-08", 1500)])  # 区间内补录
    data = _rebase(c, "2026-01-05", basis_hash=pv["basis_hash"],
                   prices_hash=pv["prices_hash"]).json()
    assert data["stale"] is True
    assert "补录" in data["note"]
    assert data["prices_hash"] != pv["prices_hash"]
    assert data["basis_hash"] == pv["basis_hash"]  # 公告未动


def test_rebase_detects_announcement_correction(client):
    """公告更正后换日：stale 并提示公告已变更。"""
    c, _ = client
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    _ann(c, ann_no="A", revision=2, effective_date="2026-01-10",
         old_shares=1, new_shares=4)
    data = _rebase(c, "2026-01-05", basis_hash=pv["basis_hash"],
                   prices_hash=pv["prices_hash"]).json()
    assert data["stale"] is True
    assert "公告" in data["note"]
    assert data["basis_hash"] != pv["basis_hash"]


def test_lock_rejected_after_backfill_then_refresh_succeeds(client):
    """补录后送审 409 拒绝；刷新预览拿到新指纹后送审成功。"""
    c, _ = client
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    _import_prices(c, [("2026-01-08", 1500)])  # 补录
    stale = _lock(c, "2026-01-25", pv["basis_hash"], pv["prices_hash"])
    assert stale.status_code == 409
    assert "补录" in stale.json()["detail"]["message"]

    fresh = _preview(c, "2026-01-25")
    ok = _lock(c, "2026-01-25", fresh["basis_hash"], fresh["prices_hash"])
    assert ok.status_code == 200
    by_date = {p["trade_date"]: p["adjusted_cents"] for p in ok.json()["points"]}
    # 补录日对两事件均处旧口径，因子 1/2 * 4 = 2：1500 -> 3000
    assert by_date["2026-01-08"] == 3000


def test_lock_rejected_after_announcement_correction(client):
    """公告更正后，旧指纹送审 409（basis 与 prices 双重校验各自独立）。"""
    c, _ = client
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    _ann(c, ann_no="A", revision=2, effective_date="2026-01-10",
         old_shares=1, new_shares=4)
    r = _lock(c, "2026-01-25", pv["basis_hash"], pv["prices_hash"])
    assert r.status_code == 409
    assert "公告" in r.json()["detail"]["message"]


# ---------------------------------------------------------- 双列送审

def test_both_columns_lock_via_same_entry(client):
    """两列均可走原送审入口；各自返回序列与各自基准日预览逐点一致。"""
    c, _ = client
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    for base in ("2026-01-25", "2026-01-05"):
        locked = _lock(c, base, pv["basis_hash"], pv["prices_hash"])
        assert locked.status_code == 200
        expect = _preview(c, base)["points"]
        assert locked.json()["points"] == expect  # 与所选列一致
        assert locked.json()["prices_hash"] == pv["prices_hash"]
    # 两次送审各自留底，凭号重开仍各自重现
    ids = [ _lock(c, b, pv["basis_hash"], pv["prices_hash"]).json()["submission_id"]
            for b in ("2026-01-25", "2026-01-05") ]
    for sub_id, base in zip(ids, ("2026-01-25", "2026-01-05")):
        reopened = c.get(f"/api/submissions/{sub_id}").json()
        assert reopened["base_date"] == base
        assert reopened["points"] == _preview(c, base)["points"]


def test_lock_without_prices_hash_keeps_backward_compat(client):
    """旧客户端不带 prices_hash 仍可送审（仅校验公告指纹）。"""
    c, _ = client
    _seed_chain(c)
    pv = _preview(c, "2026-01-25")
    r = _lock(c, "2026-01-25", pv["basis_hash"])
    assert r.status_code == 200
    assert r.json()["prices_hash"] == pv["prices_hash"]
