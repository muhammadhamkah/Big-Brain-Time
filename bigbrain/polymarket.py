"""Polymarket market making, measured before it is trusted: record the live market, then replay it with you as the maker.

On Polymarket a maker (a resting limit order) pays no fee and earns rebates; a taker pays up to 1.75% of the
notional at 50 cents. So "tiny profit, many times a day, no fees" is market making: rest a buy just under the
price and a sell just over it, and earn the gap whenever both fill. The risk is the other side of every fill:
resting orders get filled most often when someone knows more (news breaks, the underlying moves), and the
price then keeps going. Only data can say whether the gap earned beats what those fills lose.

**Record.** ``Recorder`` polls Polymarket's public CLOB, with no account or key: every few seconds the order
books of the chosen markets (``POST /books``), and their recent trades (``GET /live-activity/events/{market}``),
written as JSON lines. If a poll returns only trades it has never seen, some may have been missed between
polls; that is written down as a gap, never hidden.

**Replay.** ``simulate`` walks a recording in time order with a simple maker:

* it rests a bid at the best bid (or ``improve`` ticks better) and, once it holds shares, an ask at the best ask
  (or better); it never sells shares it does not hold, and stops buying at ``max_inventory``;
* a quote goes live ``latency`` seconds after the book it reacted to;
* **fills come only from recorded trades.** A trade *through* a quote's price fills it (the seller who traded
  below our bid had to pass it first). A trade *at* its price fills it only after the volume that was queued
  there before it has traded (``queue`` model); the stricter ``through`` model never fills at the price;
* after every fill the mid price 10 and 60 seconds later is recorded: the markout, the cost of being picked off;
* anything still held at the end is valued at the mid and, more honestly, as if sold to the best bid as a taker,
  fee included.

Maker rebates and liquidity rewards are not in the profit and loss: they depend on everyone else's volume and
on Polymarket's scoring. The report shows the time quotes were eligible for rewards and each market's daily pool.
"""

from __future__ import annotations

import bisect
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
END_CURSOR = "LTE="
# Public sources of a market's recent trades, tried in order: an endpoint that answers 404 is retired and the next is used.
TRADE_SOURCES = (f"{DATA}/v2/trades?condition={{cid}}&limit=100", f"{DATA}/trades?market={{cid}}&limit=100",
                 f"{CLOB}/live-activity/events/{{cid}}")


# ------------------------------------------------------------------- http
def _get(path: str):
    """GET a CLOB path, or any full https URL."""
    from bigbrain.net import HTTPStatusError, http_json

    for attempt in range(5):
        try:
            return http_json(path if path.startswith("https://") else CLOB + path)
        except HTTPStatusError as exc:
            if exc.status in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(1.0 * 2 ** attempt)
                continue
            raise


def _status(exc: Exception) -> int | None:
    return getattr(exc, "status", None)


def _post(path: str, body):
    from bigbrain.net import HTTPStatusError, http_post_json

    for attempt in range(5):
        try:
            return http_post_json(CLOB + path, body)
        except HTTPStatusError as exc:
            if exc.status in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(1.0 * 2 ** attempt)
                continue
            raise


# ----------------------------------------------------------------- markets
@dataclass
class Market:
    condition_id: str
    question: str
    slug: str
    tokens: list[dict]  # [{"token_id", "outcome", "price"}]
    tick: float = 0.01
    min_size: float = 5.0
    reward_max_spread: float | None = None  # cents from the mid within which a quote can earn liquidity rewards
    reward_min_size: float | None = None
    reward_per_day: float = 0.0  # the market's daily reward pool, shared by every eligible maker
    end_date: str = ""


def parse_market(raw: dict) -> Market | None:
    """A CLOB market, or None if it cannot be quoted (closed, not accepting orders, or not a two-outcome market)."""
    if not raw or raw.get("closed") or raw.get("active") is False or raw.get("accepting_orders") is False:
        return None
    tokens = [{"token_id": str(t.get("token_id")), "outcome": str(t.get("outcome", "")), "price": float(t.get("price") or 0)}
              for t in raw.get("tokens") or [] if t.get("token_id")]
    if len(tokens) != 2:
        return None
    rewards = raw.get("rewards") or {}
    rates = rewards.get("rates") or []
    per_day = sum(float(r.get("rewards_daily_rate") or 0) for r in rates if isinstance(r, dict))
    return Market(str(raw.get("condition_id")), str(raw.get("question", "")), str(raw.get("market_slug", "")), tokens,
                  float(raw.get("minimum_tick_size") or 0.01), float(raw.get("minimum_order_size") or 5),
                  float(rewards["max_spread"]) if rewards.get("max_spread") is not None else None,
                  float(rewards["min_size"]) if rewards.get("min_size") is not None else None,
                  per_day, str(raw.get("end_date_iso") or ""))


def reward_markets(limit: int = 10, lo: float = 0.10, hi: float = 0.90, fetch=_get, max_pages: int = 30) -> list[Market]:
    """Markets in Polymarket's liquidity-rewards program, priced away from the extremes (where a cent of spread is
    most of the price and fills are rare), richest reward pool first."""
    out, cursor = [], "MA=="
    for _ in range(max_pages):
        page = fetch(f"/sampling-markets?next_cursor={cursor}")
        for raw in page.get("data") or []:
            m = parse_market(raw)
            if m and lo <= m.tokens[0]["price"] <= hi:
                out.append(m)
        cursor = page.get("next_cursor") or END_CURSOR
        if cursor == END_CURSOR:
            break
    out.sort(key=lambda m: -m.reward_per_day)
    return out[:limit]


def market_by_id(condition_id: str, fetch=_get) -> Market | None:
    return parse_market(fetch(f"/markets/{condition_id}"))


# ------------------------------------------------------------------ record
def _num(x) -> float:
    return float(x) if x not in (None, "") else 0.0


def _when(ts) -> float:
    """A trade time in epoch seconds, whether it came as seconds, milliseconds or an ISO date."""
    if ts in (None, ""):
        return 0.0
    try:
        v = float(ts)
        return v / 1000 if v > 1e12 else v
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0


def _first(d: dict, *names):
    for n in names:
        if d.get(n) not in (None, ""):
            return d[n]
    return None


def normalize_trade(e: dict, m: Market) -> dict | None:
    """One trade from any of the trade sources, in one shape. The sources name the same fields differently
    (asset or asset_id, transactionHash or transaction_hash, timestamp or match_time), so every known name is read."""
    if not isinstance(e, dict):
        return None
    mk = e.get("market") if isinstance(e.get("market"), dict) else {}
    asset = _first(e, "asset", "asset_id", "assetId", "token_id", "tokenId") or mk.get("asset_id")
    if not asset:
        idx = _first(e, "outcomeIndex", "outcome_index")
        if idx is not None and 0 <= int(idx) < len(m.tokens):
            asset = m.tokens[int(idx)]["token_id"]
        else:
            name = str(e.get("outcome", "")).lower()
            asset = next((t["token_id"] for t in m.tokens if t["outcome"].lower() == name and name), None)
    price, size = _first(e, "price"), _first(e, "size", "shares")
    if not asset or price is None or size is None:
        return None
    return {"asset": str(asset), "side": e.get("side"), "price": _num(price), "size": _num(size),
            "ts": _when(_first(e, "timestamp", "match_time", "matchTime", "time", "created_at", "createdAt")),
            "tx": _first(e, "transactionHash", "transaction_hash", "tx_hash", "txHash"), "id": _first(e, "id", "trade_id"),
            "outcome": e.get("outcome"), "fee_bps": _first(e, "fee_rate_bps", "feeRateBps")}


def _levels(rows, descending: bool, depth: int) -> list[list[float]]:
    lv = [[_num(r.get("price")), _num(r.get("size"))] for r in rows or [] if _num(r.get("size")) > 0]
    lv.sort(key=lambda x: -x[0] if descending else x[0])
    return lv[:depth]


class Recorder:
    """Writes the books and trades of ``markets`` to ``out_dir`` as JSON lines, one file per UTC day."""

    def __init__(self, markets: list[Market], out_dir: str | Path, book_every: float = 2.0, trade_every: float = 1.0,
                 depth: int = 10, fetch=_get, post=_post, clock=time.time, sleep=time.sleep) -> None:
        self.markets, self.out = markets, Path(out_dir)
        self.book_every, self.trade_every, self.depth = book_every, trade_every, depth
        self.fetch, self.post, self.clock, self.sleep = fetch, post, clock, sleep
        self.seen: dict[str, set] = {m.condition_id: set() for m in markets}
        self.stats = {"books": 0, "trades": 0, "gaps": 0, "errors": 0}
        self.source = 0  # index into TRADE_SOURCES
        self.sampled = False
        self.log = None
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "markets.json").write_text(json.dumps([asdict(m) for m in markets], indent=1))

    def _write(self, row: dict) -> None:
        day = datetime.fromtimestamp(row["t"], timezone.utc).strftime("%Y-%m-%d")
        with (self.out / f"{day}.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def poll_books(self) -> None:
        tokens = [t["token_id"] for m in self.markets for t in m.tokens]
        now = self.clock()
        for start in range(0, len(tokens), 20):
            for b in self.post("/books", [{"token_id": t} for t in tokens[start:start + 20]]) or []:
                self._write({"type": "book", "t": now, "asset": str(b.get("asset_id")), "market": str(b.get("market", "")),
                             "bids": _levels(b.get("bids"), True, self.depth), "asks": _levels(b.get("asks"), False, self.depth),
                             "tick": _num(b.get("tick_size")) or None, "min": _num(b.get("min_order_size")) or None})
                self.stats["books"] += 1

    def _fetch_trades(self, m: Market) -> list:
        """Recent trades from the first source that still exists. A 404 means the endpoint was retired: move on."""
        while self.source < len(TRADE_SOURCES):
            url = TRADE_SOURCES[self.source].format(cid=m.condition_id)
            try:
                reply = self.fetch(url)
            except Exception as exc:  # noqa: BLE001
                if _status(exc) in (404, 410):
                    if self.log:
                        self.log(f"  trade source {url.split('?')[0]} answered {_status(exc)}; trying the next one")
                    self.source += 1
                    continue
                raise
            if isinstance(reply, dict):
                reply = reply.get("data") or reply.get("trades") or []
            return reply if isinstance(reply, list) else []
        raise RuntimeError("no public trade source answered: Polymarket has moved its trades endpoint again")

    def poll_trades(self, m: Market) -> None:
        now = self.clock()
        raw = self._fetch_trades(m)
        events = [(e, normalize_trade(e, m)) for e in raw]
        if raw and not self.sampled:  # show the first trade, so a changed format is caught on the first run
            self.sampled = True
            (self.out / "sample-trade.json").write_text(json.dumps({"source": TRADE_SOURCES[self.source].format(cid=m.condition_id), "raw": raw[0]}, indent=1, default=str))
            if self.log:
                ok = events[0][1]
                self.log(f"  trades come from {TRADE_SOURCES[self.source].split('?')[0]}; fields: {', '.join(sorted(raw[0]))[:200]}")
                self.log(f"  first trade read as: {ok}" if ok else "  WARNING: could not read the first trade; send sample-trade.json to fix the format")
        events = [(e, t) for e, t in events if t]
        seen = self.seen[m.condition_id]
        fresh = []
        for e, t in events:
            key = (t["id"], t["tx"], t["asset"], t["side"], t["price"], t["size"], t["ts"])
            if key not in seen:
                fresh.append((key, t))
        if seen and events and len(fresh) == len(events):
            self._write({"type": "gap", "t": now, "market": m.condition_id})  # every trade returned is new: some may have been missed
            self.stats["gaps"] += 1
        for key, t in sorted(fresh, key=lambda x: x[1]["ts"]):
            seen.add(key)
            self._write({"type": "trade", "t": now, "ts": t["ts"] or now, "asset": t["asset"], "market": m.condition_id, "side": t["side"],
                         "price": t["price"], "size": t["size"], "outcome": t["outcome"], "fee_bps": t["fee_bps"]})
            self.stats["trades"] += 1

    def run(self, hours: float, log=None) -> dict:
        self.log = log
        end = self.clock() + hours * 3600
        next_books, next_trade, k, last_log = 0.0, 0.0, 0, self.clock()
        while self.clock() < end:
            now = self.clock()
            if now >= next_books:
                try:
                    self.poll_books()
                except Exception as exc:  # noqa: BLE001  a network hiccup must not end a day-long recording
                    self.stats["errors"] += 1
                    if log:
                        log(f"  book poll failed ({exc}); retrying")
                next_books = now + self.book_every
            if now >= next_trade and self.markets:
                m = self.markets[k % len(self.markets)]
                k += 1  # markets take turns, one per trade poll; a failure moves on to the next market
                next_trade = now + self.trade_every
                try:
                    self.poll_trades(m)
                except RuntimeError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.stats["errors"] += 1
                    if log:
                        log(f"  trade poll failed for {m.question[:40]} ({exc})")
            if log and now - last_log >= 300:
                log(f"  {datetime.now():%H:%M} recorded {self.stats['books']} books, {self.stats['trades']} trades, {self.stats['gaps']} possible gaps")
                last_log = now
            self.sleep(max(0.05, min(next_books, next_trade) - self.clock()))
        return self.stats


# ---------------------------------------------------------------- simulate
@dataclass
class Quote:
    price: float
    size: float
    queue: float  # shares resting at this price ahead of ours when we joined
    live_at: float


@dataclass
class Fill:
    t: float
    side: str  # BUY or SELL
    price: float
    size: float
    mid10: float | None = None
    mid60: float | None = None


@dataclass
class Book:
    t: float
    bid: float
    ask: float
    bid_size: dict
    ask_size: dict

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class SimResult:
    asset: str
    question: str = ""
    outcome: str = ""
    hours: float = 0.0
    books: int = 0
    trades: int = 0
    gaps: int = 0
    fills: list = field(default_factory=list)
    inventory: float = 0.0
    cash: float = 0.0
    pnl_mid: float = 0.0
    pnl_dump: float = 0.0
    quoted_s: float = 0.0
    reward_s: float = 0.0
    reward_per_day: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "fills"} | {"fills": len(self.fills)}


def taker_fee(shares: float, price: float, rate: float) -> float:
    """Polymarket's taker fee: shares x rate x p x (1 - p), rounded to 5 decimals."""
    return round(shares * rate * price * (1 - price), 5)


def load(directory: str | Path) -> tuple[list[dict], dict]:
    """Every row of a recording, oldest first, and its markets by token id."""
    d = Path(directory)
    rows = []
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    rows.sort(key=lambda r: r.get("ts", r["t"]) if r.get("type") == "trade" else r["t"])
    markets = {}
    mfile = d / "markets.json"
    if mfile.exists():
        for m in json.loads(mfile.read_text()):
            for i, t in enumerate(m["tokens"]):
                markets[t["token_id"]] = {**m, "outcome": t["outcome"], "index": i}
    return rows, markets


def simulate(rows: list[dict], asset: str, size: float = 10.0, improve: int = 0, max_inventory: float = 50.0,
             latency: float = 1.0, queue_model: str = "queue", taker_rate: float = 0.07, lo: float = 0.05, hi: float = 0.95,
             meta: dict | None = None) -> SimResult:
    """Replay one token's recorded books and trades with a maker quoting it. See the module docstring for the rules."""
    meta = meta or {}
    r = SimResult(asset, meta.get("question", ""), meta.get("outcome", ""), reward_per_day=meta.get("reward_per_day", 0.0) or 0.0)
    books: list[Book] = []
    bid: Quote | None = None
    ask: Quote | None = None
    gap_market = meta.get("condition_id")
    last_t = None
    for row in rows:
        kind = row.get("type")
        if kind == "gap" and row.get("market") == gap_market:
            r.gaps += 1
            continue
        if row.get("asset") != asset:
            continue
        if kind == "book":
            if not row["bids"] or not row["asks"]:
                continue
            b = Book(row["t"], row["bids"][0][0], row["asks"][0][0], {p: s for p, s in row["bids"]}, {p: s for p, s in row["asks"]})
            if books and last_t is not None:  # time spent quoting since the last book, and of it time eligible for rewards
                dt = b.t - last_t
                if bid or ask:
                    r.quoted_s += dt
                    rs, rm = meta.get("reward_max_spread"), meta.get("reward_min_size")
                    if bid and rs is not None and (books[-1].mid - bid.price) * 100 <= rs and bid.size >= (rm or 0):
                        r.reward_s += dt
            last_t = b.t
            books.append(b)
            r.books += 1
            tick = row.get("tick") or meta.get("tick") or 0.01
            if not (lo <= b.mid <= hi):
                bid = ask = None
                continue
            want_bid = round(min(b.bid + improve * tick, b.ask - tick), 6)
            want_ask = round(max(b.ask - improve * tick, want_bid + tick), 6)
            if r.inventory + size <= max_inventory + 1e-9:
                if bid is None or abs(bid.price - want_bid) > 1e-9:
                    bid = Quote(want_bid, size, b.bid_size.get(want_bid, 0.0), b.t + latency)
            else:
                bid = None
            if r.inventory > 1e-9:
                if ask is None or abs(ask.price - want_ask) > 1e-9:
                    ask = Quote(want_ask, min(size, r.inventory), b.ask_size.get(want_ask, 0.0), b.t + latency)
                else:
                    ask.size = min(ask.size, r.inventory)
            else:
                ask = None
        elif kind == "trade" and books:
            t, p, s = row["ts"], row["price"], row["size"]
            r.trades += 1
            best = books[-1]
            if bid and t >= bid.live_at:
                filled = 0.0
                if p < bid.price - 1e-9:
                    filled = bid.size
                elif abs(p - bid.price) < 1e-9 and queue_model == "queue" and bid.price < best.ask:
                    bid.queue -= s
                    if bid.queue < 0:
                        filled = min(bid.size, -bid.queue)
                        bid.queue = 0.0
                if filled:
                    r.fills.append(Fill(t, "BUY", bid.price, filled))
                    r.inventory += filled
                    r.cash -= filled * bid.price
                    bid.size -= filled
                    if bid.size <= 1e-9:
                        bid = None
            if ask and t >= ask.live_at:
                filled = 0.0
                if p > ask.price + 1e-9:
                    filled = min(ask.size, r.inventory)
                elif abs(p - ask.price) < 1e-9 and queue_model == "queue" and ask.price > best.bid:
                    ask.queue -= s
                    if ask.queue < 0:
                        filled = min(ask.size, -ask.queue, r.inventory)
                        ask.queue = 0.0
                if filled:
                    r.fills.append(Fill(t, "SELL", ask.price, filled))
                    r.inventory -= filled
                    r.cash += filled * ask.price
                    ask.size -= filled
                    if ask.size <= 1e-9 or r.inventory <= 1e-9:
                        ask = None
    if books:
        r.hours = (books[-1].t - books[0].t) / 3600
        times = [b.t for b in books]
        for f in r.fills:  # markout: where the mid went after each fill
            for lag, attr in ((10, "mid10"), (60, "mid60")):
                j = bisect.bisect_left(times, f.t + lag)
                if j < len(books):
                    setattr(f, attr, books[j].mid)
        last = books[-1]
        r.pnl_mid = r.cash + r.inventory * last.mid
        r.pnl_dump = r.cash + r.inventory * last.bid - taker_fee(r.inventory, last.bid, taker_rate)
    return r


def markout(fills: list[Fill], attr: str = "mid60") -> float | None:
    """Average cents per share the mid moved in our favour after a fill (negative: we were picked off)."""
    vals = [((getattr(f, attr) - f.price) if f.side == "BUY" else (f.price - getattr(f, attr))) * 100 * f.size
            for f in fills if getattr(f, attr) is not None]
    shares = sum(f.size for f in fills if getattr(f, attr) is not None)
    return sum(vals) / shares if shares else None


def run(directory: str | Path, both: bool = False, **kw) -> list[SimResult]:
    rows, markets = load(directory)
    assets = sorted({r["asset"] for r in rows if r.get("type") == "book"})
    out = []
    for a in assets:
        meta = markets.get(a, {})
        if not both and meta.get("index", 0) != 0:
            continue  # quote one outcome per market: a bid on YES is the same as an ask on NO
        out.append(simulate(rows, a, meta=meta, **kw))
    return out


def format_report(results: dict[str, list[SimResult]], size: float) -> str:
    """``results``: queue model -> one result per token."""
    lines = []
    first = next(iter(results.values()), [])
    hours = max((r.hours for r in first), default=0.0)
    days = max(hours / 24, 1e-9)
    lines.append(f"Polymarket maker replay: {len(first)} markets, {hours:.1f} hours recorded, quoting {size:g} shares a side "
                 f"(one cent of spread on {size:g} shares is {size * 0.01:.2f} USDC)")
    gaps = sum(r.gaps for r in first)
    if gaps:
        lines.append(f"warning: {gaps} possible gaps where trades may have been missed; fills there are undercounted (record with a shorter --trade-every)")
    for model, rs in results.items():
        lines += ["", f"fill model '{model}'" + (" (fills only when a trade goes through the quote: the strict bound)" if model == "through"
                                                  else " (a trade at the quote's price fills it once the queue ahead of it has traded)"),
                  f"  {'market':44} {'fills':>5} {'bought':>7} {'sold':>7} {'held':>6} {'P&L mid':>9} {'P&L sold':>9} {'markout60':>9} {'quoted':>7}"]
        for r in sorted(rs, key=lambda r: r.pnl_dump):
            mk = markout(r.fills)
            bought = sum(f.size for f in r.fills if f.side == "BUY")
            sold = sum(f.size for f in r.fills if f.side == "SELL")
            q = f"{r.quoted_s / 3600 / max(r.hours, 1e-9):.0%}" if r.hours else "-"
            lines.append(f"  {(r.question or r.asset)[:44]:44} {len(r.fills):5} {bought:7.0f} {sold:7.0f} {r.inventory:6.0f} "
                         f"{r.pnl_mid:+9.2f} {r.pnl_dump:+9.2f} {(f'{mk:+.2f}c' if mk is not None else '-'):>9} {q:>7}")
        total_mid, total_dump = sum(r.pnl_mid for r in rs), sum(r.pnl_dump for r in rs)
        fills = [f for r in rs for f in r.fills]
        mk = markout(fills)
        trips = min(sum(f.size for f in fills if f.side == "BUY"), sum(f.size for f in fills if f.side == "SELL")) / size
        lines.append(f"  total: {len(fills)} fills, about {trips:.0f} round trips of {size:g} shares; P&L {total_mid:+.2f} USDC at the mid, "
                     f"{total_dump:+.2f} if what is still held were sold to the bid with the taker fee; "
                     f"about {total_dump / days:+.2f} USDC a day")
        if mk is not None:
            lines.append(f"  markout: 60 seconds after a fill the mid had moved {mk:+.2f} cents per share "
                         f"{'in our favour' if mk > 0 else 'against us (the cost of being picked off)'}")
    pool = sum(r.reward_per_day for r in first)
    eligible = sum(r.reward_s for r in first) / 3600
    lines += ["", f"not counted: maker rebates, and liquidity rewards (these markets share {pool:.0f} USDC a day among all eligible makers; "
                  f"your quotes were reward-eligible for {eligible:.1f} market-hours)",
              "a real maker also competes on speed: this replay assumes your quote rests where you placed it and nobody reacts to you"]
    return "\n".join(lines)
