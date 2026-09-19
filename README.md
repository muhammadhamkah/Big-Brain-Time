# Big Brain Time

A virtual brain for trading knowledge. It reads research papers, market data, indicators and strategies, stores each thing it learns as a **cell**, and wires every new cell to everything it already knows through weighted **synapses**. Ask it a question and activation spreads across those links, so the answer draws on concepts, papers, live observations and backtest results together. Cells that fire together wire together, so the more it is used, the better organised it gets.

The goal is a brain that keeps learning until it reasons about markets like a professor. This is version one: the learning, linking and recall machinery, four ingestion pipelines, and an optional Claude-powered cortex that reasons over what the brain has recalled.

## How it works

```
                   ┌──────────────┐
  papers (arXiv) ─►│              │
  market data ────►│   ingest     │──► cells ──┐
  indicators ─────►│  pipelines   │            │  concept extraction
  strategies ─────►│              │            ▼
  your notes ─────►└──────────────┘      ┌──────────────┐
                                         │  the brain   │  every new cell links to
                                         │ cells+synapse│  every related cell
                                         └──────┬───────┘
                        question ───────────────►│ recall: score + spread activation
                                                 │ reinforce: co-recalled cells wire together
                                                 ▼
                                         ┌──────────────┐
                                         │   cortex     │  offline briefing, or
                                         └──────────────┘  Claude reasoning over recalled cells
```

* **Cells** are units of knowledge with a kind: `concept`, `paper`, `observation`, `lesson`, `note`.
* **Concept extraction** (`bigbrain/concepts.py`) maps text onto a lexicon of about 60 trading ideas (trend, RSI, Kelly criterion, overfitting, market microstructure, ...). Anything outside the lexicon still links through term statistics, so the brain can connect on vocabulary it was never taught.
* **Linking** combines concept overlap (Jaccard) with tf-idf cosine similarity. A new cell keeps its 30 strongest links above a threshold.
* **Recall** scores cells against a question, then spreads activation across synapses so a question about "trend following on AAPL" reaches the moving average concept, the AAPL death-cross observation and the SMA crossover backtest even when they share no words with the question.
* **Hebbian reinforcement**: cells recalled together strengthen their synapses, and activations are counted, so frequently useful knowledge becomes central.
* **Storage** is one SQLite file. The brain persists and grows between runs.

## Quickstart

No dependencies beyond Python 3.10+.

```bash
pip install -e .            # installs the `bigbrain` command

bigbrain seed               # curriculum plus every knowledge pack (hundreds of written lessons)
bigbrain feed               # go online: papers, Reddit, GitHub, blogs, reference pages; 10 workers
bigbrain ask "should I use trend following on BTC right now?"
```

Or feed it one sense at a time:

```bash
bigbrain learn papers -q "momentum crypto" --max 10          # recent arXiv q-fin papers
bigbrain learn reddit --sub algotrading --sub quant --time month   # top posts and best comments
bigbrain learn github -q "topic:backtesting" --max 10          # most-starred repos and their READMEs
bigbrain learn feed                                            # curated quant blogs (RSS)
bigbrain learn url https://www.tradingview.com/chart/BTCUSD/xxxx/   # any page: idea, blog post, docs
bigbrain learn market --from binance --symbol BTCUSDT          # 1000 daily candles, no key needed
bigbrain learn market --from binance --symbol ETHUSDT --interval 4h
bigbrain learn market --from stooq --symbol aapl.us            # free daily history for stocks and indices
bigbrain learn market --csv data/AAPL.csv --symbol AAPL        # your own OHLCV data
bigbrain learn market                                          # or synthetic data for a demo
bigbrain learn text "Turtle rules" "The Turtles bought 20-day breakouts and sized by ATR..."
bigbrain learn file notes/*.md

bigbrain trade                                                 # the brain trades the top 100 USDT perps, long and short, 1000 USDT, learns from every trade
bigbrain trade --market spot --book spot                       # a second book on spot (long only), separate wallet and beliefs
bigbrain dashboard                                             # live dashboard in a second terminal: equity, positions, beliefs, feed
bigbrain portfolio --recent 20                                 # equity, open positions, closed trades with findings
bigbrain beliefs                                               # what it now believes about each signal in each context
bigbrain reset                                                 # close everything, wallet back to start, learning kept (--forget wipes it)
bigbrain watch --symbol BTCUSDT --interval 15m               # one market: paper trades 4 strategies, grades signals
bigbrain paper --recent 10                                     # virtual accounts: equity, drawdown, trades
bigbrain calls --recent 10                                     # scorecard: which signals actually work here
bigbrain recall "position sizing in high volatility" -v
bigbrain explain "Kelly criterion"
bigbrain stats --journal 10
bigbrain graph --format dot -o brain.dot && dot -Tsvg brain.dot -o brain.svg
```

The database defaults to `.brain/brain.db`; override with `--db` or `BIGBRAIN_DB`. Learning is idempotent, so running `bigbrain feed` every day only adds what is new. `feed` fetches from up to ten sources at once (`--workers`), while staying polite to each host.

### Knowledge packs

`bigbrain/knowledge/` holds written knowledge, one module per domain: technical analysis, risk management, quantitative strategies, microstructure and execution, options and volatility, futures and macro, forex, crypto, research methodology, and trading psychology and history. Each pack is 35-40 dense lessons plus curated sources for that domain (feeds, subreddits, GitHub queries, arXiv topics, reference pages). `bigbrain seed` learns every pack; `bigbrain feed` reads every pack's sources. Adding a domain is adding a file.

CSV files need `date,open,high,low,close,volume` columns (any case, oldest first or newest first).

### The senses

| Source | How | Cell kind | Trust |
|---|---|---|---|
| Built-in curriculum | 51 written lessons | concept | 1.0 |
| arXiv | public API, quantitative finance categories | paper | 1.0 |
| Market data | Binance candles, Stooq daily history, CSV or synthetic bars; indicators and backtests | observation, lesson | 1.0 |
| Your notes and files | `learn text`, `learn file` | note | 1.0 |
| Blogs and RSS feeds | public feeds; `--full` fetches the whole article | article | 0.9 |
| GitHub | public API; set `GITHUB_TOKEN` for 5000 requests/hour instead of 60 | code | 0.8 |
| Reddit | public JSON listings, no account; top posts plus best comments | discussion | 0.7 |
| Any web page | `learn url`; works when the text is in the HTML | article | 0.9 |

Trust scales a cell's recall score, so a forum thread still surfaces but a paper or a measured backtest on the same topic outranks it. TradingView has no public API and forbids scraping, so the brain reads TradingView pages only when you hand it a URL.

All requests are polite: one host at a time with a minimum interval between calls, curl-shaped headers (some edges reject Python's default request shape with 406), and `HTTPS_PROXY` is honoured.

### Watching a market in real time

`bigbrain watch` polls a symbol at every candle close (Binance, no key), recomputes the indicators, and turns each event into a *call* with an explicit hypothesis: RSI crossing below 30 says "up within 12 bars", a death cross says "down", and so on. When the horizon has passed, the call is graded against what the market did. After five graded calls per signal the running scorecard becomes a lesson cell ("on BTCUSDT 15m, RSI oversold fired 14 times, 43% hit rate, average move -0.3%: hypothesis refuted so far"), so the brain learns which signals mean something on this market at this timeframe. `ask` shows the live state of every watched market. This is the outcome loop: knowledge that is checked against reality gains or loses weight.

**Paper trading.** The watcher also runs every built-in strategy on a virtual account of 10,000 per strategy. At each closed candle the rule is re-evaluated; when its target position changes, the account trades at that close and pays 0.1% per side, the same assumptions the backtester makes, so a candle-by-candle replay reproduces the backtest exactly (there is a test for that). Open positions are marked to market every candle, so drawdown is measured on real equity. Every ten closed trades the running record becomes a lesson: equity, Sharpe, win rate, drawdown, against buy and hold over the same candles. `bigbrain paper` shows the accounts. No real orders are ever sent.

Run it in a terminal you leave open, or with `--once` from cron every 15 minutes.

### The brain trades, and learns from every trade

`bigbrain trade` runs a virtual 1,000 USDT wallet across the top 100 USDT pairs by volume on Binance (refreshed every candle, no key needed), on 15-minute candles. The default market is **perps** (USDT-margined perpetuals): the brain trades long and short, pays 0.02% maker / 0.05% taker, and is charged the exchange's real funding rate at every 00:00, 08:00 and 16:00 UTC it holds a position through. `--market spot` runs long only at 0.1% taker. Effective leverage is always one-times, so the wallet can never be liquidated. Every candle, every pair:

1. It sees which signals fired (RSI oversold, band breaks, moving-average crosses, MACD flips) and the context: trend regime (50 vs 200 average), volatility bucket relative to the pair's own history, RSI, band position.
2. It consults its **belief table**: what that signal has done in that context in its own closed trades. Positive expectancy after costs: full size. Not enough evidence: explore at half size (a brain that never tries anything never learns). Evidence says it loses: stand aside, with an occasional small re-test so a changed market can change the belief.
3. Size is risk-based: 1% of equity at risk to a stop 2 ATR away, capped at 25% of equity. Any number of positions: on perps each posts margin of a third of its notional (gross exposure up to 3x the wallet), total risk at stops is capped at 10% of equity, and a liquidation check runs every candle (equity below 0.5% of gross notional closes everything, as the exchange would). On perps, RSI overbought, death cross and MACD turning down are short entries; on spot they are exits only.
4. **Costs**: VIP 0 fees for the market (perps 0.05% taker, spot 0.1% taker, each way, market orders), funding on perps, and slippage modelled from liquidity: half the estimated spread (about 1bp for BTC, wider for thin pairs) plus impact from the order's share of the candle's volume. Stops fill at the stop, or at the open when a candle gaps through it.
5. **Every closed trade gets a post-mortem**, win or loss: context at entry, best and worst point while open, how it exited, gross versus net, funding paid or received. Diagnostic lenses produce findings (fought the trend, breakout against regime, stop inside the noise, gave back an open gain, whipsaw, thesis never developed, costs ate the edge, funding drag; trend aligned, dip in uptrend, pop in downtrend, rode the move, fast resolution, near miss, funding tailwind). Each finding updates the belief table, instructive trades become post-mortem cells the brain can recall, and every ten trades per signal it rewrites a "what I have learned" lesson.

6. **Exit learning.** Every position keeps its bar path. When it closes, the brain replays that path under five exit styles (the signal's rule, take profit at 1R or 2R, break-even after 1R, trailing stop after 1R) and records what each would have returned net of costs. Once a signal has twenty trades, if a style beats the rule by at least 0.1% per trade it becomes that signal's live exit; if it stops winning, the policy reverts. The replay and the live position share one step function, so what was measured is exactly what is then done.

`bigbrain portfolio` shows the book; `bigbrain beliefs` shows the table that now drives its decisions and the exit policies; `bigbrain ask "why did you lose on SOLUSDT"` recalls the post-mortems. Separate books (`--book`) keep separate wallets and beliefs. No real orders are ever sent.

### Running for months

The trader only manages positions while it runs, so a long run needs a process that survives closed terminals and restarts. On a Mac:

```bash
scripts/macos-service.sh install          # launchd service: starts at login, restarts on failure, keeps the Mac awake
scripts/macos-service.sh status
scripts/macos-service.sh log              # follow .brain/trade.log
scripts/macos-service.sh uninstall
```

A MacBook still sleeps when the lid closes unless it is plugged in with an external display; a Mac mini, an always-on laptop, or a small cloud server is the reliable home for a six-month run. `bigbrain dashboard` works in any terminal while the service runs.

**Backups.** The brain is one file, `.brain/brain.db`, never committed to the main branch. `bigbrain backup` takes a consistent snapshot while the trader runs (SQLite's online backup), gzips it into `.brain/backups/`, and keeps the last 14. `bigbrain backup --push` also publishes the snapshot to a `brain-backup` branch on GitHub that always holds exactly one commit, so the repository stays small. `scripts/macos-service.sh install-backup --push` does that every six hours. `bigbrain restore` brings back the latest local snapshot, a named file, or `--from-github`. Stop the trader before restoring.

### Claude-powered cortex

By default `bigbrain ask` prints a structured briefing built from the recalled cells. To have the brain *reason* over them:

```bash
pip install -e ".[cortex]"
export ANTHROPIC_API_KEY=...      # or `ant auth login`
bigbrain ask "compare mean reversion and trend following for DEMO given its current regime"
```

The cortex sends Claude only the cells the brain recalled, and asks it to answer like a professor citing those cells, to flag disagreements between them, and to say what the brain still needs to learn rather than invent facts. It uses Claude Opus 5 with server-side refusal fallbacks enabled, so a declined request is retried on a fallback model automatically. Pass `--offline` to skip the model, or `--claude` to require it.

## Library use

```python
from bigbrain import Brain
from bigbrain.ingest.textbook import seed
from bigbrain.ingest.market import synthetic, learn_bars
from bigbrain.ingest.strategies import learn_backtests

brain = Brain("my.db")
seed(brain)
bars = synthetic("SYNTH", n=400)
learn_bars(brain, "SYNTH", bars)
learn_backtests(brain, "SYNTH", bars)

for hit in brain.recall("is SYNTH oversold?", k=5):
    print(hit.score, hit.cell.title, "via", hit.via)
```

## Layout

```
bigbrain/
  brain.py            the Brain: learn / recall / reinforce / export
  cells.py            Cell, Synapse, Recall dataclasses
  concepts.py         trading lexicon, concept extraction, tokenizer
  cortex.py           offline and Claude-powered answering
  watch.py            real-time watcher: signals -> calls -> graded outcomes -> scorecard lessons
  paper.py            paper trading: a virtual account per strategy, traded at candle closes, marked to market
  trader.py           the brain trades: universe, costs, margin and risk caps, belief-driven entries, exits, one wallet
  dashboard.py        live terminal dashboard: equity, positions at live prices, beliefs, trades, feed
  postmortem.py       lenses that explain each closed trade, the belief table, post-mortem and summary lessons
  exits.py            exit learning: counterfactual exits on every trade, per-signal exit policies
  cli.py              the `bigbrain` command
  net.py              polite HTTP: curl-shaped requests, per-host rate limits, proxy support
  sources.py          the brain's diet: builds fetch jobs from defaults + packs, runs them in parallel
  knowledge/          knowledge packs: written lessons and curated sources per domain
  ingest/
    textbook.py       seed curriculum (51 concepts)
    indicators.py     SMA, EMA, RSI, MACD, Bollinger, ATR, volatility, drawdown, Sharpe
    market.py         CSV / synthetic OHLCV -> indicator observations
    strategies.py     SMA crossover, RSI mean reversion, Bollinger breakout, buy & hold + backtester
    papers.py         arXiv q-fin fetcher and parser
    reddit.py         subreddit listings and top comments -> discussion cells
    github.py         repository search and READMEs -> code cells
    web.py            any page (HTML -> text) and RSS/Atom feeds -> article cells
tests/                unittest suite (python -m unittest discover -s tests)
```

## Roadmap

1. **More senses**: price data from live APIs, full-text PDFs, earnings transcripts, news sentiment, order-book data, a scheduler so `feed` runs itself.
2. **Smarter linking**: embedding similarity next to the lexicon, contradiction detection between cells, and confidence that decays for observations as they age.
3. **Learning from outcomes**: track what the brain said and what the market did, so lessons get reinforced or weakened by results.
4. **Strategy evolution**: let the brain propose parameter changes and new rule combinations, backtest them walk-forward, and keep only what survives.
5. **A visual cortex**: a graph explorer to watch the brain grow and see which knowledge is firing.

A brain is only as good as what it is fed. Feed it well.
