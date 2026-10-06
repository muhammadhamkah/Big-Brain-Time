# Big Brain Time

A virtual brain for trading knowledge. It reads research papers, market data, indicators and strategies, stores each thing it learns as a **cell**, and wires every new cell to everything it already knows through weighted **synapses**. Ask it a question and activation spreads across those links, so the answer draws on concepts, papers, live observations and backtest results together. Cells that fire together wire together, so the more it is used, the better organised it gets. What it reads enriches its answers; what it trades is decided only by its own recorded trades.

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
* **Hebbian reinforcement**: cells recalled together strengthen their synapses, and activations are counted, so frequently recalled knowledge becomes central. This measures association, not correctness: a strong link says two things are often useful together, not that either is true. Only the trading loop below checks anything against the market.
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

`bigbrain trade` runs a virtual 1,000 USDT wallet across the top 100 USDT pairs by volume on Binance (refreshed every candle, no key needed), on 15-minute candles. The default market is **perps** (USDT-margined perpetuals): the brain trades long and short, pays 0.02% maker / 0.05% taker, and is charged the exchange's real funding rate at every 00:00, 08:00 and 16:00 UTC it holds a position through. `--market spot` runs long only at 0.1% taker. On perps every position posts margin of a third of its notional, so gross exposure can reach three times equity; a liquidation check runs every candle (equity below 0.5% of gross notional closes everything, as the exchange would). Every candle, every pair:

1. It sees which signals fired (RSI oversold, band breaks, moving-average crosses, MACD flips) and the context: trend regime (50 vs 200 average), volatility bucket relative to the pair's own history, RSI, band position.
2. It consults its **belief table**: what that signal has done in that context in its own closed trades. The verdict sets the size. Trades closed on the same day move together (one crypto market, many pairs), so the uncertainty of the expectancy is measured across UTC days, not across trades. Full size needs 20 trades over at least 4 separate days with the mean minus one standard error above zero; a positive mean short of that gets half size; too little evidence means exploring at half size some of the time; a mean plus one standard error below zero means standing aside, with an occasional small re-test so a changed market can change the belief.
3. Size is risk-based: 1% of equity at risk to a stop 2 ATR away (never closer than 0.5% or six round-trip costs), capped at 25% of equity. Any number of positions, with total risk at stops capped at 10% of equity. On perps, RSI overbought, death cross and MACD turning down are short entries; on spot they are exits only. One pair is never long and short at once.
4. **Execution is strictly chronological.** A signal is known only after its candle closes, so nothing fills at a price that has already passed. Live, an order fills at the current bid/ask (a timestamped book ticker fetched at decision time: buys lift the ask, sells hit the bid) plus market impact; if the quote is missing or older than a minute, the entry is skipped and an exit is queued for the next candle's open. In replays (tests, catch-up after an outage) there are no quotes, so every order queues and fills at the next candle's open; a decision that belongs to a missed candle is never filled at today's quote. Stops, trailing stops and targets are resting orders: they fill at their level, or at the open when a candle gaps through it. Every trade records how it was filled. Slippage on each candle is computed from the candles known at the time (the trailing day and the trailing twenty), so a candle caught up after an outage pays exactly the liquidity it had, not today's.
5. **Costs**: VIP 0 fees for the market (perps 0.05% taker, spot 0.1% taker, each way, market orders), funding on perps, and slippage from liquidity: impact from the order's share of the candle's volume, plus half the estimated spread (about 1bp for BTC, wider for thin pairs) whenever there is no live quote to carry the spread.
6. **Every closed trade gets a post-mortem**, win or loss: context at entry, best and worst point while open, how it exited, gross versus net, funding paid or received. Diagnostic lenses produce findings (fought the trend, breakout against regime, stop inside the noise, gave back an open gain, whipsaw, thesis never developed, costs ate the edge, funding drag; trend aligned, dip in uptrend, pop in downtrend, rode the move, fast resolution, near miss, funding tailwind). Each finding updates the belief table, instructive trades become post-mortem cells the brain can recall, and every ten trades per signal it rewrites a "what I have learned" lesson.
7. **Exit learning.** Every position keeps its bar path and the funding it paid along it. When the signal's own rule exits, the brain replays that path under five exit styles (the rule, take profit at 1R or 2R, break-even after 1R, trailing stop after 1R). Each style is filled the way the live trader fills it (its own slippage at its own level with the liquidity of the candle it exits on, its own exit fee, the entry fee once, the funding accrued to its own exit), so a replayed exit at the same price as a real one books the same return to the last digit. Under a learned exit the original rule is tracked from the first candle, including the candle the learned exit closes on; if the rule has not exited yet, a *shadow* carries it to its own exit, so the rule baseline is always the rule and every style is scored on one complete path. Once a signal has twenty trades, if a style beats the rule by at least 0.1% per trade it becomes that signal's live exit; if it stops winning, the policy reverts. Every adoption and reversal is logged with the evidence at the time (`bigbrain beliefs`). The replay and the live position share one step function, so what was measured is exactly what is then done.

**Bookkeeping.** A symbol's whole step (queued fills, exits, the trade record, belief update, saved wallet) is one database transaction, so a crash can never leave the money and the evidence out of step. A trade is identified by book, symbol, signal and entry time; recording one twice changes nothing. When the execution or evaluation model changes, the results version is bumped (it is 4 now): the brain snapshots the database (and refuses to retire anything if that snapshot cannot be written), retires the old exit statistics and beliefs, migrates open positions and shadows to the new layout, and stamps every trade with the version its position was opened under (a position from before the bump finishes without counting as evidence), so evidence measured under an old model never counts as evidence about the new one: sample counts, context selection, expectancy and its uncertainty all come from the same current-version trades.

**Is the learning worth anything?** `bigbrain trade --with-frozen` runs a second book, `<book>-frozen`, in the same process: same market, candles, quotes, funding, starting capital, risk limits and costs, but fixed size, the signal's own exits, beliefs ignored. Anything the adaptive book earns above it is what the learning added; anything below is what it cost. Compare them with `bigbrain portfolio --book main` and `--book main-frozen`.

`bigbrain portfolio` shows the book; `bigbrain beliefs` shows the table that drives its decisions, with days of evidence and standard errors, plus the exit policies and their change log; `bigbrain ask "why did you lose on SOLUSDT"` recalls the post-mortems. Separate books (`--book`) keep separate wallets and beliefs. No real orders are ever sent.

### The lab: testing a rule before anyone pays for it

`bigbrain lab` runs a trading rule over months of candles with the live trader's honesty (decisions after the close, fills at the next open, stops and stop-and-reverse orders filled at their level or the open on a gap, fee plus half the spread plus slippage on every fill) and then does what a backtest screenshot never does:

```bash
bigbrain lab --rule psar --symbol EURUSDT --interval 1m --days 30 --fee 0 --spread-bps 0.35 \
             --grid "start=0.01,0.02,0.04;increment=0.01,0.02,0.04;maximum=0.1,0.2,0.4"
```

* **Grid with a plateau score.** Every parameter setting is scored by its own profit factor or its neighbours' average, whichever is lower. A real edge is a region where nearby settings all work; a lone spike is a fit to one stretch of noise.
* **Walk-forward.** The candles are cut into windows; for each window the parameters are chosen on everything before it and judged on it alone. The stitched out-of-sample record, with its profit factor and how many standard errors its mean trade sits from zero, is the only number that says anything about next week.
* **A verdict the brain keeps.** "no edge", "noise", or "worth a look" becomes a lesson cell, so `bigbrain ask "does parabolic SAR work on 1m EURUSD"` answers from evidence.

Rules are small functions of past candles and parameters (`bigbrain/lab.py`). Adding one is adding a function. The lab runs on one market or a whole universe (`--symbols top:30` pools the trades and cuts the walk-forward by date, so every symbol is judged on the same days), and on perps it pays or receives funding at the real historical rate. The rules it ships with:

| rule | what it tests | run it on |
|---|---|---|
| `playbook` | the live trader's own eight signals, managed exactly as the trader manages them (`--grid "signal=*"`) | 15m, 4h and 1d, a universe |
| `trend_vt` | daily trend with volatility targeting: the boring edge | 1d, the majors |
| `funding_carry` | shorting when the crowd pays to be long, and the reverse, collecting funding | 4h perps, a universe |
| `xs_momentum` | cross-sectional momentum: long the strongest of the universe, short the weakest, market neutral | 1d, a universe |
| `psar`, `sma_cross`, `rsi_reversion` | the textbook indicators, for calibration | anything |
| `vwap_fade`, `sweep`, `squeeze` | the scalping setups (below) | 1m to 15m, liquid perps |

The lab cannot tune a rule to catch tops and bottoms; anything that appears to has been fitted to the past, and the walk-forward is there to show it. What it can do is reject nineteen ideas in twenty cheaply, so the live book only ever trades the twentieth.

### Scalping: does any small, fast edge survive the toll?

A scalper takes many small trades, so its edge per trade is a few basis points, and the round-trip cost (two fees, the spread, slippage) is of the same size. Most scalping ideas fail because they lose to that cost, not because they never predict anything. `bigbrain scalp` answers both questions at once:

```bash
bigbrain scalp                                                     # three setups on BTC, ETH, SOL perps, 1m and 5m, 14 days
bigbrain scalp --symbols top:10 --intervals 1m,5m,15m --days 30    # wider and longer
bigbrain scalp --fee 0.0002                                        # what if every fill paid the maker fee
bigbrain scalp --rules sweep --intervals 5m --detail               # one setup, with the full lab report
bigbrain scalp --synthetic                                         # offline demo on fair random walks: everything should fail
```

The three setups, each with a resting stop, a resting target and a time limit, long and short (long only on spot), optionally filtered by a trend average:

| setup | the idea | stop | target |
|---|---|---|---|
| `vwap_fade` | a close stretched `entry_z` standard deviations from the rolling VWAP that starts to turn back | `sl_atr` ATRs away | the VWAP |
| `sweep` | a candle runs the stops beyond the last `lookback` candles' low (high) and closes back in the top (bottom) half of its range: the breakout failed | just past the wick | `tp_r` times the risk |
| `squeeze` | after a Bollinger squeeze, a close outside the band on `vol_mult` times average volume | `sl_atr` ATRs | `tp_atr` ATRs |

Each runs through the lab's grid and walk-forward on the pooled symbols. The scan prints, out of sample: trades per day, win rate, profit factor, net per trade, and the split that matters for scalping, **gross** (what the setup caught between quoted prices) against **cost** (fees, spread, slippage, funding). A setup with a real gross edge smaller than its cost is a setup that needs a cheaper venue, maker fills or a slower timeframe; a setup with no gross edge needs nothing but deleting. Every verdict and cost check becomes a lesson the brain can recall.

Two guards keep the numbers honest. A setup tracks its stop, target and time limit with the simulator's own fill order, so it can never believe it is flat while the simulator holds a position. And a test runs every setup and a random-entry bracket over fair random walks built tick by tick, where nothing can earn anything: an average out of line there means look-ahead. That test is also why `--synthetic` uses its own walk. On candles whose wicks are drawn independently of the path (or that have none), a crossed stop is a stop the price keeps running through, and filling it at its level books a phantom edge of about ten basis points per trade.

Expect most scans to end in "no edge". At VIP 0 perps taker fees the toll is about 12 bp per round trip, and one-minute candles rarely move far enough between a sensible stop and target to pay it. If a setup comes back "worth a look", the next step is a small paper book, not money.

### A decision model on top of the brain: the decider experiment

The trader's signals propose trades; something has to decide which to take. `bigbrain decide` replays months of real candles and puts four deciders in front of exactly the same proposals:

| decider | sees | decides by |
|---|---|---|
| `take_all` | nothing | taking every signal: the baseline |
| `beliefs` | the brain's memory | the trader's rule: skip a setup whose record in this context is losing |
| `jev_blind` | the setup and the market, in words | [Jev](https://typesafe.ai), TypeSafe's decision model: LONG, SHORT or SKIP with probabilities |
| `jev` | the same plus the brain's memory | Jev again, now shown this setup's record in this context, the record of fading it, and the lessons the brain recalls |
| `llm_blind`, `llm` | the same as the two Jev deciders | a free model running on your own computer through [Ollama](https://ollama.com) (Llama, Qwen, ...), with `--llm MODEL` |

```bash
export TYPESAFE_API_KEY=...                         # from typesafe.ai; without it Jev sits out and the rest still runs
bigbrain decide                                     # 8 majors, 15m, 120 days, the last 1000 proposals decided
bigbrain decide --symbols top:20 --days 180 --decisions 2000
bigbrain decide --synthetic                         # offline: random walks, where no decider should make money
bigbrain decide --llm llama3.1:8b --decisions 300   # a free local model instead of (or beside) Jev
```

For the local model, install Ollama from ollama.com, then `ollama pull llama3.1:8b` (about 5 GB; `qwen2.5:7b` is a good alternative). It is shown exactly what Jev is shown and must answer with one of the same three choices, constrained to that form by Ollama. A laptop answers a question in a second or more, so start with a few hundred decisions; answers are cached, so a rerun is free.

Jev cannot be trained: it is one hosted model, every call starts fresh, and it knows only the state it is handed. So the learning lives in the brain. Every proposal is graded when its trade would have closed (taken or not, so every decider sees the same memory), its outcome joins the record for that setup and context, and every ten trades the brain rewrites its lesson about it. `jev` against `jev_blind` is what the memory is worth; both against `take_all` and against zero is whether the decisions are worth anything; `beliefs` asks whether a plain rule on the same memory does as well for free. The report also shows the gain by quarter, to answer the question that matters most: does it get better as memory grows?

The guards: an outcome enters memory only after its exit candle; Jev never sees a symbol or a date, so it cannot lean on anything it remembers about a market's history; only evergreen knowledge (concepts, papers, articles) is recalled from the main brain, never its lessons, some of which were written after the replayed candles; every number is computed in code and handed over as words, because Jev does not do arithmetic; and every standard error is computed across days, because trades open on the same day ride the same market. Without that last one, a single falling week makes any long signal look "clearly losing" with a t-statistic of four. Jev's answers are cached under `.brain/decider/`, so a rerun is free, and every decision is logged there with its probabilities and outcome.

### Studying past papers: training a decision model on years of trades

Jev cannot be trained, but [Laya](https://huggingface.co/convaiinnovations/laya) (Convai Innovations, Apache 2.0, about 1 GB, runs on a Mac) can. `bigbrain study` treats years of history like past exam papers: the model studies thousands of old trade proposals with what actually happened, then sits exams on stretches of time it never studied. Only the exams count.

```bash
pip install -e ".[laya]"                  # PyTorch and Laya; the model downloads on first use
bigbrain study --smoke                    # a few minutes: proves Laya trains and answers on this machine
bigbrain study --practice                 # practice papers with a hidden rule: can the learners find it?
bigbrain study                            # 5 years of 15m candles on 8 majors, three exams (hours on a Mac)
bigbrain study --learners rules           # the simple-rules learner alone, in a couple of minutes
```

* **Past papers.** Every signal the trader's playbook fired, described in words (the setup, the market, and the brain's record of that setup up to that moment), never with a symbol or a date.
* **The answer key** is not one right answer: each of LONG, SHORT and SKIP gets a probability that grows with what it would have earned after costs, so a trade that made 80 bp says "take it" firmly and one that made 3 bp barely leans. Laya is trained on these soft targets exactly as its own Apple-silicon script trains it.
* **Exams.** The first 40% of the timeline is study only; the rest is cut into exams. Each exam is sat by a fresh copy trained only on trades that had closed before the exam began.
* **Classmates.** `take_all`, `beliefs`, and `rules`: simple rules learned from the same papers ("a long signal while momentum is positive and rising: take it"), adopted only at three standard errors across days because many candidate rules are checked. If Laya cannot beat the simple rules, it has added nothing.
* **A test of the test.** `--practice` plants a rule in synthetic papers. The rules learner finds it and makes money on every exam; on random walks it adopts no rules at all. A learner that cannot find a planted rule will not find a real one.

### Polymarket market making: record first, replay, then decide

On Polymarket a maker pays no fee and earns rebates, while a taker pays up to 1.75% of the notional at 50 cents. So "a tiny profit, many times a day, without paying fees" means market making: rest a buy just under the price and a sell just over it, and earn the gap when both fill. The danger is the other side of each fill: resting orders are hit most often when someone knows more, and the price keeps going. Only data can say whether the gap beats that.

```bash
bigbrain poly markets                       # markets in the liquidity-rewards program, their reward rules
caffeinate -i bigbrain poly record          # record 8 of them for 24 hours; public data, no account or key
bigbrain poly simulate                      # replay the latest recording with you as the maker
bigbrain poly simulate --size 20 --improve 1 --max-inventory 100
caffeinate -i bigbrain poly record --all    # every reward market, a book a minute
bigbrain poly simulate --size 200           # at the reward minimum size: estimated share of each reward pool
```

* **Recording.** Order books every 2 seconds and every trade, from Polymarket's public CLOB, as JSON lines under `.brain/polymarket/`. When a poll returns only trades never seen before, some may have been missed, and that is written down as a gap.
* **Replay.** A maker rests a bid at the best bid (or `--improve` ticks better) and, once it holds shares, an ask at the best ask; it never sells what it does not hold and stops buying at `--max-inventory`. Quotes go live `--latency` seconds after the book they reacted to. Fills come only from recorded trades: a trade through the quote fills it; a trade at its price fills it once the shares queued ahead of it have traded. The strict model, which never fills at the price, is reported beside it.
* **The cost of being picked off.** After every fill the report records where the mid was 10 and 60 seconds later. Anything still held at the end is valued at the mid and as if sold to the bid with the taker fee.
* **Liquidity rewards, estimated.** With `--size` at or above a market's minimum reward size, the replay scores a bid on each outcome with Polymarket's published formula (each sampled minute, every order within the maximum spread scores ((v - s) / v)^2 x size; side one is Yes bids plus No asks, side two the reverse; one-sided quoting counts a third, and only two-sided quoting counts outside 10 to 90 cents) against every other order in the recorded book, and reports the share and USDC a day as a range: the books add orders up by price, not by maker. `--reward-offset` rests the bids a set number of cents from the mid instead of at the best bid.
* **Every market.** `bigbrain poly record --all` records every reward market, with a book a minute (rewards are sampled once a minute) into compressed files; the replay ranks markets by estimated rewards plus spread.
* **Not counted:** maker rebates, which depend on everyone else's volume.

### Trading what survived: a strategy book

`bigbrain strategy` trades a lab rule live in its own book, with everything the playbook trader has (fills at live quotes or the next open, real funding, fees and slippage, atomic bookkeeping, the dashboard) and none of its learning: the rule decides, the book executes. Parameters are re-chosen every 30 days on all the history the book can see, with the same selection the lab's walk-forward used, so what runs is what was tested. A buy-and-hold control of the same coins from the same start runs beside it and `bigbrain portfolio --book trend` shows both.

```bash
bigbrain strategy --rule trend_vt --symbols BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,SOLUSDT --target-vol 0.15 --wallet 1000
scripts/macos-service.sh install strategy --rule trend_vt --target-vol 0.15     # the same, as the background service
```

This is the one rule that survived the lab: daily trend following on the majors, sized to a volatility target. Expect a few decisions a month per coin, long flat stretches, and returns that arrive in bursts. Two days shows the machinery; months show the performance.

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
  lab.py              the lab: honest simulator, rule grid with plateau score, walk-forward, verdicts the brain keeps
  scalp.py            scalping setups (VWAP fade, stop-sweep reversal, squeeze breakout) and the scan that says where their edge went
  decider.py          the decider experiment: Jev (TypeSafe's decision model) reading the brain's memory, against the belief rule and taking every signal
  study.py            study: train Laya (and simple rules) on years of past trades, grade them on years they never saw
  polymarket.py       Polymarket market making: record live books and trades, replay them with you as the maker
  strategy.py         a strategy book: a lab rule traded live with the trader's machinery, monthly refit, buy-and-hold control
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
