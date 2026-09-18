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
