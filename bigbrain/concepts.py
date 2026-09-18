"""Concept extraction: the brain's vocabulary of trading ideas.

Concepts are the glue between cells. Two cells that mention the same concept
are candidates for a synapse. The lexicon below maps a canonical concept name
to the phrases that signal it. Anything not in the lexicon still contributes
through plain term statistics (see ``tokenize``), so the brain can link on
vocabulary it was never explicitly taught.
"""

from __future__ import annotations

import re
from collections import Counter

# canonical concept -> aliases (lowercase). Keep aliases specific enough that
# they rarely fire by accident in unrelated prose.
LEXICON: dict[str, tuple[str, ...]] = {
    # --- price action & market structure
    "trend": ("trend", "trending", "uptrend", "downtrend", "trend following", "trend-following"),
    "momentum": ("momentum", "momentum factor", "relative strength"),
    "mean reversion": ("mean reversion", "mean-reversion", "mean reverting", "reversion to the mean", "reverting", "oversold", "overbought", "stretched", "buy the dip", "buying the dip", "fade", "fading", "snap back"),
    "support and resistance": ("support", "resistance", "support level", "resistance level"),
    "breakout": ("breakout", "break out", "breakdown", "range break"),
    "volatility": ("volatility", "vol", "volatile", "realized volatility", "implied volatility", "vix"),
    "liquidity": ("liquidity", "liquid", "illiquid", "bid-ask", "bid ask spread", "order book", "market depth"),
    "volume": ("volume", "trading volume", "turnover", "vwap"),
    "market regime": ("regime", "market regime", "regime change", "bull market", "bear market", "sideways"),
    "seasonality": ("seasonality", "seasonal", "calendar effect", "day-of-week", "turn of the month"),
    # --- indicators
    "moving average": ("moving average", "sma", "ema", "simple moving average", "exponential moving average", "golden cross", "death cross", "crossover"),
    "rsi": ("rsi", "relative strength index", "overbought", "oversold"),
    "macd": ("macd", "moving average convergence divergence", "signal line"),
    "bollinger bands": ("bollinger", "bollinger bands", "bollinger band", "lower band", "upper band", "band touch"),
    "atr": ("atr", "average true range", "true range"),
    "stochastic": ("stochastic", "stochastic oscillator"),
    "candlestick patterns": ("candlestick", "doji", "engulfing", "hammer", "shooting star"),
    "chart patterns": ("head and shoulders", "double top", "double bottom", "triangle pattern", "flag pattern", "cup and handle"),
    "fibonacci": ("fibonacci", "fib retracement", "retracement"),
    # --- risk & portfolio
    "risk management": ("risk management", "risk control", "risk limit", "risk budget", "risk per trade", "per trade", "risk more than", "never risk", "risk of ruin", "capital preservation"),
    "position sizing": ("position sizing", "position size", "size positions", "size a position", "how much to risk", "% of equity", "percent of equity", "% of capital", "percent of capital", "% per trade", "sizing", "kelly", "kelly criterion", "fractional kelly", "bet size"),
    "stop loss": ("stop loss", "stop-loss", "stop", "trailing stop", "protective stop"),
    "take profit": ("take profit", "profit target", "price target"),
    "drawdown": ("drawdown", "max drawdown", "maximum drawdown", "underwater"),
    "sharpe ratio": ("sharpe", "sharpe ratio", "risk-adjusted return", "risk adjusted return", "sortino"),
    "diversification": ("diversification", "diversified", "correlation", "uncorrelated", "portfolio construction"),
    "leverage": ("leverage", "leveraged", "margin", "margin call"),
    "expected value": ("expected value", "expectancy", "edge", "positive expectancy", "win rate", "payoff ratio", "reward to risk", "risk reward"),
    "hedging": ("hedge", "hedging", "hedged"),
    # --- execution & costs
    "transaction costs": ("transaction cost", "transaction costs", "commission", "commissions", "fees", "cost of trading"),
    "slippage": ("slippage", "market impact", "price impact", "execution cost"),
    "order types": ("limit order", "market order", "stop order", "iceberg order"),
    # --- research method
    "backtesting": ("backtest", "backtesting", "back-test", "back test", "historical simulation"),
    "overfitting": ("overfit", "overfitting", "curve fitting", "curve-fitting", "data snooping", "data mining bias", "look-ahead bias", "lookahead bias"),
    "walk-forward": ("walk-forward", "walk forward", "out-of-sample", "out of sample", "cross-validation", "cross validation"),
    "statistical significance": ("statistical significance", "p-value", "t-statistic", "significance", "hypothesis test"),
    "survivorship bias": ("survivorship bias", "survivorship"),
    "machine learning": ("machine learning", "deep learning", "neural network", "reinforcement learning", "random forest", "gradient boosting", "lstm", "transformer"),
    "sentiment": ("sentiment", "news sentiment", "social media", "twitter", "reddit"),
    "factor investing": ("factor", "factors", "value factor", "size factor", "quality factor", "fama-french", "fama french", "alpha", "beta"),
    "market efficiency": ("efficient market", "market efficiency", "emh", "random walk", "anomaly", "anomalies"),
    "behavioral finance": ("behavioral", "behavioural", "loss aversion", "overconfidence", "herding", "disposition effect", "psychology"),
    # --- instruments & venues
    "options": ("option", "options", "call option", "put option", "greeks", "delta", "gamma", "theta", "vega"),
    "futures": ("futures", "futures contract", "contango", "backwardation"),
    "forex": ("forex", "fx", "currency pair", "eurusd", "eur/usd", "gbpusd", "usdjpy", "carry trade"),
    "crypto": ("crypto", "cryptocurrency", "cryptocurrencies", "bitcoin", "btc", "ethereum", "eth", "defi", "altcoin", "altcoins", "btcusdt", "btcusd", "ethusdt", "ethusd", "solusdt", "usdt", "stablecoin", "perpetual", "perps", "binance", "coinbase"),
    "equities": ("equity", "equities", "stock", "stocks", "shares", "s&p 500", "sp500", "nasdaq", "etf"),
    "bonds": ("bond", "bonds", "treasury", "yield curve", "interest rate", "interest rates", "fed", "central bank"),
    "market microstructure": ("microstructure", "market microstructure", "high frequency", "high-frequency", "hft", "tick data", "order flow"),
    "arbitrage": ("arbitrage", "statistical arbitrage", "stat arb", "pairs trading", "cointegration"),
    "macroeconomics": ("macro", "macroeconomic", "inflation", "gdp", "unemployment", "recession"),
    "earnings": ("earnings", "earnings announcement", "post-earnings", "guidance", "eps"),
    "fundamental analysis": ("fundamental", "fundamentals", "valuation", "p/e", "price to earnings", "book value", "cash flow"),
    "technical analysis": ("technical analysis", "technicals", "chart analysis", "price action"),
    "algorithmic trading": ("algorithmic trading", "algo trading", "systematic trading", "quantitative trading", "quant", "trading system"),
    "market making": ("market making", "market maker", "market makers"),
    "portfolio optimization": ("portfolio optimization", "mean-variance", "mean variance", "efficient frontier", "markowitz", "risk parity"),
    "tail risk": ("tail risk", "fat tails", "black swan", "crash", "kurtosis", "skew", "skewness"),
    # --- styles, tooling and platforms (what forums and repositories talk about)
    "day trading": ("day trading", "daytrading", "day trader", "intraday", "scalping", "scalp", "scalper"),
    "swing trading": ("swing trading", "swing trade", "swing trader", "position trading"),
    "paper trading": ("paper trading", "paper trade", "demo account", "sim trading"),
    "trading platform": ("tradingview", "pine script", "pinescript", "metatrader", "mt4", "mt5", "ninjatrader", "thinkorswim", "interactive brokers", "ibkr", "alpaca", "binance", "bybit", "coinbase", "kraken", "robinhood", "webull"),
    "trading library": ("backtrader", "zipline", "vectorbt", "backtesting.py", "quantconnect", "lean engine", "ccxt", "ta-lib", "talib", "pandas-ta", "freqtrade", "jesse", "hummingbot", "qlib", "yfinance"),
    "open source": ("open source", "open-source", "github", "repository", "repo", "library", "framework", "python package"),
    "prop trading": ("prop firm", "prop trading", "funded account", "ftmo", "evaluation account"),
    "market data": ("market data", "ohlcv", "tick data", "level 2", "candles", "historical data", "data feed", "api key", "websocket"),
    "execution": ("execution", "order execution", "latency", "fill", "fills", "order routing", "smart order routing"),
}

def _alias_pattern(alias: str) -> re.Pattern[str]:
    # Word boundaries on both sides, except an alias that starts with a symbol
    # such as "% per trade" may follow a digit directly ("1% per trade").
    lead = "" if not alias[0].isalnum() else r"(?<![\w/])"
    return re.compile(lead + re.escape(alias) + r"(?![\w/])", re.IGNORECASE)


_ALIAS_PATTERNS: list[tuple[str, re.Pattern[str]]] = sorted(
    ((concept, _alias_pattern(alias)) for concept, aliases in LEXICON.items() for alias in aliases),
    key=lambda pair: -len(pair[1].pattern),
)

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his how i if in into is
    it its it's may might more most much must not of on or our over per she should so some such than that the their
    them then there these they this those through to under up us very was we were what when where which while who
    whom why will with within without would you your also both each either just like many one only other same
    such use used using very via well yet paper papers study studies results result show shows shown propose
    proposed method methods approach approaches model models data based using""".split()
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-']{2,}")
_SUFFIXES = ("ization", "ations", "ation", "ingly", "ities", "ness", "ments", "ment", "ings", "ing", "ies", "ers", "ed", "es", "ly", "s")


def stem(token: str) -> str:
    """Crude suffix stripping so 'positions', 'positioning' and 'position' share a stem."""
    if token.endswith("ss") or len(token) <= 4:
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def extract_concepts(text: str) -> list[str]:
    """Return the canonical concepts signalled by ``text``, ordered by first appearance."""
    found: dict[str, int] = {}
    for concept, pattern in _ALIAS_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        pos = match.start()
        if concept not in found or pos < found[concept]:
            found[concept] = pos
    return [concept for concept, _ in sorted(found.items(), key=lambda kv: kv[1])]


def tokenize(text: str) -> Counter[str]:
    """Bag of informative lowercase terms. Feeds the term-statistics side of linking."""
    counts: Counter[str] = Counter()
    for token in _TOKEN_RE.findall(text.lower()):
        token = token.strip("-'")
        if len(token) < 3 or token in STOPWORDS:
            continue
        counts[stem(token)] += 1
    return counts


_RSI_READING = re.compile(r"\brsi\s*(?:\(\d+\))?\s*(?:at|of|is|=|:|reads|reading|around|near)?\s*(\d{1,3})(?:\.\d+)?\b", re.IGNORECASE)
_BAND_BELOW = re.compile(r"\b(below|under|beneath|outside)\b.{0,25}\b(lower)\b.{0,15}\bband", re.IGNORECASE)
_BAND_ABOVE = re.compile(r"\b(above|over|outside)\b.{0,25}\b(upper)\b.{0,15}\bband", re.IGNORECASE)


def interpret(text: str) -> str:
    """Translate indicator readings into the words traders use for them.

    "RSI at 28" means oversold; "below the lower Bollinger band" is a stretched move.
    Recall runs this on questions so a numeric description reaches the lessons
    written in trading vocabulary.
    """
    extra: list[str] = []
    for m in _RSI_READING.finditer(text):
        value = int(m.group(1))
        if value <= 30:
            extra.append("oversold mean reversion")
        elif value >= 70:
            extra.append("overbought momentum")
    if _BAND_BELOW.search(text):
        extra.append("oversold stretched move mean reversion volatility")
    if _BAND_ABOVE.search(text):
        extra.append("overbought breakout volatility")
    return f"{text} {' '.join(extra)}" if extra else text


def all_concepts() -> list[str]:
    return sorted(LEXICON)
