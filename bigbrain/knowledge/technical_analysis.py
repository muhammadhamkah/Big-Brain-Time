"""Knowledge pack: technical analysis and indicators, in depth.

Goes beyond the curriculum's introductory cells (moving averages, RSI, MACD,
Bollinger bands, support and resistance, volume and VWAP) into the wider
indicator toolbox, the charting methods traders argue about, and how to test
whether any of it has an edge. Every lesson says what a tool measures, its
standard parameters, how it is used, where it fails, and what the evidence says.
"""

from __future__ import annotations

DOMAIN = "Technical analysis and indicators"

LESSONS: list[tuple[str, str]] = [
    (
        "Ichimoku cloud",
        "The Ichimoku cloud (Ichimoku Kinko Hyo) is a trend following system of five lines built from midpoints of high-low ranges rather than closes. The conversion line is the 9-period midpoint, the base line the 26-period midpoint, and the two span lines that form the cloud are the average of those two lines and the 52-period midpoint, both plotted 26 bars ahead; the lagging span is the close plotted 26 bars back. Price above a rising cloud is bullish, inside the cloud is a range, and conversion-base crossovers give entries much like moving average crossovers. The cloud acts as dynamic support and resistance. Because the settings were tuned for a six-day Japanese trading week and the lines lag badly, Ichimoku whipsaws in sideways markets, and backtests show it behaves like any other lagging trend filter: profitable only where trends persist."
    ),
    (
        "Keltner channels",
        "Keltner channels plot an exponential moving average, typically 20 periods, with bands set a multiple of Average True Range above and below, usually 1.5 to 2 times ATR over 10 or 20 periods. Unlike Bollinger bands, which use standard deviation, ATR bands expand more smoothly and are less distorted by single outlier bars. Trend followers buy closes above the upper band and use the midline or lower band as a trailing stop; mean reversion traders fade touches of the bands inside ranges. The channel is essentially a volatility-scaled envelope, so it inherits the strengths and weaknesses of ATR: it adapts to regime changes with a lag and says nothing about direction on its own. Keltner channels are most useful as the container in the Bollinger squeeze comparison and as a stop loss framework."
    ),
    (
        "Donchian channels",
        "Donchian channels mark the highest high and lowest low of the last N bars, with 20 and 55 periods the classic Turtle trader settings. A close above the upper channel is a breakout entry; the opposite channel or a shorter 10-period channel is the exit. This is the purest form of trend following: no smoothing, no parameters beyond lookback, and entries only after price has already proved itself. Long term studies of futures markets show Donchian breakouts with volatility-based position sizing have earned positive returns for decades with low win rates and long flat periods. Weaknesses are many false breakouts in ranges, giving back open profits before the trailing exit fires, and heavy drawdowns when many markets chop at once. Channel width is also a simple volatility measure."
    ),
    (
        "Stochastic oscillator",
        "The stochastic oscillator measures where the close sits within the high-low range of the last N bars, scaled 0 to 100. The fast %K uses 14 periods; the slow version smooths %K with a 3-period average and adds a 3-period signal line %D. Readings above 80 are called overbought and below 20 oversold, and the standard signals are %K crossing %D from those zones and divergence from price. Like RSI it is a momentum oscillator that reads mean reversion in ranges but stays pinned in strong trends, so blindly selling above 80 loses money in bull markets. Tests find modest edge for stochastic mean reversion in equity indices when combined with a trend filter, and no reliable edge for the raw crossovers. Its true measurement is short-term position in range, not value."
    ),
    (
        "Williams %R",
        "Williams %R is the stochastic oscillator inverted: it measures how far the close is below the highest high of the last 14 bars, scaled from 0 to -100. Readings from 0 to -20 are overbought and from -80 to -100 oversold. Because it uses no smoothing it reacts faster than slow stochastic and is noisier, which suits short-term mean reversion in liquid stocks and indices, for example buying an index ETF when %R drops below -90 and exiting when it rises above -30. Larry Williams designed it for swing trading with a trend filter, and that is where tests find a small edge; used alone in trending markets it generates a stream of premature counter-trend signals. It is mathematically redundant with stochastic %K, so using both adds nothing."
    ),
    (
        "Commodity Channel Index (CCI)",
        "The Commodity Channel Index measures how far the typical price (the average of high, low and close) has moved from its 20-period simple moving average, divided by 0.015 times the mean absolute deviation. Roughly three quarters of readings fall between -100 and +100, so moves beyond those levels flag unusually strong momentum. Donald Lambert designed it for cyclical commodities, but it is used on any market either as a breakout tool, buying when CCI crosses above +100, or as a mean reversion tool, fading readings beyond 200. Those two uses contradict each other, and which one works depends on regime, exactly as with Bollinger bands. CCI is a z-score of price around a moving average in disguise, so it offers little that a normalized moving average distance does not."
    ),
    (
        "ADX and the directional movement index",
        "Welles Wilder's directional movement system measures trend strength without direction. Plus DI and minus DI capture the share of the true range that came from upward and downward expansion; ADX is a smoothed average of the difference between them, usually over 14 periods, scaled 0 to 100. ADX below 20 marks a range, above 25 a developing trend and above 40 a strong one; a DI crossover gives direction. Traders use ADX as a filter: run trend following when it is rising and mean reversion when it is low, which is one of the few indicator combinations that tests well. ADX lags substantially because of double smoothing, so it confirms trends late and often peaks near their end. A falling ADX does not mean reversal, only weakening trend."
    ),
    (
        "Parabolic SAR",
        "Parabolic SAR (stop and reverse) plots a trailing stop that accelerates toward price as a trend lasts. The stop starts at the prior extreme and moves each bar by an acceleration factor, beginning at 0.02 and rising by 0.02 with each new high or low up to a maximum of 0.20. When price crosses the dots the position flips. It is always in the market, so it is a pure trend following exit mechanism rather than an entry signal. It works well in smooth trends and is destroyed by ranges, where it flips every few bars and pays the spread each time. Most practitioners use it only for the trailing stop leg, filtered by ADX or a moving average, and tests confirm that the stop-and-reverse version alone loses after costs."
    ),
    (
        "On-balance volume and accumulation/distribution",
        "On-balance volume adds the bar's volume when the close is up and subtracts it when down, producing a running total whose slope is meant to show whether volume favors buyers or sellers. The accumulation/distribution line refines this by weighting volume by where the close sits within the bar's range, so a close near the high counts as accumulation. Both are compared to price: new price highs without new OBV highs are a bearish divergence suggesting a rally on thin participation. They are momentum measures dressed as flow measures, since volume is assigned by direction rather than by actual trade initiation. Evidence for their standalone predictive power is weak, and they are unreliable on assets with fragmented or wash-traded volume such as crypto. They add value mainly as confirmation for breakout trading."
    ),
    (
        "Money flow index",
        "The money flow index is a volume-weighted RSI: it multiplies typical price by volume, splits the result into positive and negative flow by whether typical price rose or fell, and maps the ratio over 14 periods to 0-100. Readings above 80 are overbought and below 20 oversold, and divergence from price is the classic signal. Weighting by volume is meant to catch moves that lack participation, but in practice MFI tracks RSI closely and shares its behavior, staying overbought through strong trends. Tests show its oversold readings support short-term mean reversion in liquid equities about as well as RSI does, and no better. Because it depends on volume it is unusable on spot forex and unreliable on thinly traded assets. Treat it as a variant of RSI, not an independent signal."
    ),
    (
        "VWAP bands and anchored VWAP",
        "Standard VWAP resets each session, which makes it meaningless on longer horizons. Anchored VWAP instead starts the volume-weighted average at a chosen event such as an earnings gap, a swing low, an IPO or a year's first bar, giving the average price paid by everyone who traded since that moment. Price above an anchored VWAP means the average holder since the anchor is in profit, which is why these levels act as support and resistance and as reference points for institutions working large orders. VWAP bands add one, two or three standard deviations of price around VWAP, used for intraday mean reversion toward the average. The choice of anchor is subjective and invites hindsight, so anchored VWAP should be tested with anchors defined by rule, and the bands behave like Bollinger bands with the same regime dependence."
    ),
    (
        "Volume profile and point of control",
        "A volume profile is a histogram of volume traded at each price over a chosen window, rather than over time. The price with the most volume is the point of control, the range holding about 70 percent of volume is the value area, and thin regions are low volume nodes. High volume nodes act as support and resistance because many positions were opened there; price tends to move quickly through low volume nodes, which makes them targets for breakout trading. Traders fade moves back toward the point of control inside balanced sessions and trade acceptance outside the value area as trend. Profiles need real exchange volume, so they are weak in forex and fragmented crypto, and the value area is only as meaningful as the window chosen. Systematic tests of profile levels as support are scarce and mixed."
    ),
    (
        "Market profile",
        "Market profile, developed by Peter Steidlmayer at the Chicago Board of Trade, organizes each session into 30-minute time-price opportunities so the chart shows how long price spent at each level. A bell-shaped profile is a balanced day where price rotated around value; a long thin profile is a trend day. The initial balance is the first hour's range, and extensions beyond it signal directional conviction. The framework gives traders a vocabulary for auction theory: price advertises, time validates, volume confirms. It is a lens on order flow rather than a signal generator, and its day-type classification is useful for choosing mean reversion inside balance or breakout trading outside it. Evidence for the specific patterns such as poor highs and single prints is anecdotal, and most practitioners today use volume profile instead."
    ),
    (
        "Fibonacci retracements and extensions",
        "Fibonacci retracements divide a swing into 23.6, 38.2, 50, 61.8 and 78.6 percent levels and treat them as likely support or resistance for a pullback; extensions at 127.2 and 161.8 percent give profit targets. The 50 percent level is not a Fibonacci ratio at all, and the theory that markets obey the golden ratio has no mechanism behind it. Controlled tests that compare Fibonacci levels with randomly placed levels find no difference in how often price reverses there, which means the levels have no intrinsic edge. They still appear to work because pullbacks of a third to two thirds are common in any trend, the levels are widely watched so stop losses cluster near them, and traders remember hits and forget misses. Use them as a pullback-depth heuristic within trend following, not as a forecast."
    ),
    (
        "Pivot points",
        "Floor pivot points compute a central pivot as the average of the prior session's high, low and close, then support and resistance levels S1, S2, R1 and R2 as reflections of that range around the pivot. Variants include Camarilla, Woodie and Fibonacci pivots, differing only in the multipliers. Pivots are popular with intraday futures and forex traders as pre-computed reference levels for mean reversion toward the pivot and breakout trading through R1 or S1. Their usefulness comes from being a deterministic function of yesterday's range, so they scale with volatility and are watched by many participants, giving them a mild self-fulfilling quality. Backtests show the pivot itself has some support and resistance effect in index futures but the outer levels add little. They are reference points, not a strategy."
    ),
    (
        "Elliott wave theory",
        "Elliott wave theory claims markets move in five waves with the trend and three against it, nested at every degree from minutes to centuries, with wave lengths related by Fibonacci ratios. Practitioners count waves to forecast the current position in the cycle. The main criticism is that the rules allow so many alternative counts that almost any price history fits after the fact, while forecasts made in real time diverge widely between analysts. Wave counts are routinely revised when price disagrees, which makes the theory unfalsifiable and unfit for a backtest. There is no published systematic evidence of edge, and the pattern's popularity persists through hindsight bias. Its salvageable content is the observation that trends unfold in impulse and correction phases, which trend following and pullback strategies exploit without needing the wave labels."
    ),
    (
        "Wyckoff method",
        "The Wyckoff method, from Richard Wyckoff in the 1930s, reads price and volume as the footprints of a composite operator who accumulates before markups and distributes before markdowns. An accumulation range has a selling climax, an automatic rally, secondary tests, a spring below support that traps sellers, and a sign of strength on rising volume; distribution mirrors this with an upthrust. The three laws are supply and demand, cause and effect (range width sets the move size), and effort versus result (volume without price progress warns of reversal). Wyckoff is a structured way of reading ranges before breakout trading, and its core ideas survive in volume profile and supply and demand zones. It is discretionary and the phases are identifiable mainly in hindsight, so its edge as a rule set is unproven."
    ),
    (
        "Supply and demand zones",
        "Supply and demand zones are price areas from which a sharp move began, on the theory that unfilled institutional orders remain there and will react when price returns. A demand zone is the base before a rally, drawn from the last bearish candle's range; a supply zone is the base before a decline. Traders buy the first return to a fresh demand zone with a stop loss just beyond it, aiming for a payoff ratio of two or three to one. The concept overlaps heavily with support and resistance and order blocks, differing mostly in vocabulary. Zones weaken with each test, since the orders that made them get filled. The idea that resting orders persist for weeks is a story, not a measured fact, but the trade structure, defined risk near a level with room to run, is sound risk management."
    ),
    (
        "Order blocks and smart money concepts",
        "Smart money concepts (SMC, ICT) describe price as engineered by institutions to hunt retail stop losses: order blocks are the last opposing candle before an impulsive move, fair value gaps are three-candle imbalances expected to fill, and liquidity sweeps are wicks beyond obvious highs and lows before reversal. The vocabulary is new but the content maps onto supply and demand zones, gaps and stop hunts, which have long histories. The narrative that a coordinated smart money entity manipulates every move is not supported by market microstructure evidence, although stop clustering and liquidity-seeking execution are real. Because the rules are flexible and rarely quantified, SMC is difficult to backtest, and no published tests show edge beyond that of ordinary breakout and pullback logic. Its strength is teaching traders to place stops away from obvious levels."
    ),
    (
        "Divergence, regular and hidden",
        "Divergence compares swings in price with swings in a momentum oscillator such as RSI, MACD or stochastic. Regular bearish divergence is a higher price high with a lower oscillator high, suggesting momentum is fading and a reversal is possible; regular bullish divergence is the mirror. Hidden divergence points the other way: in an uptrend a higher price low with a lower oscillator low signals continuation. Divergence is a mean reversion signal for the regular type and a trend following signal for the hidden type. Its weakness is that momentum fades long before trends end, so divergences can persist through several more legs, and identifying swing points involves subjective choices. Tests show regular divergence alone has low accuracy, improving when combined with a support level and a stop loss beyond the extreme."
    ),
    (
        "Multi-timeframe analysis",
        "Multi-timeframe analysis reads the same market on two or three timeframes, usually spaced by a factor of four to six, such as daily, four-hour and one-hour. The higher timeframe defines trend and key support and resistance, the middle one shows the setup, and the lowest times the entry with a tight stop loss. Trading only in the direction of the higher timeframe trend is the practical implementation of momentum at one horizon and mean reversion at another. Benefits are better payoff ratios and fewer counter-trend trades; costs are fewer signals and conflicting readings that tempt discretionary overrides. When systematized, the higher timeframe filter is a moving average or ADX condition, and tests show such filters improve most breakout and pullback strategies. Indicators on the higher timeframe update less often, which introduces lag."
    ),
    (
        "Trend lines and channels",
        "A trend line connects two or more swing lows in an uptrend or swing highs in a downtrend; a channel adds a parallel line through the opposite swings. Traders buy pullbacks to a rising trend line, sell at the channel top for mean reversion, and treat a close through the line as a breakout signal. Lines drawn through wicks and through closes give different results, and choosing which swings to connect is subjective, so two analysts rarely draw the same line. Steep lines break quickly; shallow ones are more durable. Algorithmic versions use linear regression channels or lines fit to detected pivots, and tests of those show trend line breaks behave like other breakout signals, with edge that depends on the trend regime. The line itself has no magic; it is a visual proxy for trend slope and volatility."
    ),
    (
        "Gaps: breakaway, runaway and exhaustion",
        "A gap is empty space between one bar's range and the next, usually from overnight news. Breakaway gaps leave a consolidation on high volume and start trends; runaway or measuring gaps occur mid-trend and roughly mark its halfway point; exhaustion gaps come late on climactic volume and are followed by reversal. Common gaps inside ranges fill quickly. The classification is only clear afterward, so live traders rely on context: gap size relative to ATR, volume, and where price sits in the trend. The gap fill statistic is real but misleading: most small gaps fill within days, while large gaps on earnings often do not, and post-earnings announcement drift means large gaps tend to continue. Gap fades and gap-and-go breakouts are both tradable but need volume filters and stop losses beyond the gap."
    ),
    (
        "Opening range breakout",
        "The opening range breakout takes the high and low of the first 5, 15, 30 or 60 minutes of a session and buys a break above the range or sells a break below, with a stop loss at the opposite side and a time-based exit near the close. It targets the tendency of trend days to declare direction early after the opening auction absorbs overnight order flow. Toby Crabel's work on narrow-range days and inside days found that breakouts after volatility contraction have higher follow-through. Published tests on index futures and large stocks show a modest edge that has decayed over time and depends on filtering for volatility, gap direction and relative volume. Costs are heavy because entries occur in the fastest part of the day; the strategy loses on rotational days where both sides trigger."
    ),
    (
        "Heikin-Ashi and Renko charts",
        "Heikin-Ashi candles replace each bar with averages: the close becomes the mean of open, high, low and close, and the open becomes the midpoint of the previous Heikin-Ashi candle. This smooths trends into long runs of one color and delays reversals, which makes trend following easier to read but hides the real prices, so entries and stop losses must still be set from ordinary bars. Renko charts drop time entirely and draw a new brick only when price moves a fixed amount or an ATR multiple, filtering noise in ranges and highlighting breakouts. Both are smoothing filters in disguise; they lag exactly like a moving average and repaint the current brick or candle until it completes. Backtests on these charts are easy to get wrong because the chart's own construction leaks future information."
    ),
    (
        "Bollinger squeeze and TTM squeeze",
        "A Bollinger squeeze occurs when band width, the distance between the upper and lower bands divided by the middle, reaches a multi-month low, indicating volatility contraction that often precedes expansion. John Carter's TTM squeeze formalizes this by flagging when the Bollinger bands, at 20 periods and two standard deviations, fall inside Keltner channels at 20 periods and 1.5 ATR, then uses a momentum histogram to pick direction when the squeeze releases. The squeeze itself predicts volatility, not direction, which is why the setup is paired with breakout trading in the direction of the first strong close. Volatility clustering makes the contraction-expansion cycle genuine, but the direction call is a coin flip improved only by trend context, and false releases are common. The squeeze is a good time to buy options rather than sell them."
    ),
    (
        "Moving average ribbons and the Hull moving average",
        "A moving average ribbon plots six to twelve moving averages of increasing length, for example 10 through 60 periods, so their spacing and order show trend maturity: a fanning ribbon is an accelerating trend, a compressed and tangled one a range. The Guppy multiple moving average separates short averages representing traders from long averages representing investors. Ribbons give a richer view of the same information as one crossover and share its lag. The Hull moving average attacks that lag by combining weighted moving averages of the full period and half period, then smoothing with the square root of the period; it tracks price closely with little overshoot but is prone to whipsaw in ranges precisely because it reacts fast. Every smoothing method trades lag for noise, and none escapes that constraint."
    ),
    (
        "Rate of change and momentum oscillators",
        "Rate of change is the percentage change in price over N bars, typically 10 to 20 for swing trading or 250 for the annual momentum used in factor investing. The momentum indicator is the same measure as a price difference. Unlike RSI or stochastic they are unbounded, so overbought is defined relative to history rather than fixed thresholds. Traders use zero-line crosses for trend, extremes for mean reversion, and divergence with price. Their key property is that a single large bar dropping out of the window can swing the reading, producing signals unrelated to current action. Rate of change over 6 to 12 months is the foundation of the cross-sectional momentum anomaly and the time series momentum strategy, both of which have robust evidence; short lookbacks are noisier and lean toward reversal."
    ),
    (
        "Relative strength versus a benchmark",
        "Relative strength in the comparative sense, distinct from RSI, divides an asset's price by a benchmark such as the S&P 500 or a sector index, and plots the ratio. A rising ratio means outperformance regardless of absolute direction. It is the basis of cross-sectional momentum: rank assets by relative performance over 3 to 12 months and buy the leaders. Relative rotation graphs extend this by plotting ratio momentum against ratio level to show rotation among sectors. The evidence for relative strength ranking is strong over intermediate horizons and across asset classes, with the caveat that momentum crashes follow sharp bear market rebounds. Weaknesses are that a strong ratio can come from the benchmark falling faster than the asset, and that leadership changes abruptly at regime shifts. It should be paired with an absolute trend filter."
    ),
    (
        "Supertrend indicator",
        "Supertrend draws a single trailing line at the midpoint of the bar plus or minus a multiple of ATR, usually 10-period ATR times 3, and flips sides when price closes through it. It is a volatility-adjusted stop and reverse, functionally similar to Parabolic SAR and to Keltner-based stops but simpler to read. Trend followers use it as both trend filter and trailing stop loss, and its ATR scaling adapts to regime better than fixed percentage stops. Weaknesses are the ones shared by every always-in-the-market trend tool: repeated flips and losses in ranges, and late exits that give back much of a trend. Backtests on liquid futures and crypto show it earns like a slow moving average crossover, with performance dominated by the multiplier choice and by costs. Shorter multipliers trade more and lose more in chop."
    ),
    (
        "Aroon indicator",
        "The Aroon indicator measures how many bars have passed since the highest high and the lowest low within a window, usually 25 periods. Aroon up is 100 when the high is the latest bar and falls toward zero as it ages; Aroon down does the same for lows. Both near 100 and 0 in opposite pairs describe a strong trend; both low means consolidation, and crossovers signal a change of leadership. It is a time-based rather than price-based trend measure, which makes it insensitive to the size of moves and quick to identify new highs, the same event that drives Donchian channel breakouts. In practice its signals are close to those of Donchian breakouts and ADX, so it adds little when either is already used. It has no volatility information and does not size stops."
    ),
    (
        "Point and figure charts",
        "Point and figure charts ignore time and plot columns of X for rising prices and O for falling prices, with a box size such as one percent or one ATR and a reversal threshold, typically three boxes, before a new column starts. They filter noise, make support and resistance and breakout trading unambiguous, and give price targets through vertical and horizontal counts derived from column height and congestion width. Because columns only form after price commits, signals are equivalent to a filtered breakout with the box size acting as the volatility filter, which shares the lag and whipsaw of every breakout method. Studies of point and figure signals on equities find results similar to Donchian channels. The count targets have no statistical basis, and the chart's construction depends entirely on box size, so parameter choice must be tested."
    ),
    (
        "Market breadth indicators",
        "Breadth indicators measure participation across an index rather than the index price itself. The advance-decline line cumulates the daily count of advancing minus declining stocks; the McClellan oscillator is the difference between 19-day and 39-day exponential averages of that net figure, and its cumulative version is the summation index. Percent of stocks above their 50-day or 200-day moving average and new highs minus new lows are the other standards. Divergence between an index at new highs and deteriorating breadth signals a narrowing rally led by a few large stocks, a pattern seen before several major tops, and washed-out breadth marks capitulation for mean reversion. Breadth is a regime input for position sizing rather than an entry signal; lead times are long and variable, and cap-weighted indices can rise on narrow breadth for many months."
    ),
    (
        "Chaikin money flow",
        "Chaikin money flow sums the accumulation/distribution volume of each bar, which weights volume by where the close sits in the bar's range, over 20 or 21 periods and divides by total volume, giving a bounded reading between -1 and +1. Sustained readings above 0.05 or below -0.05 are read as buying or selling pressure, and crosses of zero as shifts in control. The Chaikin oscillator instead takes the difference of 3 and 10 period exponential averages of the accumulation/distribution line and is used for divergence. Both depend on the assumption that a close near the high means buyers were dominant, which conflates intrabar momentum with order flow. Tests show CMF mostly confirms what price momentum already says; it is useful as a confirmation filter for breakout trading and unreliable where volume data is fragmented."
    ),
    (
        "Indicator lag and repainting",
        "Every indicator built from past prices lags them; a 20-period simple moving average is centered ten bars in the past, and smoothing on top of smoothing, as in ADX or slow stochastic, compounds the delay. Lag is the price of noise reduction, and no filter escapes the tradeoff, although adaptive averages and the Hull moving average shift the balance. Repainting is a different and worse problem: an indicator whose historical values change as new bars arrive, because it uses future data, centered smoothing, pivots that require later bars to confirm, or higher timeframe values that are not yet closed. Repainting indicators look perfect in hindsight and fail live, and are the leading cause of backtests that cannot be reproduced. Any indicator must be evaluated with values as they were known at each bar."
    ),
    (
        "Combining indicators without redundancy",
        "Stacking indicators that measure the same thing, such as RSI, stochastic, Williams %R and CCI, creates the illusion of confirmation while adding no information, because all of them are transformations of recent price momentum. A useful combination pairs indicators from different families: one for trend direction such as a moving average or Ichimoku, one for trend strength or regime such as ADX or band width, one for timing such as an oscillator or pullback rule, and volume or breadth for confirmation. Each filter added removes trades, so the payoff must be a better expectancy per trade, not just a higher win rate on fewer trades. The test is correlation: if two indicators' signals agree more than 80 percent of the time, drop one. Complexity also multiplies parameters, which invites overfitting."
    ),
    (
        "How to test an indicator's edge",
        "To test an indicator, define a signal rule precisely, then measure forward returns after the signal over several horizons and compare them with returns after random entries or with the unconditional average, using enough events to be statistically significant. Report the win rate, average return in units of risk, and the distribution of outcomes, not just the total. Test across markets and years, sweep parameters to check that neighbors of the chosen setting also work, and include transaction costs and slippage. Walk-forward or out-of-sample validation guards against overfitting, and the number of variants tried should be recorded so the result can be deflated. Most indicators fail this test as standalone signals; those that pass are usually measuring trend, volatility or short-term reversal, the effects with real evidence."
    ),
    (
        "Intraday volume seasonality",
        "Intraday volume follows a U shape: heavy in the first thirty to sixty minutes as overnight information is priced in, a lull around midday, and a surge into the close driven by index funds, closing auctions and VWAP algorithms. Volatility and bid-ask spreads follow the same curve, spreads widest at the open. This seasonality matters for execution, since a given order costs more at the open, and for signals, since a volume spike must be judged against the norm for that time of day, which is why relative volume compares current volume to the average for the same minute. It also shapes strategies: opening range breakout exploits the early volatility, midday mean reversion exploits the lull, and end-of-day imbalances feed closing auction effects. Scheduled macro releases and options expirations add predictable spikes."
    ),
]

SOURCES = {
    "feeds": [
        "https://stockcharts.com/articles/atom.xml",
        "https://alvarezquanttrading.com/feed/",
        "https://www.adamhgrimes.com/feed/",
        "https://www.newtraderu.com/feed/",
        "https://quantifiedstrategies.substack.com/feed",
        "https://tradeciety.com/rss.xml",
    ],
    "subreddits": [
        "TechnicalAnalysis",
        "Daytrading",
        "swingtrading",
        "Forex",
        "algotrading",
    ],
    "github_queries": [
        "topic:technical-analysis",
        "topic:technical-indicators",
        "technical analysis indicators library python",
        "volume profile market profile trading",
        "ichimoku supertrend backtest",
    ],
    "arxiv_topics": [
        "technical analysis profitability",
        "technical trading rules data snooping",
        "moving average trading rules",
        "volume weighted average price execution",
        "intraday volume patterns",
    ],
    "urls": [
        "https://school.stockcharts.com/doku.php?id=technical_indicators:ichimoku_cloud",
        "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays",
        "https://www.investopedia.com/terms/t/technicalanalysis.asp",
        "https://www.investopedia.com/terms/f/fibonacciretracement.asp",
        "https://www.cmegroup.com/education/courses/technical-analysis",
        "https://en.wikipedia.org/wiki/Technical_analysis",
    ],
}
