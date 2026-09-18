"""Knowledge pack: quantitative and systematic strategies, in depth.

Goes deeper than the seed curriculum's "Trend following", "Momentum",
"Mean reversion", "Factor investing" and "Pairs trading" cells: concrete rule
sets, typical parameters, honest performance history and failure modes.
"""

from __future__ import annotations

DOMAIN = "Quantitative and systematic strategies"

LESSONS: list[tuple[str, str]] = [
    (
        "Time-series versus cross-sectional momentum",
        "Time-series momentum asks whether an asset's own past return, usually over 12 months, is positive and goes long if so and short otherwise; cross-sectional momentum ranks assets against each other and buys the top decile while shorting the bottom, regardless of whether the market as a whole rose. Time-series momentum across 50 or more futures markets with volatility scaling has delivered roughly Sharpe 0.7 to 1.0 gross since the 1980s and is what managed futures funds mostly run. Cross-sectional equity momentum earns a similar premium with worse tail risk because it is a bet on dispersion, not direction. The two overlap heavily: much cross-sectional profit is explained by time-series exposure. Both suffer in sharp reversals and both need transaction costs controlled because turnover is high."
    ),
    (
        "Dual momentum",
        "Dual momentum, popularized by Gary Antonacci, combines relative and absolute momentum in a monthly rotation. Compare the 12-month return of a few broad assets such as US and international equities, hold the stronger one, but only if its return beats Treasury bills; otherwise move to bonds. The absolute momentum filter cuts drawdown, taking the portfolio to cash before most of the 2008 decline, while the relative filter adds a modest return edge. Backtests from 1974 show roughly double the Sharpe ratio of buy and hold with maximum drawdown near 20% instead of 50%. Failure modes are whipsaws in choppy years like 2011, 2015 and 2020, selling after a drop and buying back higher, and the tiny asset count, which makes results sensitive to the exact lookback and rebalance day."
    ),
    (
        "Momentum crashes and hedging them",
        "Momentum crashes are short, violent losses of the long-short momentum portfolio when a bear market reverses sharply. The short leg fills with high-beta, beaten-down stocks whose beta explodes on the rebound: in 2009 the US momentum factor lost around 70% in three months, with similar events in 1932 and 2020. Daniel and Moskowitz showed the losses are predictable: they follow negative two-year market returns and high volatility. Hedges include volatility scaling the momentum portfolio to a constant risk target, capping the beta of the short side, buying index calls, or cutting exposure when past market return is negative and the VIX is high. Dynamic scaling roughly doubles raw momentum's Sharpe ratio in backtests. No hedge removes the risk; they trade a little average return for a thinner left tail."
    ),
    (
        "Short-term reversal at one week and one month",
        "Short-term reversal is the tendency of stocks with the worst return over the last week or month to outperform the best over the next week or month. It is the opposite of momentum and is why 12-month momentum skips the most recent month. The driver is liquidity provision: uninformed selling pushes prices below fair value and whoever absorbs it gets paid. Gross paper returns are large, around 1% per month for the decile spread, but turnover exceeds 100% per month, so after transaction costs most of it disappears for anyone who is not a market maker. Improvements include reversing only the residual return after removing industry and factor moves, weighting by volume to trade where liquidity was demanded, and skipping stocks with news, since movement after earnings surprises is drift, not reversal."
    ),
    (
        "Value factor construction",
        "The value factor buys cheap assets and sells expensive ones on measures like book to price, earnings yield, cash flow yield or enterprise value to EBITDA. Fama and French's HML sorts stocks by book to market and holds top and bottom 30% portfolios rebalanced yearly. Modern constructions average several ratios, rank within industry to avoid permanent sector bets, and use point-in-time fundamentals lagged several months to avoid look-ahead bias. Value earned about 4% a year long-short from 1927 with a Sharpe ratio near 0.4, but lost money for most of 2007 to 2020, a drawdown over 50%, before rebounding sharply in 2021 and 2022. It works because cheap stocks carry distress risk and investors extrapolate growth too far; it fails in long liquidity-driven growth regimes and when accounting stops describing intangible-heavy businesses."
    ),
    (
        "Quality and profitability factors",
        "Quality strategies buy companies with high, stable profitability, low leverage, conservative accounting and steady growth, and sell their opposites. Novy-Marx showed gross profits to assets predicts returns about as well as book to market and is negatively correlated with value, so combining them raises the Sharpe ratio. Fama and French added profitability (RMW) and investment (CMA) factors to their five-factor model; AQR's quality minus junk combines profitability, growth, safety and payout. Long-short quality delivers roughly 3-5% a year with Sharpe near 0.5 and, unlike value and momentum, tends to gain in bear markets because junk falls hardest. Its weakness is that quality is well known, so the long side trades at a premium and junk rallies like 2009 and 2020-2021 hurt. Quality is most useful as a screen against value traps."
    ),
    (
        "Low volatility and betting against beta",
        "The low volatility anomaly is that stocks with low beta or low realized volatility earn higher risk-adjusted returns than high-beta stocks, contradicting the CAPM. Betting against beta (Frazzini and Pedersen) goes long low-beta assets levered to beta one and short high-beta assets delevered to beta one, earning a Sharpe ratio near 0.7 across equities, bonds and futures. The explanation is leverage constraints: investors who cannot borrow bid up high-beta stocks, and lottery preferences push the same way. Minimum-variance and low-volatility ETFs implement the long side only. The strategy lags in strong speculative rallies, carries hidden interest rate and value exposure, and its long leg became crowded and expensive after 2016. Betas should be estimated from about a year of daily data and shrunk toward one."
    ),
    (
        "Carry across asset classes",
        "Carry is the return an asset earns if its price does not change: the interest differential in currencies, roll yield from the futures term structure in commodities, term premium plus roll-down in bonds, dividend yield minus financing in equities, and implied minus realized volatility in options. Koijen, Moskowitz, Pedersen and Vrugt showed that buying high-carry and selling low-carry assets within each class earns Sharpe around 0.5-0.8, and diversifying across classes raises it further. Carry works because it compensates for crash risk and for providing liquidity to hedgers. It fails in risk-off events like 2008 and March 2020, when high-carry currencies and backwardated commodities collapse together, drawdowns of 30% or more. Carry and trend following are negatively correlated in crises, which is why systematic macro funds run both."
    ),
    (
        "Donchian breakouts and the Turtle rules",
        "The Turtle rules, taught by Richard Dennis in 1983, are the best documented complete trend following system. System one enters on a 20-day Donchian breakout, a new 20-day high or low, skipping the signal if the previous breakout would have been profitable; system two uses a 55-day breakout with no filter. Exits are a 10-day low for system one and a 20-day low for system two. Position size is one unit per 1% of equity divided by N, the 20-day ATR, pyramiding every half N up to four units with a stop two N from entry, plus caps on correlated exposure. The original traders made about 80% a year in the 1980s, but simple Donchian systems on diversified futures now run Sharpe roughly 0.5-0.8 with 20-30% drawdowns, win rates near 40% and long flat stretches in ranging markets."
    ),
    (
        "Moving average and breakout systems for futures trend following",
        "Institutional trend following combines several signal types across lookbacks. Moving average crossovers such as 8 and 32, 16 and 64 or 32 and 128 days become forecasts scaled by volatility and averaged; breakout signals compare price with the middle of its recent range; time-series momentum uses the sign of past 1, 3 and 12-month returns. Forecasts are capped, volatility scaled so each market contributes equal risk, and traded across 40-100 futures spanning equities, bonds, currencies, energy, metals and agriculture. Rob Carver's published framework is a public example. Backtests to the 1970s show gross Sharpe near 0.8-1.0; the live SG Trend Index shows roughly 0.4-0.6 net. Fast systems catch turns but churn; slow ones have lower costs but suffer in V-shaped reversals. Diversification across speeds and markets matters more than the indicator."
    ),
    (
        "Managed futures and CTA performance history",
        "Commodity trading advisors are mostly systematic trend followers on futures. The SG CTA and Barclay indices show annualized returns near 7-8% with 10-12% volatility from 1990 to 2010, Sharpe around 0.5-0.7, and famous crisis gains of 15-20% in 2008 while equities lost 40%. From 2009 to 2019 returns were roughly flat, a long drawdown driven by suppressed volatility, quick central bank reversals and crowding, and many allocators gave up. 2022 was the best year in decades, over 20% for the SG Trend Index, as rates and commodities trended. The key property is convexity: near-zero correlation to stocks on average and positive returns in prolonged bear markets, called crisis alpha. It does not protect against fast crashes like October 1987 or early March 2020, and fees have eaten much of gross returns."
    ),
    (
        "Volatility risk premium harvesting",
        "The volatility risk premium is the persistent gap between implied and subsequently realized volatility: S&P 500 implied exceeds realized in about 85% of months by an average of 3-4 points. Harvesting strategies sell one-month delta-hedged straddles or strangles, sell variance swaps, write covered calls or cash-secured puts like the Cboe PUT index, or short VIX futures in contango. The PUT index has matched equity returns with lower volatility since 1986, Sharpe near 0.7. The premium exists because investors overpay for crash insurance. The cost is a fat left tail: short volatility lost years of gains in October 1987, October 2008 and February 2018, when the XIV product fell 96% in a day. Survivable versions size to the worst historical loss, cap vega, and hedge tails with far out-of-the-money puts or a trend following overlay."
    ),
    (
        "Dispersion trading",
        "Dispersion trading sells index volatility and buys single-stock volatility on the index's constituents, usually with delta-hedged straddles or variance swaps weighted by index weight. The bet is on correlation: index implied volatility embeds an implied correlation between constituents that is typically priced above realized because investors buy index protection. The trade earns when stocks move independently, for example around earnings season, and loses when everything moves together in a crash and correlation jumps toward one. Practical versions trade the largest 30-50 names, are vega or correlation neutral, and rebalance hedges daily. Returns resemble a volatility risk premium strategy with somewhat less tail risk, since the long single-stock volatility partly offsets the short index leg. Transaction costs across many options make it an institutional strategy rather than a retail one."
    ),
    (
        "Cointegration tests for pairs selection",
        "Two non-stationary price series are cointegrated if a linear combination of them is stationary, which is the statistical basis of pairs trading. The Engle-Granger method regresses one price on the other to get the hedge ratio, then runs an augmented Dickey-Fuller test on the residual; it is simple but depends on which asset is the dependent variable. The Johansen test treats all series symmetrically, handles baskets of more than two assets, and returns eigenvectors that are the cointegrating weights. A p-value below 0.05 in sample means little alone: screening thousands of pairs guarantees false positives, so pairs should share an economic link such as the same industry, dual listings, or an ETF and its constituents. Relationships estimated on two years of daily data often break; the practical checks are a walk-forward test and a tradeable spread half-life."
    ),
    (
        "Ornstein-Uhlenbeck spread models and half-life",
        "A mean-reverting spread is often modeled as an Ornstein-Uhlenbeck process: the change in the spread equals theta times the distance from its long-run mean plus noise. Theta is estimated by regressing daily changes on lagged levels, and the half-life of a deviation is ln(2) divided by theta. A half-life of 5-20 trading days suits daily pairs trading; over 60 days ties up capital and gives the relationship time to break. Entry rules use the z-score of the spread against its rolling mean and standard deviation, entering beyond 2 and exiting near 0, with a stop at 3-4 or after two or three half-lives elapse. Optimal thresholds can be solved from OU parameters, but estimates are noisy, so wide bands and small positions are more robust. In-sample mean reversion overstates itself; halve the backtest's expected profit."
    ),
    (
        "Index rebalancing and inclusion trades",
        "Index funds must buy additions and sell deletions at the close on the effective date, creating predictable flow. Historically S&P 500 additions rose 5-8% between announcement and inclusion and partially reversed afterward; Russell reconstruction in June moved small caps similarly. The trade buys announced additions and shorts deletions, closing into the rebalance when passive demand arrives, or provides liquidity on the other side of the closing auction. The anomaly has decayed as arbitrageurs crowded in: S&P inclusion effects shrank to near zero after 2010 and the run-up moved earlier. Remaining edge lies in less-followed indices, closing-auction imbalances, and predicting additions from index methodology before announcement. Risk is concentrated in a few days, so one surprise announcement or a crowded unwind can erase a year of small gains."
    ),
    (
        "Post-earnings announcement drift as a strategy",
        "Post-earnings announcement drift means stocks with a large positive earnings surprise keep outperforming for one to three months, and negative surprises keep underperforming. The strategy computes standardized unexpected earnings, the surprise scaled by its historical standard deviation, or uses the announcement-day abnormal return, then goes long the top decile and short the bottom for about 60 trading days. Published spreads in the 1980s and 1990s were around 4-6% per quarter; today the effect is small in large caps but persists in small, illiquid and analyst-neglected stocks. The drift comes from under-reaction to news and slow analyst revisions. It is a short-horizon momentum trade with high turnover: the announcement-day gap cannot be captured, the drift after it is modest, and combining the announcement return with estimate revisions and volume improves it."
    ),
    (
        "Overnight versus intraday return anomaly",
        "Almost all of the US equity index return since the 1990s has accrued overnight, from the close to the next open, while the intraday session has averaged near zero or slightly negative. The pattern is strongest in high-beta, retail-favored and high-short-interest stocks and reverses in many individual names. Explanations include news released outside trading hours, market makers shedding inventory at the close and rebuying at the open, and retail flow arriving at the open. Holding only overnight has earned equity-like returns with lower volatility in backtests, but it pays the bid-ask spread twice a day, which consumes most of the edge outside the most liquid ETFs and futures. It is more useful as timing information, buying weakness late in the day and selling strength at the open, than as a standalone system."
    ),
    (
        "Intraday momentum in the last half hour",
        "Gao, Han, Li and Zhou documented that the S&P 500 ETF's return in the first half hour predicts the return of the last half hour: a positive open predicts a positive close and vice versa, with a t-statistic above 4 from 1993 to 2013. The strategy trades SPY in the last 30 minutes in the direction of the first 30-minute return, earning around 6-7% a year with Sharpe near 1 before costs and gaining most on volatile days. Drivers are late-day rebalancing by levered ETFs, which buy after up days and sell after down days, and traders who wait for information before committing. Trading once a day in the most liquid instrument keeps costs low. The signal weakened after publication, and on calm days the expected move is smaller than the spread, so a threshold on the morning return helps."
    ),
    (
        "Turn-of-month and FOMC drift strategies",
        "The turn-of-month effect is that equity returns concentrate in the last trading day and first three days of each month, driven by pension contributions, salary investment and month-end rebalancing; US stocks earned most of their return in this window over decades. The pre-FOMC announcement drift, documented by Lucca and Moench, is that the S&P 500 rose about 0.5% on average in the 24 hours before scheduled Federal Reserve decisions from 1994 to 2011, a large share of the equity premium. Both are calendar strategies that hold the index only in the window and sit in cash otherwise. Entries are few so costs are low, but the pre-FOMC drift shrank after publication and reversed in some years, and the turn-of-month effect is small enough that one bad month dominates. Use them as exposure overlays, not standalone systems."
    ),
    (
        "Calendar spreads and term-structure trades",
        "A calendar spread is long one futures month and short another in the same market, isolating the term structure from the outright price. Examples are buying a backwardated commodity front month against the deferred to earn roll yield, selling the VIX front month against the second or third month in contango, and rate curve trades such as long two-year against short ten-year futures. The spread moves on storage costs, seasonal demand, inventory shocks and financing, and is usually far less volatile than the outright, so margins are lower and leverage higher. Systematic versions rank markets by curve slope and go long the steepest backwardation and short the steepest contango, a form of carry. Risks are squeezes near expiry, delivery mechanics, and regime shifts like the April 2020 negative oil settlement, when the front spread moved more in a day than in years."
    ),
    (
        "ETF arbitrage and creation-redemption",
        "ETF prices stay near net asset value because authorized participants create shares by delivering the underlying basket when the ETF trades at a premium, and redeem shares for the basket when it trades at a discount. This is a mean reversion trade on the premium, hedged with the constituents or index futures. In liquid equity ETFs the spread is a few basis points and captured only by high frequency market makers. Larger, slower opportunities appear when the basket is hard to trade: bond ETFs traded at discounts of 5% or more in March 2020 as bonds went illiquid, and international ETFs show stale-price premiums after their home markets close. Leveraged ETFs create end-of-day rebalancing flow that others front-run. The risk is that an apparent discount is the fair price of an untradeable basket, leaving the arbitrageur unhedged."
    ),
    (
        "Cross-asset lead-lag signals",
        "Some markets incorporate information faster than others, creating exploitable leads. Credit spreads and high-yield bond prices often lead equities by days to weeks, since credit investors react to balance sheet stress first; a widening in CDX or a falling HYG relative to Treasuries is a risk signal for stocks. The VIX term structure is a widely used regime gauge: contango means calm and supports carry and short volatility, while inversion marks stress and precedes both crashes and rebounds. Copper against gold, the yen and the dollar index proxy global growth; large caps lead small caps and the US close leads Asian opens. These signals are strongest at weekly horizons and in stress, noisy day to day, and best used as filters that scale exposure. Relationships drift, so they must be retested regularly."
    ),
    (
        "Risk-on risk-off regime models",
        "Risk-on risk-off models compress the state of markets into one indicator that scales exposure across a portfolio. Inputs include the VIX level and term structure, credit spreads, equities against their 200-day moving average, the performance of defensive assets such as Treasuries, gold and the yen against cyclical ones, and cross-asset correlation, which rises in risk-off. Rules are simple: when the composite is risk-off, halve equity and carry exposure or move to bonds; when risk-on, run full size. Such filters have historically cut maximum drawdown roughly in half at the cost of some return and many false alarms, and they lag, switching off after part of the decline and on after part of the rebound. They are trend following on risk, so they share its whipsaw; several inputs with smoothed thresholds reduce churn."
    ),
    (
        "Hidden Markov regime switching for allocation",
        "A hidden Markov model assumes returns come from a few unobserved states, each with its own mean and volatility, with fixed switching probabilities. Fitted to daily or weekly equity returns with two or three states, it typically finds a calm state with positive drift and a turbulent state with negative drift and two to three times the volatility, matching Hamilton's original regime switching results. Allocation rules scale exposure by the filtered probability of the calm state, or run different strategies per state: trend and carry in calm, cash or long volatility in turbulence. Gains over volatility-only rules are modest and mostly in drawdown. The model overfits with more than three states, its state labels shift across refits, and it detects switches only after several observations, so it mostly smooths the same information as realized volatility and drawdown filters."
    ),
    (
        "Strategy ensembles and signal combination",
        "Combining weakly correlated signals or strategies is the most reliable way to raise a Sharpe ratio: ten uncorrelated strategies with Sharpe 0.4 each give a portfolio Sharpe above 1.2 in theory, around 0.8 after realistic correlations of 0.2-0.3. Methods range from equal weighting of z-scored forecasts, to risk parity by inverse volatility, to weights proportional to each signal's out-of-sample Sharpe. Equal weighting is hard to beat because return and correlation estimates are noisy and optimization overweights whatever was lucky in sample. Combine at the forecast stage, before position sizing, and cap forecasts so one strong signal cannot dominate. Diversifying across speeds, horizons and asset classes matters more than adding a tenth equity signal, and strategy correlations rise in crises, so ensembles reduce ordinary volatility more than the worst drawdown."
    ),
    (
        "Position netting across strategies",
        "When several strategies trade the same instruments, their desired positions should be netted into one target per instrument before execution. A momentum system long crude and a mean reversion system short crude cancel, saving two sets of transaction costs and margin while each strategy's paper track record stays intact. Netting needs a portfolio layer that receives target weights from each strategy, applies capital allocations and risk limits, sums them, and generates trades against actual holdings with a buffer so small changes in the net target do not trigger orders. The complication is attribution: profit and loss must still be allocated to each strategy at model prices so decay can be detected. Netting hides gross exposure, so risk limits must be checked at both gross and net levels, and offsetting positions in correlated but different contracts need explicit handling."
    ),
    (
        "Signal decay and the half-life of alpha",
        "Every signal has a horizon over which its predictive power fades, its half-life. Order flow imbalance forecasts minutes ahead, short-term reversal a few days, earnings drift weeks, value years. Measure it by regressing forward returns at increasing horizons on the signal and plotting the information coefficient against horizon. Half-life sets the required trading speed: a signal that decays in two days cannot be traded weekly, and one that decays over a year should not be traded daily because turnover costs exceed the gain. Alpha also decays across time as strategies become known: McLean and Pontiff found published anomalies lose about a third of their return after publication and more than half after adoption, and premia like size and short-term reversal have shrunk to near zero net of costs. Monitoring the rolling information coefficient and live Sharpe against backtest catches decay before it becomes a drawdown."
    ),
    (
        "Capacity and crowding",
        "Capacity is the capital a strategy can run before market impact eats its return. It scales with the liquidity of the traded assets, the holding period and how spread out the trades are: a daily reversal strategy in small caps may hold tens of millions, while slow value or trend strategies in futures run tens of billions. Crowding is many managers holding the same positions, which creates unwind risk. The August 2007 quant crash saw long-short equity factor portfolios lose 20-30% in three days as one fund's liquidation forced others to sell the same stocks, then mostly recover within a week. Crowding indicators include short interest in the short leg, factor valuation spreads, correlation of returns across managers, and price impact per unit traded. Defenses are trading slowly, capping positions against average daily volume, and diversifying into less crowded signals and markets."
    ),
    (
        "Strategy lifecycle from discovery to decay",
        "A systematic strategy moves through discovery, validation, deployment and decay. Discovery starts from an economic hypothesis or an observed pattern; validation tests it with point-in-time data, realistic transaction costs, walk-forward evaluation and a Sharpe ratio deflated for the number of variants tried. Deployment starts small, paper trading then a fraction of target size, comparing live fills and returns with the backtest for months. Live performance is typically half the backtest, from overfitting, costs and market change. Monitoring compares rolling Sharpe, drawdown and turnover with expectations and flags a strategy when it falls outside its historical distribution, for example a drawdown deeper than twice the backtest maximum. Decay means downsizing or retirement when edge no longer exceeds costs. Durable strategies are slow structural premia like trend, value and carry; microstructure and anomaly trades die fastest."
    ),
    (
        "Lessons from Quantopian and QuantConnect",
        "Quantopian ran a crowdsourced quant fund from 2011 to 2020, letting hundreds of thousands of users backtest and submit long-short equity algorithms, with the best allocated capital, and closed after the fund underperformed. Its public lessons: in a study of 888 submitted algorithms, in-sample backtest Sharpe had almost no relationship with out-of-sample performance, more backtests run on a strategy meant worse live results, and most submissions were factor exposures or overfit noise. QuantConnect continues as an open platform with its Lean engine, and its Alpha Streams market found similar decay. The takeaways are to limit the variants tested, keep truly untouched holdout data, demand an economic reason, prefer simple rules, model costs and borrow properly, and treat a spectacular backtest as a warning. Both platforms remain useful for learning, data and free infrastructure."
    ),
    (
        "Renaissance, AQR and Two Sigma in the public record",
        "Renaissance Technologies' Medallion fund, run by Jim Simons with mathematicians and physicists, reportedly returned about 66% a year gross and 39% net from 1988 to 2018 with fees of 5 and 44%, on a capped few billion dollars, from thousands of short-horizon signals in liquid markets, heavy leverage and careful execution; its public funds, which trade slower, have performed like ordinary hedge funds. AQR, founded by Cliff Asness, runs value, momentum, carry, defensive and trend strategies at scale, publishes much of its research, and endured large value drawdowns in 2018-2020 before recovering. Two Sigma applies machine learning to broad data. The public lessons: fast, capacity-limited alpha is where extreme Sharpe ratios live, slow factor premia scale but suffer multi-year drawdowns, infrastructure and data are the real edge, and nothing public reveals their actual signals."
    ),
    (
        "Crypto basis and funding-rate arbitrage",
        "Perpetual swaps pay a funding rate, usually every eight hours, from longs to shorts when the perpetual trades above spot and the reverse when below. The cash-and-carry trade buys spot and shorts the perpetual or a dated future to collect funding or the basis with near-zero price exposure. In bull markets annualized funding has run 20-40% and dated futures have traded at 10-30% annualized premiums; in bear markets funding turns negative and the trade pays nothing. Realistic returns after fees, spread and idle periods have been around 5-15% a year with low volatility, and the trade is crowded since 2021. The risks are counterparty and operational rather than market: exchange failure like FTX, liquidation of the short leg during a squeeze when collateral sits elsewhere, stablecoin depegs and withdrawal halts. Hold a large margin buffer on the short side."
    ),
    (
        "Statistical arbitrage in crypto versus equities",
        "Equity stat-arb trades hundreds of stocks on residual mean reversion after removing sector and factor exposure, holds one to ten days, has borrow available and costs of a few basis points, and runs gross Sharpe of 1-2 for a well run book. Crypto stat-arb faces a few hundred liquid tokens dominated by one factor, bitcoin beta, so residuals are small and the neutral leg is hard to build. Compensations are that inefficiencies are larger and slower: cross-exchange price gaps, lead-lag between bitcoin and altcoins, and funding dislocations persist for hours rather than seconds, and shorting is easy through perpetuals. Costs are higher, 10-20 basis points per side, with 24/7 trading and no closing auction. Survivorship bias is extreme since most tokens go to zero, and exchange risk dominates. Returns are high in retail-driven bull markets and thin when volume dries up."
    ),
    (
        "Market-neutral construction and beta hedging",
        "A market-neutral portfolio removes exposure to the overall market so returns come from the spread between longs and shorts. Dollar neutrality, equal long and short notional, is not enough because shorts are often higher beta; beta neutrality sets the sum of long betas equal to the sum of short betas, with betas estimated over one to two years of daily data, shrunk toward one and updated as they drift. Sector, size and momentum exposures are neutralized by constraining factor loadings to zero or hedging with ETFs and futures. Residual returns have lower volatility but lower expected return, so market-neutral books run two to four times gross leverage. Failures come from hidden exposures that surface in stress, such as short high-beta stocks squeezing in a rally, borrow costs, and correlations that rise in crashes; the August 2007 unwind hit portfolios that looked neutral on every measured factor."
    ),
    (
        "Long-short factor portfolio construction",
        "A long-short factor portfolio ranks a universe on a signal, buys the top and shorts the bottom, and the design choices matter as much as the signal. The universe is usually the largest 1,000-3,000 stocks by capitalization and liquidity, ranked within industry to avoid sector bets. Portfolios hold the top and bottom decile or quintile, or weight every stock by rank for a smoother book. Weights are capitalization or equal, capped near 1-2% per name and at a fraction of average daily volume. Signals are z-scored and winsorized so outliers do not dominate. The book is set dollar, beta and often sector neutral, targeted to a fixed volatility, and rebalanced monthly for slow factors or weekly for faster ones. Academic factor portfolios ignore shorting costs, borrow and impact, so paper Sharpe ratios of 0.5-0.8 shrink to 0.3-0.5 in practice."
    ),
    (
        "Turnover control and trading rules for factor portfolios",
        "Turnover is the fraction of the portfolio traded per period, and for a factor strategy it decides whether the paper return survives transaction costs. Rules that cut turnover with little signal loss include no-trade bands, where a stock is sold only when it falls out of the top 30% rather than the top 10%; smoothing the signal with an exponential average; trading only part way toward the target each rebalance; and capping the names replaced per month. Optimization approaches penalize expected trading cost in the objective. Annual one-way turnover is typically 20-50% for value and quality, 100-300% for momentum, and above 1,000% for short-term reversal, which is why reversal rarely survives costs. Buffer rules can halve momentum turnover for a tenth of its return. Estimate costs per stock from spread and volume, since small caps concentrate both factor return and cost."
    ),
    (
        "Factor momentum and factor timing",
        "Factor timing tries to overweight factors about to perform well. The most robust version is factor momentum: factors that did well over the last one to twelve months tend to keep doing well the next month, and much stock-level momentum is explained by momentum in the factors stocks load on. Valuation timing, overweighting value when the spread between cheap and expensive stocks is unusually wide, works at multi-year horizons but signaled early through the value drawdown of 2017-2020 before paying off in 2021-2022. Volatility timing, cutting a factor when its own volatility spikes, mostly helps momentum. Across studies factor timing adds little Sharpe after costs and its signals are slow, so most practitioners limit it to modest tilts around a diversified multi-factor core, accepting that holding several factors is the main protection against a bad factor decade."
    ),
]

SOURCES = {
    "feeds": [
        "https://quantocracy.com/feed/",
        "https://alphaarchitect.com/feed/",
        "https://robotwealth.com/feed/",
        "https://quantpedia.com/feed/",
        "https://blog.thinknewfound.com/feed/",
        "https://qoppac.blogspot.com/feeds/posts/default",
    ],
    "subreddits": [
        "quant",
        "algotrading",
        "quantfinance",
        "systematictrading",
        "CryptoMarkets",
    ],
    "github_queries": [
        "topic:quantitative-finance topic:trading-strategies",
        "topic:statistical-arbitrage",
        "topic:trend-following futures",
        "topic:factor-investing",
        "funding rate arbitrage perpetual",
    ],
    "arxiv_topics": [
        "time series momentum",
        "cross-sectional momentum crashes",
        "statistical arbitrage cointegration",
        "volatility risk premium",
        "regime switching asset allocation",
    ],
    "urls": [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "https://quantpedia.com/screener",
        "https://wholesale.banking.societegenerale.com/en/prime-services-indices/",
        "https://www.turtletrader.com/rules/",
        "https://www.cboe.com/tradable-products/vix/term-structure/",
        "https://vixcentral.com/",
        "https://qoppac.blogspot.com/p/systematic-trading-start-here.html",
        "https://www.aqr.com/Insights/Research",
    ],
}
