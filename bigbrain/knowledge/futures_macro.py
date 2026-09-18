"""Knowledge pack: futures, commodities, rates and macro trading.

Dense lessons on contract mechanics, the major futures markets, rates and
central banks, commodity fundamentals, spreads, bond math, cross-asset macro
and the practical differences between ways of getting exposure.
"""

from __future__ import annotations

DOMAIN = "Futures, commodities, rates and macro trading"

LESSONS: list[tuple[str, str]] = [
    (
        "Futures contract specifications and tick values",
        "Every futures contract is defined by its specification: underlying, contract size (multiplier), minimum tick, tick value in dollars, trading hours, listed months, last trading day and settlement method (physical delivery or cash). Tick value is contract size times tick size: the E-mini S&P 500 (ES) is $50 per index point with a 0.25 tick worth $12.50; crude oil (CL) is 1,000 barrels with a $0.01 tick worth $10; gold (GC) is 100 ounces with a $0.10 tick worth $10; the 10-year Treasury note (ZN) is $100,000 face quoted in 32nds with a half-32nd tick worth $15.625. Traders use notional value (price times multiplier) to gauge leverage across markets and tick value to convert a stop distance into dollars. Specifications change, so check the exchange page rather than memory.",
    ),
    (
        "Margin, mark-to-market and variation margin",
        "Futures margin is a performance bond, not a down payment. The clearing house sets initial margin, often 3-12% of notional and derived from recent volatility by SPAN-style models, plus a slightly lower maintenance level. Positions are marked to market daily: gains are credited and losses debited in cash as variation margin, so a futures position never carries unrealized loss the way a stock does. Equity below maintenance triggers a margin call and possible liquidation. Brokers offer intraday margins as low as $500 per ES contract that invite extreme leverage; overnight margins revert to exchange levels. Exchanges raise margins in volatility spikes, which forces deleveraging and amplifies moves. Size by volatility and drawdown tolerance, never by the minimum margin allowed.",
    ),
    (
        "Roll yield and continuous contract construction",
        "Futures expire, so a long history must be stitched from successive contracts. At the roll the front and next contract trade at different prices; the gap is the roll yield and reflects contango or backwardation. Unadjusted concatenation keeps true prices but shows fake jumps at every roll and corrupts backtests. Back-adjusting (the Panama method) shifts earlier prices by the additive gap, preserving dollar profit per contract but distorting percentage returns and sometimes producing negative historical prices in markets like natural gas. Ratio adjusting multiplies earlier prices by the price ratio, preserving percentage returns but not tick values. Use back-adjusted series for stop distances and dollar risk, ratio-adjusted for return statistics, apply the roll rule (days before expiry or open interest crossover) consistently, and test sensitivity to it.",
    ),
    (
        "The major futures contracts and their personalities",
        "The most traded contracts behave differently. ES (E-mini S&P 500) is the deepest equity market, drifts upward with sharp downside gaps and reacts to earnings, Fed policy and the VIX. NQ (Nasdaq 100) is roughly 1.3-1.5 times as volatile as ES with more sensitivity to interest rates. CL (WTI crude) is driven by OPEC, inventories and geopolitics, gaps over weekends and has strong term structure effects. GC (gold) follows real yields, the dollar and central bank buying and trends for months. ZN (10-year note) moves in 32nds and concentrates its volatility around 8:30 ET data releases and FOMC. 6E (euro) is a slower, mean-reverting currency market driven by rate differentials and risk appetite. Know each market's typical daily range in ticks and dollars before sizing.",
    ),
    (
        "Equity index futures and the cash-futures basis",
        "Index futures price the cash index plus cost of carry: fair value is spot times one plus the financing rate minus the dividend yield, scaled by time to expiry. When rates exceed dividends the future trades above spot (a positive basis, a form of contango) and the basis decays to zero at expiry, converging on the settlement print. Index arbitrageurs buy the cheap leg and sell the rich one when the basis deviates from fair value by more than costs, which is why ES tracks the S&P 500 so tightly. The basis dislocates when funding is scarce or dealer balance sheets are constrained, as in March 2020. Traders read overnight premium or discount to fair value as a guide to the cash open, and use futures to hedge stock portfolios without selling shares.",
    ),
    (
        "Interest rate futures and yield curve trades",
        "Treasury futures (ZT 2-year, ZF 5-year, ZN 10-year, TN ultra 10-year, ZB 30-year, UB ultra bond) deliver against a basket of eligible bonds, and the cheapest-to-deliver bond drives pricing and duration. Prices move inversely to yields. A steepener buys the short end and sells the long end (long ZT, short ZN) expecting the 2s10s spread to widen, typically ahead of Fed cuts; a flattener does the reverse during hiking cycles. Legs are weighted by DV01 so the position is neutral to parallel shifts. The 2s10s curve was inverted from July 2022 to September 2024, reaching about minus 108 basis points, and early steepeners paid negative carry for two years. Curve trades earn or bleed carry and roll-down daily, so timing the pivot matters as much as direction.",
    ),
    (
        "Fed policy, FOMC and rate expectations",
        "The FOMC meets eight times a year, sets the target range for the federal funds rate, issues a statement and, quarterly, projections including the dot plot. Markets price the policy path in fed funds futures (ZQ, price equals 100 minus the expected monthly average effective rate, worth $4,167 per basis point) and SOFR futures. The implied probability of a 25 basis point move at a meeting is roughly the implied rate change divided by 25. What moves markets is the surprise relative to that pricing plus the guidance in the press conference. Rate expectations move bond futures, the dollar, gold and growth equities together. Futures pricing embeds term premium and hedging demand, so it is not a pure forecast; the market repeatedly priced cuts that did not arrive, as in early 2024.",
    ),
    (
        "SOFR futures and the end of Eurodollars",
        "Three-month SOFR futures (SR3) replaced Eurodollar futures when LIBOR ended; open Eurodollar positions were converted in April 2023. SR3 settles to 100 minus compounded SOFR over a three-month IMM period, with $2.5 million notional and $25 per basis point; one-month SOFR futures (SR1) cover calendar months. A strip of consecutive contracts traces the expected policy path, and calendar spreads between contracts are the cleanest way to trade the timing of hikes and cuts without duration risk; packs and bundles trade several at once. Because SOFR is a secured repo rate it lacks the bank credit component LIBOR had, so it does not widen in stress the way Eurodollars did; funding stress now shows up as repo spikes instead. SR3 is the natural hedge for floating-rate exposure.",
    ),
    (
        "Inflation data and market reaction",
        "CPI is released monthly by the BLS around mid-month at 8:30 ET; core CPI excludes food and energy, and shelter is roughly a third of the index. The Fed targets PCE inflation, released later in the month, which weights healthcare more and shelter less. Markets trade the surprise versus consensus: a hot core print lifts front-end yields, the dollar and volatility while hitting long-duration equities and gold; a soft print reverses that. In 2022 headline CPI peaked at 9.1% in June and CPI days produced the largest equity moves of the year. Inflation surprises are where the stock-bond correlation turns positive, so balanced portfolios lose on both legs. Systematic traders often cut size into the release, since the first minutes are dominated by algorithms and slippage.",
    ),
    (
        "Payrolls and the macro calendar",
        "Nonfarm payrolls, usually released the first Friday of the month at 8:30 ET, report jobs added, the unemployment rate and average hourly earnings; revisions to prior months often matter as much as the headline. Other high-impact releases are CPI, PCE, ISM surveys, retail sales, GDP, weekly jobless claims on Thursday, the EIA petroleum report on Wednesday, the EIA gas storage report on Thursday, Treasury auctions and FOMC decisions. Together they shape intraday volatility in rate, currency and index futures. Traders compare the print with consensus and the spread of forecasts to judge what is priced, then trade the reaction rather than the number, since strong data can be sold if it implies tighter policy. Holding through releases is discrete event risk that needs explicit sizing, and pre-release liquidity is thin.",
    ),
    (
        "Commodity fundamentals: inventories and seasonality",
        "Commodity prices clear physical supply and demand, and inventories are the buffer between them, so stocks relative to consumption are the most useful fundamental variable. When inventories are low the curve backwardates to pull supply forward and prices turn volatile with upside skew; when stocks are high the curve goes into contango and price is capped by storage economics. Seasonality comes from predictable demand and supply cycles: gasoline in summer driving season, heating oil and natural gas in winter, grains around planting and harvest. Energy seasonality is already embedded in the term structure, so buying the front month at the seasonal low captures little. Weekly EIA petroleum and gas reports and monthly USDA reports are the scheduled data, and the surprise versus survey expectations drives the immediate reaction.",
    ),
    (
        "Crude oil market structure and OPEC",
        "WTI crude (CL on NYMEX) is physically delivered at Cushing, Oklahoma, while Brent (ICE) is cash-settled and prices most seaborne crude; their spread reflects US export and pipeline capacity. OPEC and the wider OPEC+ group including Russia manage output quotas to defend a price range, so their meetings, compliance and spare capacity are central inputs. Shale producers respond to price with a lag of months and hedge forward, capping rallies through selling in deferred contracts. Weekly EIA data on Wednesday and API data on Tuesday move the front of the curve. Crude carries gap risk from geopolitics and violent term structure shifts; contango deep enough to pay for storage, as in 2008-09 and 2020, attracts floating storage arbitrage. Speculators crowd the first two contracts, so roll and expiry weeks carry extra flow risk.",
    ),
    (
        "Natural gas seasonality and storage",
        "Henry Hub natural gas futures (NG) are 10,000 MMBtu with a $0.001 tick worth $10, and are among the most volatile liquid contracts. The market runs on an annual storage cycle: injection season from April to October builds inventories, withdrawal season from November to March draws them down, and the EIA weekly storage report on Thursday at 10:30 ET is compared with the five-year average. Two-week weather forecasts move prices intraday. The March-April calendar spread, nicknamed the widowmaker, captures end-of-winter scarcity and sank Amaranth Advisors in 2006 with about $6 billion of losses. LNG exports now link Henry Hub to global gas. Gas has extreme skew, hard storage limits and a persistent contango that makes long-only ETF exposure a losing proposition over time.",
    ),
    (
        "Gold as a macro asset",
        "Gold pays no yield, so its opportunity cost is the real interest rate: historically gold moves inversely with the 10-year TIPS yield and inversely with the dollar, which sets its price for other buyers. It also responds to central bank reserve purchases, which surged after 2022 as countries diversified away from Treasuries, and to safe-haven demand. GC futures are 100 ounces; the gold-silver ratio and gold in other currencies give relative signals. The real-yield model broke in 2023-25 when gold hit records despite positive real yields, showing that official flows and fiscal worries can dominate for years. Gold trends persistently, which suits trend following, but it is sold hard when margin calls elsewhere force liquidation, as in March 2020, so it is a partial hedge, not a reliable one.",
    ),
    (
        "Base metals and China",
        "Copper, aluminium, zinc, nickel and lead trade on the London Metal Exchange with delivery into a global warehouse network; copper also trades as COMEX HG (25,000 pounds) and in Shanghai. China consumes roughly half of global base metals, so its property construction, grid spending, credit impulse and PMI data drive demand, while mine supply responds over years. Copper is called Dr Copper because it tends to lead global industrial activity. Exchange inventories, cancelled warrants and the cash-to-three-month spread show tightness. Metals are sensitive to the dollar and to smelting energy costs. The March 2022 LME nickel squeeze, when prices doubled in a day and trades were cancelled, shows that physical markets can break and that shorts in tight metals carry unbounded risk. Seasonality is weak compared with energy or grains.",
    ),
    (
        "Agricultural cycles and weather",
        "Grain and oilseed futures (corn ZC and soybeans ZS at 5,000 bushels, wheat ZW, plus meal, oil, cotton, sugar, coffee, cattle and hogs) follow the crop year. In the northern hemisphere planting runs April to May, weather risk peaks with pollination in July and August, and harvest from September to November pressures prices; Brazil and Argentina supply the other half of the year, so soybeans effectively have two harvests. The monthly USDA WASDE report and quarterly stocks and acreage reports are scheduled shocks, and drought adds a weather premium that fades if crops develop normally. Daily price limits can lock markets and trap positions. Ag markets are smaller than energy or financial futures, dominated by commercial hedgers, and their seasonality is largely priced into calendar spreads, so naive seasonal longs often just pay carry.",
    ),
    (
        "Commodity term structure and carry trading",
        "The slope of the futures curve is the most robust predictor of commodity futures returns: backwardated markets (spot above deferred) have earned positive roll yield and deeply contangoed ones have lost it, largely independent of spot direction. Carry strategies rank markets by the annualized spread between the first and a later contract, buy the most backwardated, sell the most contangoed, and rebalance monthly with volatility scaling. Backwardation signals scarcity and convenience yield, so it overlaps with momentum. Gorton, Hayashi and Rouwenhorst, and Koijen, Moskowitz, Pedersen and Vrugt, document the premium across asset classes. It fails when a demand shock crushes spot faster than the curve adjusts, as in crude in 2020, and when funds crowd the same rolls. Choosing which point on the curve to hold is where much of the edge and the cost live.",
    ),
    (
        "Commodity index rolls and the Goldman roll",
        "Passive commodity money tracks indices such as the S&P GSCI, which rolls over the fifth to ninth business day of the month, and the Bloomberg Commodity Index, which rolls over the sixth to tenth. Tens of billions of dollars sell the expiring contract and buy the next on a published schedule, temporarily depressing the front and lifting the second. Traders learned to front-run this Goldman roll by shorting the front against the next contract before the window and reversing after; the effect has faded as funds moved to optimized rolls that pick the best carry point. Retail products such as USO roll monthly and were forced in 2020 to spread holdings across months. Roll schedules concentrate flow in calendar spreads, so a spread trader must know when index money moves.",
    ),
    (
        "The Commitments of Traders report",
        "The CFTC publishes the Commitments of Traders report every Friday at 3:30 ET with positions as of the prior Tuesday. The legacy format splits open interest into commercials (hedgers), large speculators and small traders; the disaggregated format separates producers and merchants, swap dealers, managed money and other reportables; the Traders in Financial Futures report classifies asset managers, leveraged funds and dealers in rate, index and currency futures. Traders normalize net positions with a COT index over one to three years and read extreme speculator length as a contrarian warning and extreme commercial length as bullish. The data is three days old, positioning can stay extreme for months and hedgers are not forecasters, so COT is a crowding filter rather than a timing signal. Managed money positioning also proxies trend follower exposure and liquidation risk.",
    ),
    (
        "Spread trading in futures",
        "A spread is a long in one contract against a short in a related one, so profit comes from the relative price rather than the outright level. Calendar spreads trade different months of the same commodity and express storage and seasonality views; inter-commodity spreads pair related markets such as Brent against WTI, gold against silver, corn against wheat, or the NOB spread of 10-year notes against 30-year bonds. Exchanges grant margin offsets, so a calendar spread may need a tenth of outright margin, and many trade as a single listed instrument, removing legging risk. Spreads are less exposed to macro shocks but move on their own fundamentals and can trend for months. They carry basis risk when the relationship breaks, and small tick values tempt oversizing that a blowout, as in nickel in 2022, turns into large losses.",
    ),
    (
        "Crack and crush spreads",
        "Refiners and processors hedge their margin rather than a single price. The crack spread is refined product value minus the crude used to make it; the standard 3:2:1 crack buys three crude contracts against two gasoline (RB) and one ULSD heating oil (HO), converting cents per gallon to dollars per barrel by multiplying by 42. A widening crack signals refining tightness, often into driving season or after hurricanes. The soybean crush is meal (ZM) plus oil (ZL) minus soybeans: a bushel yields roughly 44 pounds of meal and 11 pounds of oil, so the board crush is meal price times 0.022 plus oil price times 0.11 minus beans, traded in a 10:11:9 contract ratio. Traders buy the crush when margins are depressed and sell when rich. Both move violently when one product has a supply shock and the other does not.",
    ),
    (
        "Bond math for traders: duration, convexity and DV01",
        "Duration measures price sensitivity to yield: modified duration of 7 means a 1% rise in yield loses about 7% of price. DV01, the dollar value of one basis point, is duration times price times 0.0001 and is the unit rate traders think in; a 10-year Treasury futures contract has a DV01 of roughly $60-80 depending on the cheapest-to-deliver bond. Hedge ratios and curve trades are set by matching DV01, so a 2s10s steepener holds roughly two to three 2-year contracts per 10-year contract. Convexity is curvature: ordinary bonds gain more when yields fall than they lose when yields rise, so long convexity is worth paying for in volatile markets, while mortgage bonds have negative convexity that forces hedging flows. Duration is a linear approximation that breaks for large moves, and futures duration jumps when the cheapest-to-deliver switches.",
    ),
    (
        "Credit spreads and high yield as risk signals",
        "The credit spread is the extra yield corporate bonds pay over Treasuries, measured as option-adjusted spread. Investment grade normally sits near 100-150 basis points and high yield near 300-500; high yield below 300 signals complacency, as in 2007 and 2021, while readings above 800-1,000 mark stress, reaching roughly 2,000 in late 2008 and 1,100 in March 2020. Spreads widen as default risk and risk aversion rise, and because high yield issuers are leveraged and cyclical, widening often leads or confirms equity drawdowns. Traders watch HYG, JNK, the CDX indices and the ratio of high yield to investment grade spreads as risk-on and risk-off gauges. The signal fails when spreads move for reasons other than credit, as in 2022 when rates drove losses while defaults stayed low, and when central bank backstops compress spreads regardless of fundamentals.",
    ),
    (
        "The dollar index and global liquidity",
        "The US Dollar Index (DXY, traded as DX futures at $1,000 times the index) weights the euro 57.6%, yen 13.6%, pound 11.9%, Canadian dollar 9.1%, krona 4.2% and franc 3.6%, so it is mostly a euro cross; trade-weighted and emerging market dollar indices are broader. Because global trade and much emerging market debt are dollar-denominated, a rising dollar tightens financial conditions worldwide: it raises the cost of dollar liabilities, depresses commodities priced in dollars and pulls capital from emerging markets. The dollar smile describes strength both when US growth outperforms and when global risk aversion spikes, with weakness in between. Rate differentials, Fed policy and risk appetite drive it, and intervention is rare but decisive. Dollar trends persist for years, but turning points are hard to time and calls for its decline have mostly been early.",
    ),
    (
        "Risk-on, risk-off and cross-asset correlations",
        "In risk-on periods equities, commodities, emerging markets, high-yield credit and carry currencies rise together while Treasuries, the yen, the franc and the dollar lag; risk-off reverses all of it at once. This one-factor behaviour dominated 2008-2012 and returns in every panic, when cross-asset correlations converge toward one and diversification evaporates. The stock-bond correlation is the key regime variable: negative from about 2000 to 2020, so bonds hedged equities, but positive in the inflationary 2022 environment when both fell. The VIX, credit spreads, the dollar and the copper-to-gold ratio serve as risk appetite gauges. Traders use the framework to avoid stacking exposures that look diversified but are one bet on calm, such as long equities, short bonds and long carry. Correlations estimated in calm periods understate crisis correlation, so stress tests should assume the worst episode.",
    ),
    (
        "Central bank divergence trades",
        "When central banks move in opposite directions the rate differential between two currencies shifts persistently, producing some of the longest trends in macro. In 2014-15 the Fed ended quantitative easing while the ECB and Bank of Japan expanded theirs, and EUR/USD fell from 1.39 to about 1.05. In 2022 the Fed hiked aggressively while the Bank of Japan held yield curve control, driving USD/JPY from 115 to 150 and prompting intervention. The trade is expressed in currency futures (6E, 6J), in rate futures spreads between countries, or in relative equity indices. It works because policy paths are gradual and telegraphed, which suits trend following. It fails abruptly when the lagging bank pivots, as when the Bank of Japan's 2024 hike helped trigger the August 2024 carry unwind, or when intervention caps the move. Positioning data shows when it is crowded.",
    ),
    (
        "Macro regimes and asset performance",
        "A useful map crosses growth (rising or falling) with inflation (rising or falling). Rising growth with falling inflation, the 2010s, favours equities and credit. Rising growth with rising inflation favours commodities and cyclical equities while bonds suffer. Falling growth with rising inflation, stagflation as in the 1970s and 2022, is the hardest regime: commodities, trend following and gold hold up while stocks and bonds fall together. Falling growth with falling inflation, deflationary recessions like 2008, rewards long Treasuries and the dollar. Bridgewater's All Weather portfolio balances risk across these four quadrants. Regimes are read in real time through inflation surprises, PMI direction, curve moves and central bank stance, but they are clear only in hindsight and the transitions are where most macro losses occur. The practical use is to see which positions are secretly the same regime bet.",
    ),
    (
        "The April 2020 negative oil price event",
        "On 20 April 2020 the May WTI contract settled at minus $37.63 a barrel, the day before expiry. Pandemic demand had collapsed, storage at Cushing was nearly full, and holders of expiring longs faced physical delivery with nowhere to put the oil; CME had warned two weeks earlier that its systems would allow negative prices. USO had already rolled out of May, but Bank of China's Crude Oil Treasure product settled clients at the negative price, and Interactive Brokers absorbed over $100 million in client losses because its software could not display negative values. The lessons: physically delivered contracts can decouple violently from spot near expiry, the front month should be exited well before delivery, risk systems must handle prices with no lower bound, and contango deep enough to pay for storage is a warning, not a bargain.",
    ),
    (
        "The 2022 rate shock and the 60/40 portfolio",
        "In 2022 the Fed raised the funds rate from near zero to 4.25-4.5% by December, including four consecutive 75 basis point hikes, and continued to 5.25-5.5% in 2023. Ten-year yields rose from 1.5% to over 4%, long Treasury ETFs lost around 30%, the S&P 500 fell about 19%, and a 60/40 stock-bond portfolio lost roughly 16-17%, its worst year since the 1930s. The diversification that had worked for two decades failed because the shock was inflation, which hurts both assets, and the stock-bond correlation turned positive. Winners were commodities, the dollar and trend following on futures, with CTA indices up around 20%, because moves were persistent across many markets. The episode revived interest in managed futures and inflation hedges and showed that the bond hedge works in growth shocks, not inflation shocks.",
    ),
    (
        "Discretionary global macro",
        "Discretionary global macro, in the style publicly associated with Stanley Druckenmiller and George Soros, forms views on growth, inflation, liquidity and policy and expresses them in the most liquid instruments: index, bond and currency futures, options and commodities. Public accounts of Druckenmiller's approach stress that liquidity and central bank behaviour, not earnings, move markets; that one should concentrate heavily when conviction is high and stay small otherwise; that being wrong is fine but staying wrong is not; and that returns come from a few large trades, like the 1992 sterling short, rather than many small ones. He reportedly compounded around 30% a year for three decades without a losing year, yet lost heavily in the 2000 tech bubble by abandoning his own process. The approach relies on judgment that cannot be backtested and fails when concentration meets a reversal.",
    ),
    (
        "Systematic macro and trend on futures",
        "Systematic macro applies rules across 50-150 futures markets spanning equity indices, bonds, short rates, currencies, energy, metals and agriculture. The core signal is time-series momentum: go long markets that rose over the past one to twelve months and short those that fell, size each position to a fixed volatility contribution, and rebalance frequently. Moskowitz, Ooi and Pedersen documented the effect across 58 markets back to 1985. Managers layer carry, value, seasonality and fundamental macro signals and cap risk at the portfolio level by correlation and sector. The strategy earns modest Sharpe ratios of roughly 0.5-0.8 with long flat stretches, made bearable by low correlation to stocks and bonds and strong performance in extended crises. It loses in sharp reversals, in post-crisis whipsaws and in range-bound stretches like 2011-2019 that thinned the industry.",
    ),
    (
        "Managed futures and crisis alpha",
        "Crisis alpha is the tendency of trend following on futures to profit during prolonged equity bear markets, because falling stocks, rising bonds, moving currencies and trending commodities all generate signals. The Barclay CTA index gained about 14% in 2008 while stocks fell 37%, and trend indices rose 20-27% in 2022 while stocks and bonds both fell. Crises are persistent multi-month moves across many markets, exactly what the strategy captures, and positions are cut quickly when trends reverse. The limits are real: in sudden shocks like March 2020 many CTAs were caught long equities and recovered only later, V-shaped recoveries whipsaw them, and between crises they grind sideways for years, so investors abandon them just before they are needed. A 10-20% allocation to trend is the usual compromise for portfolio protection.",
    ),
    (
        "Index futures seasonality around holidays",
        "Equity index futures show mild but persistent holiday patterns. The sessions before Thanksgiving, Christmas and Independence Day have historically drifted higher on low volume and narrow ranges, the Santa Claus rally covers the last five trading days of December and the first two of January, and the turn of the month around payroll and pension inflows is positive. Holiday sessions are often half days with thin books, so small orders move price and stops get hunted. Mechanically, low participation means less selling pressure and positive drift. These effects are small relative to daily noise, have faded as they became known, and are overwhelmed by any macro event; the holiday drift trade lost badly in December 2018. They work best as filters, such as avoiding shorts into a holiday or cutting size in thin markets, not as standalone signals.",
    ),
    (
        "Overnight versus regular session behaviour",
        "ES trades almost 23 hours a day, but the regular session from 9:30 to 16:00 ET carries most of the volume and variance, while the overnight Globex session from 18:00 to 9:30 ET carries less volume and a disproportionate share of the return. Studies of the overnight versus intraday split find that the equity risk premium has been earned largely overnight in recent decades, with intraday returns near zero, attributed to retail order imbalance at the open, news released outside hours and market maker inventory management. Overnight liquidity is thinner, so foreign data, earnings and geopolitical headlines gap prices, and the overnight high and low become reference levels for the day. Intraday traders use the overnight range, the gap from settlement and the first hour's range as setups. The overnight premium is not tradable without frictions and vanishes in prolonged crises.",
    ),
    (
        "Contract expiration, quad witching and rebalancing flows",
        "Equity index futures expire on the third Friday of March, June, September and December, settling to a Special Opening Quotation built from each component's opening price. Quad witching, when index futures, index options, stock options and single-stock futures expire together, brings very heavy volume at the open and close and coincides with the quarterly S&P rebalance, so closing auctions show large imbalances. Most volume rolls to the next contract in the week before expiry, typically from the Thursday eight days ahead, and the calendar spread reflects funding and dividends. Month-end and quarter-end bring pension and mutual fund rebalancing between stocks and bonds that flows through futures. These events are predictable and mostly priced, but they distort intraday patterns, widen spreads and can pin prices, so traders discount expiry-day price action.",
    ),
    (
        "Micro futures for small accounts",
        "Micro contracts are one-tenth the size of the E-minis: MES is $5 per S&P point with a $1.25 tick, MNQ is $2 per Nasdaq point, MCL is 100 barrels, MGC is 10 ounces, and M2K, MYM, micro FX and yield-based micro Treasury contracts complete the set. Launched from 2019, they let traders with a few thousand dollars size positions in true risk units, test strategies live with real fills, and scale in gradually rather than in lumps of $200,000-plus notional. The tradeoffs are that commissions are a larger share of profit, so a round trip of $1 or more against a $1.25 tick punishes scalping, and liquidity is lower than in the parent contracts, though deep in MES and MNQ. Micros do not make an undercapitalised account safe; leverage per dollar of notional is identical to the E-mini.",
    ),
    (
        "Futures versus CFDs versus ETFs for exposure",
        "Futures are exchange-traded, centrally cleared, transparent in price and open interest, capital efficient through margin, and trade nearly around the clock, but they expire, require rolls, come in fixed sizes and demand margin management. Contracts for difference are over-the-counter bets with a broker, sized freely in small units, but they charge daily financing, embed wider spreads, carry broker counterparty risk and are banned for US retail. ETFs fit ordinary brokerage accounts but carry fees and tracking error; commodity ETFs such as USO lose to contango through their rolls, and leveraged ETFs decay from daily rebalancing in choppy markets. For a directional multi-week view in a liquid market, futures are usually cheapest; for a long-term equity allocation an index ETF wins; CFDs mostly suit sizes too small for anything else.",
    ),
    (
        "Tax treatment of futures: the 60/40 rule",
        "In the United States regulated futures and broad-based index options are Section 1256 contracts. Gains and losses are taxed 60% as long-term and 40% as short-term capital gains regardless of holding period, giving a maximum blended federal rate of about 26.8% versus 37% for short-term stock trades, and open positions are marked to market at year end as if closed on 31 December. The wash sale rule does not apply, and net 1256 losses can be carried back three years against prior 1256 gains, reported on Form 6781. Single-stock futures, CFDs and most crypto derivatives do not qualify, and rules differ entirely outside the US, where futures may fall under ordinary capital gains, income tax, or be exempt for individuals. This is general information, not tax advice; rules change and depend on entity and residence.",
    ),
    (
        "Treasury auctions and supply as a rate driver",
        "The US Treasury sells bills weekly and notes and bonds on a published calendar, with 2-, 5- and 7-year auctions late in the month and 3-, 10- and 30-year auctions early, usually at 13:00 ET. Results are judged by the tail (auction yield versus the when-issued yield at the deadline), the bid-to-cover ratio and the indirect bidder share, a proxy for foreign demand. A weak long-bond auction can move ZB and ZN by a point in minutes, as in October 2023 when a poor 30-year auction pushed 10-year yields through 5%. The quarterly refunding statement sets issuance sizes and has become a macro event because deficits are large; shifting supply toward bills eased term premium in late 2023. Dealers hedge auctions with futures, so pre-auction concessions and post-auction squeezes create tradable intraday patterns.",
    ),
]

SOURCES = {
    "feeds": [
        "https://www.eia.gov/rss/todayinenergy.xml",
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://libertystreeteconomics.newyorkfed.org/feed/",
        "https://fredblog.stlouisfed.org/feed/",
        "https://www.calculatedriskblog.com/feeds/posts/default",
        "https://www.cftc.gov/RSS/RSSGP/rssgp.xml",
        "https://alphaarchitect.com/feed/",
        "https://www.rcmalternatives.com/feed/",
    ],
    "subreddits": [
        "FuturesTrading",
        "commodities",
        "bonds",
        "economics",
        "quant",
    ],
    "github_queries": [
        "topic:futures-trading",
        "topic:commodities",
        "continuous futures contract back-adjusted roll python",
        "commitments of traders CFTC parser",
        "time series momentum futures backtest",
    ],
    "arxiv_topics": [
        "time series momentum futures",
        "commodity futures term structure carry",
        "yield curve trading strategies",
        "macroeconomic announcements asset prices",
        "managed futures crisis alpha",
    ],
    "urls": [
        "https://www.cmegroup.com/education.html",
        "https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html",
        "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "https://www.bls.gov/schedule/news_release/",
        "https://www.eia.gov/petroleum/weekly/",
        "https://www.eia.gov/naturalgas/storage/dashboard/",
        "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
    ],
}
