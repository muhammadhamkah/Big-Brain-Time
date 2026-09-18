"""Knowledge pack: crypto trading in depth.

Instruments, derivatives mechanics, exchange and custody risk, on-chain data,
DeFi trading, catalysts, strategy evidence, infrastructure and psychology for
24/7 markets. Written to link back to the core curriculum (momentum, mean
reversion, leverage, volatility, position sizing, backtesting, market making).
"""

from __future__ import annotations

DOMAIN = "Crypto trading"

LESSONS: list[tuple[str, str]] = [
    (
        "Spot, perpetuals, dated futures and options in crypto",
        "Crypto trades in four main forms. Spot is direct ownership of the coin, settled instantly and withdrawable to a wallet. Perpetual swaps (perps) are futures with no expiry that track spot through a funding rate paid between longs and shorts, usually every eight hours; they carry most crypto derivatives volume, often three to five times spot. Dated futures expire quarterly and trade at a basis to spot that reflects financing and sentiment; CME lists cash-settled bitcoin and ether contracts for institutions. Options, dominated by Deribit, give leveraged exposure to direction and implied volatility. Inverse contracts are margined in the coin itself, so a long loses collateral value as price falls, which adds convexity a trader must size for. Leverage, liquidity and counterparty risk differ sharply across these instruments.",
    ),
    (
        "Funding rate mechanics",
        "A perpetual has no expiry, so exchanges anchor it to spot with a funding rate: when the perp trades above the index, longs pay shorts; when below, shorts pay longs. Binance computes it as a premium index plus a clamped interest component, with a baseline of 0.01% per eight hours, roughly 11% annualized, and caps that vary by contract. Some venues settle hourly. Funding is paid on notional position value, not margin, so a 20x long pays twenty times more relative to its collateral. Persistently high positive funding shows crowded longs and often precedes liquidation cascades; deeply negative funding after crashes marks crowded shorts and short squeezes. Funding is both a cost to model in any backtest and a sentiment and positioning signal in its own right.",
    ),
    (
        "Cash and carry funding arbitrage",
        "Cash and carry in crypto buys spot and shorts an equal notional of the perpetual, collecting funding while price exposure nets to zero. During bull frenzies annualized funding on bitcoin has exceeded 30-50% for weeks, while in quiet or bearish periods it falls to zero or negative and the trade bleeds. Practical returns after fees are usually single digits to low teens annualized. Risks are counterparty (the short leg and its collateral sit on an exchange, which is what hurt FTX carry traders), liquidation of the short if collateral is not the spot coin itself or cross-margined, funding flipping negative for long stretches, and exit slippage when everyone unwinds together. It is a yield strategy with tail risk, not free money, and it is what the spot bitcoin ETF basis trade hedge funds run at scale.",
    ),
    (
        "Basis trading and its risks",
        "Basis is the gap between a dated future and spot, quoted as an annualized rate. In early 2021 bitcoin quarterlies traded at 20-40% annualized contango; after the spot bitcoin ETF launch in 2024 the CME basis of 10-15% drew hedge funds into long ETF, short CME futures positions worth tens of billions. The trade earns basis convergence at expiry. Its risks are mark-to-market: basis can widen further before it converges, forcing margin calls on the short leg, as happened in the May 2021 and March 2020 dislocations, and backwardation appears suddenly in crashes. Financing cost, exchange risk and roll cost at each expiry eat the return. Basis level is also a signal: extreme contango marks leveraged euphoria, and backwardation marks panic.",
    ),
    (
        "Liquidation mechanics and open interest",
        "A leveraged position is liquidated when its margin falls below the maintenance requirement, typically 0.4-1% of notional on bitcoin at the lowest tiers and higher for altcoins. Exchanges use a mark price, an index across spot venues plus a smoothed basis, not the last traded price, to reduce manipulation. The liquidation engine takes over the position and closes it with market orders; any shortfall is covered by an insurance fund, and if that is exhausted profitable traders are auto-deleveraged. Liquidation fees of roughly 0.5% feed the fund. Open interest is the total outstanding contract notional; rising open interest with rising price shows new leveraged longs, and high open interest relative to market cap is fuel for cascades. Liquidation heatmaps estimate where forced orders cluster, which is where price often gets pulled.",
    ),
    (
        "Liquidation cascades",
        "A liquidation cascade is a feedback loop: a price drop liquidates over-leveraged longs, the forced market sells push price into the next band of liquidation levels, and so on until leverage is flushed. The 19 May 2021 crash liquidated around eight to ten billion dollars in a day; August 2024 and December 2024 each cleared over one billion; the 10 October 2025 crash was reported at roughly nineteen billion dollars, the largest on record. Cascades are most violent when open interest is high, funding is stretched and liquidity is thin, such as weekends and Asian hours. Prices typically overshoot and partly retrace within hours, which is a short-horizon mean reversion opportunity for traders with dry powder and a trap for anyone holding leverage. Watch open interest, funding and stablecoin order book depth as early warnings.",
    ),
    (
        "Leverage on crypto exchanges and the 100x trap",
        "Crypto exchanges offer 50x to 125x leverage on bitcoin and 20x to 50x on altcoins. At 100x the maintenance margin of about 0.5% means a move of well under 1% against the position triggers liquidation, and bitcoin moves more than 1% within an hour on most days. The position is therefore a coin flip that pays fees on 100 times the collateral, and the liquidation fee takes the remainder. Effective leverage should come from volatility targeting and Kelly logic, not the exchange maximum: for an asset with 60% annualized volatility even 3x is aggressive. Isolated margin caps the loss at the collateral assigned; cross margin risks the whole account. Many exchanges profit directly from liquidations, so the product is designed to encourage oversizing.",
    ),
    (
        "Exchange risk and custody",
        "Crypto held on an exchange is an unsecured claim on that company. Mt. Gox lost roughly 850,000 bitcoin in 2014 and creditors waited a decade; FTX collapsed in November 2022 with an eight billion dollar hole after customer funds were lent to its trading affiliate; QuadrigaCX, Cryptopia and others failed too. Proof of reserves audits show assets but rarely liabilities. Mitigations: keep only working capital on exchanges, spread across venues, withdraw profits to self-custody with a hardware wallet, prefer regulated venues with segregated accounts, and treat exchange withdrawal halts as the first sign of insolvency. Self-custody replaces counterparty risk with operational risk: lost seed phrases are unrecoverable. Any carry or funding strategy must include exchange failure in its tail risk estimate, because that is how most such strategies have actually lost money.",
    ),
    (
        "Stablecoins and depeg risk",
        "Stablecoins are the quote currency of crypto trading. Tether (USDT) and USD Coin (USDC) are fiat-backed and together exceed 200 billion dollars; DAI is over-collateralized by crypto; algorithmic designs back the peg with a sister token. TerraUSD (UST) held about eighteen billion dollars in May 2022, propped by 20% yields on Anchor, and collapsed to near zero in a week, taking LUNA and around forty billion dollars with it. USDC briefly fell to 0.87 dollars in March 2023 when 3.3 billion of its reserves were stuck in Silicon Valley Bank. Trading implications: collateral denominated in a stablecoin carries issuer and banking risk, depegs spread through leveraged positions and DeFi lending, and stablecoin premiums or discounts on specific venues signal capital flow and regional demand, for example USDT premiums in Asia during rallies.",
    ),
    (
        "On-chain analytics assessed",
        "Public blockchains expose flows that traditional markets hide. Exchange net flows treat inflows as sell pressure, but labels are incomplete and internal transfers and derivatives collateral movements add noise. Active addresses proxy adoption but are gamed. MVRV divides market cap by realized cap (each coin valued at its last move); values above 3.5 have marked bitcoin cycle tops and below 1 bottoms. SOPR, the ratio of sale price to purchase price of spent coins, above 1 means holders are realizing profit and resets to 1 act as support in uptrends. Honest assessment: these are valuation and regime descriptors fitted to three or four cycles, many are transforms of price itself, and few have shown out-of-sample forecasting power at horizons under months. They are useful context for sizing, not standalone signals, and the best data is behind Glassnode or CryptoQuant paywalls.",
    ),
    (
        "The bitcoin halving cycle assessed",
        "Every 210,000 blocks, about four years, bitcoin's block reward halves: 50, 25, 12.5, 6.25 and, since April 2024, 3.125 bitcoin, so new supply is around 450 coins a day and annual inflation below 1%. The narrative says this supply shock drives a four-year cycle with peaks 12-18 months after each halving, which fits 2013, 2017 and 2021. The evidence is four observations, the 2020 peak coincided with global stimulus, and a scheduled event should be priced in under any efficient market view. Miner selling of 450 coins a day is tiny next to tens of billions in daily volume. The stock-to-flow model built on halvings predicted over 100,000 dollars by the end of 2021 and failed. Treat the cycle as a weak prior on regime and sentiment, and momentum and macro liquidity as the actual drivers.",
    ),
    (
        "BTC dominance and altcoin seasons",
        "Bitcoin dominance is bitcoin's share of total crypto market cap. It peaked near 70% in early 2021, fell toward 40% during the altcoin season that followed, sank below 40% in 2022, and climbed back above 60% in 2025 as capital concentrated in bitcoin and ETFs. Altcoins behave as high-beta bitcoin: a typical large altcoin has a beta of 1.5 to 3 to bitcoin with far worse drawdowns of 80-95% in bear markets. An altcoin season is a late-cycle phase when retail flows chase smaller tokens and dominance falls; it is short, violent and ends with most tokens never recovering their highs. Over full cycles most altcoins underperform bitcoin, and survivorship bias hides the ones that went to zero. Rotation strategies use dominance trend plus momentum in cross-section rather than calling seasons by narrative.",
    ),
    (
        "Crypto correlation with Nasdaq and macro liquidity",
        "Before 2020 bitcoin's correlation with equities was near zero; since the pandemic stimulus it has traded as a high-beta risk asset, with 90-day correlations to the Nasdaq reaching 0.5-0.7 in 2022 and staying positive since. Bitcoin sells off on hawkish Federal Reserve surprises, rising real yields and a strong dollar, and rallies on liquidity expansion, so FOMC and CPI releases are volatility events for crypto too. The popular global M2 overlay charts are lag-fitted and unreliable. Correlation falls during crypto-specific shocks such as FTX, and rises in broad risk-off moves, which is when diversification is needed and fails. For portfolio construction crypto should be treated as a leveraged growth equity exposure, not as an uncorrelated hedge, with position sizing scaled to its much higher volatility.",
    ),
    (
        "Weekend and low-liquidity moves",
        "Crypto trades 24/7 but liquidity does not: weekend volumes on major exchanges run 30-50% below weekday levels, order books are thinner, and spreads widen, so the same order moves price further. Many large liquidation cascades and sharp squeezes have happened on Saturday and Sunday or in early Asian hours when market makers reduce quotes. CME bitcoin futures close from Friday to Sunday evening, creating gaps that are often but not always filled, a pattern traders track without strong statistical support. Since 2024 spot ETF flows only arrive on weekdays, concentrating institutional flow there. Practical rules: reduce size or leverage into weekends, avoid large market orders, and expect fake breakouts that reverse when weekday liquidity returns.",
    ),
    (
        "CEX versus DEX trading",
        "Centralized exchanges (Binance, Coinbase, OKX, Bybit) run central limit order books, custody funds, require identity checks, offer fiat rails and the deepest liquidity. Decentralized exchanges execute on-chain from the user's own wallet: automated market makers such as Uniswap and Raydium, or on-chain order books such as dYdX and Hyperliquid. DEX spot share has grown to roughly a fifth of volume by 2025 and is where new tokens trade first. DEX trading costs gas plus pool fees and price impact, is exposed to MEV and smart contract exploits, and has no customer support or reversals; CEX trading carries custody and insolvency risk instead. Arbitrage between the two keeps prices aligned within fees. Most systematic traders use CEX liquidity for execution and DEX data for early discovery.",
    ),
    (
        "AMMs, slippage and price impact on DEXs",
        "An automated market maker holds two tokens in a pool and prices trades by a formula, classically constant product x times y equals k. Price impact grows with trade size relative to pool depth: buying 1% of a pool's token moves price about 2%, and buying 10% moves it over 20%. Uniswap v3 concentrated liquidity lets providers quote within ranges, deepening liquidity near the current price but vanishing outside it. Fee tiers are typically 0.05%, 0.3% or 1% of trade value. The slippage tolerance setting caps the accepted price, and a wide tolerance is what sandwich bots exploit. Aggregators such as 1inch and Jupiter split orders across pools. Check pool depth, not reported volume, before sizing, and remember thin memecoin pools can be drained by a single exit.",
    ),
    (
        "MEV, sandwich attacks and front-running",
        "Maximal extractable value is profit block builders and searchers capture by ordering transactions. A sandwich attack sees a pending swap in the public mempool, buys just before it to push the price up, lets the victim fill at the worse price inside their slippage tolerance, then sells right after. Front-running and back-running arbitrage between pools are related. Cumulative sandwich extraction on Ethereum and Solana runs into hundreds of millions of dollars. Defenses: tight slippage tolerance, private transaction relays such as Flashbots Protect or MEV-protected RPCs, splitting large trades, and trading through aggregators that route privately. MEV is a hidden transaction cost that must be added to any DEX backtest, and it makes public-mempool trading of size structurally worse than a centralized order book.",
    ),
    (
        "Impermanent loss for liquidity providers",
        "A liquidity provider in a constant product pool ends up with less value than simply holding the two tokens whenever their relative price moves; the gap is impermanent loss, permanent once withdrawn. A 2x price change costs about 5.7% versus holding, 3x costs 13.4%, 4x costs 20% and 5x costs 25.5%. Fees must exceed this to make providing worthwhile, and concentrated liquidity positions amplify both fees and loss because they behave like short straddles that are fully converted once price exits the range. Studies of Uniswap v3 found roughly half of positions lost money versus holding. Being an LP is short volatility, so it pays in ranges and loses in trends, exactly like grid trading. Stablecoin pairs minimize the loss but carry depeg risk.",
    ),
    (
        "Yield farming and its risks",
        "Yield farming deposits tokens into DeFi protocols to earn rewards, usually paid in the protocol's own governance token. Quoted APYs of 100% or more mostly come from token emissions whose price falls as farmers sell, so realized returns are far lower and often negative. Layers of risk: smart contract exploits (over three billion dollars stolen in 2022 alone), oracle manipulation, rug pulls where developers drain liquidity, stablecoin depegs, impermanent loss in the underlying pool, and reflexive collapses like Anchor's 20% on UST. Real yield from trading fees or lending interest is sustainable; emissions are not. Assess the source of the yield, protocol age, audits, total value locked trend and how quickly you can exit, and size any farm as a venture bet, not as cash management.",
    ),
    (
        "Airdrop farming",
        "Protocols reward early users with free token airdrops: Uniswap gave 400 UNI per user in 2020, Arbitrum, Optimism, Starknet and EigenLayer followed, and Hyperliquid distributed about 31% of its supply in November 2024, worth over a billion dollars at launch. Farmers generate activity across many wallets to qualify. Costs are gas, capital locked at risk, time and sybil filtering that disqualifies obvious multi-wallet patterns; points programs now make allocation opaque and many expected airdrops never arrive or are tiny. Post-airdrop prices usually fall as recipients sell, so the trade is often to sell on receipt or short the perpetual against unvested tokens. Expected value per wallet has dropped as farming became crowded. Treat it as paid labor with lottery upside and count the smart contract and exchange risk on deposited funds.",
    ),
    (
        "Token unlocks and supply schedules as catalysts",
        "Most tokens launch with a small circulating float and large allocations to team, investors and foundation that vest over years, often with a one-year cliff followed by monthly unlocks. Low float, high fully diluted valuation tokens face steady selling as supply arrives. Research on hundreds of unlocks found the majority preceded negative returns, with most of the price pressure occurring in the weeks before the date as holders hedge, and unlocks above 1% of supply mattering most while team and investor unlocks hurt more than ecosystem ones. Trades include shorting perpetuals ahead of large unlocks, watching borrow rates and funding turn negative, and fading the post-unlock bounce. Schedules are public on unlock trackers, so the edge is in sizing and timing, and in avoiding tokens whose supply inflation exceeds any plausible demand.",
    ),
    (
        "Exchange listing effects",
        "A listing on a top exchange expands access and often produces a pump: studies of Coinbase listings found average gains of tens of percent in the days around announcement, and Binance listings frequently spike 20-50% on the announcement itself. The evidence also shows most of the move happens within minutes, before retail can buy, followed by reversal; many 2024 Binance listings traded below their listing price within weeks. Announcements leak, and a Coinbase product manager was convicted of insider trading on listing information in 2022. Tradable edges are thin: front-running rumors is insider-adjacent, and buying the announcement is buying the top. Delistings produce the mirror image with sustained selling. Listing is a liquidity event, not a fundamental one, so momentum from it decays fast.",
    ),
    (
        "Memecoins and pump-and-dump dynamics",
        "Memecoins are tokens with no product whose price is pure attention: Dogecoin, Shiba Inu, Pepe and thousands launched daily on bonding curve platforms like pump.fun, where the vast majority never reach a real market and over 98% go to zero. Pump-and-dump groups coordinate buying, insiders and sniping bots hold large supply from launch, and developers can pull liquidity in a rug pull. Celebrity coins such as LIBRA in February 2025 collapsed 90% within hours of launch. The few winners produce extreme returns, which fuels survivorship-biased hype. Trading them means accepting that you are the exit liquidity unless you are early or automated, sizing positions as lottery tickets, taking profit mechanically, checking holder concentration and locked liquidity, and expecting spreads and price impact to swallow small pools.",
    ),
    (
        "Wash trading and fake volume on exchanges",
        "Reported crypto volume is unreliable. A 2019 Bitwise study estimated that 95% of reported bitcoin volume on unregulated exchanges was fake, and academic work by Cong and coauthors found wash trading above 70% of volume on many unregulated venues, used to climb rankings and attract listing fees. NFT and DEX markets show the same, with wash trades inflating airdrop points and token metrics. Tells include volume far out of proportion to order book depth and spread, round-trip trades between linked accounts, and trade size distributions that violate expected statistical patterns. For trading, rely on depth and slippage from your own test orders, use exchange trust scores from data providers such as Kaiko and CoinGecko, and exclude suspect venues from backtests because fake volume makes a strategy appear far more scalable than it is.",
    ),
    (
        "Crypto spreads and liquidity provision on exchanges",
        "Bitcoin against USDT on Binance quotes at about one basis point wide with millions of dollars near the top of book; a mid-cap altcoin quotes at 10-50 basis points with a few hundred thousand dollars of depth, and small tokens are far worse. Professional firms such as Wintermute, GSR and Jump provide most of that liquidity, earning maker rebates and cross-exchange arbitrage while managing inventory and adverse selection from faster arbitrageurs. Token projects pay market makers with loans of tokens plus call options, which creates incentives to sell the token at agreed prices. Spreads widen sharply in volatility and on weekends. For a trader the practical rules are to use post-only limit orders where possible, size to visible depth on several levels, and treat quoted volume as less informative than measured slippage.",
    ),
    (
        "Grid bots and DCA bots assessed",
        "A grid bot places buy and sell limit orders at fixed intervals across a range and pockets each round trip; it earns in sideways markets and loses when price leaves the range, accumulating a losing inventory on a breakdown or missing the move on a breakout. It is a short volatility, short gamma position much like providing liquidity. DCA bots on platforms such as 3Commas buy fixed amounts on schedule, and their safety order variants add to losers at set intervals, which is martingale averaging down and can produce enormous drawdowns in an altcoin bear market. Plain scheduled dollar cost averaging into bitcoin is a reasonable accumulation method, not a trading edge. Backtests of grids look excellent because ranges are chosen in hindsight; test across regimes with fees and include the trend that ends the range.",
    ),
    (
        "Crypto momentum and trend evidence",
        "Momentum is the best-documented anomaly in crypto. Liu, Tsyvinski and Wu found that one- to four-week cross-sectional momentum in coins earned significant returns, and time-series momentum on bitcoin at daily to weekly horizons has been robust across studies. Simple trend following, such as holding bitcoin only above its 20-week or 200-day moving average, has historically captured most of the upside with drawdowns near 30-40% instead of 80%. Caveats: transaction costs and borrow constraints erode altcoin cross-sectional returns, the effect has weakened as the market institutionalized, and momentum crashes follow sharp rebounds from capitulation lows. Volatility scaling and sizing for gaps matter more than the entry rule, and combining trend with funding and open interest filters helps avoid buying late into crowded longs.",
    ),
    (
        "Short-horizon mean reversion in crypto",
        "At horizons of minutes to a day, crypto shows mean reversion, especially in altcoins: overreaction to news, liquidation cascades that overshoot and retrace within hours, and cross-exchange dislocations that arbitrage closes. Academic studies find intraday and one-day reversal in altcoins alongside weekly momentum, consistent with the coexistence seen in equities. Practical evidence is that reversals are strongest after extreme funding, large liquidations and wide deviations from a short moving average, and weakest for bitcoin at daily frequency. The edge is thin after fees of 5-10 basis points per side and slippage in thin books, so it survives only on liquid pairs with maker orders. The main risk is the same as everywhere: the stretched move is the start of a crash, so stops and small size are essential.",
    ),
    (
        "Volatility regimes and sizing in crypto",
        "Bitcoin realized volatility ran 60-100% annualized for most of its history and has drifted down to 40-60% since spot ETF flows arrived; large altcoins run 80-150% and memecoins far more. Volatility clusters, with quiet weeks followed by 10-20% daily moves, and regime shifts are abrupt around liquidation cascades and macro events. Volatility targeting is the single most useful sizing rule: position size equals target volatility divided by recent realized volatility, so exposure shrinks automatically when the market gets wild. A 10% annualized volatility target implies roughly 0.2x bitcoin exposure and less in altcoins. Stops should be ATR-based, and correlation across coins near 0.8-0.9 in stress means holding ten altcoins is barely more diversified than holding one, so the whole book should be sized as one position.",
    ),
    (
        "Crypto options and implied volatility on Deribit",
        "Deribit carries most of crypto options open interest, with bitcoin open interest in the tens of billions of dollars and quarterly expiries at 08:00 UTC on the last Friday of the month; Coinbase acquired it in 2025 and CME and other venues are growing. Implied volatility for bitcoin typically sits at 45-80%, tracked by the DVOL index. Unlike equities, skew often favors calls in bull markets and flips to puts in stress. Implied volatility has on average exceeded realized volatility, giving a positive volatility risk premium to option sellers, punctuated by crashes that wipe out months of premium, the classic short volatility profile. Options are coin-settled, so payoffs are convex in coin terms. Traders use options for defined-risk directional bets, hedging spot or carry books, and reading positioning from open interest by strike and the max pain level, which is a weak signal.",
    ),
    (
        "Spot bitcoin ETF flows and their market impact",
        "US spot bitcoin ETFs launched on 11 January 2024; BlackRock's IBIT gathered assets faster than any ETF in history and the group held well over a million bitcoin within about eighteen months, with ether ETFs following in July 2024. Daily creation and redemption flows are published each evening and correlate strongly with same-day returns, but have weak power to predict the next day; heavy inflow days cluster after rallies. A share of the inflows is the hedge fund basis trade, long ETF against short CME futures, which is delta neutral and adds no directional demand while it is on, and unwinds sharply when the basis compresses. ETFs concentrated institutional flow into weekday US hours, reduced realized volatility, and tightened correlation with equities. Flows are a useful confirmation of trend, not a leading indicator.",
    ),
    (
        "Regulation as a risk factor",
        "Regulatory action moves crypto prices as much as any fundamental. China's 2021 mining and trading ban, the SEC's 2023 lawsuits against Binance and Coinbase and its long case against Ripple, and the 2025 reversal that dropped most of those cases each produced double-digit moves and lasting shifts in which tokens exchanges would list. Tokens labeled securities get delisted in the US; in the EU the MiCA regime, in force for stablecoins since mid-2024 and for service providers since the end of 2024, led venues to delist non-compliant stablecoins such as USDT for European users. Stablecoin laws such as the 2025 GENIUS Act, tax reporting rules and exchange licensing determine where liquidity can exist. Treat regulatory headlines as event risk with gap potential, and diversify across jurisdictions and instruments so that a single ruling cannot freeze the whole book.",
    ),
    (
        "Crypto taxes at a high level",
        "In most jurisdictions crypto is property, so every disposal is a taxable event, including swapping one token for another, spending it, and receiving a different asset from a bridge or wrap; staking rewards, airdrops and mining are usually income at receipt. The United States distinguishes short-term and long-term gains at one year and has required broker reporting on new forms from 2025, while the wash sale rule has historically not applied to crypto, allowing loss harvesting. Germany exempts coins held over a year, the UK applies capital gains with pooling rules, and some countries tax nothing. Traders with thousands of transactions need software that tracks cost basis across exchanges and wallets from day one, because reconstructing years of DeFi activity later is close to impossible. High turnover strategies must be evaluated after tax, and this is not tax advice.",
    ),
    (
        "APIs, bots and rate limits",
        "Exchange APIs expose REST endpoints for orders and account state and websocket streams for trades, order books and user events. The ccxt library wraps over a hundred exchanges in one interface for spot and derivatives, at the cost of lowest-common-denominator features. Rate limits are strict: Binance counts request weight per minute and orders per second and per day, returns HTTP 429 when exceeded and bans the IP with 418 on repeat, so bots must throttle and back off. Reliable bots use client order IDs to make retries idempotent, reconcile positions from the exchange rather than local memory after any disconnect, synchronize clocks for signed requests, handle partial fills and maintenance windows, and test on a testnet or with tiny size first. Most bot losses come from engineering failures such as double orders and stale data, not from the signal.",
    ),
    (
        "Backtesting crypto without survivorship bias",
        "Survivorship bias is extreme in crypto: most tokens in the top hundred of 2018 have since fallen out or gone to zero, so a universe built from today's rankings makes any buy-the-dip or altcoin momentum strategy look brilliant. Use point-in-time listings, including delisted pairs, from sources such as exchange historical archives, Kaiko, CoinGecko or CryptoCompare, and record when each pair became tradable on the venue you would actually use. Include realistic fees of roughly 0.1% spot taker and 0.02-0.05% futures, slippage from book depth, funding payments for perpetual positions, and exchange outages during crashes. Data has gaps, timestamp inconsistencies and stale ticks from wash trading. There are only a few full cycles of history, so walk-forward tests and parameter stability matter more than in-sample Sharpe, and any result that depends on 2017 or 2021 altcoin mania is suspect.",
    ),
    (
        "Security hygiene for trading accounts",
        "Most crypto losses by individuals come from security failures, not markets. Create API keys with trading permission only and withdrawals disabled, restrict them to whitelisted IP addresses, rotate them, and never commit them to code repositories. Use hardware or app-based two-factor authentication, never SMS, because SIM swap attacks are routine, and enable withdrawal address whitelists with time locks. Hold long-term coins in a hardware wallet with the seed phrase stored offline in more than one place. In DeFi, revoke unused token approvals, verify contract addresses, beware address poisoning that plants look-alike addresses in transaction history, and assume every unsolicited message is phishing. Use exchange sub-accounts to isolate bots. A trading edge is irrelevant if the account is drained, so this belongs in the risk budget.",
    ),
    (
        "Psychology of 24/7 markets",
        "Crypto never closes, so the discipline problems of trading are amplified: there is no end of day to reset, price alerts interrupt sleep, and the biggest moves land at 3 a.m. or on Sunday. Fear of missing out is constant because something is always pumping, and social feeds are engineered to trigger it. Sustainable traders fix trading hours, use standing stop and take-profit orders or automation so positions are managed while they sleep, size so that an overnight 20% move is survivable, and take full days away from screens. Fatigue produces the classic errors of revenge trading and oversizing after wins. Journaling trades and reviewing them weekly exposes patterns such as losing mostly at night or on weekends. The market's availability is a cost to manage, not an opportunity to trade more.",
    ),
    (
        "Common crypto trader failure modes",
        "The same mistakes recur across cycles. Oversized leverage that turns normal volatility into liquidation. Holding altcoins through a bear market in which they fall 90% and never recover. Keeping the whole stack on one exchange that fails. Chasing listings, airdrops and memecoins after the move has happened. Paying funding of 50% annualized to hold a long that goes nowhere. Farming yields whose source was token emissions. Trusting reported volume and getting stuck in thin books. Running a bot that was overfit to 2021 or that double-ordered after a disconnect. Skipping stops because crypto always comes back, until it does not for that token. Ignoring taxes and security until both bite. Every one of these is a position sizing, custody or process failure rather than a forecasting failure, which is why rules beat conviction here.",
    ),
    (
        "Perpetual DEXs and on-chain derivatives",
        "Perpetual futures now trade on-chain: Hyperliquid runs an order book on its own chain and captured well over half of decentralized perpetual volume in 2025, dYdX uses an app chain, and GMX uses a pooled liquidity model where the pool takes the other side of traders. Advantages are self-custody, no identity checks and permissionless listings; funding rates, liquidations and leverage work as on centralized venues. Risks are specific: oracle manipulation, thin liquidity in long-tail markets, pooled vaults that absorb losses from manipulated positions such as the JELLY episode on Hyperliquid in March 2025, exploits like the 2025 GMX drain, validator concentration and front-running through ordering. Funding and basis on perp DEXs diverge from centralized exchanges when liquidity is stressed, which creates arbitrage for well-capitalized traders and gap risk for everyone else.",
    ),
]

SOURCES = {
    "feeds": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://insights.glassnode.com/rss/",
        "https://www.theblock.co/rss.xml",
        "https://cointelegraph.com/rss",
        "https://insights.deribit.com/feed/",
    ],
    "subreddits": [
        "CryptoCurrency",
        "Bitcoin",
        "BitcoinMarkets",
        "CryptoMarkets",
        "defi",
        "algotrading",
    ],
    "github_queries": [
        "topic:crypto-trading-bot",
        "topic:cryptocurrency-trading",
        "ccxt trading bot",
        "funding rate arbitrage",
        "topic:market-making crypto",
    ],
    "arxiv_topics": [
        "cryptocurrency momentum",
        "bitcoin volatility forecasting",
        "perpetual futures funding rate",
        "decentralized exchange automated market maker",
        "maximal extractable value sandwich attack",
    ],
    "urls": [
        "https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031",
        "https://www.binance.com/en/futures/funding-history/perpetual/real-time-funding-rate",
        "https://www.deribit.com/kb/deribit-introduction-policy",
        "https://insights.deribit.com/education/",
        "https://learn.bybit.com/",
        "https://academy.binance.com/en",
        "https://docs.ccxt.com/",
        "https://docs.uniswap.org/concepts/protocol/concentrated-liquidity",
    ],
}
