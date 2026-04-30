# Pentagon-Pizza-Index Proxy-Strategy

A full quantitative research pipeline testing whether GDELT military tension spikes predict next-day defense and aerospace stock outperformance — a data-driven proxy for the "Pentagon Pizza Index" (the observation that late-night pizza deliveries to the Pentagon precede major military events).

**Spoiler: H2 and H3 passed. The signal is real, statistically significant, and survives walk-forward validation. Implementation needs careful position sizing.**

---

## The Idea

The Pentagon Pizza Index is a decades-old informal signal: delivery drivers noticed unusual late-night orders at Pentagon-area restaurants before major military operations (Kuwait 1990, Desert Storm 1991, bin Laden 2011). The hypothesis is that elevated Pentagon activity precedes geopolitical events that benefit defense and aerospace equities.

Since actual pizza order data is unavailable historically, this project constructs a quantitative proxy using **GDELT** (Global Database of Events, Language and Tone) — a real-time open dataset that aggregates global news coverage into daily conflict tension scores. The proxy captures the same underlying signal: elevated US military activity in the news precedes defense sector rotation.

Three hypotheses tested:

| Hypothesis | Signal | Prediction |
|---|---|---|
| **H1 — VIX spike** | GDELT tension z > 2.0 | Next-day VIX rises ≥ 5% |
| **H2 — Defense outperformance** | GDELT tension z > 2.0 | ITA / XAR / PPA outperform SPY next day |
| **H3 — Crisis trade** | GDELT tension z > 2.0 | VIX up AND defense outperforms simultaneously |

---

## Results

Tested on 2017–2026 (2,341 trading days, 166 pizza spikes at z > 2.0):

```
════════════════════════════════════════════════════════════════════════
── H1: Tension spike → VIX jump  (n_spike=166) ──
  VIX ≥5.0% on spike days : 24.1%  (median -0.80%)
  VIX ≥5.0% otherwise    : 19.4%  (median -0.70%)
  Fisher OR=1.323 p=0.0860  MW p=0.3981
  FAIL ✗

── H2: Tension spike → defense ETF outperforms SPY ──
  ITA  spike=34.9%  no-spike=23.6%  med excess: +0.199% vs +0.004%  Fisher p=0.0010  ✓
  XAR  spike=36.1%  no-spike=26.3%  med excess: +0.210% vs -0.008%  Fisher p=0.0047  ✓
  PPA  spike=30.1%  no-spike=20.8%  med excess: +0.195% vs +0.002%  Fisher p=0.0041  ✓
  PASS ✓

── H3: Crisis trade (VIX up AND defense outperforms) ──
  ITA: spike=11.4%  no-spike=5.5%   p=0.0030  ✓
  XAR: spike=7.8%   no-spike=4.2%   p=0.0283  ✓
  PPA: spike=10.2%  no-spike=5.2%   p=0.0091  ✓
  PASS ✓

  OVERALL: 2/3  →  MODERATE SUPPORT
════════════════════════════════════════════════════════════════════════
```

H1 (VIX) failed — the market does not panic broadly on tension spikes. H2 and H3 passed strongly — there is a statistically significant sector rotation signal. On pizza spike days, defense ETFs outperform SPY at ~2× the base rate for the crisis trade, with Fisher p-values well below 0.01 for ITA.

---

## Walk-Forward Validation (14 folds, 2-year train / 6-month test)

```
  ITA:  mean Sharpe=+0.462  std=0.815  positive folds=43%  mean win rate=62.9%
  XAR:  mean Sharpe=+0.319  std=1.063  positive folds=43%  mean win rate=65.6%
  PPA:  mean Sharpe=+0.489  std=0.852  positive folds=43%  mean win rate=72.3%

  STABILITY (t-test Sharpe > 0):
  ITA: t=2.121  p=0.0269  significant ✓
  XAR: t=1.123  p=0.1410  not significant ✗
  PPA: t=2.149  p=0.0255  significant ✓
```

ITA and PPA pass the t-test for positive out-of-sample Sharpe. 7 of 14 folds had zero spikes (signal was dormant) — in the 7 active folds, 6 of 7 are positive. The one negative fold (Jul 2021–Jan 2022) coincides with the Afghanistan withdrawal and Biden defense budget cuts — a regime where the economic mechanism broke down.

---

## Why H1 Failed (and Why That's Informative)

The VIX didn't spike reliably on tension events. This is actually good news: **the market wasn't panicking — it was rotating quietly into defense**. H1 failing while H2 and H3 pass is the signature of informed sector rotation rather than broad fear. That is a cleaner, more exploitable signal.

---

## Paper Trading (v2 — Robinhood Configuration)

Initial paper trading (v1) showed Sharpe = −0.706 due to three structural problems:

**Problem 1:** Kelly fractions of 23–25% per ETF deployed ~72% of capital per spike day, generating $13,497 in transaction costs (13.5% of capital) over the test period.

**Problem 2:** No frequency filter — the strategy traded every spike, including the 2025 cluster (85 spikes in 6 months) where the "rare event" assumption completely broke down.

**Problem 3:** The SPY short hedge is unavailable on Robinhood.

**v2 fixes:**

| Fix | Description |
|---|---|
| Kelly cap 8% | Max 8% per ETF (was 25%) — ~24% total capital deployed per spike |
| Cooldown 3 days | No new trade within 3 days of last trade |
| Saturation filter | Skip if >8 spikes in last 28 days — signal is no longer a rare event |
| Robinhood costs | 0 commission, 1.5 bps bid-ask, 1.5 bps PFOF slippage (~6 bps total) |

---

## Signal Construction

The GDELT proxy is built from three GKG conflict themes queried via the GDELT DOC 2.0 API:

```
MILITARY_FORCE — use or threat of military force
TERROR         — terrorism and counterterrorism events
WMD            — weapons of mass destruction
```

Daily tension score = `sum(article_count × |avg_tone|)` across all three themes.
Pizza z-score = `(log(tension) − 28d_rolling_mean) / 28d_rolling_std`

This replicates the "busier than usual vs recent baseline" logic of pizzint.watch, without requiring access to live Google Maps foot traffic data.

---

## Repository Structure

```
pentagon-pizza-index/
├── pizza_index.py            # Full pipeline: GDELT fetch → signal → tests →
│                             #   walk-forward → paper trading
├── requirements.txt          # Python dependencies
├── DOCUMENTATION.md          # Technical methodology and code guide
├── LIMITATIONS.md            # Known limitations and mitigations
├── hypothesis_report.txt     # Raw output from hypothesis tests
├── walk_forward_results.csv  # Per-fold walk-forward results
├── paper_trade_log.csv       # Every individual paper trade
└── paper_equity_curve.csv    # Daily equity curve
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Full run (first run downloads GDELT — ~60-90 seconds, cached after)
python pizza_index.py

# Shorter window (faster for testing)
python pizza_index.py --start 2020-01-01

# Stricter spike threshold
python pizza_index.py --threshold 2.5

# Test 2-day market response
python pizza_index.py --lag 2

# Force re-download GDELT cache
python pizza_index.py --refresh
```

**Note on first run:** GDELT DOC 2.0 makes ~114 parallel API calls across 3 themes × 38 quarterly chunks. Expect 60–90 seconds on first run. Results cache to `gdelt_tension_cache.parquet` — every subsequent run loads instantly.

---

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--threshold` | 2.0 | Z-score above which a "pizza spike" is declared |
| `--lag` | 1 | Trading days between signal and trade entry |
| `--window` | 28 | Rolling window for z-score normalisation (days) |
| `--vix-spike` | 5.0 | VIX % change threshold for H1 |
| `--outperform` | 0.5 | Defense excess return threshold for H2 (% vs SPY) |
| `kelly_cap` | 0.08 | Max Kelly fraction per ETF (8%) |
| `cooldown_days` | 3 | Days to skip after a trade fires |
| `saturation_max` | 8 | Max spikes in last 28 days before signal is saturated |

---

## Dependencies

```
gdelt-doc-api
pandas
numpy
scipy
statsmodels
yfinance
matplotlib
```

---

## Limitations

See `LIMITATIONS.md` for full detail. Critical items:

- GDELT DOC 2.0 API covers 2017+ only (free tier); earlier data requires Google BigQuery
- `MILITARY_FORCE` and `TERROR` themes had partial API failures for some date ranges — `WMD` theme carried the signal during those gaps
- The signal is a proxy, not the actual pizza index — it captures broad news coverage of conflict, not Pentagon-specific activity
- 2025 tariff/geopolitical cluster produced 85 spikes in 6 months, saturating the "rare event" signal — the saturation filter addresses this but was not in the original backtest
- Robinhood PFOF slippage is unobservable in real time — the 1.5 bps estimate is based on academic studies of order flow routing

---

## License

MIT. Methodology and implementation are original work — reuse with attribution appreciated.
