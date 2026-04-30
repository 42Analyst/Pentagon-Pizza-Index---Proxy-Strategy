# =============================================================================
# Pentagon Pizza Index — Quantitative Proxy Pipeline  v2
# =============================================================================
#
# WHAT CHANGED FROM v1
# ────────────────────
# v1 used GDELT 1.0 raw daily CSV files via direct HTTP download.
# Those files return 403 Forbidden from automated clients (IP blocking).
# v2 uses the GDELT DOC 2.0 API via the gdelt-doc-api Python client, which:
#   - Is specifically designed for programmatic access (no IP blocking)
#   - Returns pre-aggregated timeline data (no parsing 57-column CSVs)
#   - Is ~50x faster because we fetch quarterly chunks, not daily files
#   - Supports GKG themes — cleaner signal than raw CAMEO codes
#
# SPEED FIX
# ─────────
# v1: sequential HTTP GET per day → ~7,000 requests → 20+ min
# v2: parallel quarterly batches via concurrent.futures.ThreadPoolExecutor
#     → ~12-40 API calls total → typically under 2 minutes
#     Results are cached to disk (gdelt_tension_cache.parquet) so
#     subsequent runs are instant.
#
# SIGNAL CONSTRUCTION
# ───────────────────
# Three GDELT GKG themes queried:
#   MILITARY_FORCE  — use or threat of military force
#   TERROR          — terrorism and counterterrorism
#   WMD             — weapons of mass destruction (escalation signal)
#
# For each quarterly chunk we fetch:
#   timelinevolraw  — raw article count per day
#   timelinetone    — average tone (negative = alarming coverage)
#
# Tension score = article_count × |avg_tone|
# Then: log1p → 28-day rolling z-score → spike flag at z > threshold
#
# HYPOTHESIS
# ──────────
# H1 — GDELT tension spike → next-day VIX spike (fear)
# H2 — GDELT tension spike → defense ETF (ITA, XAR, PPA) outperforms SPY
# H3 — Both simultaneously (the "crisis trade")
#
# NOTE ON DATE RANGE
# ──────────────────
# GDELT DOC 2.0 API covers ~2017-01-01 onward (free, no API key).
# Default start date is 2017-01-01.
#
# INSTALL
# ───────
#   pip install gdelt-doc-api pandas numpy scipy statsmodels yfinance matplotlib
#
# USAGE
# ─────
#   python pizza_index.py                        # 2017-today
#   python pizza_index.py --start 2020-01-01     # shorter window
#   python pizza_index.py --threshold 1.5        # lower spike bar
#   python pizza_index.py --lag 2                # 2-day market response
#   python pizza_index.py --workers 8            # more parallel threads
#   python pizza_index.py --refresh              # ignore cache
# =============================================================================

from __future__ import annotations

import argparse
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yfinance as yf
from gdeltdoc import GdeltDoc, Filters
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("PizzaIndex")


# =============================================================================
# CONFIG
# =============================================================================

TENSION_THEMES = [
    "MILITARY_FORCE",
    "TERROR",
    "WMD",
]

DEFENSE_ETFS = {
    "ITA": "iShares US Aerospace & Defense (large-cap, cap-weighted)",
    "XAR": "SPDR S&P Aerospace & Defense (equal-weight)",
    "PPA": "Invesco Aerospace & Defense (broadest, 52 stocks)",
}

DEFENSE_NAMES  = ["LMT", "RTX", "NOC", "GD", "LHX", "BA"]
SPY_TICKER     = "SPY"
VIX_TICKER     = "^VIX"

DEFAULT_SPIKE_Z    = 2.0
DEFAULT_VIX_SPIKE  = 5.0
DEFAULT_OUTPERFORM = 0.5
DEFAULT_LAG        = 1
DEFAULT_WINDOW     = 28
DEFAULT_WORKERS    = 6

CACHE_FILE = Path("gdelt_tension_cache.parquet")


# =============================================================================
# GDELT DOC 2.0 DATA LAYER
# =============================================================================

def _date_chunks(start: date, end: date,
                 chunk_days: int = 90) -> list[tuple[date, date]]:
    chunks, cur = [], start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _fetch_chunk(theme: str, start: date, end: date,
                 retries: int = 3) -> pd.DataFrame | None:
    """
    Fetches one (theme, date-range) from GDELT DOC 2.0.
    Returns DataFrame with columns [datetime, volume, tone, theme] or None.
    """
    gd = GdeltDoc()
    f  = Filters(
        theme      = theme,
        start_date = start.isoformat(),
        end_date   = end.isoformat(),
    )
    for attempt in range(retries):
        try:
            vol  = gd.timeline_search("timelinevolraw", f)
            tone = gd.timeline_search("timelinetone",   f)

            if vol is None or vol.empty:
                return None

            vol_col  = [c for c in vol.columns  if c != "datetime"][0]
            tone_col = ([c for c in tone.columns if c != "datetime"][0]
                        if tone is not None and not tone.empty else None)

            result = vol.rename(columns={vol_col: "volume"})
            if tone_col:
                result = result.merge(
                    tone.rename(columns={tone_col: "tone"}),
                    on="datetime", how="left",
                )
            else:
                result["tone"] = -5.0

            result["theme"]    = theme
            result["datetime"] = pd.to_datetime(result["datetime"])
            return result

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.warning("Failed %s %s→%s: %s", theme, start, end, e)
                return None


def fetch_gdelt_signal(start: date, end: date,
                       themes: list[str] = TENSION_THEMES,
                       workers: int = DEFAULT_WORKERS,
                       force_refresh: bool = False) -> pd.Series:
    """
    Downloads GDELT tension signal using parallel quarterly batches.
    Caches result to CACHE_FILE so re-runs are instant.

    Tension score = sum over themes of (volume × |tone|)
    """
    # Check cache
    if not force_refresh and CACHE_FILE.exists():
        cached = pd.read_parquet(CACHE_FILE)
        cached_end = cached.index.max().date()
        if cached_end >= end - timedelta(days=3):
            log.info("Cache up to date (%s)", cached_end)
            return cached["tension"].loc[str(start):str(end)]
        log.info("Extending cache from %s to %s …", cached_end, end)
        start = cached_end + timedelta(days=1)

    chunks = _date_chunks(start, end)
    tasks  = [(t, s, e) for t in themes for (s, e) in chunks]
    log.info("Fetching GDELT: %d themes × %d chunks = %d API calls (parallel)",
             len(themes), len(chunks), len(tasks))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_chunk, t, s, e): (t, s, e)
                   for t, s, e in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % max(1, len(futures) // 5) == 0 or done == len(futures):
                log.info("  %d / %d API calls done", done, len(futures))
            df = future.result()
            if df is not None:
                results.append(df)

    if not results:
        raise RuntimeError(
            "All GDELT API calls failed.\n"
            "Check: pip install gdelt-doc-api\n"
            "and that your internet connection is active."
        )

    combined = pd.concat(results, ignore_index=True)
    combined["tone_abs"] = combined["tone"].abs()
    combined["score"]    = combined["volume"] * combined["tone_abs"]

    daily = (
        combined
        .groupby("datetime")["score"]
        .sum()
        .rename("tension_raw")
        .sort_index()
    )

    # Merge with existing cache if extending
    if not force_refresh and CACHE_FILE.exists():
        existing = pd.read_parquet(CACHE_FILE)["tension"].rename("tension_raw")
        daily = pd.concat([existing, daily])
        daily = daily[~daily.index.duplicated(keep="last")].sort_index()

    daily.to_frame("tension").to_parquet(CACHE_FILE)
    log.info("Tension series: %d days → %s", len(daily), CACHE_FILE)
    return daily


# =============================================================================
# SIGNAL CONSTRUCTION
# =============================================================================

def build_pizza_index(tension: pd.Series,
                      window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    # Strip UTC timezone if present (GDELT DOC 2.0 returns UTC-aware index)
    if hasattr(tension.index, "tz") and tension.index.tz is not None:
        tension = tension.copy()
        tension.index = tension.index.tz_convert(None)
    df = pd.DataFrame({"tension_raw": tension})
    df["tension_log"]  = np.log1p(df["tension_raw"])
    roll = df["tension_log"].rolling(window, min_periods=window // 2)
    df["tension_mean"] = roll.mean()
    df["tension_std"]  = roll.std()
    df["pizza_z"] = (
        (df["tension_log"] - df["tension_mean"]) / df["tension_std"]
    )
    return df


# =============================================================================
# MARKET DATA
# =============================================================================

def fetch_market_data(start: date, end: date) -> pd.DataFrame:
    tickers = [VIX_TICKER, SPY_TICKER] + list(DEFENSE_ETFS.keys()) + DEFENSE_NAMES
    log.info("Fetching market data …")
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=5)).isoformat(),
        auto_adjust=True, progress=False,
    )["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={VIX_TICKER: "VIX", SPY_TICKER: "SPY"})
    log.info("Market data: %d trading days, %d series", len(raw), len(raw.columns))
    return raw


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = pd.DataFrame(index=prices.index)
    for col in prices.columns:
        if col == "VIX":
            rets["VIX_pct"] = prices["VIX"].pct_change() * 100
        else:
            rets[f"{col}_ret"] = np.log(prices[col] / prices[col].shift(1)) * 100
    return rets.dropna()


def align_signal_to_market(pizza: pd.DataFrame,
                            returns: pd.DataFrame,
                            lag: int) -> pd.DataFrame:
    # GDELT returns UTC-aware timestamps; yfinance is tz-naive.
    # Strip timezone so reindex can compare them.
    idx = pizza.index
    if hasattr(idx, "tz") and idx.tz is not None:
        pizza = pizza.copy()
        pizza.index = idx.tz_convert(None)
    pizza_td = pizza[["pizza_z"]].reindex(returns.index, method="ffill")
    combined = pd.concat([pizza_td.shift(lag), returns], axis=1).dropna()
    log.info("Aligned dataset: %d trading days", len(combined))
    return combined


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

def _guard(df: pd.DataFrame, min_rows: int = 10) -> None:
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} aligned observations — need {min_rows}+. "
            "Use --start 2017-01-01 or lower --threshold."
        )


def test_h1_vix(df: pd.DataFrame, spike_z: float,
                vix_threshold: float) -> dict:
    _guard(df)
    spike  = df["pizza_z"] >= spike_z
    vix_up = df["VIX_pct"] >= vix_threshold
    n_sp, n_nsp = int(spike.sum()), int((~spike).sum())
    n_vsp  = int((spike  & vix_up).sum())
    n_vnsp = int((~spike & vix_up).sum())

    cont = np.array([[n_vsp,  max(n_sp  - n_vsp,  0)],
                     [n_vnsp, max(n_nsp - n_vnsp, 0)]])
    fisher_or, fisher_p = stats.fisher_exact(cont, alternative="greater")
    _, mw_p  = stats.mannwhitneyu(df.loc[spike,  "VIX_pct"].dropna(),
                                   df.loc[~spike, "VIX_pct"].dropna(),
                                   alternative="greater")
    r, corr_p = stats.pointbiserialr(spike.astype(int), df["VIX_pct"])
    counts = np.array([n_vsp, n_vnsp])
    nobs   = np.array([max(n_sp, 1), max(n_nsp, 1)])
    _, prop_p = proportions_ztest(counts, nobs, alternative="larger")

    return dict(
        n_spike=n_sp, n_no_spike=n_nsp,
        pct_vix_spike=round(100*n_vsp/max(n_sp,1),1),
        pct_vix_no_spike=round(100*n_vnsp/max(n_nsp,1),1),
        med_vix_spike=round(df.loc[spike,"VIX_pct"].median(),2),
        med_vix_no_spike=round(df.loc[~spike,"VIX_pct"].median(),2),
        fisher_or=round(fisher_or,3), fisher_p=round(fisher_p,4),
        mw_p=round(mw_p,4), r=round(r,4),
        corr_p=round(corr_p,4), prop_p=round(prop_p,4),
        passes=fisher_p < 0.05 and mw_p < 0.05,
    )


def test_h2_defense(df: pd.DataFrame, spike_z: float,
                    outperform: float) -> dict:
    _guard(df)
    spike   = df["pizza_z"] >= spike_z
    results = {}
    for etf in DEFENSE_ETFS:
        col = f"{etf}_ret"
        if col not in df.columns:
            continue
        excess   = df[col] - df["SPY_ret"]
        out_flag = excess >= outperform
        n_sp, n_nsp   = int(spike.sum()), int((~spike).sum())
        n_osp, n_onsp = int((spike&out_flag).sum()), int((~spike&out_flag).sum())
        cont = np.array([[n_osp,  max(n_sp -n_osp, 0)],
                         [n_onsp, max(n_nsp-n_onsp,0)]])
        _, fp = stats.fisher_exact(cont, alternative="greater")
        _, mp = stats.mannwhitneyu(excess[spike].dropna(),
                                    excess[~spike].dropna(),
                                    alternative="greater")
        r, _ = stats.pointbiserialr(spike.astype(int), excess)
        results[etf] = dict(
            description=DEFENSE_ETFS[etf],
            pct_out_spike=round(100*n_osp/max(n_sp,1),1),
            pct_out_no_spike=round(100*n_onsp/max(n_nsp,1),1),
            med_excess_spike=round(excess[spike].median(),3),
            med_excess_no_spike=round(excess[~spike].median(),3),
            fisher_p=round(fp,4), mw_p=round(mp,4),
            r=round(r,4), passes=fp<0.05 and mp<0.05,
        )
    return results


def test_h3_combined(df: pd.DataFrame, spike_z: float,
                     vix_threshold: float, outperform: float) -> dict:
    _guard(df)
    spike  = df["pizza_z"] >= spike_z
    vix_up = df["VIX_pct"] >= vix_threshold
    results = {}
    for etf in DEFENSE_ETFS:
        col = f"{etf}_ret"
        if col not in df.columns:
            continue
        excess  = df[col] - df["SPY_ret"]
        crisis  = vix_up & (excess >= outperform)
        n_sp, n_nsp  = int(spike.sum()), int((~spike).sum())
        n_csp, n_cnsp = int((spike&crisis).sum()), int((~spike&crisis).sum())
        cont = np.array([[n_csp,  max(n_sp -n_csp, 0)],
                         [n_cnsp, max(n_nsp-n_cnsp,0)]])
        _, fp = stats.fisher_exact(cont, alternative="greater")
        results[etf] = dict(
            pct_crisis_spike=round(100*n_csp/max(n_sp,1),1),
            pct_crisis_no_spike=round(100*n_cnsp/max(n_nsp,1),1),
            fisher_p=round(fp,4), passes=fp<0.05,
        )
    return results


def test_idiosyncratic(df: pd.DataFrame, spike_z: float) -> dict:
    spike   = df["pizza_z"] >= spike_z
    results = {}
    for name in DEFENSE_NAMES:
        col = f"{name}_ret"
        if col not in df.columns:
            continue
        cv = lambda s: s.std() / (abs(s.mean()) + 1e-6)
        cv_sp  = cv(df.loc[spike,  col].dropna())
        cv_nsp = cv(df.loc[~spike, col].dropna())
        results[name] = dict(
            cv_spike=round(cv_sp,2),
            cv_no_spike=round(cv_nsp,2),
            cv_ratio=round(cv_sp/max(cv_nsp,0.01),2),
        )
    return results


# =============================================================================
# REPORT
# =============================================================================

def print_report(h1, h2, h3, idio, cfg):
    sep = "═" * 72
    print(f"\n{sep}")
    print("PENTAGON PIZZA INDEX — GDELT DOC 2.0 — RESULTS")
    print(f"Period: {cfg['start']} → {cfg['end']}  |  "
          f"Spike z>{cfg['spike_z']}  |  Lag={cfg['lag']}d")
    print(sep)

    print(f"\n── H1: Tension spike → VIX jump  (n_spike={h1['n_spike']}) ──")
    print(f"  VIX ≥{cfg['vix_spike']}% on spike days : {h1['pct_vix_spike']}%  "
          f"(median {h1['med_vix_spike']:+.2f}%)")
    print(f"  VIX ≥{cfg['vix_spike']}% otherwise    : {h1['pct_vix_no_spike']}%  "
          f"(median {h1['med_vix_no_spike']:+.2f}%)")
    print(f"  Fisher OR={h1['fisher_or']:.3f} p={h1['fisher_p']:.4f}  "
          f"MW p={h1['mw_p']:.4f}  r={h1['r']:.4f}")
    print(f"  {'PASS ✓' if h1['passes'] else 'FAIL ✗'}")

    print(f"\n── H2: Tension spike → defense ETF outperforms SPY ──")
    any_h2 = False
    for etf, r in h2.items():
        v = "✓" if r["passes"] else "✗"
        print(f"  {etf}  spike={r['pct_out_spike']}%  "
              f"no-spike={r['pct_out_no_spike']}%  "
              f"med excess: {r['med_excess_spike']:+.3f}% vs "
              f"{r['med_excess_no_spike']:+.3f}%  "
              f"Fisher p={r['fisher_p']:.4f}  {v}")
        if r["passes"]: any_h2 = True
    print(f"  {'PASS ✓' if any_h2 else 'FAIL ✗'}")

    print(f"\n── H3: Crisis trade (VIX up AND defense outperforms) ──")
    any_h3 = False
    for etf, r in h3.items():
        v = "✓" if r["passes"] else "✗"
        print(f"  {etf}: spike={r['pct_crisis_spike']}%  "
              f"no-spike={r['pct_crisis_no_spike']}%  "
              f"p={r['fisher_p']:.4f}  {v}")
        if r["passes"]: any_h3 = True
    print(f"  {'PASS ✓' if any_h3 else 'FAIL ✗'}")

    print(f"\n── Idiosyncratic risk (CV ratio > 1.5 = contamination) ──")
    for name, r in idio.items():
        flag = "  ⚠ high dispersion" if r["cv_ratio"] > 1.5 else ""
        print(f"  {name}: spike={r['cv_spike']:.2f}  "
              f"no-spike={r['cv_no_spike']:.2f}  "
              f"ratio={r['cv_ratio']:.2f}{flag}")

    score = sum([h1["passes"], any_h2, any_h3])
    v = {3:"STRONG",2:"MODERATE",1:"WEAK",0:"NOT SUPPORTED"}[score]
    print(f"\n{sep}")
    print(f"  OVERALL: {score}/3  →  {v} SUPPORT")
    print(sep)


# =============================================================================
# PLOTS
# =============================================================================

def plot_results(pizza, df, prices, h1, h2, spike_z):
    spike_mask = df["pizza_z"] >= spike_z
    spike_days = df.index[spike_mask]
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle("Pentagon Pizza Index — GDELT DOC 2.0", fontsize=14)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(pizza.index, pizza["tension_log"],  lw=0.7, color="#378ADD", alpha=0.7, label="Log tension")
    ax1.plot(pizza.index, pizza["tension_mean"], lw=1.5, color="#888780", ls="--", label="28d mean")
    ax1.set_title("GDELT military tension (log)", fontsize=10)
    ax1.set_ylabel("Log score"); ax1.legend(fontsize=8); ax1.grid(alpha=0.15)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(pizza.index, pizza["pizza_z"], lw=0.6, color="#534AB7", alpha=0.8)
    ax2.axhline(spike_z, color="#E24B4A", lw=1.2, ls="--", label=f"z={spike_z}")
    ax2.fill_between(pizza.index, spike_z, pizza["pizza_z"],
                     where=pizza["pizza_z"] >= spike_z, alpha=0.3, color="#E24B4A")
    ax2.set_title(f"Pizza z-score ({len(spike_days)} spikes)", fontsize=10)
    ax2.set_ylabel("Z-score"); ax2.legend(fontsize=8); ax2.grid(alpha=0.15)

    ax3 = fig.add_subplot(gs[1, 0])
    windows = []
    for sd in spike_days:
        try:
            loc = df.index.get_loc(sd)
            sl  = df["VIX_pct"].iloc[max(0,loc-3):min(len(df),loc+6)].values
            if len(sl) == 9:
                windows.append(sl)
        except KeyError:
            pass
    if windows:
        mat = np.array(windows); days = range(-3, 6)
        ax3.plot(days, np.median(mat, axis=0), color="#E24B4A", lw=2, label="Median")
        ax3.fill_between(days, np.percentile(mat,25,axis=0),
                         np.percentile(mat,75,axis=0), alpha=0.2, color="#E24B4A", label="IQR")
        ax3.axvline(0, color="gray", lw=0.8, ls="--")
        ax3.axvline(1, color="#1D9E75", lw=0.8, ls=":", label="Trade day")
        ax3.set_title("VIX Δ event study", fontsize=9)
        ax3.set_xlabel("Days vs spike"); ax3.set_ylabel("VIX %Δ")
        ax3.legend(fontsize=7); ax3.grid(alpha=0.15)

    ax4 = fig.add_subplot(gs[1, 1])
    colors = ["#378ADD", "#1D9E75", "#EF9F27"]
    for i, etf in enumerate(DEFENSE_ETFS):
        col = f"{etf}_ret"
        if col not in df.columns: continue
        excess = df[col] - df["SPY_ret"]
        bp = ax4.boxplot([excess[~spike_mask].dropna(), excess[spike_mask].dropna()],
                         positions=[i*3, i*3+1.2], widths=0.9, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black", lw=2))
        for patch in bp["boxes"]: patch.set_facecolor(colors[i]+"80")
    ax4.axhline(0, color="gray", lw=0.8)
    ax4.set_xticks([0.6, 3.6, 6.6])
    ax4.set_xticklabels(list(DEFENSE_ETFS.keys()), fontsize=9)
    ax4.set_title("Defense excess return: no-spike vs spike", fontsize=9)
    ax4.set_ylabel("Excess vs SPY (%)"); ax4.grid(alpha=0.15)

    ax5 = fig.add_subplot(gs[1, 2])
    for etf, color in zip(DEFENSE_ETFS, colors):
        col = f"{etf}_ret"
        if col not in df.columns: continue
        cum = (1 + df.loc[spike_mask, col].fillna(0)/100).cumprod() - 1
        ax5.plot(cum.reset_index(drop=True), lw=1.5, label=etf, color=color)
    ax5.axhline(0, color="gray", lw=0.8)
    ax5.set_title("Cumulative return on spike days only", fontsize=9)
    ax5.set_xlabel("Spike #"); ax5.set_ylabel("Cumulative return")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.15)

    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(df["pizza_z"], df["VIX_pct"], alpha=0.12, s=6, color="#534AB7")
    m  = np.polyfit(df["pizza_z"].dropna(), df["VIX_pct"].dropna(), 1)
    xr = np.linspace(df["pizza_z"].min(), df["pizza_z"].max(), 100)
    ax6.plot(xr, np.polyval(m, xr), color="#E24B4A", lw=1.8)
    r, _ = stats.pearsonr(df["pizza_z"].dropna(), df["VIX_pct"].dropna())
    ax6.axvline(spike_z, color="#E24B4A", lw=0.8, ls="--")
    ax6.set_title(f"Pizza z vs next-day VIX Δ  r={r:.3f}", fontsize=9)
    ax6.set_xlabel("Pizza z"); ax6.set_ylabel("VIX %Δ"); ax6.grid(alpha=0.15)

    ax7 = fig.add_subplot(gs[2, 1])
    monthly = (pizza["pizza_z"] >= spike_z).resample("ME").sum()
    ax7.bar(monthly.index, monthly.values, width=20, color="#EF9F27", alpha=0.8)
    ax7.set_title("Monthly spike frequency", fontsize=9)
    ax7.set_ylabel("Spike days / month"); ax7.grid(alpha=0.15, axis="y")

    ax8 = fig.add_subplot(gs[2, 2])
    if "ITA" in prices.columns:
        norm = (prices[["ITA","SPY"]].dropna() / prices[["ITA","SPY"]].dropna().iloc[0] * 100)
        ax8.plot(norm.index, norm["ITA"], lw=1.2, color="#378ADD", label="ITA")
        ax8.plot(norm.index, norm["SPY"], lw=1.2, color="#888780", ls="--", label="SPY")
        for sd in spike_days[-30:]:
            if sd in norm.index:
                ax8.axvline(sd, color="#E24B4A", lw=0.4, alpha=0.5)
        ax8.set_title("ITA vs SPY (red = pizza spikes)", fontsize=9)
        ax8.set_ylabel("Indexed"); ax8.legend(fontsize=8); ax8.grid(alpha=0.15)

    plt.savefig("pizza_index_results.png", dpi=150, bbox_inches="tight")
    log.info("Plot → pizza_index_results.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pentagon Pizza Index — GDELT DOC 2.0 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pizza_index.py
  python pizza_index.py --start 2020-01-01
  python pizza_index.py --threshold 1.5
  python pizza_index.py --lag 2
  python pizza_index.py --refresh
        """,
    )
    parser.add_argument("--start",      default="2017-01-01")
    parser.add_argument("--end",        default=date.today().isoformat())
    parser.add_argument("--threshold",  type=float, default=DEFAULT_SPIKE_Z)
    parser.add_argument("--vix-spike",  type=float, default=DEFAULT_VIX_SPIKE)
    parser.add_argument("--outperform", type=float, default=DEFAULT_OUTPERFORM)
    parser.add_argument("--lag",        type=int,   default=DEFAULT_LAG)
    parser.add_argument("--window",     type=int,   default=DEFAULT_WINDOW)
    parser.add_argument("--workers",    type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--refresh",    action="store_true")
    args = parser.parse_args()

    start_dt = date.fromisoformat(args.start)
    end_dt   = date.fromisoformat(args.end)
    cfg = dict(start=args.start, end=args.end,
               spike_z=args.threshold, vix_spike=args.vix_spike,
               outperform=args.outperform, lag=args.lag, window=args.window)

    tension = fetch_gdelt_signal(start_dt, end_dt,
                                  workers=args.workers,
                                  force_refresh=args.refresh)
    pizza   = build_pizza_index(tension, window=args.window)

    prices  = fetch_market_data(start_dt, end_dt)
    returns = compute_returns(prices)
    df      = align_signal_to_market(pizza, returns, lag=args.lag)

    n_spikes = int((df["pizza_z"] >= args.threshold).sum())
    log.info("Spikes: %d (%.1f%% of sample)", n_spikes,
             100 * n_spikes / max(len(df), 1))

    if n_spikes < 5:
        log.warning("Only %d spikes — try --threshold 1.5", n_spikes)

    h1   = test_h1_vix(df, args.threshold, args.vix_spike)
    h2   = test_h2_defense(df, args.threshold, args.outperform)
    h3   = test_h3_combined(df, args.threshold, args.vix_spike, args.outperform)
    idio = test_idiosyncratic(df, args.threshold)

    print_report(h1, h2, h3, idio, cfg)
    plot_results(pizza, df, prices, h1, h2, args.threshold)

    # ── Walk-forward validation ───────────────────────────────────────────────
    log.info("Running walk-forward validation ...")
    wf = run_walk_forward(df, train_days=504, test_days=126)
    print_walk_forward(wf)
    plot_walk_forward(wf)
    wf.to_csv("walk_forward_results.csv", index=False)
    log.info("Walk-forward results -> walk_forward_results.csv")

    # ── Paper trading simulation ──────────────────────────────────────────────
    log.info("Running paper trading simulation ...")
    pt = run_paper_trader(df, spike_z=args.threshold, capital=100_000.0,
                          bid_ask_bps=1.5, commission_bps=0.0, slippage_bps=1.5,
                          use_spy_hedge=False, train_frac=0.5,
                          kelly_cap=0.08, cooldown_days=3,
                          saturation_window=28, saturation_max=8)
    print_paper_trader(pt)
    plot_paper_trader(pt)
    pt["trade_log"].to_csv("paper_trade_log.csv", index=False)
    pt["equity_curve"].to_csv("paper_equity_curve.csv")
    log.info("Paper trade log -> paper_trade_log.csv")
    log.info("Equity curve   -> paper_equity_curve.csv")

    df.to_csv("pizza_index_aligned.csv")
    pizza.to_csv("pizza_tension_series.csv")
    log.info("Saved CSVs.")



# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def _score_threshold(df_train, z, etfs):
    """Score a spike threshold on training data. Returns mean Fisher p (lower = better)."""
    spike = df_train["pizza_z"] >= z
    if spike.sum() < 5 or (~spike).sum() < 20:
        return 1.0
    ps = []
    for etf in etfs:
        col = f"{etf}_ret"
        if col not in df_train.columns:
            continue
        excess   = df_train[col] - df_train["SPY_ret"]
        out_flag = excess >= DEFAULT_OUTPERFORM
        n_sp, n_nsp = int(spike.sum()), int((~spike).sum())
        n_osp  = int((spike & out_flag).sum())
        n_onsp = int((~spike & out_flag).sum())
        cont = np.array([[n_osp,  max(n_sp -n_osp, 0)],
                         [n_onsp, max(n_nsp-n_onsp,0)]])
        _, fp = stats.fisher_exact(cont, alternative="greater")
        ps.append(fp)
    return float(np.mean(ps)) if ps else 1.0


def run_walk_forward(df, train_days=504, test_days=126, z_candidates=None):
    """
    Expanding walk-forward: optimise spike_z on train, evaluate on test.
    Each test window is strictly out-of-sample.
    train_days ~2yr, test_days ~6mo.
    """
    if z_candidates is None:
        z_candidates = [1.2, 1.5, 1.8, 2.0, 2.2, 2.5]
    etfs   = [e for e in DEFENSE_ETFS if f"{e}_ret" in df.columns]
    folds  = []
    n      = len(df)
    fold_n = 0
    start_i = train_days

    while start_i + test_days <= n:
        fold_n += 1
        train = df.iloc[:start_i]
        test  = df.iloc[start_i: start_i + test_days]

        best_z     = min(z_candidates, key=lambda z: _score_threshold(train, z, etfs))
        train_score = _score_threshold(train, best_z, etfs)
        spike_test  = test["pizza_z"] >= best_z
        n_spikes    = int(spike_test.sum())

        record = {
            "fold": fold_n,
            "train_end":     df.index[start_i - 1].date(),
            "test_start":    df.index[start_i].date(),
            "test_end":      df.index[start_i + test_days - 1].date(),
            "best_z":        best_z,
            "train_p":       round(train_score, 4),
            "n_spikes_test": n_spikes,
        }

        for etf in etfs:
            col       = f"{etf}_ret"
            excess    = test[col] - test["SPY_ret"]
            strat_ret = excess.where(spike_test, 0.0)
            ann_ret   = strat_ret.mean() * 252
            ann_vol   = strat_ret.std()  * np.sqrt(252)
            sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
            win_rate  = float((excess[spike_test] > 0).mean()) if n_spikes > 0 else np.nan
            cum_ret   = float((1 + strat_ret / 100).prod() - 1)
            record.update({
                f"{etf}_sharpe":   round(sharpe, 4),
                f"{etf}_win_rate": round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
                f"{etf}_cum_ret":  round(cum_ret, 6),
            })

        folds.append(record)
        start_i += test_days

    return pd.DataFrame(folds)


def print_walk_forward(wf):
    etfs = [e for e in DEFENSE_ETFS if f"{e}_sharpe" in wf.columns]
    sep  = "=" * 72
    print(f"\n{sep}")
    print("WALK-FORWARD VALIDATION RESULTS")
    print(f"  {len(wf)} folds  |  train~2yr  |  test~6mo  |  expanding window")
    print(sep)

    hdr = f"  Fold  Test window                   z*    Spk"
    for etf in etfs:
        hdr += f"  {etf:>8}"
    print(hdr)
    print(f"  {'-'*68}")

    for _, r in wf.iterrows():
        row = (f"  {int(r['fold']):<5} "
               f"{str(r['test_start'])}->{ str(r['test_end'])}  "
               f"{r['best_z']:>4.1f}  {int(r['n_spikes_test']):>4}")
        for etf in etfs:
            s = r.get(f"{etf}_sharpe", np.nan)
            row += f"  {s:>+8.3f}"
        print(row)

    print(f"\n  SUMMARY:")
    for etf in etfs:
        sharpes  = wf[f"{etf}_sharpe"].dropna()
        wr_vals  = wf[f"{etf}_win_rate"].dropna()
        pos_rate = (sharpes > 0).mean()
        print(f"  {etf}:  mean Sharpe={sharpes.mean():+.3f}  "
              f"std={sharpes.std():.3f}  "
              f"positive folds={pos_rate*100:.0f}%  "
              f"mean win rate={wr_vals.mean()*100:.1f}%")

    print(f"\n  STABILITY (t-test Sharpe > 0):")
    for etf in etfs:
        sharpes = wf[f"{etf}_sharpe"].dropna()
        t_stat, p_val = stats.ttest_1samp(sharpes, 0)
        sig = "significant v" if p_val/2 < 0.05 else "not significant x"
        print(f"  {etf}: t={t_stat:.3f}  p={p_val/2:.4f}  {sig}")
    print(sep)


def plot_walk_forward(wf):
    etfs   = [e for e in DEFENSE_ETFS if f"{e}_sharpe" in wf.columns]
    colors = {"ITA": "#378ADD", "XAR": "#1D9E75", "PPA": "#EF9F27"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Walk-Forward Validation — Pentagon Pizza Index", fontsize=13)
    x = wf["fold"].values

    ax = axes[0, 0]
    for etf in etfs:
        ax.plot(x, wf[f"{etf}_sharpe"], marker="o", ms=5,
                label=etf, color=colors.get(etf, "gray"))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Out-of-sample Sharpe per fold", fontsize=10)
    ax.set_xlabel("Fold"); ax.set_ylabel("Annualised Sharpe")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.bar(x, wf["best_z"], color="#534AB7", alpha=0.8, width=0.6)
    ax.axhline(DEFAULT_SPIKE_Z, color="#E24B4A", lw=1.2, ls="--",
               label=f"Default z={DEFAULT_SPIKE_Z}")
    ax.set_title("Optimal z threshold per fold", fontsize=10)
    ax.set_xlabel("Fold"); ax.set_ylabel("Best z*")
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")

    ax = axes[1, 0]
    for etf in etfs:
        cum_rets = wf[f"{etf}_cum_ret"].values
        ax.plot(x, np.cumprod(1 + cum_rets) - 1,
                marker="o", ms=4, label=etf, color=colors.get(etf, "gray"))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Cumulative return across folds (OOS)", fontsize=10)
    ax.set_xlabel("Fold"); ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for etf in etfs:
        ax.plot(x, wf[f"{etf}_win_rate"].fillna(0) * 100,
                marker="s", ms=4, label=etf, color=colors.get(etf, "gray"))
    ax.axhline(50, color="gray", lw=0.8, ls="--", label="50%")
    ax.set_title("Win rate on spike days per fold", fontsize=10)
    ax.set_xlabel("Fold"); ax.set_ylabel("Win rate (%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("walk_forward_results.png", dpi=150, bbox_inches="tight")
    log.info("Walk-forward plot -> walk_forward_results.png")
    plt.show()


# =============================================================================
# PAPER TRADING SIMULATOR
# =============================================================================

def _kelly_fraction(win_rate, avg_win, avg_loss, max_f=0.25):
    """Half-Kelly position size, capped at max_f."""
    if avg_loss <= 0 or win_rate <= 0:
        return 0.02
    b = avg_win / avg_loss
    q = 1 - win_rate
    f = (win_rate * b - q) / b
    return max(0.0, min(f * 0.5, max_f))


def run_paper_trader(df, spike_z=DEFAULT_SPIKE_Z, capital=100_000.0,
                     bid_ask_bps=1.5, commission_bps=0.0, slippage_bps=1.5,
                     stop_loss_pct=2.0, use_spy_hedge=False, train_frac=0.5,
                     kelly_cap=0.08, cooldown_days=3,
                     saturation_window=28, saturation_max=8):
    """
    Paper trading simulation — Robinhood configuration with three fixes.

    ROBINHOOD COST MODEL:
      commission = 0 bps  (genuinely free)
      bid_ask    = 1.5 bps (still pay half the spread, unavoidable)
      slippage   = 1.5 bps (payment for order flow — invisible but real)
      Total round-trip ~ 6 bps per leg
      use_spy_hedge = False (Robinhood cannot short)

    THREE STRUCTURAL FIXES vs v1:
      Fix 1 — kelly_cap=0.08    Cap Kelly at 8% per ETF (was 25%).
                                 Prevents 70%+ capital deployment on cluster
                                 days which grinds down capital via cost drag.
      Fix 2 — cooldown_days=3   Skip trading for 3 days after any trade.
                                 Edge degrades when spikes cluster — this
                                 enforces the "rare event" assumption.
      Fix 3 — saturation filter  Skip if >saturation_max spikes in last
                                 saturation_window days. When the signal fires
                                 constantly it is no longer a rare event signal
                                 (e.g. 2025 tariff chaos: 85 spikes in 6mo).
    """
    etfs     = [e for e in DEFENSE_ETFS if f"{e}_ret" in df.columns]
    cost_pct = (bid_ask_bps + commission_bps + slippage_bps) * 2 / 10_000
    split    = int(len(df) * train_frac)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    # Calibrate Kelly on training half (capped at kelly_cap)
    kelly_params = {}
    for etf in etfs:
        col         = f"{etf}_ret"
        spike_train = df_train["pizza_z"] >= spike_z
        excess      = (df_train[col] - df_train["SPY_ret"])[spike_train].dropna()
        if len(excess) < 5:
            kelly_params[etf] = 0.02
            continue
        wins     = excess[excess > 0]
        losses   = excess[excess <= 0].abs()
        wr       = len(wins) / len(excess)
        avg_win  = wins.mean()  if len(wins)  > 0 else 0.01
        avg_loss = losses.mean() if len(losses) > 0 else 0.01
        kelly_params[etf] = _kelly_fraction(wr, avg_win, avg_loss,
                                             max_f=kelly_cap)

    log.info("Kelly fractions (cap=%.0f%%): %s", kelly_cap * 100,
             {k: f"{v:.1%}" for k, v in kelly_params.items()})
    log.info("Filters: cooldown=%dd  saturation=max %d spikes/%dd",
             cooldown_days, saturation_max, saturation_window)

    equity       = capital
    peak_equity  = capital
    equity_curve = []
    trade_log    = []
    last_trade_i = -999  # for cooldown tracking

    for i, (ts, row) in enumerate(df_test.iterrows()):
        z           = row["pizza_z"]
        is_spike    = z >= spike_z
        daily_pnl   = 0.0
        skip_reason = None

        if is_spike:
            # Fix 2: Cooldown filter
            if i - last_trade_i < cooldown_days:
                skip_reason = "cooldown"
                is_spike    = False

            # Fix 3: Saturation filter
            if skip_reason is None:
                window_start  = max(0, i - saturation_window)
                recent_spikes = int(
                    (df_test.iloc[window_start:i]["pizza_z"] >= spike_z).sum()
                )
                if recent_spikes >= saturation_max:
                    skip_reason = f"saturated({recent_spikes}/{saturation_window}d)"
                    is_spike    = False

        if is_spike:
            last_trade_i = i
            for etf in etfs:
                col       = f"{etf}_ret"
                kelly_f   = kelly_params.get(etf, 0.02)   # Fix 1: capped at kelly_cap
                notional  = equity * kelly_f
                etf_ret   = row[col] / 100
                gross_ret = etf_ret
                if etf_ret * 100 < -stop_loss_pct:        # stop loss
                    gross_ret = -stop_loss_pct / 100
                gross_pnl  = notional * gross_ret
                trade_cost = notional * cost_pct           # no SPY hedge leg
                net_pnl    = gross_pnl - trade_cost
                daily_pnl += net_pnl
                trade_log.append({
                    "timestamp":   ts,
                    "etf":         etf,
                    "pizza_z":     round(z, 3),
                    "kelly_f":     round(kelly_f, 4),
                    "notional":    round(notional, 2),
                    "etf_ret_pct": round(etf_ret * 100, 3),
                    "excess_pct":  round(gross_ret * 100, 3),
                    "gross_pnl":   round(gross_pnl, 2),
                    "cost":        round(trade_cost, 2),
                    "net_pnl":     round(net_pnl, 2),
                    "skip_reason": "",
                })
        elif skip_reason:
            trade_log.append({
                "timestamp": ts, "etf": "SKIPPED",
                "pizza_z": round(z, 3), "kelly_f": 0,
                "notional": 0, "etf_ret_pct": 0, "excess_pct": 0,
                "gross_pnl": 0, "cost": 0, "net_pnl": 0,
                "skip_reason": skip_reason,
            })

        equity      += daily_pnl
        peak_equity  = max(peak_equity, equity)
        equity_curve.append({
            "timestamp":   ts,
            "equity":      round(equity, 2),
            "daily_pnl":   round(daily_pnl, 2),
            "drawdown":    round((equity - peak_equity) / peak_equity, 4),
            "is_spike":    is_spike,
            "pizza_z":     round(z, 3),
            "skip_reason": skip_reason or "",
        })

    ec = pd.DataFrame(equity_curve).set_index("timestamp")
    tl = pd.DataFrame(trade_log)

    # Filter to actual trades only for stats
    tl_trades = tl[tl["etf"] != "SKIPPED"]
    daily_rets = ec["daily_pnl"] / capital
    ann_ret    = daily_rets.mean() * 252
    ann_vol    = daily_rets.std()  * np.sqrt(252)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0
    total_ret  = (equity - capital) / capital

    if len(tl_trades) > 0:
        winners       = tl_trades[tl_trades["net_pnl"] > 0]
        losers        = tl_trades[tl_trades["net_pnl"] <= 0]
        win_rate      = len(winners) / len(tl_trades)
        profit_factor = (winners["net_pnl"].sum() /
                         max(abs(losers["net_pnl"].sum()), 1e-6))
        total_cost    = tl_trades["cost"].sum()
    else:
        win_rate = profit_factor = total_cost = 0.0

    n_skipped_cooldown   = int((tl["skip_reason"] == "cooldown").sum())
    n_skipped_saturation = int(tl["skip_reason"].str.startswith("saturated").sum())

    metrics = {
        "capital":             capital,
        "end_equity":          round(equity, 2),
        "total_return":        round(total_ret * 100, 2),
        "ann_return":          round(ann_ret * 100, 2),
        "ann_vol":             round(ann_vol * 100, 2),
        "sharpe":              round(sharpe, 3),
        "max_drawdown":        round(float(ec["drawdown"].min()) * 100, 2),
        "n_trades":            len(tl_trades),
        "n_spike_days":        int(ec["is_spike"].sum()),
        "n_skipped_cooldown":  n_skipped_cooldown,
        "n_skipped_saturation":n_skipped_saturation,
        "win_rate":            round(win_rate * 100, 1),
        "avg_win":             round(winners["net_pnl"].mean(), 2) if len(tl_trades)>0 and len(winners)>0 else 0,
        "avg_loss":            round(losers["net_pnl"].mean(),  2) if len(tl_trades)>0 and len(losers)>0  else 0,
        "profit_factor":       round(profit_factor, 3),
        "total_cost_drag":     round(total_cost / capital * 100, 3),
        "train_period":        f"{df.index[0].date()} -> {df.index[split].date()}",
        "test_period":         f"{df.index[split].date()} -> {df.index[-1].date()}",
    }
    return {"metrics": metrics, "equity_curve": ec,
            "trade_log": tl, "kelly_params": kelly_params}



def print_paper_trader(result):
    m  = result["metrics"]
    kp = result["kelly_params"]
    sep = "=" * 72
    print(f"\n{sep}")
    print("PAPER TRADING SIMULATION — OUT-OF-SAMPLE")
    print(f"  Train: {m['train_period']}")
    print(f"  Test : {m['test_period']}")
    print(sep)
    print(f"  Kelly sizes : {', '.join(f'{e}={v:.1%}' for e,v in kp.items())}")
    print(f"  Strategy    : Long defense ETFs (Kelly-weighted, no short)  |  Hold 1 day")
    print(f"  Cost model  : Robinhood (~6 bps round-trip, 0 commission)")
    print(f"  Filters     : cooldown=3d  saturation=max 8 spikes/28d")
    print(f"\n  PERFORMANCE:")
    print(f"    Starting capital : ${m['capital']:>12,.2f}")
    print(f"    Ending equity    : ${m['end_equity']:>12,.2f}")
    print(f"    Total return     : {m['total_return']:>+8.2f}%")
    print(f"    Ann. return      : {m['ann_return']:>+8.2f}%")
    print(f"    Ann. volatility  : {m['ann_vol']:>8.2f}%")
    print(f"    Sharpe ratio     : {m['sharpe']:>8.3f}")
    print(f"    Max drawdown     : {m['max_drawdown']:>8.2f}%")
    print(f"\n  TRADES:")
    print(f"    Total trades     : {m['n_trades']:>6}")
    print(f"    Spike days traded: {m['n_spike_days']:>6}")
    print(f"    Skipped-cooldown : {m['n_skipped_cooldown']:>6}")
    print(f"    Skipped-saturated: {m['n_skipped_saturation']:>6}")
    print(f"    Win rate         : {m['win_rate']:>6.1f}%")
    print(f"    Avg win          : ${m['avg_win']:>8.2f}")
    print(f"    Avg loss         : ${m['avg_loss']:>8.2f}")
    print(f"    Profit factor    : {m['profit_factor']:>8.3f}")
    print(f"    Cost drag        : {m['total_cost_drag']:>6.3f}% of capital")
    verdict = ("LIVE-WORTHY" if m["sharpe"] > 1.0 else
               "PROMISING"  if m["sharpe"] > 0.5 else
               "MARGINAL"   if m["sharpe"] > 0.0 else "NOT VIABLE")
    print(f"\n  VERDICT: {verdict}  (Sharpe={m['sharpe']:.3f})")
    print(sep)


def plot_paper_trader(result):
    ec = result["equity_curve"]
    tl = result["trade_log"]
    m  = result["metrics"]
    colors = {"ITA": "#378ADD", "XAR": "#1D9E75", "PPA": "#EF9F27"}

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"Paper Trading — Pizza Index  "
        f"[Sharpe={m['sharpe']:.3f}  Return={m['total_return']:+.2f}%  "
        f"MaxDD={m['max_drawdown']:.2f}%]",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(ec.index, ec["equity"], lw=1.5, color="#378ADD", label="Strategy")
    ax.axhline(m["capital"], color="gray", lw=0.8, ls="--", label="Start capital")
    ax.fill_between(ec.index, m["capital"], ec["equity"],
                    where=ec["equity"] >= m["capital"], alpha=0.15, color="#1D9E75")
    ax.fill_between(ec.index, m["capital"], ec["equity"],
                    where=ec["equity"] <  m["capital"], alpha=0.15, color="#E24B4A")
    spike_idx = ec.index[ec["is_spike"]]
    ax.vlines(spike_idx, ec["equity"].min(), ec["equity"].max(),
              color="#534AB7", alpha=0.08, lw=0.6, label="Spike day")
    ax.set_title("Equity curve (out-of-sample)", fontsize=10)
    ax.set_ylabel("Portfolio ($)"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.fill_between(ec.index, ec["drawdown"] * 100, 0,
                    alpha=0.7, color="#E24B4A")
    ax.set_title("Drawdown (%)", fontsize=10)
    ax.set_ylabel("Drawdown (%)"); ax.grid(alpha=0.2)

    ax = axes[1, 0]
    if len(tl) > 0:
        for etf in DEFENSE_ETFS:
            etf_tl = tl[tl["etf"] == etf]
            if etf_tl.empty: continue
            ax.scatter(etf_tl["timestamp"], etf_tl["net_pnl"],
                       alpha=0.5, s=15, label=etf, color=colors.get(etf, "gray"))
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("Net P&L per trade", fontsize=10)
        ax.set_ylabel("Net P&L ($)"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1, 1]
    if len(tl) > 0:
        for etf in DEFENSE_ETFS:
            etf_tl = tl[tl["etf"] == etf]
            if etf_tl.empty: continue
            ax.scatter(etf_tl["pizza_z"], etf_tl["excess_pct"],
                       alpha=0.35, s=12, label=etf, color=colors.get(etf, "gray"))
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(DEFAULT_SPIKE_Z, color="#E24B4A", lw=0.8, ls="--")
        ax.set_title("Excess return vs pizza z", fontsize=10)
        ax.set_xlabel("Pizza z"); ax.set_ylabel("Excess vs SPY (%)")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[2, 0]
    if len(tl) > 0:
        pnl_s = tl.groupby("timestamp")["net_pnl"].sum() / m["capital"]
        roll_s = (pnl_s.rolling(30).mean() / pnl_s.rolling(30).std() * np.sqrt(252))
        ax.plot(roll_s.index, roll_s, lw=1.2, color="#534AB7", label="30-trade rolling Sharpe")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axhline(1, color="#1D9E75", lw=0.8, ls=":", label="Sharpe=1")
        ax.set_title("Rolling 30-trade Sharpe", fontsize=10)
        ax.set_ylabel("Sharpe"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[2, 1]
    if len(tl) > 0:
        ax.scatter(tl["gross_pnl"], tl["net_pnl"], alpha=0.3, s=10, color="#EF9F27")
        lims = [min(tl["gross_pnl"].min(), tl["net_pnl"].min()),
                max(tl["gross_pnl"].max(), tl["net_pnl"].max())]
        ax.plot(lims, lims, color="gray", lw=0.8, ls="--", label="No cost line")
        ax.set_title("Gross vs net P&L (cost drag)", fontsize=10)
        ax.set_xlabel("Gross P&L ($)"); ax.set_ylabel("Net P&L ($)")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("paper_trader_results.png", dpi=150, bbox_inches="tight")
    log.info("Paper trader plot -> paper_trader_results.png")
    plt.show()

if __name__ == "__main__":
    main()