"""
Deriv EXPIRYRANGE MC Bot v3 — 1HZ10V only (RDBEAR dropped, see v3 notes below)
================================================
Fixes applied vs v1 (confirmed from Railway logs):

  BUG 1 — GARCH vol unit error (ROOT CAUSE of ±0.30 barriers on 9648-priced asset)
    GARCH is fitted on RELATIVE returns scaled by 1000.
    vol_per_tick from GARCH came out as ~0.000018 (relative units / 1000).
    The MC used this directly as absolute price units → barrier = 0.40 * 0.000018 * sqrt(120)
    = 0.000079 price units → all 75K paths trivially inside → win=1.000.
    FIX: convert relative vol back to absolute: abs_vol = (garch_vol / scale) * price.
    Also added a hard sanity check: if abs_vol_per_tick < 0.01 * price/1000, abort and
    fall back to std(price_diffs) directly (no relative-return conversion).

  BUG 2 — Proposal API payout misread ($2.18 returned instead of $0.18)
    Deriv's proposal response for EXPIRYRANGE returns:
      { "proposal": { "ask_price": 0.35, "payout": 2.53, ... } }
    "payout" is the TOTAL payout if won (stake + profit), not net profit.
    Code did total - BASE_STAKE = 2.53 - 0.35 = $2.18 — WRONG because ask_price
    for EXPIRYRANGE is NOT always equal to stake (Deriv may adjust it).
    FIX: net_profit = payout - ask_price (use the actual ask price from the response).
    Also: cap sanity check — if net_profit > stake * 20, something is still wrong → skip.

  BUG 3 — Supabase column mismatch (PGRST204 on 'actual_payout')
    The SQL schema used column name 'actual_payout' but the original schema.sql
    had different column names. The schema.sql we provided had 'actual_payout'
    but the user's Supabase table was created from the FIRST schema which used
    'breach_prob' and 'ev_conservative' etc without 'actual_payout', 'mc_win_prob',
    'mc_ci_lower', 'barrier_sigma', 'hawkes_val'.
    FIX: simplified log record to only columns that exist in the ORIGINAL schema,
    plus a graceful column-mismatch fallback that logs what it can.

  BUG 4 — ValueError: too many values to unpack (expected 2)
    After refactor, execute_expiryrange returned 3-tuple (won, profit, placed)
    but some call sites still unpacked 2 values.
    FIX: all call sites updated; consistent 3-tuple everywhere.

  BUG 5 — Watchdog restarting too aggressively (4.5 min timeout, GARCH takes 3+ min)
    FIX: WATCHDOG_TIMEOUT raised to 15 minutes; last_activity reset after GARCH fit.

  BUG 6 — Bootstrap only loading 1000 ticks (HISTORY_BOOTSTRAP was 5000 but API
    silently returned 1000 — Deriv caps ticks_history at 5000 max but the count=5000
    call was working; however GARCH needs MIN_TICKS_FOR_FIT=200 and was getting it.
    The real issue: GARCH on RELATIVE returns of a 9648-priced asset with scale=1000
    gives conditional vol in units of (relative_return * 1000), not price.
    This is fixed by BUG 1 fix.

  BUG 7 — Unicode box-drawing chars split by Railway logger into individual lines
    FIX: all separators switched to plain ASCII hyphens/equals.

ARCHITECTURE (unchanged):
  4-stage filter: Gate → MC optimizer → Proposal API payout check → Execute
  Daily self-improvement: midnight UTC, Bayesian reweighting of dur/barrier prefs
  Supabase: all trades logged, config warm-started on restart

v3 CHANGES (2026-07-27) — fixes candidate scarcity, adds HMM/GBM engine:

  SCARCITY ROOT CAUSE — three independent binary gates were stacked, each
  cutting the surviving set, and the tightest of them left almost no room:
    1. MC_REQUIRED_WIN=0.58 vs MC_MAX_WIN_PROB~0.628 (derived from
       MIN_NET_PAYOUT/BASE_STAKE fair-odds math) left only a ~4.8-point-wide
       win_prob band as analytically acceptable, BEFORE any live pricing
       was even checked.
    2. structural_gate()'s 5 conditions (ADX/vol_trust/MBS/Bollinger/Hawkes)
       must ALL pass simultaneously.
    3. The live proposal then had to clear a fixed-dollar MIN_NET_PAYOUT
       floor, all-or-nothing.
    4. classify_regime() (rule-based on ADX/Hawkes thresholds) narrowed the
       duration/sigma grid further depending on which of 4 discrete regime
       buckets the market happened to be in.
  Any one of these being restrictive was fine; ALL FOUR compounding is why
  qualifying candidates were rare even with a reasonably fine grid.

  CHANGE 1 — SYMBOLS restricted to 1HZ10V only (RDBEAR dropped). Frees up
    the full per-cycle compute/API budget for one symbol.

  CHANGE 2 — Finer, wider sweep grids (DURATION_CANDIDATES, BARRIER_SIGMAS)
    so more of the space is sampled per cycle without touching the edge
    bar itself.

  CHANGE 3 — The analytical MC_REQUIRED_WIN/MC_MAX_WIN_PROB band is now a
    loose sanity pre-filter only (avoids wasting proposal-API calls on
    obviously-bad candidates). The actual accept/reject decision moved
    downstream to the LIVE proposal EV, which is the only number that
    reflects Deriv's real (house-margin-inclusive) pricing anyway.

  CHANGE 4 — rank_candidates_by_ev() no longer hard-excludes candidates
    below a fixed MIN_NET_PAYOUT floor. Any candidate with genuine positive
    EV trades; edge magnitude (edge_frac = ev/ask_price) continuously scales
    the stake instead of gating trade/no-trade. NOTE: Deriv's own minimum
    stake ($0.35 = BASE_STAKE) is a hard floor -- "trade marginal edges
    small" cannot mean below $0.35, so scaling runs BASE_STAKE upward with
    edge strength (capped at STAKE_MULT_MAX) rather than downward. This
    still stops good-but-marginal edges from being thrown away; it just
    can't go below the platform's own minimum.

  CHANGE 5 — Added SimpleHMM2 (a dependency-free 2-state Gaussian HMM,
    consistent with this file's self-built-statistics style -- GARCH via
    `arch` remains the only external stats dependency) and a regime-
    conditional GBM sampler (hmm_gbm_terminal_samples). Each MC cycle now
    blends TWO terminal-distribution sources at equal weight: the existing
    historical block-bootstrap (empirical shape/fat-tails) and the new
    HMM/GBM sampler (regime-persistent lognormal paths, drift/vol drawn
    from whichever of the 2 fitted states the filter currently favours).
    This directly makes the HMM/GBM model a first-class input to which
    duration/barrier gets selected, not just a logging label.
    classify_regime()'s ADX/Hawkes buckets are kept as a DIAGNOSTIC print
    only now -- the full grid sweeps every cycle regardless of bucket,
    since dropping RDBEAR freed the compute budget this used to protect.
"""

import asyncio, contextlib, io, json, math, os, random, sys, time, warnings
import base64
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import websockets
from scipy.stats import norm
from arch import arch_model

# v9: LSTM barrier/duration win-probability classifier (see
# expiryrange_lstm_model.py). Imported defensively -- if torch isn't
# installed or the import otherwise fails, the bot falls back to the
# bootstrap+HMM/GBM blend only (pre-v8 behaviour), rather than crashing on
# startup over an optional component.
try:
    import torch
    from expiryrange_lstm_model import BarrierWinClassifier
    _LSTM_AVAILABLE = True
except Exception as _lstm_import_err:
    _LSTM_AVAILABLE = False
    print(f"[LSTM] torch/model import failed, LSTM component disabled: {_lstm_import_err}")

warnings.filterwarnings("ignore")

# ── Deriv connection ──────────────────────────────────────────────────────
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID   = os.getenv("DERIV_ACCOUNT_ID") or None
API_BASE           = "https://api.derivws.com"
ACCOUNTS_PATH      = "/trading/v1/options/accounts"
OTP_PATH           = "/trading/v1/options/accounts/{account_id}/otp"

# ── Supabase ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Symbols ───────────────────────────────────────────────────────────────
# v3: restricted to 1HZ10V only (RDBEAR dropped) -- see header notes.
SYMBOLS = ["1HZ10V"]

# ── Contract parameters ───────────────────────────────────────────────────
BASE_STAKE        = 0.35     # Deriv minimum (was mistakenly 1.0 -- see v3 notes
                              # below; MIN_NET_PAYOUT was already calibrated
                              # against $0.35 and MC_FAIR_ODDS_CEIL's own
                              # "~0.658" comment only checks out at 0.35, so
                              # this corrects a pre-existing inconsistency,
                              # not just a v3 preference)
MIN_NET_PAYOUT    = 0.11      # v6 (2026-07-29): was 0.182 (52% of $0.35).
                               # User-specified correction: minimum payout
                               # per $0.35 stake should be $0.11 (~31%), not
                               # $0.182. Reference value only now (see
                               # MC_FAIR_ODDS_CEIL below and rank_candidates_
                               # by_ev's edge_frac gating) -- not a hard
                               # floor by itself, but BARRIER_SIGMAS below is
                               # now calibrated so real Deriv payouts land
                               # around this figure instead of $0.01-0.04.
WATCHDOG_TIMEOUT  = 15 * 60  # 15 min — accounts for GARCH + bootstrap time
HISTORY_BOOTSTRAP = 5000
MIN_TICKS_FOR_FIT = 200
MIN_TICKS_LIVE    = 60
GARCH_SCALE       = 1000.0   # scale factor for GARCH fitting on relative returns

# ── Martingale staking (per-symbol, independent streak tracking) ──────────
MG_ENABLED        = True
MG_TRIGGER_LOSSES = 1      # only escalate after this many CONSECUTIVE losses
MG_MAX_STEPS      = 3      # cap — step 4 onward stays at step-3 stake
MG_FACTOR         = 1.22
MG_MAX_STAKE      = BASE_STAKE * (MG_FACTOR ** MG_MAX_STEPS) * 1.05  # hard ceiling
                                                                       # (safety margin
                                                                       # for rounding)

# ── Continuous edge-scaled staking (v3) ────────────────────────────────────
# Replaces the old binary "net_payout >= MIN_NET_PAYOUT or discard" gate.
# edge_frac = ev / ask_price (expected return on stake, from a LIVE Deriv
# proposal) is a unit-normalized edge measure, comparable across candidates
# regardless of stake. Any candidate with edge_frac > MIN_EDGE_FRAC trades;
# how strong the edge is scales the stake continuously instead of being a
# pass/fail cliff.
#
# IMPORTANT ASYMMETRY: Deriv's own minimum stake is BASE_STAKE ($0.35) --
# that is a hard platform floor, not a tunable. "Let marginal edges trade
# small instead of discarding them" therefore can't mean sub-$0.35 stakes; it
# means marginal-but-positive edges trade AT the $0.35 floor instead of being
# thrown out, while stronger edges scale the stake UP from that floor
# (capped at STAKE_MULT_MAX, independent of and applied on top of the
# existing martingale loss-recovery scaling).
MIN_EDGE_FRAC     = 0.02   # minimum EV/stake ratio to trade at all -- guards
                            # against calibration noise, not a profitability
                            # bar (any genuinely positive edge above this
                            # trivial floor is worth taking at $1).
TARGET_EDGE_FRAC  = 0.15   # edge_frac at which a candidate earns full
                            # STAKE_MULT_MAX sizing; scales linearly below this.
STAKE_MULT_MIN    = 1.00   # never below Deriv's own stake floor
STAKE_MULT_MAX    = 1.75   # cap on edge-driven upsizing (separate from and
                            # multiplicative with martingale recovery scaling)

# Explicit hard-dollar ceiling on any SINGLE trade's stake, independent of
# and enforced AFTER all the martingale/edge-scaling math above. A
# deliberate second, dumb, unconditional backstop: account-size risk
# hygiene that holds even if a future change to any of the constants above
# (or a bug in how they combine) would otherwise let a single stake climb
# further than intended. Override via env var if your account size can
# support more.
ABS_MAX_STAKE = float(os.getenv("ABS_MAX_STAKE", "2.00"))

# v15 (2026-07-31): stake growth of ANY kind (martingale escalation, edge-
# scaled upsizing) is disabled below this account balance -- trades flat
# at BASE_STAKE until the account has built a cushion. See next_stake()'s
# docstring for the real-log evidence motivating this (constant escalation
# to the stake cap after 6 consecutive losses, ~31% realized win rate, on
# what was likely still a small account).
STAKE_GROWTH_UNLOCK_BALANCE = float(os.getenv("STAKE_GROWTH_UNLOCK_BALANCE", "15.00"))

# ── Signal confirmation (reduces trade frequency / false positives) ───────
CONFIRM_REQUIRED      = 2      # consecutive passes the top candidate must survive
CONFIRM_MIN_GAP_SECS  = 60     # minimum time between confirmation checks
CONFIRM_MAX_AGE_SECS  = 600    # abandon a confirmation streak if it's been open
                                # this long without completing (stale signal)
CONFIRM_DUR_TOLERANCE = 60     # candidate is "the same" signal if its duration
                                # is within this many seconds of the prior pick
CONFIRM_SIGMA_TOLERANCE = 0.15 # and its barrier_sigma is within this of the prior pick

# ── MC engine ─────────────────────────────────────────────────────────────
MC_SIMULATIONS   = 75_000
MC_CI_PERCENTILE = 5
# v3: these are now a LOOSE analytical pre-filter only, not the accept/reject
# decision. Their old job (guaranteeing a candidate can clear MIN_NET_PAYOUT)
# is now done for real against Deriv's live proposal in rank_candidates_by_ev
# -- that's the only place house margin is actually known. Keeping a wide
# band here just avoids burning proposal-API calls on candidates that are
# obviously a coin flip (win_prob too low) or obviously unpriceable
# ("no return" territory, win_prob too high). See MC_FAIR_ODDS_CEIL below
# for where the old tight ceiling came from.
MC_REQUIRED_WIN  = 0.50
MC_REQUIRED_CI   = 0.46
MC_FAIR_ODDS_CEIL = BASE_STAKE / (BASE_STAKE + MIN_NET_PAYOUT)   # ~ 0.658 --
                                                 # kept only as a reference
                                                 # point / log annotation now.
MC_MAX_WIN_PROB   = 0.90   # wide sanity ceiling; the real "no return" cutoff
                            # is enforced live via the proposal API, not
                            # estimated here.
MC_BATCH_SIZE    = 75_000

# ── Sweep grids ───────────────────────────────────────────────────────────
# v3: widened to 30s steps end-to-end (was a hand-picked 7-value list) so the
# optimizer samples far more of the duration space per cycle -- this is the
# "widen the opportunity set" fix, independent of the edge-bar changes above.
DURATION_CANDIDATES = list(range(120, 481, 30))   # v15: was 60,90,...,480;
                                                     # floor raised to 120s
                                                     # per explicit request --
                                                     # 120,150,...,480 (13 values)
# Barrier expressed as multiples of terminal vol. v3: finer step (0.05 instead
# of the old hand-picked list) across the same overall range, plus extending
# down to 0.30 now that the win_prob ceiling isn't artificially truncating
# the useful zone.
# v6 (2026-07-29): re-centered based on real log data + explicit user
# calibration. The 2026-07-29 log showed EV ranking consistently selecting
# the WIDEST sigmas in the old 0.30-2.00 grid (1.35-2.00 observed) because
# those have the highest win_prob and win_prob drives weighted_score/
# candidate ranking before live pricing is known -- but Deriv's real payout
# at those widths was $0.01-0.04 on a $0.35 stake (some even "offers no
# return"), giving strongly negative edge_frac every time (down to -0.078).
# The barriers were too safe/wide to be worth trading, and the ranking logic
# was actively steering toward exactly that failure mode.
#
# User-specified fix: absolute barrier should be +-1.69 to +-2.30 (price
# units) at the 2-minute (120s) duration. Converting via that cycle's own
# logged abs_vol_per_tick (0.16998, GARCH): vol_terminal(120s) =
# 0.16998 * sqrt(120) = 1.862, so target sigma = barrier_abs / vol_terminal
# = 1.69/1.862 .. 2.30/1.862 = 0.91 .. 1.24. Since sigma is ALREADY defined
# relative to vol_terminal (which itself scales as sqrt(duration)), this
# same sigma window is the best available generalization to every other
# duration in DURATION_CANDIDATES too -- vol_terminal does the duration
# scaling automatically, sigma is what stays constant across durations for
# a given "how strict" setting.
#
# Grid: 0.85-1.35 (mild padding either side of the 0.91-1.24 target so EV
# ranking still has real candidates to compare, not a single point) in 0.025
# steps -- finer granularity than the old grid despite covering under a
# third of its range, so this isn't a step backward on "widen the
# opportunity set" from the earlier fix, just correctly relocated.
BARRIER_SIGMAS = [round(0.85 + 0.025 * i, 3) for i in range(21)]   # 0.85..1.35
BARRIER_ABS_MIN = 0.3    # minimum absolute barrier (price units)

# ── Asymmetric barrier config ─────────────────────────────────────────────
# Bias signal: measured over this many ticks of recent price history
BIAS_LOOKBACK      = 60      # ticks to measure directional drift
BIAS_MAX           = 0.35    # cap bias magnitude (prevents degenerate barriers)
# v15 (2026-07-31): symmetric-only trades, per explicit request -- asymmetric
# barriers add a second dimension of variance to what the LSTM classifier
# and MC engine both need to learn/estimate well, on top of duration and
# barrier width. Collapsing ASYM_RATIO_GRID to a single (1.0, 1.0) pair
# means asym_pairs below always has exactly one entry -- every candidate is
# a symmetric barrier, and the whole asymmetric sweep/scoring machinery
# just runs once per duration instead of over a 7x7 grid (also a nice
# side-effect: ~49x fewer candidates to evaluate per duration, meaning more
# compute budget per cycle for what's left). If you want asymmetric
# barriers back later, restore the original 7-value grid here -- nothing
# else needs to change.
ASYM_RATIO_GRID    = [1.00]
# Neither side can be less than this fraction of the symmetric barrier_abs
ASYM_SIDE_MIN_FRAC = 0.60

# ── Directional overlay (CALL/PUT alongside EXPIRYRANGE) ─────────────────
# Only fires when |bias| hits BIAS_MAX exactly (saturated cap = max observed
# drift signal). At lower bias values the directional edge is too weak on
# these mean-reverting synthetic indices — data showed |bias| barely reached
# 0.025 on average; only cap-saturated events are worth betting directionally.
DIR_OVERLAY_ENABLED    = True
DIR_OVERLAY_BIAS_FLOOR = 0.004             # |bias| must be >= this to trigger
                                             # the overlay. Derived from actual
                                             # trade data (129 trades): max
                                             # observed bias was 0.0254, only 2
                                             # trades exceeded 0.020 (top 1.5%).
                                             # BIAS_MAX=0.35 cap was never hit —
                                             # using it would mean overlay never
                                             # fires. 0.020 = the real top-end
                                             # signal on these symbols.
DIR_OVERLAY_STAKE_FRAC = 0.50              # CALL/PUT stake = 50% of EXPIRYRANGE
                                             # stake (secondary position)
DIR_OVERLAY_MIN_PAYOUT = 0.05              # lower payout floor for CALL/PUT

# ── Per-symbol gate thresholds ────────────────────────────────────────────
SYMBOL_CONFIG = {
    "1HZ10V": {
        "ticks_per_sec":     1.0,
        "max_adx":           13,    # v11 (2026-07-30, 1h48m/2002-gate-check log):
                                     # was 10. That earlier fix was based on only 4
                                     # minutes of data and said as much ("enough to
                                     # see it was miscalibrated, not enough to
                                     # perfectly recalibrate"). This log showed real
                                     # ADX up to 14.0 (mean 10.9 among blocks) --
                                     # raising to 13 lets ~94% of previously-blocked
                                     # fail_adx cases through while still catching
                                     # genuinely extreme readings above what's been
                                     # observed.
        "min_vol_trust":     0.85,
        "max_mbs":           0.60,  # v11: was 0.50. Observed failing values 0.50-0.62
                                     # (mean 0.54) -- same "threshold sits inside the
                                     # normal range" symptom, not a rare-tail one.
                                     # 0.60 lets ~97% of previously-blocked cases through.
        "boll_width_factor": 1.80,  # v11: was 1.20 -- this turned out to be the
                                     # DOMINANT blocker in the 2026-07-30 log (1923/2002
                                     # blocked gate checks, 96%), not a rare-spike
                                     # filter as the v5 comment assumed. Real
                                     # cur_std/med_std for 1HZ10V ranged 1.20-4.49
                                     # (median 1.52, 5th percentile only 1.221) --
                                     # i.e. "current vol > 1.2x medium-term median"
                                     # is closer to this instrument's NORMAL state
                                     # than an exceptional one. 1.80 lets ~76% of
                                     # previously-blocked cases through while still
                                     # blocking the top quartile of genuine spikes
                                     # (up to 4.49x). Re-check against a longer log --
                                     # same caveat as the max_adx note above.
        "max_hawkes":        0.50,
        "cooldown_secs":     150,
        "barrier_dp":        2,    # Deriv: max 2 decimal places for 1HZ10V barriers
    },
}

DAILY_TUNE_HOUR_UTC = 0

# ── Market regime detection ──────────────────────────────────────────────
# structural_gate() already computes ADX / Hawkes / MBS / Bollinger-width
# per evaluation cycle, but historically only used them as a binary pass/
# fail gate. classify_regime() turns those same readings (expressed as
# fractions of each symbol's own gate thresholds, so it's comparable across
# symbols with different SYMBOL_CONFIG limits) into one of four explicit
# states, and mc_auto_optimize() uses the state to gate which duration /
# barrier_sigma combos are even considered -- instead of always sweeping the
# full grid regardless of conditions. This directly targets the pattern
# flagged in the daily self-improvement log ("51 trades flagged negative EV
# at entry actually won 64.7%... positive EV trades won only 33.3%"), which
# looks like a strategy shape (duration/sigma choice) being right for one
# regime and wrong for another, not a calibration problem.
REGIME_TREND        = "TREND"         # ADX elevated relative to this symbol's own ceiling
REGIME_HIGH_VOL      = "HIGH_VOL"     # Hawkes clustering elevated and/or Bollinger width expanding
REGIME_LOW_VOL       = "LOW_VOL"      # both ADX and Hawkes low, Bollinger width contracted
REGIME_MEAN_REVERT   = "MEAN_REVERT"  # default/friendliest regime for EXPIRYRANGE

# Duration/sigma subsets swept per regime. Falling through to the full grid
# in MEAN_REVERT (the regime EXPIRYRANGE is fundamentally built for) keeps
# today's behaviour there; the other three regimes narrow the sweep to the
# combos that make structural sense for that regime, which also cuts the
# number of MC calls per cycle (relevant now that MC uses bootstrap
# resampling instead of a single vectorized Gaussian draw -- see
# mc_asymmetric_estimate).
REGIME_DURATIONS = {
    REGIME_TREND:       [120, 180, 240],
    REGIME_HIGH_VOL:    [120, 180],
    REGIME_LOW_VOL:     [300, 360, 420, 480, 240],
    REGIME_MEAN_REVERT: DURATION_CANDIDATES,
}
REGIME_SIGMAS = {
    # Trending / high-vol: price is more likely to travel far, so only
    # consider the wider barriers -- narrow ones are a coin flip against a
    # real directional or volatility move, not a genuine edge.
    REGIME_TREND:       [0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00],
    REGIME_HIGH_VOL:    [0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00],
    # Low-vol: price isn't going far, so narrow barriers are still safe and
    # pay much better than in the other regimes.
    REGIME_LOW_VOL:     [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90],
    REGIME_MEAN_REVERT: BARRIER_SIGMAS,
}


def classify_regime(symbol: str, gate_info: dict) -> str:
    """
    Classifies the current market state from the SAME readings
    structural_gate() already computed this cycle (no extra indicator
    work). Thresholds are expressed as fractions of each symbol's own
    SYMBOL_CONFIG gate ceilings so TREND/HIGH_VOL mean the same relative
    thing on 1HZ10V and RDBEAR even though their raw ADX/Hawkes ranges
    differ.

    Because structural_gate() must have already passed (gate_ok=True) for
    this function to be called, adx_frac and hawkes_frac are always < 1.0
    by construction -- this classifies WHERE inside the passing band the
    market currently sits, not whether it passed.
    """
    cfg = SYMBOL_CONFIG[symbol]
    adx_frac    = gate_info.get("adx_val", 0.0)    / max(cfg["max_adx"],    1e-9)
    hawkes_frac = gate_info.get("hawkes_val", 0.0) / max(cfg["max_hawkes"], 1e-9)
    cur_std     = gate_info.get("cur_std", 0.0)
    med_std     = gate_info.get("med_std", 0.0)
    boll_ratio  = (cur_std / med_std) if med_std > 0 else 1.0
    # boll_ratio's natural "calm" baseline is 1.0 (current vol == historical
    # median), and structural_gate only allows up to boll_width_factor
    # (~1.15-1.20) above that -- a much narrower band than ADX/Hawkes range
    # from 0. So rather than comparing boll_ratio to the raw ceiling
    # (which would misclassify anything at-or-above the 1.0 baseline as
    # HIGH_VOL), measure how far it has moved from 1.0 TOWARD that ceiling.
    boll_elevation = max(0.0, (boll_ratio - 1.0) /
                          max(cfg["boll_width_factor"] - 1.0, 1e-9))

    # SYMBOL_CONFIG's own tuning notes record that ADX normally sits at
    # 0.57-1.0 of max_adx even under everyday passing conditions (observed
    # live range 5.1-9.2 against max_adx=9, mean 7.1 -> frac ~0.79) -- so a
    # 0.60 cutoff would classify most cycles as TREND. These thresholds are
    # set high enough (>=0.85) that only the genuinely elevated tail near
    # the gate ceiling counts as a distinct regime; "typical" conditions
    # fall through to MEAN_REVERT, which matches EXPIRYRANGE's actual
    # design assumption (range-bound) for the common case.
    if hawkes_frac >= 0.75 or boll_elevation >= 0.70:
        return REGIME_HIGH_VOL
    if adx_frac >= 0.85:
        return REGIME_TREND
    if boll_ratio <= 0.60 and hawkes_frac <= 0.25:
        return REGIME_LOW_VOL
    return REGIME_MEAN_REVERT


# =============================================================================
# SUPABASE STORE
# =============================================================================
class SupabaseStore:
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.ok  = bool(self.url and self.key)
        print(f"[Store] {'Active -> ' + self.url if self.ok else 'No creds — state will not persist.'}")

    def _hdr(self, prefer="return=minimal"):
        return {"apikey": self.key, "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json", "Prefer": prefer}

    def _upsert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                headers=self._hdr("resolution=merge-duplicates,return=minimal"),
                json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} upsert {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} upsert error: {e}")

    def _insert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                headers=self._hdr(), json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} insert {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} insert error: {e}")

    def _select(self, table, query="select=*"):
        if not self.ok: return []
        try:
            r = requests.get(f"{self.url}/rest/v1/{table}?{query}",
                headers=self._hdr("return=representation"), timeout=12)
            if r.status_code == 200: return r.json()
            print(f"[Store] {table} select {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} select error: {e}")
        return []

    def log_trade(self, rec: dict):
        """
        Maps to the FULL bot_expiryrange_log schema (see accompanying SQL).
        Covers symmetric legacy fields plus asymmetric-barrier, martingale,
        and MC-diagnostic fields added in this revision. Uses .get() with
        safe defaults so older callers (or partial records) don't crash.
        """
        payload = {
            "ts":                   datetime.now(timezone.utc).isoformat(),
            "symbol":               rec["symbol"],
            "entry_price":          round(float(rec["entry_price"]),     5),
            "upper_barrier":        round(float(rec["upper_barrier"]),   5),
            "lower_barrier":        round(float(rec["lower_barrier"]),   5),
            "barrier_width":        round(float(rec["barrier_width"]),   5),
            "upper_abs":            round(float(rec.get("upper_abs", 0)),      5),
            "lower_abs":            round(float(rec.get("lower_abs", 0)),      5),
            "upper_ratio":          round(float(rec.get("upper_ratio", 1.0)),  4),
            "lower_ratio":          round(float(rec.get("lower_ratio", 1.0)),  4),
            "bias":                 round(float(rec.get("bias", 0.0)),         4),
            "drift_per_tick":       round(float(rec.get("drift_per_tick", 0)), 6),
            "drift_total":          round(float(rec.get("drift_total", 0)),    6),
            "duration_secs":        int(rec["duration_secs"]),
            "n_steps":              int(rec.get("n_steps", 0)),
            "stake":                round(float(rec.get("stake", BASE_STAKE)), 4),
            "mg_step":              int(rec.get("mg_step", 0)),
            "mg_active":            bool(rec.get("mg_active", False)),
            "consec_losses_before": int(rec.get("consec_losses_before", 0)),
            "won":                  bool(rec["won"]),
            "profit":               round(float(rec["profit"]),          4),
            "ask_price":            round(float(rec.get("ask_price", 0)), 4),
            "breach_prob":          round(float(rec["breach_prob"]),     4),
            "win_prob":             round(float(rec.get("win_prob", 0)), 4),
            "ci_lower":             round(float(rec.get("ci_lower", 0)), 4),
            "weighted_score":       round(float(rec.get("weighted_score", 0)), 6),
            "ev_conservative":      round(float(rec.get("ev_conservative", 0)), 4),
            "ev_optimistic":        round(float(rec.get("ev_optimistic",  0)), 4),
            "vol_per_tick":         round(float(rec["vol_per_tick"]),    6),
            "vol_terminal":         round(float(rec.get("vol_terminal", 0)), 6),
            "barrier_sigma":        round(float(rec.get("barrier_sigma", 0)), 4),
            "used_garch":           bool(rec["used_garch"]),
            "adx_val":              round(float(rec.get("adx_val", 0)), 3),
            "vol_trust":            round(float(rec.get("vol_trust", 0)), 4),
            "hawkes_intensity":     round(float(rec.get("hawkes_val", 0)), 4),
            "n_sims":               int(MC_SIMULATIONS),
        }
        self._insert("bot_expiryrange_log", payload)

    def save_config(self, key, value):
        self._upsert("bot_expiryrange_config",
            {"key": key, "value": json.dumps(value),
             "updated_at": datetime.now(timezone.utc).isoformat()})

    def load_config(self, key):
        rows = self._select("bot_expiryrange_config", f"select=value&key=eq.{key}")
        if rows:
            raw = rows[0]["value"]
            return json.loads(raw) if isinstance(raw, str) else raw
        return None

    def load_lstm_model(self):
        """v8: fetches the latest trained LSTM state_dict, written by the
        separate expiryrange_lstm_train.py cron service. Returns
        (state_dict_b64, trained_at) or (None, None) if no model has been
        trained yet or the request fails -- callers must treat this as
        optional and fall back gracefully (see load_or_refresh_lstm())."""
        try:
            rows = self._select(
                "bot_expiryrange_lstm_model",
                "select=state_dict_b64,trained_at,val_loss&key=eq.current")
            if rows:
                return rows[0]["state_dict_b64"], rows[0]["trained_at"]
        except Exception as e:
            print(f"[LSTM] Supabase fetch failed: {e}")
        return None, None

    def save_daily_summary(self, date_str, symbol, n, wins, profit, best_dur, best_bar):
        self._upsert("bot_expiryrange_daily", {
            "date_utc":     date_str, "symbol": symbol,
            "n_trades":     n, "n_wins": wins,
            "win_rate":     round(wins / max(n, 1), 4),
            "total_profit": round(float(profit), 4),
            "best_duration": int(best_dur),
            "best_barrier":  round(float(best_bar), 4),
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        })

    def load_recent_trades(self, symbol, days=7):
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return self._select("bot_expiryrange_log",
            f"select=*&symbol=eq.{symbol}&ts=gte.{since}&order=ts.asc")

    def log_overlay(self, rec: dict):
        """Logs a directional overlay (CALL/PUT) trade to bot_overlay_log."""
        payload = {
            "ts":              datetime.now(timezone.utc).isoformat(),
            "symbol":          rec["symbol"],
            "direction":       rec["direction"],
            "entry_price":     round(float(rec["entry_price"]),      5),
            "duration_secs":   int(rec["duration_secs"]),
            "stake":           round(float(rec["stake"]),            4),
            "bias":            round(float(rec["bias"]),             5),
            "bias_floor_used": round(float(rec["bias_floor_used"]),  4),
            "er_win_prob":     round(float(rec["er_win_prob"]),      4),
            "er_upper_ratio":  round(float(rec["er_upper_ratio"]),   4),
            "er_lower_ratio":  round(float(rec["er_lower_ratio"]),   4),
            "won":             bool(rec["won"]),
            "profit":          round(float(rec["profit"]),           4),
            "ask_price":       round(float(rec.get("ask_price", 0)), 4),
        }
        self._insert("bot_overlay_log", payload)


# =============================================================================
# DERIV CLIENT  (identical connection layer from parent bot)
# =============================================================================
class DerivClient:
    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE     = 2.0
    RECONNECT_CAP      = 60.0

    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id        = app_id
        self.token         = token
        self.account_type  = account_type
        self.account_id    = account_id
        self.ws            = None
        self.req_id        = 0
        self.pending: dict = {}
        self.subscriptions = defaultdict(list)
        self.account       = None
        self.resubscribe_cb = None
        self._running      = False
        self._reader_task  = None
        self._ka_task      = None

    def _rest_headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Deriv-App-ID": self.app_id,
                "Content-Type": "application/json"}

    def _resolve_account_id_sync(self):
        resp = requests.get(f"{API_BASE}{ACCOUNTS_PATH}",
                            headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == self.account_type:
                aid = acc.get("account_id") or acc.get("id")
                if aid:
                    return aid
        raise RuntimeError(f"No '{self.account_type}' account found. data={data}")

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            print(f"Resolved {self.account_type} account_id = {self.account_id}")
        resp = requests.post(
            f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}",
            headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url  = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    async def connect(self):
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task     = asyncio.create_task(self._heartbeat())
        bal          = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(f"Connected ({self.account_type}). "
              f"loginid={self.account.get('loginid')} "
              f"balance=${self.account.get('balance'):.2f}")

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[Client] WS lost: {e}")

    async def _supervise(self):
        while self._running:
            if self._reader_task:
                await self._reader_task
            if self._ka_task:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WS disconnected"))
            self.pending.clear()
            self.ws = None
            if not self._running:
                break
            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(self.RECONNECT_BASE * (2 ** (attempt - 1)),
                            self.RECONNECT_CAP) + random.uniform(0, 1)
                print(f"[Client] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[Client] Reconnect failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id   = data.get("req_id")
        msg_type = data.get("msg_type")
        if msg_type == "ping":
            return
        if req_id is not None and req_id in self.pending:
            fut = self.pending.pop(req_id)
            if not fut.done():
                fut.set_result(data)
                return
        if msg_type in self.subscriptions:
            for q in self.subscriptions[msg_type]:
                q.put_nowait(data)

    async def send(self, request, timeout=20):
        self.req_id += 1
        rid = self.req_id
        request = {**request, "req_id": rid}
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q


# =============================================================================
# SYMBOL DATA BUFFER
# =============================================================================
class SymbolData:
    def __init__(self, symbol, maxlen=8000):
        self.symbol = symbol
        self.ticks  = deque(maxlen=maxlen)

    def add_tick(self, epoch, price):
        self.ticks.append((float(epoch), float(price)))

    def prices(self) -> np.ndarray:
        return np.array([p for _, p in self.ticks], dtype=float)

    def price_diffs(self) -> np.ndarray:
        """Absolute price differences (not returns) — used for vol estimation."""
        p = self.prices()
        return np.diff(p) if len(p) >= 2 else np.array([])

    def returns(self) -> np.ndarray:
        """Relative returns — used for GARCH fitting."""
        p = self.prices()
        if len(p) < 2: return np.array([])
        return np.diff(p) / p[:-1]


# =============================================================================
# BOT STATE
# =============================================================================
class BotState:
    def __init__(self):
        self.balance          = 0.0
        self.trading_locked   = False
        self.last_activity    = time.time()
        self.last_price: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.last_trade_time: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.last_daily_tune  = 0.0
        self.garch_cache: Dict[str, tuple]  = {}     # sym -> (result, fitted_at)
        self.vol_scalar: Dict[str, float]   = {s: 1.0 for s in SYMBOLS}
        # v8: LSTM terminal-price model, hot-reloaded from Supabase -- see
        # load_or_refresh_lstm(). None until the first successful load; the
        # optimizer falls back to bootstrap+HMM/GBM only until then.
        self.lstm_model             = None
        self.lstm_trained_at: Optional[str] = None
        self.lstm_last_check_time   = 0.0
        # Cold-start priors from the first 81-trade sample (pre-self-improvement):
        # 120s = 70.6% win/+0.96 net, 300s = 68.2% win/+0.70 net -> favoured.
        # 180s/360s/480s were net-negative -> down-weighted until real data overrides.
        _COLD_START_DUR_WEIGHTS = {120: 2.0, 300: 1.6, 240: 1.0,
                                    420: 1.0, 180: 0.5, 360: 0.5, 480: 0.5}
        self.duration_weights: Dict[str, Dict[int, float]]   = {
            s: dict(_COLD_START_DUR_WEIGHTS) for s in SYMBOLS}
        self.barrier_weights:  Dict[str, Dict[float, float]] = {s: {} for s in SYMBOLS}
        self.session_trades:  Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.session_wins:    Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.session_profit:  Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        # Directional overlay session tracking (separate from EXPIRYRANGE stats)
        self.overlay_trades:  Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.overlay_wins:    Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.overlay_profit:  Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        # Martingale tracking — independent per symbol, resets on any win
        self.consec_losses:  Dict[str, int] = {s: 0 for s in SYMBOLS}
        self.mg_step:        Dict[str, int] = {s: 0 for s in SYMBOLS}
        # Signal confirmation tracking — independent per symbol. A "streak"
        # is a run of consecutive evaluation passes (>=60s apart) where the
        # top-ranked MC candidate stayed roughly the same (duration + sigma
        # within tolerance). Reaching CONFIRM_REQUIRED passes clears the gate.
        self.confirm_streak:    Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.confirm_last_ts:   Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.confirm_started:   Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.confirm_signature: Dict[str, tuple] = {s: None for s in SYMBOLS}

    def check_confirmation(self, symbol: str, top_candidate: dict) -> Tuple[bool, dict]:
        """
        Returns (confirmed, info). `confirmed=True` means top_candidate has
        now survived CONFIRM_REQUIRED consecutive evaluation passes, each
        at least CONFIRM_MIN_GAP_SECS apart, with a consistent signal
        signature (duration + barrier_sigma within tolerance). Otherwise
        increments/resets the streak and returns False so the caller waits
        for the next polling cycle instead of trading immediately.

        This does NOT block the event loop — it relies on the natural
        per-symbol polling cadence already in the main loop, so the other
        symbol keeps trading normally while this one is "pending confirm".
        """
        now = time.time()
        sig = (round(top_candidate["duration_secs"] / CONFIRM_DUR_TOLERANCE),
               round(top_candidate["barrier_sigma"] / CONFIRM_SIGMA_TOLERANCE))

        prev_sig   = self.confirm_signature.get(symbol)
        started_at = self.confirm_started.get(symbol, 0.0)
        last_ts    = self.confirm_last_ts.get(symbol, 0.0)

        # Reset if streak went stale (signal hasn't been seen recently) or
        # took too long overall (market likely moved on).
        if started_at and (now - started_at) > CONFIRM_MAX_AGE_SECS:
            self.confirm_streak[symbol]    = 0
            self.confirm_signature[symbol] = None
            started_at = 0.0

        # Enforce minimum spacing between confirmation checks — if we're
        # being polled faster than CONFIRM_MIN_GAP_SECS, this pass doesn't
        # count yet (avoid the loop's natural sub-second cadence padding
        # the streak with checks that are essentially simultaneous).
        if last_ts and (now - last_ts) < CONFIRM_MIN_GAP_SECS:
            info = {"streak": self.confirm_streak.get(symbol, 0),
                    "required": CONFIRM_REQUIRED,
                    "reason": f"waiting {CONFIRM_MIN_GAP_SECS - (now - last_ts):.0f}s "
                              f"for next confirm check"}
            return False, info

        same_signal = (prev_sig == sig)

        if same_signal:
            self.confirm_streak[symbol] = self.confirm_streak.get(symbol, 0) + 1
        else:
            self.confirm_streak[symbol]    = 1
            self.confirm_signature[symbol] = sig
            self.confirm_started[symbol]   = now

        self.confirm_last_ts[symbol] = now
        streak = self.confirm_streak[symbol]

        info = {
            "streak":      streak,
            "required":    CONFIRM_REQUIRED,
            "same_signal": same_signal,
            "dur":         top_candidate["duration_secs"],
            "sigma":       top_candidate["barrier_sigma"],
        }

        if streak >= CONFIRM_REQUIRED:
            # Confirmed — reset tracking so the next signal starts fresh
            self.confirm_streak[symbol]    = 0
            self.confirm_signature[symbol] = None
            self.confirm_started[symbol]   = 0.0
            return True, info

        return False, info

    def next_stake(self, symbol: str) -> float:
        """
        Returns the stake to use for the NEXT trade on this symbol.
        Escalates only after MG_TRIGGER_LOSSES consecutive losses, capped at
        MG_MAX_STEPS, multiplied by MG_FACTOR per step. Resets to BASE_STAKE
        the instant a win occurs (see record_trade_result).

        v15 (2026-07-31): stake growth of ANY kind (martingale escalation
        here, and edge-scaled upsizing in edge_scaled_stake() below) is
        gated behind STAKE_GROWTH_UNLOCK_BALANCE -- per explicit request,
        the account trades flat at BASE_STAKE until it's built up enough of
        a cushion to absorb a losing streak without the stake climbing on
        top of a still-small balance. Real log evidence motivating this:
        a 31% realized win rate with stakes constantly escalating to the
        $1.12 cap after 6 consecutive losses, on what was likely still a
        small account -- profitable so far, but a rough combination for a
        long losing streak. Once balance clears the threshold, behaves
        exactly as before.
        """
        if not MG_ENABLED:
            return BASE_STAKE
        if self.balance < STAKE_GROWTH_UNLOCK_BALANCE:
            return BASE_STAKE
        cl = self.consec_losses.get(symbol, 0)
        if cl < MG_TRIGGER_LOSSES:
            return BASE_STAKE
        step  = min(self.mg_step.get(symbol, 0) + 1, MG_MAX_STEPS)
        stake = BASE_STAKE * (MG_FACTOR ** step)
        return round(min(stake, MG_MAX_STAKE, ABS_MAX_STAKE), 2)

    def edge_scaled_stake(self, symbol: str, edge_frac: float) -> float:
        """
        v3: applies continuous edge-based scaling on TOP of next_stake()'s
        martingale-aware base. edge_frac below MIN_EDGE_FRAC should already
        have been excluded by the caller (this function does not re-check
        that floor). Linear interpolation from STAKE_MULT_MIN at
        MIN_EDGE_FRAC to STAKE_MULT_MAX at TARGET_EDGE_FRAC, clamped beyond.

        v15: short-circuits to flat BASE_STAKE below STAKE_GROWTH_UNLOCK_
        BALANCE -- same gate as next_stake(), enforced here too so edge-
        based upsizing can't grow the stake independently of the martingale
        gate above.
        """
        if self.balance < STAKE_GROWTH_UNLOCK_BALANCE:
            return BASE_STAKE
        base = self.next_stake(symbol)
        if TARGET_EDGE_FRAC <= MIN_EDGE_FRAC:
            mult = STAKE_MULT_MIN
        else:
            t = (edge_frac - MIN_EDGE_FRAC) / (TARGET_EDGE_FRAC - MIN_EDGE_FRAC)
            mult = STAKE_MULT_MIN + t * (STAKE_MULT_MAX - STAKE_MULT_MIN)
            mult = float(np.clip(mult, STAKE_MULT_MIN, STAKE_MULT_MAX))
        return round(min(base * mult, MG_MAX_STAKE * STAKE_MULT_MAX, ABS_MAX_STAKE), 2)

    def record_trade_result(self, symbol: str, won: bool):
        """Updates consecutive-loss streak and martingale step for a symbol."""
        if won:
            self.consec_losses[symbol] = 0
            self.mg_step[symbol]       = 0
        else:
            self.consec_losses[symbol] = self.consec_losses.get(symbol, 0) + 1
            if self.consec_losses[symbol] >= MG_TRIGGER_LOSSES:
                self.mg_step[symbol] = min(
                    self.mg_step.get(symbol, 0) + 1, MG_MAX_STEPS)


# =============================================================================
# VOL ESTIMATION  — FIX FOR BUG 1
# =============================================================================
def estimate_abs_vol_per_tick(prices: np.ndarray, returns: np.ndarray,
                               garch_result, price_now: float) -> Tuple[float, float, bool]:
    """
    Returns (abs_vol_per_tick, vol_trust, used_garch).

    GARCH is fitted on relative returns * GARCH_SCALE.
    Conditional vol from GARCH is therefore in units of (relative_return * GARCH_SCALE).
    To get absolute price vol per tick:
        abs_vol = (garch_cond_vol / GARCH_SCALE) * price_now

    Sanity check: abs_vol must be between 0.001% and 5% of price_now.
    If outside that range, fall back to std(price_diffs) directly.
    """
    price_diffs  = np.diff(prices) if len(prices) >= 2 else np.array([0.0])
    baseline_abs = float(np.std(price_diffs)) if len(price_diffs) > 5 else abs(price_now) * 0.001
    baseline_abs = max(baseline_abs, 1e-6)

    lo_bound = price_now * 0.00001   # 0.001% of price
    hi_bound = price_now * 0.05      # 5% of price

    used_garch = False
    vol_trust  = 0.5

    if garch_result is not None:
        try:
            fc = garch_result.forecast(horizon=1, reindex=False)
            garch_cond_vol_scaled = math.sqrt(float(fc.variance.values[-1, 0]))
            # Convert back to absolute price units
            abs_vol = (garch_cond_vol_scaled / GARCH_SCALE) * price_now
            if lo_bound <= abs_vol <= hi_bound:
                # Compare to baseline to compute trust
                ratio     = abs_vol / max(baseline_abs, 1e-9)
                vol_trust = float(np.clip(1.0 / (1.0 + max(ratio - 1.0, 0) * 2), 0.1, 1.0))
                used_garch = True
                return float(abs_vol), vol_trust, used_garch
            else:
                print(f"[Vol] GARCH abs_vol={abs_vol:.5f} outside [{lo_bound:.5f},{hi_bound:.5f}]"
                      f" — falling back to baseline={baseline_abs:.5f}")
        except Exception as e:
            print(f"[Vol] GARCH forecast error: {e}")

    # Fallback: direct std of price differences
    return baseline_abs, 0.5, False


# =============================================================================
# GARCH FITTING
# =============================================================================
def fit_garch(returns: np.ndarray):
    """Fit GARCH(1,1) on relative returns * GARCH_SCALE. Returns result or None."""
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        scaled = returns * GARCH_SCALE
        am     = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                return am.fit(disp="off")
    except Exception as e:
        print(f"[GARCH] fit failed: {e}")
        return None


# =============================================================================
# STRUCTURAL INDICATORS
# =============================================================================
def compute_adx(prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
    if len(prices) < period * 2 + 2:
        return 20.0, 0.3
    tr_, pdm_, ndm_ = [], [], []
    for i in range(1, len(prices)):
        tr_.append(abs(prices[i] - prices[i-1]))
        pdm_.append(max(prices[i] - prices[i-1], 0.0))
        ndm_.append(max(prices[i-1] - prices[i], 0.0))
    tr_a  = np.array(tr_[-period*2:])
    pdm_a = np.array(pdm_[-period*2:])
    ndm_a = np.array(ndm_[-period*2:])
    atr   = np.mean(tr_a[-period:])
    if atr == 0:
        return 20.0, 0.3
    adx_vals = [
        100 * abs(pdm_a[i] - ndm_a[i]) / (np.mean(tr_a[max(0,i-period):i]) * period + 1e-9)
        for i in range(period, len(tr_a))
    ]
    adx = float(np.clip(np.mean(adx_vals) if adx_vals else 20.0, 0, 100))
    return adx, float(np.clip((adx - 20) / 30, 0, 1))


def compute_bollinger_width(prices: np.ndarray) -> Tuple[float, float]:
    if len(prices) < 30:
        return 1.0, 1.0
    cur = float(np.std(prices[-10:]))
    stds = [float(np.std(prices[i:i+10])) for i in range(0, len(prices)-10, 5)]
    return cur, float(np.median(stds)) if stds else cur


def compute_hawkes_proxy(price_diffs: np.ndarray) -> float:
    if len(price_diffs) < 20:
        return 0.0
    thresh  = 0.5 * np.std(price_diffs) if np.std(price_diffs) > 0 else 1e-9
    recent  = float(np.mean(np.abs(price_diffs[-20:]) > thresh))
    base    = float(np.mean(np.abs(price_diffs) > thresh)) if len(price_diffs) >= 50 else 0.1
    return float(np.clip((recent / max(base, 1e-6) - 1.0) / 3.0, 0.0, 1.0))


def compute_mbs(prices: np.ndarray, lookback: int = 50) -> float:
    if len(prices) < lookback + 5:
        return 0.0
    w    = prices[-lookback:]
    rng  = np.max(w) - np.min(w)
    if rng < 1e-9: return 0.0
    prev_hi = np.max(w[:-10])
    prev_lo = np.min(w[:-10])
    last    = prices[-1]
    return float(np.clip(max(max(0.0, last - prev_hi), max(0.0, prev_lo - last)) / rng, 0, 1))


# =============================================================================
# STRUCTURAL GATE — all 5 conditions must pass
# =============================================================================
def structural_gate(symbol: str, prices: np.ndarray, price_diffs: np.ndarray,
                    returns: np.ndarray, garch_result,
                    price_now: float) -> Tuple[bool, dict]:
    cfg  = SYMBOL_CONFIG[symbol]
    ok   = True
    info = {}

    # 1. ADX
    adx_val, _ = compute_adx(prices)
    info["adx_val"] = adx_val
    if adx_val > cfg["max_adx"]:
        ok = False
        info["fail_adx"] = f"ADX={adx_val:.1f} > {cfg['max_adx']}"

    # 2. Vol trust (need abs vol estimate first)
    abs_vol, vol_trust, _ = estimate_abs_vol_per_tick(prices, returns, garch_result, price_now)
    info["vol_trust"] = vol_trust
    info["abs_vol"]   = abs_vol
    if vol_trust < cfg["min_vol_trust"]:
        ok = False
        info["fail_vol"] = f"vol_trust={vol_trust:.3f} < {cfg['min_vol_trust']}"

    # 3. MBS
    mbs_val = compute_mbs(prices)
    info["mbs_val"] = mbs_val
    if mbs_val >= cfg["max_mbs"]:
        ok = False
        info["fail_mbs"] = f"mbs={mbs_val:.3f} >= {cfg['max_mbs']}"

    # 4. Bollinger width
    cur_std, med_std = compute_bollinger_width(prices)
    info["cur_std"] = cur_std
    info["med_std"] = med_std
    if med_std > 0 and cur_std > med_std * cfg["boll_width_factor"]:
        ok = False
        info["fail_boll"] = f"cur_std/med_std={cur_std/med_std:.2f} > {cfg['boll_width_factor']}"

    # 5. Hawkes
    hawkes_val = compute_hawkes_proxy(price_diffs)
    info["hawkes_val"] = hawkes_val
    if hawkes_val > cfg["max_hawkes"]:
        ok = False
        info["fail_hawkes"] = f"hawkes={hawkes_val:.3f} > {cfg['max_hawkes']}"

    return ok, info


# =============================================================================
# DIRECTIONAL BIAS SIGNAL
# =============================================================================
def compute_directional_bias(prices: np.ndarray, abs_vol_per_tick: float) -> float:
    """
    Returns a bias in [-1, +1].
      +1 = strong upward drift  → widen upper barrier, tighten lower
      -1 = strong downward drift → widen lower barrier, tighten upper
       0 = no detectable drift  → symmetric barriers

    Method: net displacement over BIAS_LOOKBACK ticks, normalised by the
    expected random-walk displacement (vol * sqrt(n)).  Capped at BIAS_MAX
    so extreme readings don't produce degenerate barriers.
    """
    n = min(BIAS_LOOKBACK, len(prices))
    if n < 10 or abs_vol_per_tick < 1e-9:
        return 0.0

    window        = prices[-n:]
    net_move      = float(window[-1] - window[0])
    expected_move = abs_vol_per_tick * math.sqrt(n)   # 1-sigma expected range

    raw_bias = net_move / max(expected_move, 1e-9)
    return float(np.clip(raw_bias, -BIAS_MAX / 0.01, BIAS_MAX / 0.01) * 0.01)


# =============================================================================
# MC ENGINE  — asymmetric: historical-return block bootstrap (replaces the
# old i.i.d. Gaussian terminal draw)
# =============================================================================
# Below this many ticks of history, a block bootstrap isn't trustworthy
# (too few distinct blocks -> effectively resampling the same handful of
# moves over and over), so we fall back to the old Gaussian draw exactly.
MIN_TICKS_FOR_BOOTSTRAP = 200
# Block length as a fraction of n_steps -- short enough that many distinct
# blocks exist in the history pool, long enough to preserve the tick-to-tick
# autocorrelation the Gaussian model discarded entirely.
BOOTSTRAP_BLOCK_FRACTION = 1 / 12
BOOTSTRAP_BLOCK_MIN = 5
BOOTSTRAP_BLOCK_MAX = 30


class BootstrapPool:
    """
    Precomputed resampling pool for ONE mc_auto_optimize() call.

    The historical price_diffs are demeaned (any embedded drift stripped
    out, since drift is injected separately and explicitly via
    drift_per_tick) and rescaled so their per-tick std matches
    target_vol_per_tick -- the SAME abs_vol used everywhere else this cycle
    (GARCH-derived, vol_scalar-calibrated). That keeps the calibrated vol
    level the rest of the system already trusts, while borrowing the
    empirical DISTRIBUTION SHAPE (fat tails, skew, clustering) instead of
    assuming normality like the old np.random.normal draw did.

    A cumulative-sum array lets every block-sum for every candidate this
    cycle be computed via cumsum[start+L] - cumsum[start] (O(batch*n_blocks))
    instead of materializing and summing every individual sampled tick
    (O(batch*n_blocks*block_length)) -- this is what keeps the full
    duration x sigma x asymmetry grid sweep fast enough to run every cycle,
    since this pool is built ONCE per symbol per cycle and reused across
    every candidate in the sweep (only n_steps/block_length change per
    candidate, not the underlying data).
    """
    def __init__(self, price_diffs: np.ndarray, target_vol_per_tick: float):
        self.usable = False
        n = len(price_diffs)
        if n < MIN_TICKS_FOR_BOOTSTRAP:
            return
        std_diffs = float(np.std(price_diffs))
        if std_diffs < 1e-12:
            return
        scale = target_vol_per_tick / std_diffs
        demeaned = (price_diffs - np.mean(price_diffs)) * scale
        self.demeaned = demeaned
        self.cumsum   = np.concatenate(([0.0], np.cumsum(demeaned)))
        self.n        = n
        self.usable   = True

    def sample_sums(self, n_steps: int, batch_size: int,
                     rng: np.random.Generator) -> np.ndarray:
        """Returns `batch_size` bootstrap sums of `n_steps` resampled ticks."""
        block_length = int(np.clip(round(n_steps * BOOTSTRAP_BLOCK_FRACTION),
                                    BOOTSTRAP_BLOCK_MIN, BOOTSTRAP_BLOCK_MAX))
        block_length = max(1, min(block_length, self.n))
        n_blocks     = max(1, int(math.ceil(n_steps / block_length)))
        max_start    = self.n - block_length

        if max_start <= 0:
            # History barely longer than one block -- sample individual
            # ticks with replacement instead (still empirical, no blocks).
            idx = rng.integers(0, self.n, size=(batch_size, n_steps))
            return self.demeaned[idx].sum(axis=1)

        starts = rng.integers(0, max_start + 1, size=(batch_size, n_blocks))
        block_sums = self.cumsum[starts + block_length] - self.cumsum[starts]
        total = block_sums.sum(axis=1)

        sampled_steps = n_blocks * block_length
        if sampled_steps != n_steps:
            # Blocks don't divide n_steps evenly -- rescale the sampled sum
            # so its variance matches n_steps ticks rather than
            # sampled_steps ticks (sampled_steps is always within one
            # block_length of n_steps, so this is a small correction).
            total = total * math.sqrt(n_steps / sampled_steps)

        return total


def generate_terminal_samples(abs_vol_per_tick: float,
                              duration_secs: float, ticks_per_sec: float,
                              bootstrap_pool: Optional[BootstrapPool] = None,
                              drift_per_tick: float = 0.0,
                              n_sims: int = MC_SIMULATIONS,
                              rng: Optional[np.random.Generator] = None) -> dict:
    """
    Draws `n_sims` terminal-displacement samples for one (duration, drift,
    vol) combination.

    X_terminal = drift_total + diffusion, where diffusion is drawn from a
    historical block bootstrap (BootstrapPool) when enough history is
    available, falling back to the old
    N(0, (abs_vol_per_tick * sqrt(n_steps))^2) Gaussian draw otherwise --
    this fallback is not a compromise, it's what the bot already did before
    this change, so behaviour degrades gracefully on a cold start.

    IMPORTANT: the terminal distribution depends only on (duration, drift,
    vol) -- NOT on any particular barrier. mc_auto_optimize() calls this
    ONCE per duration in its sweep and reuses the resulting samples across
    every barrier_sigma x asymmetry-ratio combo for that duration (via
    win_prob_from_samples below), instead of redrawing a fresh bootstrap
    for every single barrier width. That reuse is what keeps a full grid
    sweep fast with a resampling-based MC engine -- redrawing per-barrier
    was measured at ~60s/cycle; reusing per-duration draws is the same
    number of statistically independent MC runs (one per duration) with
    the barrier sweep on top being cheap vectorized comparisons.
    """
    n_steps      = max(1, int(round(duration_secs * ticks_per_sec)))
    vol_terminal = abs_vol_per_tick * math.sqrt(n_steps)
    drift_total  = drift_per_tick * n_steps    # total expected drift over contract

    if vol_terminal < 1e-9:
        return {"blocked": True, "reason": f"vol_terminal={vol_terminal:.2e} too small"}

    rng = rng or np.random.default_rng()
    use_bootstrap = bootstrap_pool is not None and bootstrap_pool.usable

    if use_bootstrap:
        terminal = bootstrap_pool.sample_sums(n_steps, n_sims, rng) + drift_total
    else:
        terminal = rng.normal(drift_total, vol_terminal, size=n_sims)

    return {
        "blocked":        False,
        "terminal":        terminal,
        "vol_terminal":    vol_terminal,
        "drift_total":     drift_total,
        "n_steps":         n_steps,
        "n_sims":          n_sims,
        "used_bootstrap":  use_bootstrap,
    }


# =============================================================================
# HMM/GBM ENGINE (v3)  — dependency-free 2-state Gaussian HMM + regime-
# conditional GBM sampler. Blended with the historical bootstrap above to
# form the terminal distribution mc_auto_optimize() actually sweeps against.
# =============================================================================
MIN_TICKS_FOR_HMM = 200        # below this, fall back to a flat single-state
HMM_LOOKBACK      = 1500       # ticks of returns used to (re)fit each cycle
HMM_EM_ITERS      = 25


class SimpleHMM2:
    """
    Minimal 2-state Gaussian HMM on a 1-D return series, fit via EM
    (Baum-Welch, forward-backward). No external HMM dependency, consistent
    with this file's self-built-statistics style (GARCH via `arch` is the
    only external stats package used elsewhere). State 0 is relabeled
    post-fit to always be the LOWER-variance state ("calm"), state 1 the
    higher-variance state ("volatile/trending") -- so callers don't need to
    know the arbitrary label EM converged to.

    fit() is defensive: any failure (too little data, degenerate variance,
    non-finite params) leaves `fitted=False` and callers fall back to a
    flat single-Gaussian model (see state_params()/state_probs()).
    """
    def __init__(self):
        self.fitted = False
        self.mu     = np.array([0.0, 0.0])
        self.sd     = np.array([1e-6, 1e-6])
        self.trans  = np.eye(2)
        self.filt_p = np.array([0.5, 0.5])   # filtered state prob @ last obs

    def fit(self, x: np.ndarray, n_iter: int = HMM_EM_ITERS):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < MIN_TICKS_FOR_HMM:
            return self

        try:
            # Init: split by sign of deviation from rolling std so the two
            # states start plausibly separated (calm vs volatile), not
            # randomly -- avoids EM converging to two near-identical states.
            roll = np.abs(x - np.mean(x))
            thresh = np.median(roll)
            mu    = np.array([np.mean(x[roll <= thresh]) if np.any(roll <= thresh) else np.mean(x),
                              np.mean(x[roll >  thresh]) if np.any(roll >  thresh) else np.mean(x)])
            sd    = np.array([max(np.std(x[roll <= thresh]), 1e-9) if np.any(roll <= thresh) else np.std(x) + 1e-9,
                              max(np.std(x[roll >  thresh]) * 1.5, 1e-9) if np.any(roll >  thresh) else np.std(x)*1.5 + 1e-9])
            trans = np.array([[0.95, 0.05], [0.10, 0.90]])
            pi0   = np.array([0.5, 0.5])

            for _ in range(n_iter):
                # E-step: forward-backward
                B = np.stack([norm.pdf(x, mu[k], sd[k]) + 1e-300 for k in range(2)], axis=1)  # (n,2)

                alpha = np.zeros((n, 2))
                c = np.zeros(n)
                alpha[0] = pi0 * B[0]
                c[0] = alpha[0].sum() + 1e-300
                alpha[0] /= c[0]
                for t in range(1, n):
                    alpha[t] = (alpha[t-1] @ trans) * B[t]
                    c[t] = alpha[t].sum() + 1e-300
                    alpha[t] /= c[t]

                beta = np.zeros((n, 2))
                beta[-1] = 1.0
                for t in range(n - 2, -1, -1):
                    beta[t] = (trans @ (B[t+1] * beta[t+1])) / c[t+1]

                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

                xi_sum = np.zeros((2, 2))
                for t in range(n - 1):
                    num = (alpha[t][:, None] * trans) * (B[t+1] * beta[t+1])[None, :]
                    xi_sum += num / (num.sum() + 1e-300)

                # M-step
                pi0   = gamma[0]
                trans = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)
                for k in range(2):
                    w = gamma[:, k]
                    wsum = w.sum() + 1e-300
                    mu[k] = (w @ x) / wsum
                    var_k = (w @ (x - mu[k]) ** 2) / wsum
                    sd[k] = math.sqrt(max(var_k, 1e-12))

            if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sd)) and np.all(sd > 0)):
                return self

            # Relabel so index 0 = lower-variance ("calm") state
            order = np.argsort(sd)
            self.mu = mu[order]
            self.sd = sd[order]
            self.trans = trans[np.ix_(order, order)]
            self.filt_p = gamma[-1][order]
            self.fitted = True
        except Exception as e:
            print(f"[HMM] fit failed, falling back to flat model: {e}")
        return self

    def state_params(self):
        """Returns (mu, sd) per state, always length-2 arrays even if unfit."""
        return self.mu, self.sd

    def state_probs(self) -> Tuple[float, float]:
        """Filtered P(calm), P(volatile) at the most recent observation."""
        if not self.fitted:
            return 0.5, 0.5
        return float(self.filt_p[0]), float(self.filt_p[1])


def hmm_gbm_terminal_samples(hmm: SimpleHMM2, price_now: float,
                             abs_vol_per_tick: float,
                             duration_secs: float, ticks_per_sec: float,
                             drift_per_tick_bias: float = 0.0,
                             n_sims: int = MC_SIMULATIONS,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Regime-conditional GBM sampler: each simulated path draws ONE regime
    (calm/volatile) from the HMM's current filtered state distribution and
    then behaves as pure GBM in that regime for the whole contract duration
    -- approximating regime PERSISTENCE over the trade window rather than
    switching every tick (which the transition matrix says is rare on these
    short 1-8min durations anyway, given the high diagonal of `trans`).

    True GBM (not the additive/arithmetic model the bootstrap path uses):
    `hmm.mu`/`hmm.sd` are fit on RELATIVE per-tick returns (same series
    GARCH fits on, r_t = (p_t - p_{t-1}) / p_{t-1}), which for the tiny
    per-tick moves on these synthetic indices is an adequate proxy for
    log-returns. Cumulative log-return over the contract is drawn via CLT
    as N(n_steps*mu, n_steps*sd^2) per regime, then exponentiated:
        terminal_price = price_now * exp(log_ret_total)
    so the terminal distribution is properly lognormal, not Gaussian-in-
    price-units like the bootstrap/Gaussian-fallback path.

    Returns terminal DISPLACEMENT samples (terminal_price - price_now, price
    units) in the same format as generate_terminal_samples()'s "terminal"
    array, so it can be blended directly with the bootstrap samples in
    mc_auto_optimize().

    drift_per_tick_bias is in ABSOLUTE price units per tick (matching the
    rest of the file's directional-bias convention) and is converted to a
    relative per-tick drift here before being added into log-return space.
    If the HMM isn't fitted (cold start / too little history), falls back
    to a flat single-state lognormal GBM using abs_vol_per_tick, so
    behaviour degrades gracefully rather than raising.
    """
    n_steps = max(1, int(round(duration_secs * ticks_per_sec)))
    rng = rng or np.random.default_rng()
    rel_drift_bias = (drift_per_tick_bias / price_now) if price_now > 1e-9 else 0.0

    if not hmm.fitted:
        rel_vol = (abs_vol_per_tick / price_now) if price_now > 1e-9 else 1e-6
        mu_total = rel_drift_bias * n_steps
        sd_total = max(rel_vol * math.sqrt(n_steps), 1e-9)
        log_ret  = rng.normal(mu_total, sd_total, size=n_sims)
        return price_now * np.exp(log_ret) - price_now

    p_calm, p_vol = hmm.state_probs()
    mu, sd = hmm.state_params()

    is_vol_state = rng.random(n_sims) < p_vol
    mu_path = np.where(is_vol_state, mu[1], mu[0]) * n_steps + rel_drift_bias * n_steps
    sd_path = np.where(is_vol_state, sd[1], sd[0]) * math.sqrt(n_steps)
    sd_path = np.clip(sd_path, 1e-9, None)

    log_ret_total = rng.normal(mu_path, sd_path)
    return price_now * np.exp(log_ret_total) - price_now


# =============================================================================
# LSTM TERMINAL-PRICE MODEL (v8)
# =============================================================================
# Trained OFFLINE by the separate expiryrange_lstm_train.py service (a
# Railway cron job, twice daily). This module only does INFERENCE + hot-
# reload -- no training happens in the live process. See
# expiryrange_lstm_model.py for the architecture/prediction target
# (per-tick Gaussian (mu, sigma), NOT a specific future price -- see that
# file's docstring for why).
LSTM_RELOAD_CHECK_INTERVAL_SECS = 1800   # poll Supabase for a newer model
                                          # at most every 30 min -- cheap
                                          # metadata-only query either way


def load_or_refresh_lstm(state: "BotState", store: Optional["SupabaseStore"]):
    """
    Called once per main-loop cycle; internally throttled. If torch/the
    model class failed to import at startup, this is a permanent no-op
    (state.lstm_model stays None forever, optimizer uses bootstrap+HMM/GBM
    only). If Supabase has a model newer than what's cached, loads it.
    Any failure here (bad state_dict, network error, no model trained yet)
    leaves the PREVIOUSLY cached model in place (or None) -- never crashes
    the trading loop over an optional component.
    """
    if not _LSTM_AVAILABLE or store is None:
        return
    now = time.time()
    if now - state.lstm_last_check_time < LSTM_RELOAD_CHECK_INTERVAL_SECS:
        return
    state.lstm_last_check_time = now

    b64, trained_at = store.load_lstm_model()
    if b64 is None:
        if state.lstm_model is None:
            print("[LSTM] No trained model in Supabase yet -- "
                  "using bootstrap+HMM/GBM only until the trainer's first run completes.")
        return
    if trained_at == state.lstm_trained_at:
        return   # already have this exact version loaded

    try:
        raw = base64.b64decode(b64)
        state_dict = torch.load(io.BytesIO(raw), map_location="cpu")
        model = BarrierWinClassifier()
        model.load_state_dict(state_dict)
        model.eval()
        was_first_load = state.lstm_model is None
        state.lstm_model = model
        state.lstm_trained_at = trained_at
        print(f"[LSTM] Loaded model trained_at={trained_at} "
              f"({'first load' if was_first_load else 'reload'})")
    except Exception as e:
        print(f"[LSTM] Failed to load model (trained_at={trained_at}): {e} -- "
              f"keeping previous model (or none) in place.")


# v9: LSTM_BLEND_WEIGHT controls how much the classifier's win_prob counts
# against the MC engine's (bootstrap+HMM/GBM) estimate for the SAME
# (duration, barrier_sigma) -- 0.5 = equal weight. Only applied when a
# model is loaded; MC-only otherwise (see mc_auto_optimize()).
LSTM_BLEND_WEIGHT = 0.5


def win_prob_from_samples(terminal: np.ndarray, upper_abs: float,
                          lower_abs: float) -> dict:
    """Cheap vectorized barrier check against pre-drawn terminal samples."""
    n_sims   = len(terminal)
    wins     = int(np.sum((terminal > -lower_abs) & (terminal < upper_abs)))
    win_prob = wins / n_sims
    return {
        "win_prob":    win_prob,
        "breach_prob": 1.0 - win_prob,
        "upper_abs":   upper_abs,
        "lower_abs":   lower_abs,
        "symmetric":   abs(upper_abs - lower_abs) < 1e-6,
    }


def mc_asymmetric_estimate(abs_vol_per_tick: float,
                           upper_abs: float, lower_abs: float,
                           duration_secs: float, ticks_per_sec: float,
                           bootstrap_pool: Optional[BootstrapPool] = None,
                           drift_per_tick: float = 0.0,
                           rng: Optional[np.random.Generator] = None) -> dict:
    """
    Single-shot convenience wrapper: draws samples for ONE duration and
    evaluates ONE barrier. Kept for standalone/one-off use and tests --
    mc_auto_optimize() does NOT use this in its sweep (it calls
    generate_terminal_samples once per duration and win_prob_from_samples
    per barrier, to avoid redrawing the bootstrap for every barrier).
    """
    gen = generate_terminal_samples(
        abs_vol_per_tick, duration_secs, ticks_per_sec,
        bootstrap_pool=bootstrap_pool, drift_per_tick=drift_per_tick,
        n_sims=MC_SIMULATIONS, rng=rng)
    if gen.get("blocked"):
        return gen
    wp = win_prob_from_samples(gen["terminal"], upper_abs, lower_abs)
    return {
        "blocked":        False,
        "win_prob":        wp["win_prob"],
        "breach_prob":     wp["breach_prob"],
        "vol_terminal":    gen["vol_terminal"],
        "drift_total":     gen["drift_total"],
        "n_steps":         gen["n_steps"],
        "n_sims":          gen["n_sims"],
        "upper_abs":       upper_abs,
        "lower_abs":       lower_abs,
        "symmetric":       wp["symmetric"],
        "used_bootstrap":  gen["used_bootstrap"],
    }


def ci_lower_bound(win_prob: float, n: int) -> float:
    z = norm.ppf(1 - MC_CI_PERCENTILE / 100)
    return win_prob - z * math.sqrt(max(win_prob * (1 - win_prob) / n, 1e-12))


# =============================================================================
# MC AUTO-OPTIMIZER  — asymmetric barrier sweep
# =============================================================================
def mc_auto_optimize(prices: np.ndarray, price_diffs: np.ndarray,
                     returns: np.ndarray, symbol: str,
                     garch_result, state: BotState,
                     gate_info: Optional[dict] = None) -> Optional[List[dict]]:
    """
    Sweeps (duration × barrier_sigma × asym_ratio) grid.

    For each symmetric baseline barrier_abs, the asymmetry ratio splits it:
      upper_abs = barrier_abs * upper_ratio
      lower_abs = barrier_abs * lower_ratio

    The ratio pair is chosen from ASYM_RATIO_GRID for both sides independently,
    but biased by the directional signal:
      bias > 0 (up drift) → favour upper_ratio > 1, lower_ratio < 1
      bias < 0 (down drift) → favour lower_ratio > 1, upper_ratio < 1
      bias ≈ 0             → symmetric (1.0, 1.0) gets the most weight

    The drift is injected into the MC terminal distribution as a mean shift,
    so win_prob correctly reflects both the asymmetric window AND the drift.

    `gate_info` (from structural_gate(), computed this same cycle) is used
    to classify the current market regime and gate which duration/sigma
    combos are swept at all -- see classify_regime() / REGIME_DURATIONS /
    REGIME_SIGMAS. If not provided, falls back to the full grid (old
    behaviour), so this stays backward-compatible with any other caller.

    Returns candidates sorted by win_prob ascending (lowest win = narrowest
    barrier = highest Deriv payout), or None if no candidate passes.
    """
    cfg           = SYMBOL_CONFIG[symbol]
    ticks_per_sec = cfg["ticks_per_sec"]
    price_now     = float(prices[-1]) if len(prices) > 0 else 1.0

    abs_vol, vol_trust, used_garch = estimate_abs_vol_per_tick(
        prices, returns, garch_result, price_now)
    abs_vol *= state.vol_scalar.get(symbol, 1.0)

    # ── Regime classification (v3: DIAGNOSTIC ONLY -- see header notes).
    # No longer narrows the swept grid; dropping RDBEAR freed the compute
    # budget that narrowing used to protect, and the discrete ADX/Hawkes
    # buckets are a cruder regime read than the HMM fit below anyway. Full
    # grid sweeps every cycle regardless of which bucket this reports.
    regime = classify_regime(symbol, gate_info) if gate_info else REGIME_MEAN_REVERT
    dur_grid   = DURATION_CANDIDATES
    sigma_grid = BARRIER_SIGMAS
    print(f"[Regime] {symbol}: {regime} (diagnostic only)  "
          f"sweeping full grid: {len(dur_grid)} durations x "
          f"{len(sigma_grid)} sigmas")

    # ── Bootstrap pool (built once, reused for every candidate this cycle) ─
    pool = BootstrapPool(price_diffs, abs_vol)
    rng  = np.random.default_rng()
    if not pool.usable:
        print(f"[MC] {symbol}: bootstrap pool not usable "
              f"({len(price_diffs)} ticks available, need >= "
              f"{MIN_TICKS_FOR_BOOTSTRAP}) -- falling back to Gaussian MC")

    # ── HMM fit (v3) — once per cycle, reused for every duration/barrier
    # below via hmm_gbm_terminal_samples(). Fit on relative returns (same
    # series GARCH uses), most recent HMM_LOOKBACK ticks.
    hmm = SimpleHMM2()
    hmm_input = returns[-HMM_LOOKBACK:] if len(returns) > 0 else returns
    hmm.fit(hmm_input)
    if hmm.fitted:
        p_calm, p_vol = hmm.state_probs()
        mu, sd = hmm.state_params()
        print(f"[HMM] {symbol}: P(calm)={p_calm:.2f} P(volatile)={p_vol:.2f}  "
              f"calm(mu={mu[0]:+.6f},sd={sd[0]:.6f})  "
              f"volatile(mu={mu[1]:+.6f},sd={sd[1]:.6f})")
    else:
        print(f"[HMM] {symbol}: not enough history yet "
              f"({len(hmm_input)} ticks, need >= {MIN_TICKS_FOR_HMM}) "
              f"-- GBM component uses flat fallback this cycle")

    # ── LSTM classifier hidden state (v9) — computed ONCE per cycle here,
    # reused for every (duration, barrier_sigma) combo in the sweep below
    # via predict_win_probs_batch(). The encoder doesn't depend on
    # duration/barrier at all, only the window -- see
    # expiryrange_lstm_model.py's "EFFICIENCY NOTE".
    lstm_model = state.lstm_model
    lstm_hidden = None
    if lstm_model is not None:
        try:
            lstm_hidden = lstm_model.compute_hidden(returns)
        except Exception as e:
            print(f"[LSTM] compute_hidden failed, disabling classifier "
                  f"blend this cycle: {e}")
            lstm_hidden = None

    # ── Directional bias & drift ──────────────────────────────────────────
    bias         = compute_directional_bias(prices, abs_vol)   # [-1, +1] capped
    # Convert bias to a per-tick drift in absolute price units.
    # bias=±BIAS_MAX → drift = ±BIAS_MAX * abs_vol (i.e. up to BIAS_MAX sigma/tick)
    drift_per_tick = bias * abs_vol

    bias_str = (f"UP  {bias:+.3f}" if bias >  0.02 else
                f"DN  {bias:+.3f}" if bias < -0.02 else
                f"FLAT {bias:+.3f}")
    print(f"[MC] {symbol}  bias={bias_str}  drift/tick={drift_per_tick:+.5f}  "
          f"abs_vol={abs_vol:.5f}  ({'GARCH' if used_garch else 'baseline'})")

    # ── Build asymmetry pairs biased toward drift direction ───────────────
    # For each candidate (upper_ratio, lower_ratio):
    #   if bias > 0: upper_ratio ≥ lower_ratio preferred (widen upside)
    #   if bias < 0: lower_ratio ≥ upper_ratio preferred (widen downside)
    # We generate all pairs from ASYM_RATIO_GRID and score them by alignment.
    def bias_score(ur: float, lr: float) -> float:
        """Higher = better aligned with current bias direction."""
        asym = (ur - lr)      # +ve = upper wider, -ve = lower wider
        return asym * bias    # maximised when asym aligns with bias sign

    asym_pairs = []
    for ur in ASYM_RATIO_GRID:
        for lr in ASYM_RATIO_GRID:
            score = bias_score(ur, lr)
            # TREND regime: a breakout in the drift direction is the live
            # risk, so don't waste MC calls (or risk) on ratio pairs that
            # fight the trend (score < 0 = wider on the side price is
            # moving AWAY from). HIGH_VOL/LOW_VOL/MEAN_REVERT keep the full
            # asymmetry grid -- duration/sigma narrowing already applies
            # there.
            if regime == REGIME_TREND and score < 0:
                continue
            asym_pairs.append((ur, lr, score))
    # Sort by alignment descending so aligned pairs are tried first
    asym_pairs.sort(key=lambda x: -x[2])

    candidates = []
    seen_keys  = set()   # deduplicate (dur, upper_abs_rounded, lower_abs_rounded)

    for dur_secs in dur_grid:
        n_steps      = max(1, int(round(dur_secs * ticks_per_sec)))
        vol_terminal = abs_vol * math.sqrt(n_steps)

        # Terminal displacement distribution depends only on (duration,
        # drift, vol) -- draw it ONCE here and reuse for every barrier_sigma
        # x asymmetry-ratio combo below via win_prob_from_samples (cheap
        # boolean comparisons against the same sample array), instead of
        # re-running the bootstrap for every barrier width. This is what
        # keeps the full grid sweep fast with a resampling-based MC engine.
        # v3: blend two independent terminal-distribution sources at equal
        # weight -- half the sims from the empirical block-bootstrap
        # (BootstrapPool, fat-tails/shape realism) and half from the
        # regime-conditional HMM/GBM sampler (lognormal, regime-persistent
        # drift/vol). This makes the HMM/GBM model a direct input to which
        # duration/barrier gets selected, not just a side diagnostic.
        #
        # v9: the LSTM no longer contributes SAMPLES to this blend (that
        # was v8's design -- a generic next-tick forecast extrapolated via
        # the same sqrt(duration) formula GARCH/HMM already use). It's now
        # a direct (duration, barrier_sigma) win-probability classifier
        # instead, blended at the PROBABILITY level per sigma below (see
        # classifier_wps), not the sample level -- so this stays a clean
        # two-way sample blend regardless of whether an LSTM is loaded.
        n_boot = MC_SIMULATIONS // 2
        n_gbm  = MC_SIMULATIONS - n_boot
        gen = generate_terminal_samples(
            abs_vol, dur_secs, ticks_per_sec,
            bootstrap_pool=pool, drift_per_tick=drift_per_tick,
            n_sims=n_boot, rng=rng)
        if gen.get("blocked"):
            continue
        gbm_terminal = hmm_gbm_terminal_samples(
            hmm, price_now, abs_vol, dur_secs, ticks_per_sec,
            drift_per_tick_bias=drift_per_tick, n_sims=n_gbm, rng=rng)
        terminal       = np.concatenate([gen["terminal"], gbm_terminal])
        used_bootstrap = gen["used_bootstrap"]

        # v9: one batched classifier call per duration covers every sigma
        # in the grid -- cheap, since the encoder (the expensive part) was
        # already run once for the whole cycle (lstm_hidden, above).
        classifier_wps = None
        if lstm_hidden is not None:
            try:
                classifier_wps = lstm_model.predict_win_probs_batch(
                    lstm_hidden, [(n_steps, bs) for bs in sigma_grid])
            except Exception as e:
                print(f"[LSTM] predict_win_probs_batch failed for dur={dur_secs}s, "
                      f"MC-only this duration: {e}")
                classifier_wps = None

        for bs_idx, bs in enumerate(sigma_grid):
            barrier_abs = max(bs * vol_terminal, BARRIER_ABS_MIN)

            for ur, lr, _ in asym_pairs:
                upper_abs = barrier_abs * ur
                lower_abs = barrier_abs * lr

                # Enforce minimum side size
                min_side = barrier_abs * ASYM_SIDE_MIN_FRAC
                if upper_abs < min_side or lower_abs < min_side:
                    continue

                # Deduplicate to avoid running near-identical MC calls
                key = (dur_secs, round(upper_abs, 3), round(lower_abs, 3))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                mc = win_prob_from_samples(terminal, upper_abs, lower_abs)

                wp = mc["win_prob"]
                # v9: blend in the classifier's win_prob for this (duration,
                # barrier_sigma) if available. Note the classifier was
                # trained against a SYMMETRIC barrier_abs (see
                # expiryrange_lstm_model.py / the trainer's label
                # construction) -- for asymmetric candidates this is an
                # approximation (the classifier corrects the overall width
                # calibration; win_prob_from_samples still does the exact
                # asymmetric evaluation via real MC samples either way).
                if classifier_wps is not None:
                    wp = (LSTM_BLEND_WEIGHT * float(classifier_wps[bs_idx])
                          + (1 - LSTM_BLEND_WEIGHT) * wp)
                cil = ci_lower_bound(wp, MC_SIMULATIONS)

                # Calibration-aware floor: if daily_self_improvement has had to
                # push vol_scalar above 1.0 for this symbol (MC was underestimating
                # real volatility -> overconfident win_prob), raise the required
                # win threshold proportionally until vol_scalar settles back near 1.0.
                # This stops the optimizer from chasing marginal-edge candidates
                # exactly when the MC is known to be running hot.
                calib_penalty   = max(0.0, state.vol_scalar.get(symbol, 1.0) - 1.0) * 0.10
                required_win    = MC_REQUIRED_WIN + calib_penalty
                required_ci     = MC_REQUIRED_CI  + calib_penalty

                if wp  < required_win: continue
                if cil < required_ci:  continue
                if wp  > MC_MAX_WIN_PROB: continue   # Deriv won't quote these — see MC_MAX_WIN_PROB note

                # Learned preference weights (keyed on symmetric baseline sigma)
                dw = state.duration_weights.get(symbol, {}).get(dur_secs, 1.0)
                bw = state.barrier_weights.get(symbol, {}).get(round(bs * 2) / 2, 1.0)

                # Asymmetry alignment bonus: reward candidates whose upper/lower
                # split is well-aligned with the bias signal.
                asym_alignment = 1.0 + 0.15 * abs(bias) * bias_score(ur, lr)

                # Payout-awareness penalty. EXPIRYRANGE payout falls roughly
                # as win_prob rises (wider/safer window = smaller return,
                # before house margin payout ~ stake/win_prob).
                #
                # v7 BUG FIX (2026-07-29): the previous version only started
                # discounting above wp=0.92 -- but MC_MAX_WIN_PROB was
                # loosened to 0.90 in an earlier fix (moving the real
                # accept/reject decision to live proposal EV instead of this
                # analytical ceiling). That made the 0.92 knee UNREACHABLE:
                # every candidate this optimizer can even generate now has
                # wp <= 0.90, so payout_penalty was silently 1.0 for 100% of
                # candidates, every cycle. With no counterweight, raw wp
                # dominated weighted_score completely and the shortlist
                # converged to whatever sigma sits at the TOP of
                # BARRIER_SIGMAS (wp closest to 0.90) -- confirmed live in
                # two separate logs: narrowing BARRIER_SIGMAS from 0.30-2.00
                # to 0.85-1.35 didn't change the failure mode at all, it just
                # relocated the same convergence point to the new grid's
                # ceiling (sigma 1.25-1.35 both times), and real net payout
                # there was still $0.01-0.03 with edge_frac as bad as -0.24 --
                # the untested lower 2/3 of the grid (sigma 0.85-1.20, the
                # actual user-calibrated target zone) was NEVER once reaching
                # live EV ranking.
                #
                # Fix: slope the penalty across the WHOLE achievable range
                # (MC_REQUIRED_WIN=0.50 .. MC_MAX_WIN_PROB=0.90) instead of a
                # knee near the range's edge, so lower-wp/tighter-barrier
                # candidates get a genuine chance to outrank the wp-maximal
                # ones in the shortlist.
                payout_penalty = 1.0 - 2.25 * (wp - 0.50)
                payout_penalty = max(payout_penalty, 0.05)
                implied_payout_mult = 1.0 / max(wp, 1e-6)   # logging/Bayes only

                candidates.append({
                    "duration_secs":   dur_secs,
                    "barrier_abs":     barrier_abs,      # symmetric baseline (for logging/Bayes)
                    "upper_abs":       upper_abs,        # actual upper distance from spot
                    "lower_abs":       lower_abs,        # actual lower distance from spot
                    "upper_ratio":     ur,
                    "lower_ratio":     lr,
                    "barrier_sigma":   bs,
                    "n_steps":         n_steps,
                    "win_prob":        wp,
                    "ci_lower":        cil,
                    "breach_prob":     1.0 - wp,
                    "vol_per_tick":    abs_vol,
                    "vol_terminal":    vol_terminal,
                    "used_garch":      used_garch,
                    "vol_trust":       vol_trust,
                    "drift_per_tick":  drift_per_tick,
                    "bias":            bias,
                    "drift_total":     gen["drift_total"],
                    "implied_payout_mult": implied_payout_mult,
                    "payout_penalty":  payout_penalty,
                    # weighted_score is now only a PROXY used to (a) shortlist
                    # candidates for real EV ranking against live proposal
                    # payouts, and (b) pick a stable signal for the
                    # confirmation gate. It no longer determines which
                    # candidate actually gets traded -- see
                    # rank_candidates_by_ev() / execute_expiryrange().
                    "weighted_score":  wp * dw * bw * asym_alignment * payout_penalty,
                    "symmetric":       mc["symmetric"],
                    "regime":          regime,
                    "used_bootstrap":  used_bootstrap,
                })


    if not candidates:
        return None

    # Sort by weighted_score descending: this blends win_prob with LEARNED
    # duration/barrier performance from daily self-improvement (Bayesian
    # win-rate weights) and asymmetry-bias alignment. Sorting on raw
    # win_prob alone (old behaviour) picks the narrowest/cheapest-payout
    # barrier at the MC_REQUIRED_WIN floor — these are the candidates most
    # exposed to MC/live calibration drift and were empirically the worst
    # performers (51 trades flagged "negative EV" at entry actually won
    # 64.7% of the time; "positive EV" trades won only 33.3%).
    candidates.sort(key=lambda x: -x["weighted_score"])
    return candidates


# =============================================================================
# PROPOSAL API  — BUG 2 fixed: net = payout - ask_price (not payout - BASE_STAKE)
# =============================================================================
async def fetch_proposal_payout(client: DerivClient, symbol: str,
                                upper: float, lower: float,
                                duration_secs: int, stake: float) -> Tuple[Optional[float], float]:
    """
    Returns (net_profit, ask_price).
    net_profit = proposal.payout - proposal.ask_price
    ask_price is what Deriv actually charges (may differ slightly from `stake`).
    `stake` is the dynamic stake for this trade (martingale-aware via
    BotState.next_stake). Returns (None, stake) on any failure.
    """
    try:
        bdp = SYMBOL_CONFIG[symbol].get("barrier_dp", 5)
        resp = await client.send({
            "proposal": 1, "amount": stake, "basis": "stake",
            "contract_type": "EXPIRYRANGE", "currency": "USD",
            "duration": duration_secs, "duration_unit": "s",
            "underlying_symbol": symbol,
            "barrier": str(round(upper, bdp)), "barrier2": str(round(lower, bdp)),
        }, timeout=12)

        if "error" in resp:
            err = resp["error"].get("message", str(resp["error"]))
            print(f"[Proposal] {symbol} error: {err}")
            return None, stake

        prop      = resp.get("proposal", {})
        payout    = float(prop.get("payout",    0))
        ask_price = float(prop.get("ask_price", stake))

        if payout <= 0 or ask_price <= 0:
            return None, stake

        net_profit = payout - ask_price

        # Sanity: net profit > 20x stake is impossible on EXPIRYRANGE
        if net_profit > stake * 20:
            print(f"[Proposal] {symbol}: suspicious net_profit=${net_profit:.4f} "
                  f"(payout={payout}, ask={ask_price}) — skipping")
            return None, stake

        return net_profit, ask_price

    except Exception as e:
        print(f"[Proposal] {symbol} exception: {e}")
        return None, stake


# =============================================================================
# EV-BASED CANDIDATE SELECTION  — replaces win-rate/weighted_score selection
# =============================================================================
def stratified_shortlist(candidates: List[dict], n_bins: int = 4, per_bin: int = 3) -> List[dict]:
    """
    v7: replaces flat `candidates[:12]`. weighted_score is only a PROXY (see
    its own comment block) and empirically converges to whichever end of
    BARRIER_SIGMAS has the highest win_prob, regardless of where that grid's
    bounds actually sit -- confirmed live across two separate sessions: after
    BARRIER_SIGMAS was narrowed from 0.30-2.00 to 0.85-1.35 specifically to
    stop this, the shortlist converged to sigma 1.25-1.35 (the new grid's
    ceiling) instead, and the actual user-calibrated target zone
    (sigma 0.85-1.20) was NEVER once reaching live EV ranking in either log.
    A smarter weighted_score alone isn't a robust enough fix for this --
    guaranteeing coverage of the full sigma range on every cycle is. Bins
    barrier_sigma into `n_bins` equal-width buckets spanning the full
    BARRIER_SIGMAS range and takes the top `per_bin` (by weighted_score)
    from EACH bucket, so the tightest and widest barriers both always get a
    live proposal check, not just whichever end weighted_score favours this
    cycle. Falls back to plain top-N if candidates don't span enough of the
    range to bin meaningfully.
    """
    if not candidates:
        return []
    lo, hi = min(BARRIER_SIGMAS), max(BARRIER_SIGMAS)
    if hi <= lo:
        return sorted(candidates, key=lambda x: -x["weighted_score"])[:n_bins * per_bin]

    bin_width = (hi - lo) / n_bins
    bins: List[List[dict]] = [[] for _ in range(n_bins)]
    for c in candidates:
        idx = int((c["barrier_sigma"] - lo) / bin_width)
        idx = min(max(idx, 0), n_bins - 1)
        bins[idx].append(c)

    shortlist = []
    for b in bins:
        b.sort(key=lambda x: -x["weighted_score"])
        shortlist.extend(b[:per_bin])
    return shortlist


async def rank_candidates_by_ev(client: DerivClient, state: BotState,
                                symbol: str, candidates: List[dict]
                                ) -> List[Tuple[dict, float, float, float, float]]:
    """
    Fetches a live Deriv proposal for each of `candidates` and ranks them by
    a unit-normalized edge measure:

        EV        = win_prob * net_payout - (1 - win_prob) * ask_price
        edge_frac = EV / ask_price     (expected return on stake)

    using the real net_payout/ask_price the proposal API just returned --
    not win_prob, not weighted_score, not the payout_penalty proxy.

    v3 CHANGE: no longer hard-excludes candidates below a fixed-dollar
    MIN_NET_PAYOUT floor. A candidate is excluded ONLY if its edge is
    genuinely negative-or-noise (edge_frac <= MIN_EDGE_FRAC) -- previously
    a candidate with real but modest edge (e.g. net_payout=$0.15 against a
    $0.182 floor) was discarded outright even though it was profitable in
    expectation; now it trades, sized by edge_frac via
    BotState.edge_scaled_stake() at the call site. Ranking is by edge_frac
    (not raw ev) so a strong SMALL-stake edge isn't buried under a weak
    LARGE-stake one -- edge_frac is what the stake scaling in
    execute_expiryrange() actually consumes.

    Returns a list of (candidate, net_payout, ask_price, ev, edge_frac)
    tuples for every candidate with edge_frac > MIN_EDGE_FRAC, sorted by
    edge_frac descending. Empty list if nothing cleared.

    Callers should keep `candidates` short (the caller passes the top
    weighted_score-ranked handful, not the full sweep) -- each entry costs
    one proposal API round trip.
    """
    stake = state.next_stake(symbol)
    bdp   = SYMBOL_CONFIG[symbol].get("barrier_dp", 5)
    price_now = state.last_price[symbol]

    results = []
    for cand in candidates:
        upper = round(price_now + cand["upper_abs"], bdp)
        lower = round(price_now - cand["lower_abs"], bdp)
        net_payout, ask_price = await fetch_proposal_payout(
            client, symbol, upper, lower, int(cand["duration_secs"]), stake)

        if net_payout is None or ask_price <= 0:
            continue

        wp = cand["win_prob"]
        ev = wp * net_payout - (1 - wp) * ask_price
        edge_frac = ev / ask_price

        if edge_frac <= MIN_EDGE_FRAC:
            print(f"[EV] {symbol}: dur={cand['duration_secs']}s "
                  f"sigma={cand['barrier_sigma']:.2f} net=${net_payout:.4f} "
                  f"edge_frac={edge_frac:+.3f} <= {MIN_EDGE_FRAC} -- excluded")
            continue

        print(f"[EV] {symbol}: dur={cand['duration_secs']}s "
              f"sigma={cand['barrier_sigma']:.2f} win={wp:.3f} "
              f"net=${net_payout:.4f} ev=${ev:+.4f} edge_frac={edge_frac:+.3f}")
        results.append((cand, net_payout, ask_price, ev, edge_frac))

    results.sort(key=lambda x: -x[4])
    return results


# =============================================================================
# EXECUTE CONTRACT  — BUG 4 fixed: consistent 3-tuple return
# =============================================================================
async def execute_expiryrange(client: DerivClient, state: BotState,
                               symbol: str, candidate: dict, gate_info: dict,
                               store: SupabaseStore,
                               edge_frac: Optional[float] = None,
                               cached_proposal: Optional[Tuple[float, float]] = None
                               ) -> Tuple[bool, float, bool]:
    """
    Returns (won, profit, placed).
    placed=False means payout check failed; caller tries next candidate.

    `edge_frac` (v3), if given, is the edge measure from rank_candidates_by_ev
    (EV/ask_price at the ORIGINAL martingale-aware stake). It scales the
    actual trade stake continuously via BotState.edge_scaled_stake() -- see
    the "Continuous edge-scaled staking" constants near the top of the file.
    If the scaled stake differs from the stake `cached_proposal` was quoted
    at, a fresh proposal is fetched at the final stake before buying (the
    buy call's `price` must match a real quote for that exact amount).
    Pass None to skip scaling entirely (falls back to next_stake() only,
    old behaviour) -- e.g. for any other caller of this function.

    `cached_proposal`, if given, is (net_payout, ask_price) already fetched
    moments earlier by rank_candidates_by_ev() -- reused as-is only when the
    edge-scaled stake matches what it was quoted at.
    """
    price_now     = state.last_price[symbol]
    duration_secs = int(candidate["duration_secs"])
    bdp           = SYMBOL_CONFIG[symbol].get("barrier_dp", 5)
    upper_abs     = candidate["upper_abs"]
    lower_abs     = candidate["lower_abs"]
    upper         = round(price_now + upper_abs, bdp)
    lower         = round(price_now - lower_abs, bdp)

    # Martingale-aware BASE stake for this trade (escalates only after
    # MG_TRIGGER_LOSSES consecutive losses on THIS symbol, capped at
    # MG_MAX_STEPS, resets to BASE_STAKE on any win -- see
    # BotState.record_trade_result). v3: edge_frac then scales this
    # continuously UPWARD (never below it -- see edge_scaled_stake()).
    base_stake = state.next_stake(symbol)
    if edge_frac is not None:
        stake = state.edge_scaled_stake(symbol, edge_frac)
    else:
        stake = base_stake
    mg_active = base_stake > BASE_STAKE + 1e-9
    edge_scaled = stake > base_stake + 1e-9

    # Stage 3: Proposal API payout verification. Reuse cached_proposal ONLY
    # if it was quoted at the same stake we're about to buy at -- edge
    # scaling can move the stake, and payout isn't perfectly linear in
    # stake near Deriv's own pricing edges, so a changed stake needs a
    # fresh quote.
    if cached_proposal is not None and abs(stake - base_stake) < 1e-9:
        net_payout, ask_price = cached_proposal
    else:
        net_payout, ask_price = await fetch_proposal_payout(
            client, symbol, upper, lower, duration_secs, stake)

    if net_payout is None:
        print(f"[Proposal] {symbol}: API failed — skipping candidate")
        return False, 0.0, False

    # v3: sanity floor only (guards against a bad quote / rounding dust),
    # NOT the old fixed MIN_NET_PAYOUT edge bar -- that decision already
    # happened upstream in rank_candidates_by_ev() via edge_frac.
    if net_payout <= 0:
        print(f"[Proposal] {symbol}: net=${net_payout:.4f} <= 0 at "
              f"scaled stake ${stake:.2f} — trying next candidate")
        return False, 0.0, False

    SEP = "-" * 68
    sym_tag = "SYMMETRIC" if candidate.get("symmetric") else \
              f"ASYM  u={candidate['upper_ratio']:.2f}x / l={candidate['lower_ratio']:.2f}x"
    bias_val = candidate.get("bias", 0.0)
    bias_tag = (f"UP {bias_val:+.3f}" if bias_val >  0.02 else
                f"DN {bias_val:+.3f}" if bias_val < -0.02 else
                f"FLAT {bias_val:+.3f}")
    mg_tag = (f"MARTINGALE step={min(state.mg_step.get(symbol,0)+1, MG_MAX_STEPS)}/{MG_MAX_STEPS} "
              f"(after {state.consec_losses.get(symbol,0)} consec losses)"
              if mg_active else "base stake")
    edge_tag = (f" + edge-scaled x{stake/base_stake:.2f} (edge_frac={edge_frac:+.3f})"
                if edge_scaled and edge_frac is not None else "")
    print(f"\n{SEP}")
    print(f"  EXPIRYRANGE  {symbol}  {datetime.now(timezone.utc).isoformat()}")
    print(SEP)
    print(f"  Entry         : {price_now:.5f}")
    print(f"  Upper barrier : {upper:.5f}  (+{upper_abs:.5f} from spot)")
    print(f"  Lower barrier : {lower:.5f}  (-{lower_abs:.5f} from spot)")
    print(f"  Asym profile  : {sym_tag}  |  bias={bias_tag}")
    print(f"  Drift/tick    : {candidate.get('drift_per_tick', 0.0):+.5f}  "
          f"total over {candidate['n_steps']} ticks={candidate.get('drift_total', 0.0):+.5f}")
    print(f"  Sigma base    : {candidate['barrier_sigma']:.2f}x terminal_vol "
          f"({candidate['vol_terminal']:.5f})")
    print(f"  Duration      : {duration_secs}s  ({candidate['n_steps']} ticks)")
    print(f"  Stake/Ask     : ${stake:.2f} / ${ask_price:.4f}   [{mg_tag}{edge_tag}]")
    print(f"  Net payout    : ${net_payout:.4f}  (confirmed by Deriv proposal API)")
    print(f"  MC win_prob   : {candidate['win_prob']:.3f}  "
          f"CI5={candidate['ci_lower']:.3f}  ({MC_SIMULATIONS:,} sims)")
    print(f"  Vol/tick      : {candidate['vol_per_tick']:.5f} abs price units  "
          f"({'GARCH' if candidate['used_garch'] else 'baseline'})  "
          f"vol_trust={candidate['vol_trust']:.3f}")
    print(f"  Gate          : ADX={gate_info['adx_val']:.1f}  "
          f"vol_trust={gate_info['vol_trust']:.3f}  "
          f"hawkes={gate_info['hawkes_val']:.3f}  "
          f"mbs={gate_info['mbs_val']:.3f}")
    print(SEP)

    won, profit, contract_id = False, 0.0, None
    try:
        resp = await client.send({
            "buy": "1", "price": ask_price,
            "parameters": {
                "amount":            stake,
                "basis":             "stake",
                "contract_type":     "EXPIRYRANGE",
                "currency":          "USD",
                "duration":          duration_secs,
                "duration_unit":     "s",
                "underlying_symbol": symbol,
                "barrier":           str(round(upper, bdp)),
                "barrier2":          str(round(lower, bdp)),
            },
        }, timeout=30)

        if "error" in resp:
            err = resp["error"].get("message", str(resp["error"]))
            print(f"[Buy] {symbol} error: {err}")
            return False, 0.0, False

        contract_id = resp.get("buy", {}).get("contract_id")
        if not contract_id:
            print(f"[Buy] {symbol}: no contract_id: {resp}")
            return False, 0.0, False

        print(f"[Buy] Contract id={contract_id} -- waiting {duration_secs}s...")

        # Fire directional overlay concurrently if bias is at max observed level.
        # asyncio.create_task lets it run alongside the EXPIRYRANGE settlement
        # poll below — both contracts settle at the same time since they share
        # the same duration. The overlay does NOT block EXPIRYRANGE result logging.
        bias_now = candidate.get("bias", 0.0)
        if DIR_OVERLAY_ENABLED and abs(bias_now) >= DIR_OVERLAY_BIAS_FLOOR:
            asyncio.create_task(
                execute_directional_overlay(
                    client, state, symbol, candidate,
                    duration_secs, store)
            )

        deadline = time.time() + duration_secs + 30
        while time.time() < deadline:
            await asyncio.sleep(5)
            try:
                poll = await client.send(
                    {"proposal_open_contract": 1, "contract_id": contract_id},
                    timeout=12)
                poc    = poll.get("proposal_open_contract", {})
                status = poc.get("status")
                if status == "sold" or poc.get("is_expired") or poc.get("is_settleable"):
                    profit = float(poc.get("profit", 0.0))
                    won    = profit > 0
                    break
            except Exception:
                pass

    except Exception as e:
        print(f"[Buy] {symbol} exception: {e}")
        return False, 0.0, False

    # Update martingale streak BEFORE session stats so next_stake() on the
    # next cycle reflects this outcome immediately.
    state.record_trade_result(symbol, won)

    # Update session state
    state.session_trades[symbol] += 1
    if won:
        state.session_wins[symbol] += 1
    state.session_profit[symbol] += profit
    state.last_trade_time[symbol] = time.time()
    state.last_activity           = time.time()

    wr     = state.session_wins[symbol] / max(state.session_trades[symbol], 1)
    result = f"WIN  +${profit:.4f}" if won else f"LOSS  -${ask_price:.4f}"
    print(f"\n{SEP}")
    print(f"  RESULT  {symbol}  {datetime.now(timezone.utc).isoformat()}")
    print(SEP)
    print(f"  Contract   : {contract_id}")
    print(f"  Outcome    : {result}")
    print(f"  Stake used : ${stake:.2f}  next_stake=${state.next_stake(symbol):.2f}")
    print(f"  Session    : {state.session_wins[symbol]}/{state.session_trades[symbol]} "
          f"({wr:.1%})  net=${state.session_profit[symbol]:+.2f}")
    print(SEP + "\n")

    # Refresh balance
    try:
        bal_resp      = await client.send({"balance": 1})
        state.balance = float(bal_resp["balance"]["balance"])
    except Exception:
        pass

    # Log to Supabase
    store.log_trade({
        "symbol":        symbol,
        "entry_price":   price_now,
        "upper_barrier": upper,
        "lower_barrier": lower,
        "barrier_width": upper_abs + lower_abs,   # total window width
        "upper_abs":     upper_abs,
        "lower_abs":     lower_abs,
        "upper_ratio":   candidate.get("upper_ratio", 1.0),
        "lower_ratio":   candidate.get("lower_ratio", 1.0),
        "bias":          candidate.get("bias", 0.0),
        "drift_per_tick": candidate.get("drift_per_tick", 0.0),
        "duration_secs": duration_secs,
        "stake":         stake,
        "mg_step":       state.mg_step.get(symbol, 0),
        "mg_active":     mg_active,
        "consec_losses_before": state.consec_losses.get(symbol, 0) if not won else 0,
        "won":           won,
        "profit":        profit,
        "breach_prob":   candidate["breach_prob"],
        "ev_conservative": won * net_payout - (not won) * ask_price,
        "ev_optimistic":   candidate["win_prob"] * net_payout - (1 - candidate["win_prob"]) * ask_price,
        "vol_per_tick":  candidate["vol_per_tick"],
        "used_garch":    candidate["used_garch"],
        "adx_val":       gate_info.get("adx_val", 0.0),
        "vol_trust":     gate_info.get("vol_trust", 0.0),
        "hawkes_val":    gate_info.get("hawkes_val", 0.0),
        "barrier_sigma": candidate.get("barrier_sigma", 0.0),
        "win_prob":      candidate.get("win_prob", 0.0),
        "ci_lower":      candidate.get("ci_lower", 0.0),
        "weighted_score": candidate.get("weighted_score", 0.0),
        "vol_terminal":  candidate.get("vol_terminal", 0.0),
        "drift_total":   candidate.get("drift_total", 0.0),
        "n_steps":       candidate.get("n_steps", 0),
        "ask_price":     ask_price,
    })
    return won, profit, True


# =============================================================================
# DIRECTIONAL OVERLAY  — CALL/PUT fired alongside EXPIRYRANGE on max-bias events
# =============================================================================
async def execute_directional_overlay(client: DerivClient, state: BotState,
                                      symbol: str, candidate: dict,
                                      er_duration_secs: int,
                                      store: SupabaseStore) -> None:
    """
    Fires a CALL (bias > 0, upper wider) or PUT (bias < 0, lower wider)
    contract for the same duration as the parent EXPIRYRANGE, using
    DIR_OVERLAY_STAKE_FRAC * EXPIRYRANGE_stake.

    Called ONLY when |bias| >= DIR_OVERLAY_BIAS_FLOOR. The EXPIRYRANGE
    contract has already been submitted — this runs concurrently after
    the buy call returns, so it does not block EXPIRYRANGE settlement.

    Logic:
      bias > 0  →  asymmetry pushed upper barrier up  →  CALL
                   (price drifted up over BIAS_LOOKBACK ticks and MC
                    expects it to continue; CALL wins if price ends above
                    entry at expiry)
      bias < 0  →  asymmetry pushed lower barrier down →  PUT
                   (price drifted down; PUT wins if price ends below entry)

    The two contracts are complementary but NOT a perfect hedge:
      - If price stays inside the EXPIRYRANGE window AND drifts in the
        bias direction: BOTH win.
      - If price drifts strongly in the bias direction but breaks the
        opposite barrier: EXPIRYRANGE loses, CALL/PUT wins (partial offset).
      - If price moves against the bias: EXPIRYRANGE may still win (wide
        window), CALL/PUT loses (small secondary stake = limited downside).
    """
    bias = candidate.get("bias", 0.0)
    if not DIR_OVERLAY_ENABLED:
        return
    if abs(bias) < DIR_OVERLAY_BIAS_FLOOR:
        return

    # Direction must come from the ACTUAL selected candidate's barrier
    # geometry (upper_ratio vs lower_ratio), not from the raw bias scalar.
    # `bias` is a single global drift reading shared by every candidate a
    # given mc_auto_optimize() call produces; which (upper_ratio, lower_ratio)
    # pair ends up as the top-ranked candidate is driven mostly by win_prob
    # (the asym_alignment bonus is a <=2% nudge even at max bias), so the
    # winning candidate's asymmetry can legitimately point the opposite way
    # from the bias sign. Firing the overlay off `bias` alone can then bet
    # CALL/PUT against the very barrier skew that was just priced in.
    asym = candidate.get("upper_ratio", 1.0) - candidate.get("lower_ratio", 1.0)
    if abs(asym) < 1e-9:
        return  # symmetric candidate -- no directional geometry to overlay on

    geometry_direction = "CALL" if asym > 0 else "PUT"
    bias_direction      = "CALL" if bias > 0 else "PUT"
    if geometry_direction != bias_direction:
        print(f"[Overlay] {symbol}: SKIPPED -- bias signal says {bias_direction} "
              f"({bias:+.4f}) but the selected candidate's barrier asymmetry "
              f"says {geometry_direction} (upper_ratio={candidate['upper_ratio']:.2f} "
              f"lower_ratio={candidate['lower_ratio']:.2f}) -- disagreement means "
              f"there's no confident directional read, not firing")
        return

    direction = geometry_direction
    er_stake  = state.next_stake(symbol)           # same stake logic as parent
    overlay_stake = round(er_stake * DIR_OVERLAY_STAKE_FRAC, 2)
    overlay_stake = max(overlay_stake, 0.35)       # Deriv minimum stake floor

    price_now = state.last_price[symbol]

    print(f"\n[Overlay] {symbol}: bias={bias:+.4f} >= floor={DIR_OVERLAY_BIAS_FLOOR:.3f} "
          f"-- firing {direction} overlay  "
          f"stake=${overlay_stake:.2f}  dur={er_duration_secs}s")

    # ── Proposal check ────────────────────────────────────────────────────
    try:
        prop_resp = await client.send({
            "proposal":      1,
            "amount":        overlay_stake,
            "basis":         "stake",
            "contract_type": direction,
            "currency":      "USD",
            "duration":      er_duration_secs,
            "duration_unit": "s",
            "underlying_symbol": symbol,
        }, timeout=12)

        if "error" in prop_resp:
            err = prop_resp["error"].get("message", str(prop_resp["error"]))
            print(f"[Overlay] {symbol} proposal error: {err} -- skipping overlay")
            return

        prop      = prop_resp.get("proposal", {})
        payout    = float(prop.get("payout",    0))
        ask_price = float(prop.get("ask_price", overlay_stake))
        net_profit_est = payout - ask_price

        if net_profit_est < DIR_OVERLAY_MIN_PAYOUT * (overlay_stake / 0.35):
            print(f"[Overlay] {symbol}: net=${net_profit_est:.4f} below overlay floor "
                  f"-- skipping")
            return

        print(f"[Overlay] {symbol}: proposal OK  payout=${payout:.4f}  "
              f"ask=${ask_price:.4f}  est_net=${net_profit_est:.4f}")

    except Exception as e:
        print(f"[Overlay] {symbol} proposal exception: {e} -- skipping overlay")
        return

    # ── Buy ───────────────────────────────────────────────────────────────
    won, profit, contract_id = False, 0.0, None
    try:
        buy_resp = await client.send({
            "buy":   "1",
            "price": ask_price,
            "parameters": {
                "amount":              overlay_stake,
                "basis":               "stake",
                "contract_type":       direction,
                "currency":            "USD",
                "duration":            er_duration_secs,
                "duration_unit":       "s",
                "underlying_symbol":   symbol,
            },
        }, timeout=30)

        if "error" in buy_resp:
            err = buy_resp["error"].get("message", str(buy_resp["error"]))
            print(f"[Overlay] {symbol} buy error: {err}")
            return

        contract_id = buy_resp.get("buy", {}).get("contract_id")
        if not contract_id:
            print(f"[Overlay] {symbol}: no contract_id")
            return

        print(f"[Overlay] Contract id={contract_id} -- "
              f"settling alongside EXPIRYRANGE in {er_duration_secs}s...")

        # ── Wait for settlement ───────────────────────────────────────────
        deadline = time.time() + er_duration_secs + 30
        while time.time() < deadline:
            await asyncio.sleep(5)
            try:
                poll   = await client.send(
                    {"proposal_open_contract": 1,
                     "contract_id": contract_id}, timeout=12)
                poc    = poll.get("proposal_open_contract", {})
                status = poc.get("status")
                if (status == "sold" or
                        poc.get("is_expired") or
                        poc.get("is_settleable")):
                    profit = float(poc.get("profit", 0.0))
                    won    = profit > 0
                    break
            except Exception:
                pass

    except Exception as e:
        print(f"[Overlay] {symbol} buy exception: {e}")
        return

    # ── Result ────────────────────────────────────────────────────────────
    state.overlay_trades[symbol]  = state.overlay_trades.get(symbol, 0) + 1
    state.overlay_wins[symbol]    = state.overlay_wins.get(symbol, 0) + (1 if won else 0)
    state.overlay_profit[symbol]  = state.overlay_profit.get(symbol, 0.0) + profit

    ov_wr  = state.overlay_wins[symbol] / max(state.overlay_trades[symbol], 1)
    result = f"WIN  +${profit:.4f}" if won else f"LOSS  -${ask_price:.4f}"
    SEP    = "-" * 68
    print(f"\n{SEP}")
    print(f"  OVERLAY RESULT  {symbol}  {direction}  "
          f"{datetime.now(timezone.utc).isoformat()}")
    print(SEP)
    print(f"  Contract   : {contract_id}")
    print(f"  Bias       : {bias:+.4f}  ({direction})")
    print(f"  Outcome    : {result}")
    print(f"  Overlay session: {state.overlay_wins[symbol]}/"
          f"{state.overlay_trades[symbol]} ({ov_wr:.1%})  "
          f"net=${state.overlay_profit[symbol]:+.2f}")
    print(SEP + "\n")

    # ── Log to Supabase ───────────────────────────────────────────────────
    store.log_overlay({
        "symbol":          symbol,
        "direction":       direction,
        "entry_price":     price_now,
        "duration_secs":   er_duration_secs,
        "stake":           overlay_stake,
        "bias":            bias,
        "bias_floor_used": DIR_OVERLAY_BIAS_FLOOR,
        "er_win_prob":     candidate.get("win_prob", 0.0),
        "er_upper_ratio":  candidate.get("upper_ratio", 1.0),
        "er_lower_ratio":  candidate.get("lower_ratio", 1.0),
        "won":             won,
        "profit":          profit,
        "ask_price":       ask_price,
    })


# =============================================================================
# DAILY SELF-IMPROVEMENT
# =============================================================================
def daily_self_improvement(state: BotState, store: SupabaseStore):
    print("\n" + "=" * 68)
    print("  DAILY SELF-IMPROVEMENT  " +
          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print("=" * 68)

    for symbol in SYMBOLS:
        rows = store.load_recent_trades(symbol, days=7)
        if not rows:
            print(f"[SI] {symbol}: no history in last 7 days, skipping.")
            continue

        n_total = len(rows)
        n_wins  = sum(1 for r in rows if r.get("won"))
        profit  = sum(float(r.get("profit", 0)) for r in rows)
        print(f"\n[SI] {symbol}: {n_total} trades  {n_wins} wins "
              f"({n_wins/max(n_total,1):.1%})  net=${profit:+.2f}")

        dur_stats: Dict[int,   List[int]] = defaultdict(lambda: [0, 0])
        bar_stats: Dict[float, List[int]] = defaultdict(lambda: [0, 0])
        mc_preds, actuals = [], []

        for r in rows:
            dur  = int(r.get("duration_secs", 120))
            won  = bool(r.get("won", False))
            bw   = float(r.get("barrier_width", 0))  # 2 * barrier_abs
            vpt  = float(r.get("vol_per_tick", 0))
            n_st = max(1, int(dur * SYMBOL_CONFIG[symbol]["ticks_per_sec"]))
            vt   = vpt * math.sqrt(n_st) if vpt > 0 else 1.0
            bs   = round(((bw / 2) / max(vt, 1e-9)) * 2) / 2
            bs   = float(np.clip(bs, 0.5, 3.0))

            dur_stats[dur][1] += 1
            bar_stats[bs][1]  += 1
            if won:
                dur_stats[dur][0] += 1
                bar_stats[bs][0]  += 1

            mc_wp = 1.0 - float(r.get("breach_prob", 0.5))
            mc_preds.append(mc_wp)
            actuals.append(1.0 if won else 0.0)

        alpha = 2.0

        # Duration reweighting
        raw_dw = {}
        print(f"  Duration win rates:")
        for dur, (w, t) in sorted(dur_stats.items()):
            if t == 0: continue
            bwr = (w + alpha) / (t + 2 * alpha)
            raw_dw[dur] = bwr
            print(f"    {dur}s: {w}/{t} ({w/t:.1%}) Bayes={bwr:.3f}")
        if raw_dw:
            mx, mn, sp = max(raw_dw.values()), min(raw_dw.values()), 0
            sp = mx - mn
            # Widened from [0.5, 2.0] to [0.25, 2.5]: empirically, 120s traded
            # at 75% win / +0.96 net while 180s/360s/480s were all net negative.
            # A narrower weight band let losing durations keep getting picked
            # almost as often as the winner. This starves bad buckets harder.
            state.duration_weights[symbol] = {
                d: (0.25 + 2.25 * (v - mn) / sp if sp > 0 else 1.0)
                for d, v in raw_dw.items()}

        # Barrier reweighting
        raw_bw = {}
        print(f"  Barrier sigma-slot win rates:")
        for slot, (w, t) in sorted(bar_stats.items()):
            if t == 0: continue
            bwr = (w + alpha) / (t + 2 * alpha)
            raw_bw[slot] = bwr
            print(f"    s={slot:.1f}: {w}/{t} ({w/t:.1%}) Bayes={bwr:.3f}")
        if raw_bw:
            mx, mn = max(raw_bw.values()), min(raw_bw.values())
            sp = mx - mn
            state.barrier_weights[symbol] = {
                sl: (0.5 + 1.5 * (v - mn) / sp if sp > 0 else 1.0)
                for sl, v in raw_bw.items()}

        # Vol scalar calibration
        if len(mc_preds) >= 10:
            mc_mean  = float(np.mean(mc_preds))
            act_mean = float(np.mean(actuals))
            ratio    = mc_mean / max(act_mean, 0.05)
            old_sc   = state.vol_scalar.get(symbol, 1.0)
            if ratio > 1.08:
                new_sc = float(np.clip(old_sc * min(ratio, 1.25), 0.5, 4.0))
                state.vol_scalar[symbol] = new_sc
                print(f"  Vol scalar UP: MC={mc_mean:.3f} > actual={act_mean:.3f} "
                      f"-> {old_sc:.3f} -> {new_sc:.3f}")
            elif ratio < 0.92:
                new_sc = float(np.clip(old_sc * max(ratio, 0.80), 0.5, 4.0))
                state.vol_scalar[symbol] = new_sc
                print(f"  Vol scalar DOWN: MC={mc_mean:.3f} < actual={act_mean:.3f} "
                      f"-> {old_sc:.3f} -> {new_sc:.3f}")
            else:
                print(f"  Vol scalar stable: MC={mc_mean:.3f} ~ actual={act_mean:.3f}")

        best_dur = max(dur_stats, key=lambda d: dur_stats[d][0] / max(dur_stats[d][1], 1)) if dur_stats else 120
        best_bar = max(bar_stats, key=lambda b: bar_stats[b][0] / max(bar_stats[b][1], 1)) if bar_stats else 1.0
        store.save_daily_summary(
            datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            symbol, n_total, n_wins, profit, best_dur, best_bar)

    store.save_config("duration_weights",
        {s: {str(k): v for k, v in state.duration_weights.get(s, {}).items()} for s in SYMBOLS})
    store.save_config("barrier_weights",
        {s: {str(k): v for k, v in state.barrier_weights.get(s, {}).items()} for s in SYMBOLS})
    store.save_config("vol_scalars",
        {s: state.vol_scalar.get(s, 1.0) for s in SYMBOLS})

    state.last_daily_tune = time.time()
    print("\n[SI] Config saved. Next tuning in ~24h.")
    print("=" * 68 + "\n")


def load_config_from_supabase(state: BotState, store: SupabaseStore):
    loaded = False
    dur_w = store.load_config("duration_weights")
    if dur_w:
        for s in SYMBOLS:
            if s in dur_w:
                state.duration_weights[s] = {int(k): float(v) for k, v in dur_w[s].items()}
                loaded = True
        stale = set(dur_w.keys()) - set(SYMBOLS)
        if stale:
            print(f"[Config] WARNING: duration_weights in Supabase has "
                  f"key(s) {sorted(stale)} that don't match current "
                  f"SYMBOLS={SYMBOLS} -- that data will never be used. "
                  f"If this looks like a mangled symbol name, check SYMBOLS "
                  f"for a typo (see the startup validation in main()).")
    bar_w = store.load_config("barrier_weights")
    if bar_w:
        for s in SYMBOLS:
            if s in bar_w:
                state.barrier_weights[s] = {float(k): float(v) for k, v in bar_w[s].items()}
                loaded = True
    vol_s = store.load_config("vol_scalars")
    if vol_s:
        for s in SYMBOLS:
            if s in vol_s:
                state.vol_scalar[s] = float(vol_s[s])
                loaded = True
    if loaded:
        print(f"[Config] Warm-start loaded. vol_scalars={state.vol_scalar}")
    else:
        print("[Config] Cold start.")


# =============================================================================
# TICK HELPERS
# =============================================================================
async def fetch_history(client: DerivClient, symbol: str, count: int) -> list:
    resp = await client.send(
        {"ticks_history": symbol, "count": count, "end": "latest", "style": "ticks"})
    h = resp.get("history", {})
    return list(zip(h.get("times", []), h.get("prices", [])))


async def subscribe_ticks(client: DerivClient, symbol: str) -> asyncio.Queue:
    q = client.subscribe_channel("tick")
    await client.send({"ticks": symbol, "subscribe": 1})
    return q


# =============================================================================
# WATCHDOG  — BUG 5 fixed: 15 min timeout
# =============================================================================
async def watchdog(state: BotState):
    while True:
        await asyncio.sleep(30)
        idle = time.time() - state.last_activity
        if idle > WATCHDOG_TIMEOUT:
            print(f"[Watchdog] No activity for {idle:.0f}s -- restarting.")
            os.execv(sys.executable, [sys.executable] + sys.argv)


# =============================================================================
# MAIN
# =============================================================================
async def main():
    if not DERIV_API_TOKEN:
        sys.exit("[FATAL] DERIV_API_TOKEN not set.")
    if not DERIV_APP_ID:
        sys.exit("[FATAL] DERIV_APP_ID not set.")

    # v10: fail fast if SYMBOLS contains anything without a matching
    # SYMBOL_CONFIG entry. Root-caused a real incident: a missing comma
    # between two adjacent string literals in a SYMBOLS edit (e.g.
    # ["1HZ10V" "R_10"]) is valid Python -- it silently concatenates into
    # ONE bogus symbol string ("1HZ10VR_10") instead of raising a syntax
    # error. That bogus "symbol" then gets written into every self-
    # improvement config table (duration_weights/barrier_weights/
    # vol_scalars) under a key the live bot can never look back up
    # (state.duration_weights.get("1HZ10V", ...) never finds
    # "1HZ10VR_10") -- silently discarding all learned duration/barrier
    # preference on every restart -- and Deriv likely rejects tick
    # subscriptions for a symbol name that doesn't exist, which is
    # consistent with a real multi-hour trade-frequency dead zone that
    # traced back to exactly this. Better to refuse to start than to
    # silently corrupt config and starve trading for hours before anyone
    # notices.
    for _s in SYMBOLS:
        if _s not in SYMBOL_CONFIG:
            sys.exit(f"[FATAL] SYMBOLS contains '{_s}' which has no "
                     f"SYMBOL_CONFIG entry. Check for a typo -- a missing "
                     f"comma between two adjacent string literals silently "
                     f"concatenates them (e.g. [\"1HZ10V\" \"R_10\"] becomes "
                     f"[\"1HZ10VR_10\"], not two symbols). Configured "
                     f"symbols: {list(SYMBOL_CONFIG.keys())}")

    store = SupabaseStore()
    state = BotState()
    load_config_from_supabase(state, store)
    load_or_refresh_lstm(state, store)   # v8: initial attempt; retried each
                                          # main-loop cycle (throttled) below

    client  = DerivClient(DERIV_APP_ID, DERIV_API_TOKEN,
                          DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID)
    account = await client.connect()
    state.balance = float(account.get("balance", 0))
    print(f"Balance: ${state.balance:.2f}")

    sdata: Dict[str, SymbolData] = {s: SymbolData(s) for s in SYMBOLS}

    # Bootstrap history
    print("\nBootstrapping tick history...")
    for sym in SYMBOLS:
        ticks = await fetch_history(client, sym, HISTORY_BOOTSTRAP)
        for epoch, price in ticks:
            sdata[sym].add_tick(epoch, price)
        prices = sdata[sym].prices()
        if len(prices):
            state.last_price[sym] = float(prices[-1])
        print(f"  {sym}: {len(ticks)} ticks loaded  "
              f"price={state.last_price[sym]:.5f}")
    state.last_activity = time.time()  # reset after bootstrap

    # Fit GARCH
    print("\nFitting GARCH models...")
    for sym in SYMBOLS:
        returns = sdata[sym].returns()
        if len(returns) >= MIN_TICKS_FOR_FIT:
            gr = await asyncio.to_thread(fit_garch, returns)
            state.garch_cache[sym] = (gr, time.time())
            # Sanity-check the vol immediately
            prices = sdata[sym].prices()
            abs_vol, vol_trust, used_g = estimate_abs_vol_per_tick(
                prices, returns, gr, state.last_price[sym])
            print(f"  {sym}: GARCH {'fitted' if gr else 'failed'}  "
                  f"abs_vol_per_tick={abs_vol:.5f}  "
                  f"({'GARCH' if used_g else 'baseline'})  "
                  f"vol_trust={vol_trust:.3f}")
        else:
            state.garch_cache[sym] = (None, 0.0)
            print(f"  {sym}: not enough data ({len(returns)} returns)")
    state.last_activity = time.time()  # reset after GARCH

    # Subscribe ticks
    tick_queues: Dict[str, asyncio.Queue] = {}
    for sym in SYMBOLS:
        tick_queues[sym] = await subscribe_ticks(client, sym)
    print(f"\nSubscribed to: {SYMBOLS}")

    async def resubscribe(c: DerivClient):
        for sym in SYMBOLS:
            tick_queues[sym] = await subscribe_ticks(c, sym)
        bal_resp     = await c.send({"balance": 1})
        state.balance = float(bal_resp.get("balance", {}).get("balance", state.balance))
        print("[Reconnect] Subscriptions restored.")

    client.resubscribe_cb = resubscribe

    asyncio.create_task(watchdog(state))
    state.last_activity = time.time()

    print("\n" + "=" * 68)
    print("  Bot armed -- scanning for EXPIRYRANGE setups")
    print("=" * 68 + "\n")

    garch_recal_secs = 2 * 3600

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    while True:
        load_or_refresh_lstm(state, store)   # v8: throttled internally, cheap to call every cycle

        # Drain tick queues
        for sym in SYMBOLS:
            drained = 0
            while drained < 200:
                try:
                    msg  = tick_queues[sym].get_nowait()
                    tick = msg.get("tick", {})
                    if tick.get("symbol") == sym:
                        ep = float(tick.get("epoch", 0))
                        px = float(tick.get("quote", 0))
                        sdata[sym].add_tick(ep, px)
                        state.last_price[sym] = px
                        drained += 1
                except asyncio.QueueEmpty:
                    break
            if drained == 0:
                try:
                    msg = await asyncio.wait_for(tick_queues[sym].get(), timeout=1.5)
                    tick = msg.get("tick", {})
                    if tick.get("symbol") == sym:
                        sdata[sym].add_tick(float(tick.get("epoch", 0)),
                                            float(tick.get("quote", 0)))
                        state.last_price[sym] = float(tick.get("quote", 0))
                except asyncio.TimeoutError:
                    pass

        state.last_activity = time.time()

        # Daily self-improvement check
        now_utc = datetime.now(timezone.utc)
        if (time.time() - state.last_daily_tune > 23 * 3600
                and now_utc.hour == DAILY_TUNE_HOUR_UTC):
            await asyncio.to_thread(daily_self_improvement, state, store)

        # Periodic GARCH recalibration
        for sym in SYMBOLS:
            gr, fitted_at = state.garch_cache.get(sym, (None, 0.0))
            if time.time() - fitted_at > garch_recal_secs:
                returns = sdata[sym].returns()
                if len(returns) >= MIN_TICKS_FOR_FIT:
                    gr_new = await asyncio.to_thread(fit_garch, returns)
                    state.garch_cache[sym] = (gr_new, time.time())
                    state.last_activity    = time.time()
                    prices    = sdata[sym].prices()
                    abs_vol, _, used_g = estimate_abs_vol_per_tick(
                        prices, returns, gr_new, state.last_price[sym])
                    print(f"[GARCH] {sym}: recalibrated  "
                          f"abs_vol={abs_vol:.5f}  "
                          f"({'GARCH' if used_g else 'baseline'})")

        if state.trading_locked:
            await asyncio.sleep(0.5)
            continue

        # Per-symbol evaluation
        for sym in SYMBOLS:
            sd = sdata[sym]
            if len(sd.ticks) < MIN_TICKS_LIVE:
                continue

            elapsed = time.time() - state.last_trade_time.get(sym, 0.0)
            if elapsed < SYMBOL_CONFIG[sym]["cooldown_secs"]:
                continue

            prices      = sd.prices()
            price_diffs = sd.price_diffs()
            returns     = sd.returns()
            if len(returns) < 20:
                continue

            garch_result, _ = state.garch_cache.get(sym, (None, 0.0))
            price_now       = state.last_price[sym]

            # Stage 1: Structural gate
            gate_ok, gate_info = structural_gate(
                sym, prices, price_diffs, returns, garch_result, price_now)
            if not gate_ok:
                fails = {k: v for k, v in gate_info.items() if k.startswith("fail_")}
                print(f"[Gate] {sym}: blocked -- {fails}")
                continue

            # Stage 2: MC optimizer
            print(f"\n[MC] {sym}: running {MC_SIMULATIONS:,}-sim optimizer "
                  f"(ADX={gate_info['adx_val']:.1f} "
                  f"vol_trust={gate_info['vol_trust']:.3f} "
                  f"abs_vol={gate_info['abs_vol']:.5f})...")

            t0         = time.time()
            candidates = await asyncio.to_thread(
                mc_auto_optimize, prices, price_diffs, returns,
                sym, garch_result, state, gate_info)
            dt = time.time() - t0

            if not candidates:
                print(f"[MC] {sym}: no combo cleared win>={MC_REQUIRED_WIN:.0%} "
                      f"& CI>={MC_REQUIRED_CI:.0%} in {dt:.1f}s -- waiting.")
                continue

            print(f"[MC] {sym}: {len(candidates)} passing combos in {dt:.1f}s -- "
                  f"shortlisting top candidates for live EV ranking")

            # Stage 2.5: Signal confirmation gate. Requires the top-ranked
            # candidate to remain consistent (same duration/sigma signature)
            # across CONFIRM_REQUIRED passes, each >=CONFIRM_MIN_GAP_SECS
            # apart, before a trade is allowed to fire. This prevents the
            # bot from entering on a single favorable tick window that
            # happened to clear the MC bar — the signal has to persist.
            top = candidates[0]
            confirmed, cinfo = state.check_confirmation(sym, top)
            if not confirmed:
                print(f"[Confirm] {sym}: streak {cinfo['streak']}/{cinfo['required']} "
                      f"dur={top['duration_secs']}s sigma={top['barrier_sigma']:.2f} "
                      f"-- {cinfo.get('reason', 'awaiting next confirm pass')}")
                continue
            print(f"[Confirm] {sym}: CONFIRMED after {CONFIRM_REQUIRED} consistent "
                  f"passes (dur={top['duration_secs']}s sigma={top['barrier_sigma']:.2f}) "
                  f"-- proceeding to execution")

            # Stage 3: fetch live proposals for the top weighted_score
            # candidates and rank them by ACTUAL expected value (not
            # win_prob, not weighted_score) -- see rank_candidates_by_ev().
            # Stage 4: buy whichever cleared candidate has the best EV.
            state.trading_locked = True
            placed = False
            # v3: wider shortlist (was 6) -- the widened sweep grid produces
            # a much larger candidate pool, so more get a live proposal
            # check before falling back to "nothing this cycle".
            # v7: stratified across the sigma range (was flat candidates[:12]
            # by weighted_score) -- see stratified_shortlist() docstring for
            # why the flat version was silently starving out the tightest
            # barriers every single cycle.
            shortlist = stratified_shortlist(candidates, n_bins=4, per_bin=3)
            ranked = await rank_candidates_by_ev(client, state, sym, shortlist)

            if not ranked:
                print(f"[EV] {sym}: none of the top {len(shortlist)} candidates "
                      f"had edge_frac > {MIN_EDGE_FRAC} -- skipping cycle.")
            else:
                best_cand, best_net_payout, best_ask_price, best_ev, best_edge_frac = ranked[0]
                print(f"[EV] {sym}: selected best-edge candidate -- "
                      f"dur={best_cand['duration_secs']}s "
                      f"sigma={best_cand['barrier_sigma']:.2f} "
                      f"win={best_cand['win_prob']:.3f} "
                      f"net_payout=${best_net_payout:.4f} "
                      f"ev=${best_ev:+.4f}  edge_frac={best_edge_frac:+.3f}  "
                      f"({len(ranked)}/{len(shortlist)} candidates cleared "
                      f"edge_frac > {MIN_EDGE_FRAC})")
                won, profit, ok = await execute_expiryrange(
                    client, state, sym, best_cand, gate_info, store,
                    edge_frac=best_edge_frac,
                    cached_proposal=(best_net_payout, best_ask_price))
                placed = ok

            state.trading_locked = False

            if not placed:
                print(f"[MC] {sym}: no candidate executed this cycle.")

        await asyncio.sleep(0.1)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped.")
    except Exception as e:
        print(f"[main] {type(e).__name__}: {e}")
        sys.stdout.flush()
        time.sleep(3)
        os.execv(sys.executable, [sys.executable] + sys.argv)
