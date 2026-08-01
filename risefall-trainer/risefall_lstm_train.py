"""
risefall_lstm_train.py

Standalone training service for the RISEFALL bot's LSTM layer(s). Deployed
as a Railway service on a CRON schedule, same pattern as
expiryrange_lstm_train.py -- one run, exits, safe to trigger manually.

TRAINS ONE MODEL PER RUN, PICKED BY MODEL_KIND
----------------------------------------------------------------------
MODEL_KIND=tick   -> trains RiseFallWinClassifier(kind="tick"),   window
                     is raw tick log-returns, labels use raw tick counts
                     as duration (matches CANDIDATE_DURATIONS in
                     risefall_bot_v4_hmm_gbm.py).
MODEL_KIND=minute -> trains RiseFallWinClassifier(kind="minute"), window
                     is minute-bar log-returns (see build_minute_bars()
                     below), labels use minute counts as duration.

Run this twice on the CRON schedule (two Railway cron triggers, or one
service invoked with both env values) to keep both models fresh -- they are
independent checkpoints and never share a state_dict. See
risefall_lstm_model.py's module docstring for why they're kept separate
instead of one model conditioned on a unit flag.

LABEL CONSTRUCTION -- simpler than EXPIRYRANGE, by design
----------------------------------------------------------------------
RISEFALL settles on direction alone: label = 1.0 if
price[t + duration] > price[t], else 0.0. No barrier, no local-vol term.
Exact ties (terminal == entry) are dropped rather than labeled either way --
real Deriv Rise/Fall contracts have their own tie-handling rules that don't
matter for what this classifier needs to learn (the direction, given it
actually moved).

MINUTE BARS
----------------------------------------------------------------------
Deriv's ticks_history gives a raw tick stream, not pre-built minute bars.
build_minute_bars() resamples it: for each whole minute (floor(epoch/60)),
takes the LAST tick price observed in that minute (last-observation, not
OHLC-close-vs-open -- matches how a Rise/Fall minute contract actually
settles: against the price prevailing at the expiry instant, not a
synthetic bar close). Minutes with zero ticks (should be rare on an
always-on synthetic index, but not impossible around a reconnect gap) are
forward-filled from the prior minute's last price -- NOT interpolated,
since interpolation would leak the (unknown, real) intra-gap path into a
label.

Env vars required (same as expiryrange_lstm_train.py):
  DERIV_APP_ID, DERIV_API_TOKEN, DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID
  SUPABASE_URL, SUPABASE_KEY
Plus:
  MODEL_KIND                       "tick" or "minute" (required)
  RISEFALL_TRAIN_SYMBOL            default "1HZ10V"
  LSTM_REQUIRE_BEAT_PERSISTENCE    default "true" -- see BASELINE
                                    DIAGNOSTICS below; aborts the Supabase
                                    upload (keeping whatever model is
                                    already live) if the LSTM doesn't beat
                                    a naive persistence baseline out of
                                    sample on this run's val split.

BASELINE DIAGNOSTICS -- "is the LSTM actually earning its keep?"
----------------------------------------------------------------------
Every training run also fits a naive persistence baseline, an AR(1) +
rescaled-range Hurst-exponent baseline, a HistGradientBoostingClassifier
on hand-engineered return-window features, a compact GRU, and a compact
dilated causal CNN -- all scored on the SAME purged validation split the
served LSTM ensemble is scored on, logged as a comparison table, and
uploaded alongside the model in Supabase's `baseline_comparison` column
(JSON). None of these five are served to the live bot; they exist purely
so you (or Gate 6's operator) can tell whether the LSTM's extra complexity
is finding real signal or just quietly overfitting to noise a much cheaper
model would also fit. See run_baseline_diagnostics() below.
"""
import asyncio
import base64
import gc
import io
import json
import os
import resource
import sys
import time
import traceback
from typing import Optional

import numpy as np
import requests
import websockets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier

from risefall_lstm_model import (
    RiseFallWinClassifier, WINDOW_SIZE_TICKS, WINDOW_SIZE_MINUTES,
    LSTM_HIDDEN, NUM_LSTM_LAYERS, N_ENSEMBLE_HEADS,
    CANDIDATE_DURATIONS_TICKS, CANDIDATE_DURATIONS_MINUTES,
    normalize_duration_count,
)

# =============================================================================
# CONFIG
# =============================================================================
MODEL_KIND = os.getenv("MODEL_KIND", "").strip().lower()
if MODEL_KIND not in ("tick", "minute"):
    print(f"[Trainer] FATAL -- MODEL_KIND env var must be 'tick' or 'minute', "
          f"got {MODEL_KIND!r}")
    sys.exit(1)

SYMBOL              = os.getenv("RISEFALL_TRAIN_SYMBOL", "1HZ10V")
TRAIN_HISTORY_DAYS  = float(os.getenv("LSTM_TRAIN_HISTORY_DAYS", "5" if MODEL_KIND == "tick" else "30"))
# Minute model needs far more wall-clock history to accumulate enough
# distinct minute bars -- 5 days of ticks is ~200k+ tick examples but only
# ~7200 minute bars, so it defaults to a longer pull.
MAX_TICKS           = int(os.getenv("LSTM_MAX_TICKS", "300000"))
TICKS_PER_HISTORY_CALL = 5000

EPOCHS       = int(os.getenv("LSTM_EPOCHS", "15"))
BATCH_SIZE   = int(os.getenv("LSTM_BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LSTM_LR", "1e-3"))
VAL_FRACTION  = 0.15

ANCHOR_STRIDE     = int(os.getenv("LSTM_ANCHOR_STRIDE", "5" if MODEL_KIND == "tick" else "1"))
COMBOS_PER_ANCHOR = int(os.getenv("LSTM_COMBOS_PER_ANCHOR", "4"))

# Per-head bagging: each of the N_ENSEMBLE_HEADS heads sees an independent
# Bernoulli(BAG_KEEP_PROB) mask over each batch's examples -- this (plus
# each head's own dropout) is what actually decorrelates the ensemble.
# Without it, all K heads would converge to near-identical functions since
# they see identical gradients from identical data every step.
BAG_KEEP_PROB = float(os.getenv("LSTM_BAG_KEEP_PROB", "0.8"))

DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID   = os.getenv("DERIV_ACCOUNT_ID") or None

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Separate Supabase rows per kind -- table shared, key differentiates, same
# "Prefer: resolution=merge-duplicates" upsert pattern as EXPIRYRANGE so
# each kind's cron run only ever touches its own row.
SUPABASE_TABLE = "bot_risefall_lstm_model"
SUPABASE_KEY_FIELD_VALUE = f"current_{MODEL_KIND}"

API_BASE   = "https://api.derivws.com/trading/v1/options"
ACCOUNTS_PATH = "/accounts"
OTP_PATH      = "/accounts/{account_id}/otp"

WINDOW_SIZE = WINDOW_SIZE_TICKS if MODEL_KIND == "tick" else WINDOW_SIZE_MINUTES
CANDIDATE_DURATIONS = CANDIDATE_DURATIONS_TICKS if MODEL_KIND == "tick" else CANDIDATE_DURATIONS_MINUTES


def log_peak_mem(tag: str):
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[Trainer:{MODEL_KIND}] [mem] {tag}: peak RSS so far = {peak_mb:.0f} MB")


# =============================================================================
# MINIMAL DERIV CLIENT -- identical pattern to expiryrange_lstm_train.py
# =============================================================================
class MinimalDerivClient:
    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id, self.token = app_id, token
        self.account_type, self.account_id = account_type, account_id
        self.ws = None
        self.req_id = 0
        self.pending = {}

    def _rest_headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Deriv-App-ID": self.app_id,
                "Content-Type": "application/json"}

    def _resolve_account_id_sync(self):
        resp = requests.get(f"{API_BASE}{ACCOUNTS_PATH}", headers=self._rest_headers(), timeout=15)
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
        resp = requests.post(f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}",
                             headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP missing data.url: {data}")
        return ws_url

    async def connect(self):
        ws_url = await asyncio.to_thread(self._fetch_otp_url_sync)
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        asyncio.create_task(self._read_loop())
        print(f"[Trainer:{MODEL_KIND}] Connected ({self.account_type}) for historical data pull.")

    async def _read_loop(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                rid = data.get("req_id")
                if rid is not None and rid in self.pending:
                    fut = self.pending.pop(rid)
                    if not fut.done():
                        fut.set_result(data)
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[Trainer:{MODEL_KIND}] WS closed: {e}")

    async def send(self, request, timeout=20):
        self.req_id += 1
        rid = self.req_id
        request = {**request, "req_id": rid}
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    async def close(self):
        if self.ws:
            await self.ws.close()


# =============================================================================
# PAGINATED HISTORY FETCH -- returns (times, prices), both oldest -> newest.
# EXPIRYRANGE's version only kept prices; RISEFALL's minute-bar resampling
# needs the epochs too, so both are returned here for both kinds (the tick
# path just ignores `times`).
# =============================================================================
async def fetch_full_history(client: MinimalDerivClient, symbol: str,
                             target_ticks: int):
    all_times: list = []
    all_prices: list = []
    end = "latest"
    consecutive_empty = 0

    while len(all_prices) < target_ticks and consecutive_empty < 2:
        resp = await client.send({
            "ticks_history": symbol,
            "count": TICKS_PER_HISTORY_CALL,
            "end": end,
            "style": "ticks",
        })
        h = resp.get("history", {})
        times = h.get("times", [])
        prices = h.get("prices", [])
        if not times:
            consecutive_empty += 1
            print(f"[Trainer:{MODEL_KIND}] Empty history page (end={end}) -- retry {consecutive_empty}/2")
            await asyncio.sleep(1)
            continue
        consecutive_empty = 0
        all_times = list(times) + all_times
        all_prices = list(prices) + all_prices
        end = int(times[0]) - 1
        print(f"[Trainer:{MODEL_KIND}] Fetched {len(times)} ticks (page ending {times[0]}) -- "
              f"{len(all_prices)}/{target_ticks} total")
        await asyncio.sleep(0.3)

    if len(all_prices) > target_ticks:
        all_times = all_times[-target_ticks:]
        all_prices = all_prices[-target_ticks:]
    return np.array(all_times, dtype=np.int64), np.array(all_prices, dtype=float)


def build_minute_bars(times: np.ndarray, prices: np.ndarray):
    """Resamples a raw tick series into one-per-minute last-observed-price
    bars. See module docstring for the last-observation / forward-fill
    rationale. Returns (bar_epochs, bar_prices), both minute-aligned and
    gap-free (every whole minute between the first and last tick has an
    entry, forward-filled if no tick actually landed in it)."""
    if len(times) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    minute_idx = times // 60
    first_min, last_min = int(minute_idx[0]), int(minute_idx[-1])
    n_minutes = last_min - first_min + 1

    bar_prices = np.full(n_minutes, np.nan, dtype=float)
    # Last tick observed in each minute wins -- times is sorted ascending,
    # so a simple forward scan assigning into the bucket naturally keeps
    # the LAST write per bucket.
    rel_idx = (minute_idx - first_min).astype(np.int64)
    bar_prices[rel_idx] = prices   # vectorized "last write wins" since
                                    # rel_idx is non-decreasing and later
                                    # ticks overwrite earlier ones at the
                                    # same index

    # Forward-fill any empty minutes.
    nan_mask = np.isnan(bar_prices)
    if nan_mask.any():
        idx = np.where(~nan_mask, np.arange(n_minutes), 0)
        np.maximum.accumulate(idx, out=idx)
        bar_prices = bar_prices[idx]
        # Any leading NaNs (shouldn't happen -- first_min's bucket always
        # gets the first tick) fall back to the first real price.
        if np.isnan(bar_prices[0]):
            bar_prices[:1] = prices[0]

    bar_epochs = (np.arange(n_minutes) + first_min) * 60
    return bar_epochs.astype(np.int64), bar_prices


# =============================================================================
# DATASET CONSTRUCTION
# =============================================================================
def build_labeled_examples(prices: np.ndarray, returns: np.ndarray,
                           window_size: int, candidate_durations: list,
                           rng: Optional[np.random.Generator] = None):
    """
    Builds (anchor_index, duration_norm -> up/down label) examples, direct
    direction target -- see module docstring, no barrier/vol term needed.
    Same lazy-window-storage pattern as EXPIRYRANGE's v14 fix (store only
    the anchor int, slice the window lazily in the Dataset) -- see
    RiseFallExampleDataset below.

    Samples duration counts as INTEGERS uniformly from candidate_durations'
    min/max range (continuous within that integer range, not just the
    discrete grid points) so the head learns a smooth function of duration,
    same reasoning as EXPIRYRANGE's continuous barrier_sigma sampling.
    """
    rng = rng or np.random.default_rng()
    n = len(returns)
    lo_dur, hi_dur = min(candidate_durations), max(candidate_durations)
    max_dur = hi_dur

    anchor_start = window_size
    anchor_end = n - max_dur - 1
    if anchor_end <= anchor_start:
        return (np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    anchor_ts = range(anchor_start, anchor_end, ANCHOR_STRIDE)
    n_anchors = len(anchor_ts)
    max_n = n_anchors * COMBOS_PER_ANCHOR

    anchors = np.empty((max_n,), dtype=np.int64)
    dur_norms = np.empty((max_n,), dtype=np.float32)
    labels = np.empty((max_n,), dtype=np.float32)

    print(f"[Trainer:{MODEL_KIND}] Building labeled examples: {n_anchors} anchors x "
          f"{COMBOS_PER_ANCHOR} combos (up to {max_n} examples)...")
    log_peak_mem("before building labeled examples")

    idx = 0
    progress_every = max(1, n_anchors // 5)
    kind = MODEL_KIND
    for i, t in enumerate(anchor_ts):
        if i > 0 and i % progress_every == 0:
            print(f"[Trainer:{kind}]   ...{i}/{n_anchors} anchors processed ({idx} examples so far)")
            log_peak_mem(f"anchor {i}/{n_anchors}")

        for _ in range(COMBOS_PER_ANCHOR):
            n_steps = int(rng.integers(lo_dur, hi_dur + 1))
            if t + n_steps >= n:
                continue

            entry_price = prices[t]
            terminal_price = prices[t + n_steps]
            if terminal_price == entry_price:
                continue   # exact tie -- dropped, see module docstring

            anchors[idx] = t
            dur_norms[idx] = normalize_duration_count(n_steps, kind)
            labels[idx] = 1.0 if terminal_price > entry_price else 0.0
            idx += 1

    print(f"[Trainer:{kind}] Built {idx} labeled examples.")
    log_peak_mem("after building labeled examples")

    if idx == 0:
        return (np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    return anchors[:idx], dur_norms[:idx], labels[:idx]


class RiseFallExampleDataset(torch.utils.data.Dataset):
    """Same lazy-slice-from-shared-array pattern as EXPIRYRANGE's
    BarrierExampleDataset -- `returns` is shared by reference, never
    duplicated per example."""
    def __init__(self, returns: np.ndarray, anchors: np.ndarray,
                dur_norms: np.ndarray, labels: np.ndarray, window_size: int):
        self.returns = returns
        self.anchors = anchors
        self.dur_norms = dur_norms
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        t = int(self.anchors[idx])
        window = self.returns[t - self.window_size:t]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(-1)
        d = torch.tensor([self.dur_norms[idx]], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, d, y


# =============================================================================
# SUPABASE PERSISTENCE
# =============================================================================
def save_model_to_supabase(state_dict, meta: dict):
    if not (SUPABASE_URL and SUPABASE_KEY):
        print(f"[Trainer:{MODEL_KIND}] No Supabase credentials -- skipping model upload "
              "(the live bot will keep using its current model, if any).")
        return False

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "key": SUPABASE_KEY_FIELD_VALUE,
        "kind": MODEL_KIND,
        "state_dict_b64": b64,
        "window_size": WINDOW_SIZE,
        "hidden_size": LSTM_HIDDEN,
        "num_layers": NUM_LSTM_LAYERS,
        "n_heads": N_ENSEMBLE_HEADS,
        **meta,
    }
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            data=json.dumps(payload), timeout=30,
        )
        resp.raise_for_status()
        print(f"[Trainer:{MODEL_KIND}] Model uploaded to Supabase "
              f"({len(b64)} base64 chars, val_loss={meta.get('val_loss'):.5f})")
        return True
    except Exception as e:
        print(f"[Trainer:{MODEL_KIND}] Supabase upload failed: {e}")
        return False


# =============================================================================
# TRAINING LOOP
# =============================================================================
def bagged_ensemble_loss(logits: torch.Tensor, yb: torch.Tensor,
                         loss_fn, rng: torch.Generator) -> torch.Tensor:
    """logits: (n_heads, batch). Each head's loss is computed only over a
    fresh Bernoulli(BAG_KEEP_PROB) subsample of the batch (independent draw
    per head, per step) -- this is what decorrelates the ensemble; see
    BAG_KEEP_PROB's comment above. Heads with an all-zero mask this step
    (rare at batch_size=64, p=0.8) contribute zero loss that step rather
    than dividing by zero."""
    n_heads, batch = logits.shape
    total = logits.new_tensor(0.0)
    n_active = 0
    for h in range(n_heads):
        mask = (torch.rand(batch, generator=rng) < BAG_KEEP_PROB)
        if mask.sum() == 0:
            continue
        total = total + loss_fn(logits[h][mask], yb[mask])
        n_active += 1
    return total / max(n_active, 1)


def train_model(prices: np.ndarray, returns: np.ndarray) -> tuple:
    anchors, dur_norms, y = build_labeled_examples(
        prices, returns, WINDOW_SIZE, CANDIDATE_DURATIONS)
    if len(anchors) < 500:
        raise RuntimeError(f"Only {len(anchors)} labeled examples available (need >=500) "
                           f"-- not enough history fetched to train reliably.")

    n_val = max(int(len(anchors) * VAL_FRACTION), 50)
    anchors_train, dn_train, y_train = anchors[:-n_val], dur_norms[:-n_val], y[:-n_val]
    anchors_val, dn_val, y_val       = anchors[-n_val:], dur_norms[-n_val:], y[-n_val:]

    # ── Purge gap at the train/val boundary ─────────────────────────────────
    # anchors are already chronologically ordered (see build_labeled_examples),
    # so this is a walk-forward split, not a random one -- good. But a label
    # is `price[t + n_steps] > price[t]`, looking FORWARD by up to
    # max(CANDIDATE_DURATIONS) -- a training anchor sitting right at the
    # boundary can have its label computed from raw price data that falls
    # chronologically inside the validation window, which is leakage even
    # though the split itself is time-ordered. Drop any training anchor
    # whose forward label horizon could reach into the validation region.
    val_start_t = int(anchors_val[0]) if len(anchors_val) else len(returns)
    purge_horizon = max(CANDIDATE_DURATIONS)
    keep_mask = (anchors_train + purge_horizon) < val_start_t
    n_purged = int((~keep_mask).sum())
    anchors_train = anchors_train[keep_mask]
    dn_train = dn_train[keep_mask]
    y_train = y_train[keep_mask]
    print(f"[Trainer:{MODEL_KIND}] Purged {n_purged} train anchors whose label "
          f"horizon crossed into the validation region "
          f"({len(anchors_train)} train / {len(anchors_val)} val examples remain).")

    # ── Normalization stats from TRAINING data only ─────────────────────────
    # Everything at/after val_start_t is held out, so this pool never touches
    # a single return the model will be validated or served on.
    train_return_pool = returns[:val_start_t]
    ret_mean = float(np.mean(train_return_pool)) if len(train_return_pool) else 0.0
    ret_std  = float(np.std(train_return_pool)) if len(train_return_pool) else 1.0
    print(f"[Trainer:{MODEL_KIND}] Normalization stats (train-only): "
          f"mean={ret_mean:.3e} std={ret_std:.3e}")

    gc.collect()
    log_peak_mem("before building datasets/loaders")
    train_ds = RiseFallExampleDataset(returns, anchors_train, dn_train, y_train, WINDOW_SIZE)
    val_ds   = RiseFallExampleDataset(returns, anchors_val, dn_val, y_val, WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    torch.set_num_threads(1)

    model = RiseFallWinClassifier(kind=MODEL_KIND, window_size=WINDOW_SIZE,
                                  hidden_size=LSTM_HIDDEN, num_layers=NUM_LSTM_LAYERS,
                                  n_heads=N_ENSEMBLE_HEADS)
    model.set_normalization_stats(ret_mean, ret_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    bag_rng = torch.Generator().manual_seed(1234)


    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None

    log_peak_mem("before first epoch")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, db, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb, db)                       # (n_heads, batch)
            loss = bagged_ensemble_loss(logits, yb, loss_fn, bag_rng)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for xb, db, yb in val_loader:
                val_logits = model(xb, db)                # (n_heads, batch)
                val_loss_b = loss_fn(val_logits.mean(dim=0), yb)   # eval: ensemble
                                                                     # mean vs. label,
                                                                     # matches how
                                                                     # predict_probs_
                                                                     # batch() is
                                                                     # actually used
                val_losses.append(val_loss_b.item())
                val_preds = (torch.sigmoid(val_logits.mean(dim=0)) > 0.5).float()
                val_correct += (val_preds == yb).sum().item()
                val_total += len(yb)
        val_loss = float(np.mean(val_losses))
        val_acc = val_correct / max(val_total, 1)

        train_loss = float(np.mean(train_losses))
        print(f"[Trainer:{MODEL_KIND}] epoch {epoch}/{EPOCHS}  train_bce={train_loss:.5f}  "
              f"val_bce={val_loss:.5f}  val_acc={val_acc:.3f}")
        if epoch == 1:
            log_peak_mem("after first epoch")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Re-run inference with the BEST checkpoint (not necessarily the last
    # epoch's weights) to get the val predictions the diagnostics/baseline
    # comparison below actually score against.
    model.load_state_dict(best_state)
    model.eval()
    val_probs_chunks = []
    with torch.no_grad():
        for xb, db, yb in val_loader:
            logits = model(xb, db)
            val_probs_chunks.append(torch.sigmoid(logits.mean(dim=0)).numpy())
    val_probs = np.concatenate(val_probs_chunks) if val_probs_chunks else np.array([])

    return {
        "state_dict": best_state,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "n_train": len(anchors_train),
        "n_val": len(anchors_val),
        "val_probs": val_probs,          # LSTM ensemble's own val predictions
        "anchors_train": anchors_train, "dn_train": dn_train, "y_train": y_train,
        "anchors_val": anchors_val, "dn_val": dn_val, "y_val": y_val,
        "ret_mean": ret_mean, "ret_std": ret_std,
    }


# =============================================================================
# BASELINE DIAGNOSTICS -- "is the LSTM actually earning its keep?"
# =============================================================================
# Every baseline here is scored on the EXACT SAME purged val split the LSTM
# ensemble above was scored on (same anchors_val/y_val), so the comparison
# is apples-to-apples. None of these are served to the live bot -- this is
# a diagnostic report only, logged to console and stashed in the Supabase
# meta row so you don't have to go dig through Railway logs to see it.
#
# Ordered cheapest/most-fundamental first: if the LSTM can't beat a naive
# persistence baseline, nothing below it (AR(1)/Hurst, GBM, GRU, dilated
# CNN) will tell you anything the persistence check didn't already.

def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def _accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    preds = (probs > 0.5).astype(np.float32)
    return float(np.mean(preds == y))


def _persistence_baseline(returns: np.ndarray, anchors_val: np.ndarray,
                          y_val: np.ndarray) -> dict:
    """Naive 'whatever the last observed return did, keep doing' baseline.
    p_up = 1.0 if the last return before the anchor was positive, 0.0 if
    negative, 0.5 on an exact-zero tie (rare on live tick data)."""
    last_returns = returns[anchors_val - 1]
    probs = np.where(last_returns > 0, 1.0,
                     np.where(last_returns < 0, 0.0, 0.5)).astype(np.float32)
    return {"name": "persistence", "accuracy": _accuracy(probs, y_val),
            "brier": _brier(probs, y_val), "probs": probs}


def _hurst_rs(x: np.ndarray, min_chunk: int = 16) -> Optional[float]:
    """Classic rescaled-range (R/S) Hurst exponent estimate. H≈0.5 -> the
    series is close to a random walk (no exploitable short-range memory);
    H>0.5 -> trending/persistent; H<0.5 -> mean-reverting. Purely
    descriptive here -- logged so WINDOW_SIZE_TICKS/MINUTES (in
    risefall_lstm_model.py) can be sanity-checked against how much real
    memory the process actually has, rather than being an untested guess."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_chunk * 4:
        return None
    chunk_sizes = np.unique(np.geomspace(min_chunk, max(n // 2, min_chunk + 1), num=10).astype(int))
    logs_n, logs_rs = [], []
    for size in chunk_sizes:
        if size < 2:
            continue
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_vals = []
        for i in range(n_chunks):
            chunk = x[i * size:(i + 1) * size]
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            r = dev.max() - dev.min()
            s = chunk.std()
            if s > 1e-12:
                rs_vals.append(r / s)
        if rs_vals:
            logs_n.append(np.log(size))
            logs_rs.append(np.log(np.mean(rs_vals)))
    if len(logs_n) < 3:
        return None
    slope, _ = np.polyfit(logs_n, logs_rs, 1)
    return float(slope)


def _ar1_baseline(returns: np.ndarray, train_return_pool: np.ndarray,
                  anchors_val: np.ndarray, y_val: np.ndarray) -> dict:
    """AR(1) fit on TRAINING returns only (phi = lag-1 autocorrelation),
    used to sign-predict every val example: r_hat = phi * last_return,
    p_up = 1.0 if r_hat > 0 else 0.0. Ignores exact multi-step decay
    (phi^k) since only the SIGN of the k-step forecast is needed, which
    matches sign(phi)*sign(last_return) -- a cheap sanity floor, not a
    tuned trading signal. If the LSTM can't clear this, it's not modeling
    anything AR(1) doesn't already capture linearly."""
    pool = train_return_pool[np.isfinite(train_return_pool)]
    if len(pool) < 100:
        phi = 0.0
    else:
        phi = float(np.corrcoef(pool[:-1], pool[1:])[0, 1])
        if not np.isfinite(phi):
            phi = 0.0
    last_returns = returns[anchors_val - 1]
    r_hat = phi * last_returns
    probs = np.where(r_hat > 0, 1.0, np.where(r_hat < 0, 0.0, 0.5)).astype(np.float32)
    hurst = _hurst_rs(pool)
    return {"name": "AR(1)", "accuracy": _accuracy(probs, y_val),
            "brier": _brier(probs, y_val), "probs": probs,
            "phi": phi, "hurst": hurst}


def _engineer_features(returns: np.ndarray, anchors: np.ndarray,
                       dur_norms: np.ndarray, window_size: int) -> np.ndarray:
    """Hand-crafted feature vector per anchor for the GBM baseline: return
    stats at three lookback scales, a skew estimate, current streak length,
    and the same normalized duration the LSTM sees. If this flat feature
    vector gets close to the LSTM's accuracy, the LSTM's extra encoder
    complexity isn't earning its keep."""
    scales = [10, 50, window_size]
    n = len(anchors)
    feats = np.zeros((n, len(scales) * 2 + 3), dtype=np.float32)
    for i, t in enumerate(anchors):
        t = int(t)
        col = 0
        for scale in scales:
            w = returns[max(0, t - scale):t]
            if len(w) == 0:
                feats[i, col], feats[i, col + 1] = 0.0, 0.0
            else:
                feats[i, col], feats[i, col + 1] = w.mean(), w.std()
            col += 2
        w_full = returns[max(0, t - window_size):t]
        if len(w_full) > 2 and w_full.std() > 1e-12:
            z = (w_full - w_full.mean()) / w_full.std()
            feats[i, col] = float(np.mean(z ** 3))          # skew
        col += 1
        streak = 0                                          # consecutive
        if t > 0:                                           # same-sign
            sign = np.sign(returns[t - 1])                   # returns
            j = t - 1                                        # ending at
            while j >= 0 and sign != 0 and np.sign(returns[j]) == sign:  # t-1
                streak += 1
                j -= 1
        feats[i, col] = float(streak)
        col += 1
        feats[i, col] = dur_norms[i]
    return feats


def _gbm_baseline(returns, anchors_train, dn_train, y_train,
                  anchors_val, dn_val, y_val, window_size: int) -> dict:
    X_train = _engineer_features(returns, anchors_train, dn_train, window_size)
    X_val = _engineer_features(returns, anchors_val, dn_val, window_size)
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1].astype(np.float32)
    return {"name": "GBM (engineered features)", "accuracy": _accuracy(probs, y_val),
            "brier": _brier(probs, y_val), "probs": probs}


class _DiagGRUClassifier(nn.Module):
    """Compact GRU competitor -- diagnostic only, never persisted/served.
    Same input (normalized return window + duration) as the LSTM ensemble;
    single point head instead of a K-head ensemble since this is a cheap
    sanity check, not meant to replace the LSTM's uncertainty estimate."""
    def __init__(self, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=layers,
                          batch_first=True, dropout=(0.1 if layers > 1 else 0.0))
        self.head = nn.Sequential(nn.Linear(hidden + 1, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x, d):
        _, h_n = self.gru(x)
        return self.head(torch.cat([h_n[-1], d], dim=1)).squeeze(-1)


class _DiagCausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return x + F.gelu(self.conv(F.pad(x, (self.pad, 0))))


class _DiagDilatedCNNClassifier(nn.Module):
    """Compact WaveNet-style dilated causal CNN competitor -- diagnostic
    only. Global-average-pools over time instead of an LSTM+attention
    pool, deliberately cheaper/simpler than the served model."""
    def __init__(self, channels: int = 16, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.in_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList([_DiagCausalBlock(channels, 3, d) for d in dilations])
        self.head = nn.Sequential(nn.Linear(channels + 1, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x, d):
        h = self.in_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(torch.cat([h.mean(dim=2), d], dim=1)).squeeze(-1)


def _train_diag_torch_model(model: nn.Module, train_loader, val_loader,
                            y_val: np.ndarray, epochs: int, name: str) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_probs = None
    for _ in range(1, epochs + 1):
        model.train()
        for xb, db, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb, db), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        losses, probs_chunks = [], []
        with torch.no_grad():
            for xb, db, yb in val_loader:
                logits = model(xb, db)
                losses.append(loss_fn(logits, yb).item())
                probs_chunks.append(torch.sigmoid(logits).numpy())
        val_loss = float(np.mean(losses))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_probs = np.concatenate(probs_chunks)
    print(f"[Trainer:{MODEL_KIND}] [diagnostic:{name}] best val_bce={best_val_loss:.5f}  "
          f"val_acc={_accuracy(best_probs, y_val):.3f}")
    return {"name": name, "accuracy": _accuracy(best_probs, y_val),
            "brier": _brier(best_probs, y_val), "probs": best_probs}


def run_baseline_diagnostics(result: dict, returns: np.ndarray) -> dict:
    """Runs every baseline from the model-review doc against the SAME
    purged val split the LSTM ensemble was scored on, logs a comparison
    table, and returns a JSON-safe summary (no raw prob arrays) for the
    Supabase meta row.

    Deliberately non-fatal: any individual baseline failing is caught and
    logged, never blocks the LSTM's own upload -- these are diagnostics,
    not a hard gate (except the persistence check, gated separately in
    main() via LSTM_REQUIRE_BEAT_PERSISTENCE)."""
    anchors_val, y_val, dn_val = result["anchors_val"], result["y_val"], result["dn_val"]
    anchors_train, dn_train, y_train = (result["anchors_train"], result["dn_train"],
                                        result["y_train"])
    lstm_probs = result["val_probs"]

    baselines = []
    try:
        baselines.append(_persistence_baseline(returns, anchors_val, y_val))
    except Exception as e:
        print(f"[Trainer:{MODEL_KIND}] [diagnostic] persistence baseline failed: {e}")

    try:
        val_start_t = int(anchors_val[0]) if len(anchors_val) else len(returns)
        baselines.append(_ar1_baseline(returns, returns[:val_start_t], anchors_val, y_val))
    except Exception as e:
        print(f"[Trainer:{MODEL_KIND}] [diagnostic] AR(1)/Hurst baseline failed: {e}")

    try:
        baselines.append(_gbm_baseline(returns, anchors_train, dn_train, y_train,
                                       anchors_val, dn_val, y_val, WINDOW_SIZE))
    except Exception as e:
        print(f"[Trainer:{MODEL_KIND}] [diagnostic] GBM baseline failed: {e}")

    diag_epochs = min(EPOCHS, 8)   # cheaper than the served model on purpose
    norm_returns = ((returns - result["ret_mean"]) /
                    (result["ret_std"] if result["ret_std"] > 1e-9 else 1.0)).astype(np.float32)
    diag_train_ds = RiseFallExampleDataset(norm_returns, anchors_train, dn_train, y_train, WINDOW_SIZE)
    diag_val_ds   = RiseFallExampleDataset(norm_returns, anchors_val, dn_val, y_val, WINDOW_SIZE)
    diag_train_loader = DataLoader(diag_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    diag_val_loader   = DataLoader(diag_val_ds, batch_size=BATCH_SIZE, shuffle=False)

    try:
        baselines.append(_train_diag_torch_model(
            _DiagGRUClassifier(), diag_train_loader, diag_val_loader, y_val, diag_epochs, "GRU"))
    except Exception as e:
        print(f"[Trainer:{MODEL_KIND}] [diagnostic] GRU baseline failed: {e}")

    try:
        baselines.append(_train_diag_torch_model(
            _DiagDilatedCNNClassifier(), diag_train_loader, diag_val_loader, y_val,
            diag_epochs, "dilated CNN"))
    except Exception as e:
        print(f"[Trainer:{MODEL_KIND}] [diagnostic] dilated CNN baseline failed: {e}")

    lstm_acc, lstm_brier = _accuracy(lstm_probs, y_val), _brier(lstm_probs, y_val)
    print(f"\n[Trainer:{MODEL_KIND}] ── Baseline comparison (val, n={len(y_val)}) ──────────────")
    print(f"[Trainer:{MODEL_KIND}]   {'LSTM ensemble (served)':<28} acc={lstm_acc:.3f}  brier={lstm_brier:.4f}")
    for b in baselines:
        print(f"[Trainer:{MODEL_KIND}]   {b['name']:<28} acc={b['accuracy']:.3f}  brier={b['brier']:.4f}")

    persistence_probs = next((b["probs"] for b in baselines if b["name"] == "persistence"), None)
    corr_with_persistence = None
    if persistence_probs is not None and len(lstm_probs) == len(persistence_probs) and len(lstm_probs) > 1:
        try:
            c = np.corrcoef(lstm_probs, persistence_probs)[0, 1]
            corr_with_persistence = float(c) if np.isfinite(c) else None
        except Exception:
            pass
    if corr_with_persistence is not None:
        flag = " -- HIGH: LSTM may just be learning persistence" if corr_with_persistence > 0.85 else ""
        print(f"[Trainer:{MODEL_KIND}]   correlation(LSTM, persistence) = {corr_with_persistence:.3f}{flag}")

    best_baseline = max(baselines, key=lambda b: b["accuracy"]) if baselines else None
    beats_best_baseline = best_baseline is None or lstm_acc > best_baseline["accuracy"]
    if not beats_best_baseline:
        print(f"[Trainer:{MODEL_KIND}]   WARNING: LSTM does not beat the best baseline "
              f"({best_baseline['name']}, acc={best_baseline['accuracy']:.3f}) on this val split.")
    print(f"[Trainer:{MODEL_KIND}] ────────────────────────────────────────────────────────\n")

    return {
        "lstm_accuracy": lstm_acc,
        "lstm_brier": lstm_brier,
        "baselines": [{"name": b["name"], "accuracy": b["accuracy"], "brier": b["brier"]}
                     for b in baselines],
        "corr_lstm_persistence": corr_with_persistence,
        "beats_best_baseline": bool(beats_best_baseline),
    }


# =============================================================================
# MAIN
# =============================================================================
async def main():
    print(f"[Trainer:{MODEL_KIND}] Starting RISEFALL LSTM training run ({MODEL_KIND}) "
          f"for {SYMBOL} at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    target_ticks = min(int(TRAIN_HISTORY_DAYS * 86400), MAX_TICKS)
    client = MinimalDerivClient(DERIV_APP_ID, DERIV_API_TOKEN, DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID)
    await client.connect()
    try:
        times, prices = await fetch_full_history(client, SYMBOL, target_ticks)
    finally:
        await client.close()

    if len(prices) < 1000:
        print(f"[Trainer:{MODEL_KIND}] Only {len(prices)} ticks fetched -- aborting, "
              f"not enough data to train on.")
        sys.exit(1)
    log_peak_mem("after fetching tick history")

    if MODEL_KIND == "minute":
        bar_epochs, bar_prices = build_minute_bars(times, prices)
        if len(bar_prices) < 1000:
            print(f"[Trainer:{MODEL_KIND}] Only {len(bar_prices)} minute bars built from "
                  f"{len(prices)} ticks -- aborting, need a longer TRAIN_HISTORY_DAYS pull.")
            sys.exit(1)
        prices_for_training = bar_prices.astype(np.float32)
        print(f"[Trainer:{MODEL_KIND}] Resampled {len(prices)} ticks -> {len(bar_prices)} "
              f"minute bars spanning ~{(bar_epochs[-1]-bar_epochs[0])/86400:.2f} days")
    else:
        prices_for_training = prices.astype(np.float32)
        print(f"[Trainer:{MODEL_KIND}] Fetched {len(prices)} ticks "
              f"spanning ~{len(prices)/86400:.2f} days")

    # Log-returns, not simple returns -- the earlier `diff(p)/p[:-1]` was
    # numerically close for small tick-to-tick moves but not exactly what
    # the module docstring already claimed, and log-returns are the more
    # defensible choice generally (additive across horizons, symmetric
    # treatment of up/down moves of the same magnitude). Synthetic index
    # prices are always strictly positive, but floor them defensively
    # anyway rather than let a bad tick produce a NaN that silently
    # poisons every window it touches.
    safe_prices = np.maximum(prices_for_training, 1e-9)
    returns = np.diff(np.log(safe_prices)).astype(np.float32)
    gc.collect()

    result = train_model(prices_for_training, returns)
    val_loss, val_acc = result["val_loss"], result["val_acc"]
    print(f"[Trainer:{MODEL_KIND}] Best model: val_bce={val_loss:.5f}  val_acc={val_acc:.3f}")

    diagnostics = run_baseline_diagnostics(result, returns)

    # Optional hard gate on the most basic sanity floor: if the LSTM can't
    # even beat a naive persistence baseline out-of-sample, the doc's
    # position is that "everything downstream is noise-chasing" -- default
    # ON, since shipping a model that's strictly worse than persistence to
    # a live-trading Gate 6 is a net negative, not a neutral no-op. The
    # richer AR(1)/GBM/GRU/CNN comparisons are logged + uploaded but never
    # block the upload on their own -- they're diagnostics for a human to
    # read, not gates a single val split should get veto power over.
    require_beat_persistence = os.getenv(
        "LSTM_REQUIRE_BEAT_PERSISTENCE", "true").strip().lower() not in ("0", "false", "no", "")
    persistence_entry = next(
        (b for b in diagnostics["baselines"] if b["name"] == "persistence"), None)
    if require_beat_persistence and persistence_entry is not None:
        if diagnostics["lstm_accuracy"] <= persistence_entry["accuracy"]:
            print(f"[Trainer:{MODEL_KIND}] ABORTING upload -- LSTM val_acc "
                  f"({diagnostics['lstm_accuracy']:.3f}) does not beat the persistence "
                  f"baseline ({persistence_entry['accuracy']:.3f}). The live bot keeps "
                  f"using whichever model is already in Supabase, if any. Set "
                  f"LSTM_REQUIRE_BEAT_PERSISTENCE=false to upload anyway.")
            print(f"[Trainer:{MODEL_KIND}] Done (no upload).")
            return

    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symbol": SYMBOL,
        "n_ticks_used": int(len(prices)),
        "n_train_examples": int(result["n_train"]),
        "n_val_examples": int(result["n_val"]),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "baseline_comparison": json.dumps(diagnostics),
    }
    save_model_to_supabase(result["state_dict"], meta)
    print(f"[Trainer:{MODEL_KIND}] Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        print(f"[Trainer:{MODEL_KIND}] FATAL -- training run failed with an exception:")
        traceback.print_exc()
        log_peak_mem("at failure")
        sys.exit(1)
