"""
expiryrange_lstm_train.py

Standalone training service for the EXPIRYRANGE bot's LSTM component.
Deployed as a separate Railway service on a CRON schedule (see railway.json
in this folder -- runs twice daily, e.g. 00:00 and 12:00 UTC).

This script does ONE training run and exits -- Railway's cron scheduler is
what makes it "twice a day", not a loop inside this file. That also means
it's safe/cheap to trigger manually (Railway dashboard "Run now") without
worrying about leaving a long-lived process around.

v9 REDESIGN -- trains a direct (duration, barrier) win-probability
classifier instead of a generic next-tick forecaster. See
expiryrange_lstm_model.py's docstring for the full rationale. The
practical difference here is in dataset construction (build_labeled_
examples() below): instead of one (window -> next_return) pair per tick,
each anchor point in history gets several (window, duration, barrier_sigma
-> win/loss) examples, built by looking at what ACTUALLY happened at that
point in history for randomly sampled (duration, barrier_sigma)
combinations. This needs no trade history at all -- it's constructed
entirely from price history, exactly like the live bot's own MC engine
would evaluate a barrier, just retroactively.

Pipeline:
  1. Connect to Deriv (same OTP-based auth flow as the live bot -- ticks_
     history is technically public data, but this is the one connection
     method already confirmed working for this account/app_id).
  2. Page backward through ticks_history to build a real historical tick
     series (each call is capped at 5000 ticks by Deriv; TRAIN_HISTORY_DAYS
     controls how far back we page).
  3. Build labeled (window, duration, barrier_sigma -> win/loss) examples
     by walking the historical series (see build_labeled_examples()).
  4. Train BarrierWinClassifier via binary cross-entropy against those
     realized labels.
  5. Serialize the trained state_dict (small -- a few hundred KB) to base64
     and upsert it to Supabase (bot_expiryrange_lstm_model), where the live
     bot polls for a newer model periodically and hot-reloads it.

Env vars required (same as the live bot):
  DERIV_APP_ID, DERIV_API_TOKEN, DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID
  SUPABASE_URL, SUPABASE_KEY
"""
import asyncio
import base64
import io
import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import requests
import websockets
import torch
from torch.utils.data import DataLoader

from expiryrange_lstm_model import (
    BarrierWinClassifier, WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    DURATION_TICKS_RANGE, BARRIER_SIGMA_RANGE, normalize_duration,
)

# =============================================================================
# CONFIG
# =============================================================================
SYMBOL              = os.getenv("LSTM_TRAIN_SYMBOL", "1HZ10V")
TRAIN_HISTORY_DAYS  = float(os.getenv("LSTM_TRAIN_HISTORY_DAYS", "5"))
MAX_TICKS           = int(os.getenv("LSTM_MAX_TICKS", "300000"))   # hard cap,
                                                                     # regardless
                                                                     # of DAYS,
                                                                     # to bound
                                                                     # worst-case
                                                                     # runtime/memory
TICKS_PER_HISTORY_CALL = 5000   # Deriv's ticks_history cap
EPOCHS              = int(os.getenv("LSTM_EPOCHS", "15"))
BATCH_SIZE          = int(os.getenv("LSTM_BATCH_SIZE", "256"))
LEARNING_RATE       = float(os.getenv("LSTM_LR", "1e-3"))
VAL_FRACTION        = 0.15   # held-out tail (chronological, not shuffled --
                              # shuffling would leak future info into val)

# v9: labeled-example construction (see build_labeled_examples()).
ANCHOR_STRIDE       = int(os.getenv("LSTM_ANCHOR_STRIDE", "5"))    # ticks between
                                                                     # anchor points --
                                                                     # keeps dataset
                                                                     # size manageable
                                                                     # despite heavy
                                                                     # overlap otherwise
COMBOS_PER_ANCHOR   = int(os.getenv("LSTM_COMBOS_PER_ANCHOR", "4"))  # random
                                                                       # (duration,
                                                                       # barrier_sigma)
                                                                       # samples per anchor

DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID   = os.getenv("DERIV_ACCOUNT_ID") or None

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

API_BASE   = "https://api.derivws.com/trading/v1/options"
ACCOUNTS_PATH = "/accounts"
OTP_PATH      = "/accounts/{account_id}/otp"


# =============================================================================
# MINIMAL DERIV CLIENT (connect + send only -- no trading methods needed)
# Same OTP-based auth flow as the live bot, trimmed to what training needs.
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
        print(f"[Trainer] Connected ({self.account_type}) for historical data pull.")

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
            print(f"[Trainer] WS closed: {e}")

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
# PAGINATED HISTORY FETCH
# =============================================================================
async def fetch_full_history(client: MinimalDerivClient, symbol: str,
                             target_ticks: int) -> np.ndarray:
    """
    Deriv caps a single ticks_history call at 5000 ticks. To build a real
    multi-day training series, page backward using the `end` epoch param --
    each call's oldest returned tick becomes the next call's `end`. Stops
    early if Deriv runs out of history to give (some symbols/accounts may
    not have `target_ticks` worth available) or a call fails twice in a row.
    Returns prices ordered oldest -> newest.
    """
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
            print(f"[Trainer] Empty history page (end={end}) -- retry {consecutive_empty}/2")
            await asyncio.sleep(1)
            continue
        consecutive_empty = 0
        # Prepend this page (it's older than everything fetched so far)
        all_prices = list(prices) + all_prices
        end = int(times[0]) - 1   # next page ends just before this page's oldest tick
        print(f"[Trainer] Fetched {len(times)} ticks (page ending {times[0]}) -- "
              f"{len(all_prices)}/{target_ticks} total")
        await asyncio.sleep(0.3)   # be polite to the API

    return np.array(all_prices[-target_ticks:] if len(all_prices) > target_ticks else all_prices,
                    dtype=float)


# =============================================================================
# DATASET CONSTRUCTION (v9, memory-fixed v13)
# =============================================================================
# v13 MEMORY FIX (2026-07-31): the previous version called windows.append
# (window) per example -- a full WINDOW_SIZE-length COPY of the return
# window, once per (anchor, duration, barrier_sigma) combo, even though the
# window is IDENTICAL for every combo sharing the same anchor. With
# COMBOS_PER_ANCHOR=4, that's 4x more window copies than distinct windows
# actually exist. Then `np.array(windows)` materializes ALL of them into one
# dense (n_examples, window_size) array -- at n_examples~240k this is
# ~365MB in float64, and converting that to a torch tensor for training
# makes a SECOND full copy alongside the first before the old one is
# garbage collected. That's the actual mechanism behind the "peak RSS ~676MB
# right before the first epoch" pattern -- consistent with an OOM kill on a
# memory-capped Railway plan.
#
# Fix: never materialize the window itself during dataset construction.
# Store only the lightweight anchor index (a single int) per example --
# BarrierExampleDataset (below) slices the actual window LAZILY, per
# __getitem__, directly from the shared `returns` array (a few hundred KB
# total, not duplicated). Peak memory drops from O(n_examples * window_size)
# to O(n_examples) -- for 240k examples that's roughly 640KB (three float32
# values per example: anchor index, duration_norm, barrier_sigma, label)
# instead of ~365MB+.
def build_labeled_examples(prices: np.ndarray, returns: np.ndarray,
                           window_size: int = WINDOW_SIZE,
                           rng: Optional[np.random.Generator] = None):
    """
    Builds (anchor_index, duration_norm, barrier_sigma -> win/loss) examples
    by walking historical price data. For each anchor tick t (strided by
    ANCHOR_STRIDE to bound dataset size -- consecutive anchors overlap
    heavily otherwise, adding little information for a lot more compute):

      1. recent window = returns[t-window_size : t]  (what the model sees --
         NOT stored here, see v13 note above; only `t` itself is kept)
      2. local realized vol = std(recent window) -- a fast, anchor-local
         stand-in for a full GARCH refit at every single historical point
         (which would be far too slow at this scale). This is the SAME
         window the model conditions on, so it's a reasonably grounded
         proxy for "current volatility regime" at that point in history.
      3. sample COMBOS_PER_ANCHOR random (duration_ticks, barrier_sigma)
         pairs, continuous within DURATION_TICKS_RANGE / BARRIER_SIGMA_
         RANGE (continuous sampling, not just the live bot's discrete grid
         points, so the classifier learns a smooth function instead of
         memorizing grid points).
      4. label = 1 (win) if the REAL terminal price at t+duration_ticks
         stayed within +/-barrier_abs of price[t], else 0 (loss) -- exactly
         the same symmetric win condition win_prob_from_samples() checks
         in the live bot, computed retroactively against real history.

    No trade history needed -- every label comes from price history alone.
    Returns (anchors, duration_sigma pairs, labels) as plain numpy arrays,
    still in CHRONOLOGICAL anchor order (caller does the train/val split).
    `anchors` are int64 indices into `returns`, NOT the windows themselves --
    pair with BarrierExampleDataset to get actual windows at train time.
    """
    rng = rng or np.random.default_rng()
    n = len(returns)
    max_dur = int(DURATION_TICKS_RANGE[1])
    lo_dur, hi_dur = DURATION_TICKS_RANGE
    lo_sig, hi_sig = BARRIER_SIGMA_RANGE

    anchors, ds_pairs, labels = [], [], []
    anchor_start = window_size
    anchor_end = n - max_dur - 1   # need max_dur future returns available
    if anchor_end <= anchor_start:
        return (np.empty((0,), dtype=np.int64), np.empty((0, 2)), np.empty((0,)))

    for t in range(anchor_start, anchor_end, ANCHOR_STRIDE):
        window = returns[t - window_size:t]   # cheap numpy VIEW, not copied
        local_vol_per_tick = float(np.std(window)) * float(prices[t])
        if local_vol_per_tick <= 0:
            continue

        for _ in range(COMBOS_PER_ANCHOR):
            duration_ticks = rng.uniform(lo_dur, hi_dur)
            barrier_sigma = rng.uniform(lo_sig, hi_sig)
            n_steps = max(1, int(round(duration_ticks)))
            if t + n_steps >= n:
                continue

            vol_terminal_local = local_vol_per_tick * math.sqrt(n_steps)
            barrier_abs = barrier_sigma * vol_terminal_local

            entry_price = prices[t]
            terminal_price = prices[t + n_steps]
            displacement = terminal_price - entry_price
            label = 1.0 if (-barrier_abs < displacement < barrier_abs) else 0.0

            anchors.append(t)   # NOT the window -- see v13 note above
            ds_pairs.append((normalize_duration(n_steps), barrier_sigma))
            labels.append(label)

    if not anchors:
        return (np.empty((0,), dtype=np.int64), np.empty((0, 2)), np.empty((0,)))
    return (np.array(anchors, dtype=np.int64),
            np.array(ds_pairs, dtype=np.float32),
            np.array(labels, dtype=np.float32))


class BarrierExampleDataset(torch.utils.data.Dataset):
    """
    Pairs with build_labeled_examples()'s lightweight (anchors, ds_pairs,
    labels) output. Slices each example's return window LAZILY from the
    shared `returns` array at __getitem__ time -- `returns` is one small
    array (a few hundred KB for a multi-day tick history) shared by
    reference across every example, never copied per-example. This is what
    keeps this dataset's memory footprint at O(n_examples) instead of
    O(n_examples * window_size) -- see the v13 comment above
    build_labeled_examples() for the full rationale.
    """
    def __init__(self, returns: np.ndarray, anchors: np.ndarray,
                ds_pairs: np.ndarray, labels: np.ndarray,
                window_size: int = WINDOW_SIZE):
        self.returns = returns
        self.anchors = anchors
        self.ds_pairs = ds_pairs
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        t = int(self.anchors[idx])
        window = self.returns[t - self.window_size:t]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(-1)
        ds = torch.tensor(self.ds_pairs[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, ds, y


# =============================================================================
# SUPABASE PERSISTENCE
# =============================================================================
def save_model_to_supabase(state_dict, meta: dict):
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("[Trainer] No Supabase credentials -- skipping model upload "
              "(the live bot will keep using its current model, if any).")
        return False

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "key": "current",
        "state_dict_b64": b64,
        "window_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        **meta,
    }
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/bot_expiryrange_lstm_model",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            data=json.dumps(payload), timeout=30,
        )
        resp.raise_for_status()
        print(f"[Trainer] Model uploaded to Supabase "
              f"({len(b64)} base64 chars, val_loss={meta.get('val_loss'):.5f})")
        return True
    except Exception as e:
        print(f"[Trainer] Supabase upload failed: {e}")
        return False


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_model(prices: np.ndarray, returns: np.ndarray) -> tuple:
    anchors, DS, y = build_labeled_examples(prices, returns, WINDOW_SIZE)
    if len(anchors) < 500:
        raise RuntimeError(f"Only {len(anchors)} labeled examples available (need >=500) "
                           f"-- not enough history fetched to train reliably.")

    n_val = max(int(len(anchors) * VAL_FRACTION), 50)
    anchors_train, DS_train, y_train = anchors[:-n_val], DS[:-n_val], y[:-n_val]
    anchors_val, DS_val, y_val       = anchors[-n_val:], DS[-n_val:], y[-n_val:]

    # v13: lazy datasets -- windows are sliced from `returns` on the fly per
    # __getitem__, never materialized as one giant array. See
    # BarrierExampleDataset's docstring for why this is the actual OOM fix.
    train_ds = BarrierExampleDataset(returns, anchors_train, DS_train, y_train, WINDOW_SIZE)
    val_ds   = BarrierExampleDataset(returns, anchors_val, DS_val, y_val, WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    # Val set evaluated in batches too (was one giant forward pass over the
    # whole val tensor before) -- keeps peak memory bounded regardless of
    # how large VAL_FRACTION * n_examples gets.
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BarrierWinClassifier(window_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, dsb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb, dsb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for xb, dsb, yb in val_loader:
                val_logits = model(xb, dsb)
                val_losses.append(loss_fn(val_logits, yb).item())
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_correct += (val_preds == yb).sum().item()
                val_total += len(yb)
        val_loss = float(np.mean(val_losses))
        val_acc = val_correct / max(val_total, 1)

        train_loss = float(np.mean(train_losses))
        print(f"[Trainer] epoch {epoch}/{EPOCHS}  train_bce={train_loss:.5f}  "
              f"val_bce={val_loss:.5f}  val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state, best_val_loss, best_val_acc, len(anchors_train), len(anchors_val)


# =============================================================================
# MAIN
# =============================================================================
async def main():
    print(f"[Trainer] Starting EXPIRYRANGE LSTM training run for {SYMBOL} "
          f"at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    target_ticks = min(int(TRAIN_HISTORY_DAYS * 86400), MAX_TICKS)   # 1 tick/sec assumption
    client = MinimalDerivClient(DERIV_APP_ID, DERIV_API_TOKEN, DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID)
    await client.connect()
    try:
        prices = await fetch_full_history(client, SYMBOL, target_ticks)
    finally:
        await client.close()

    if len(prices) < 1000:
        print(f"[Trainer] Only {len(prices)} ticks fetched -- aborting, "
              f"not enough data to train on.")
        sys.exit(1)

    returns = np.diff(prices) / prices[:-1]
    print(f"[Trainer] Fetched {len(prices)} ticks -> {len(returns)} returns "
          f"spanning ~{len(prices)/86400:.2f} days")

    state_dict, val_loss, val_acc, n_train, n_val = train_model(prices, returns)
    print(f"[Trainer] Best model: val_bce={val_loss:.5f}  val_acc={val_acc:.3f}")

    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symbol": SYMBOL,
        "n_ticks_used": int(len(prices)),
        "n_train_windows": int(n_train),
        "n_val_windows": int(n_val),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
    }
    save_model_to_supabase(state_dict, meta)
    print("[Trainer] Done.")


if __name__ == "__main__":
    asyncio.run(main())
