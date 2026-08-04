"""
risefall_lstm_model.py

Shared model architecture for the RISEFALL bot's LSTM layer(s). Imported by
BOTH the training service(s) (risefall_lstm_train.py) and the live bot
(risefall_bot_v4_hmm_gbm.py) -- same reasoning as expiryrange_lstm_model.py:
one class definition, so a state_dict trained by one always loads in the
other.

WHY THIS ISN'T JUST expiryrange_lstm_model.py WITH A DIFFERENT LABEL
----------------------------------------------------------------------
EXPIRYRANGE asks "does price stay inside a barrier". RISEFALL asks something
strictly simpler -- "is price higher or lower than entry at expiry" -- so a
single-scalar target isn't the interesting part of this redesign. Two other
things are:

1. RISEFALL's Monte Carlo (monte_carlo_duration() / hmm_gbm_scan() in
   risefall_bot_v4_hmm_gbm.py) currently only ever sweeps CANDIDATE_DURATIONS
   in TICKS -- duration_unit is hardcoded to "t" at the one call site that
   places a trade (execute_single_step()). Minute-expiry contracts are a
   completely untapped candidate set. But a tick window and a minute-bar
   window are not the same statistical object: a tick contract settles
   after N raw ticks (microstructure-heavy, ~1-2s apart); a minute contract
   settles after N minutes of WALL-CLOCK time regardless of how many ticks
   land in it (closer to a low-frequency OHLC-style series, autocorrelation
   and vol clustering look different at that scale). Feeding both through
   one encoder and just tacking a "duration in seconds" scalar onto the head
   asks a single small LSTM to learn two different regimes' worth of
   dynamics from one weight set. Cleaner to give each its own encoder+model,
   trained on the input representation that's actually appropriate for it,
   and let the live bot pick whichever one matches the duration_unit being
   evaluated at that point in the MC sweep.

2. A point estimate P(up) with no uncertainty is exactly the failure mode
   the rest of this bot's stack (DriftDetector, ConfidenceCalibrator,
   MetaLearner) exists to compensate for after the fact. Cheaper to have the
   model say "I don't know" up front. RiseFallWinClassifier below is a small
   deep ensemble (shared encoder, K independent heads, decorrelated via
   per-head bagging masks at train time -- see risefall_lstm_train.py) so
   inference returns (p_mean, p_std) instead of just p. p_std is a genuine
   epistemic-disagreement signal the live bot can gate on directly, the same
   way it already gates on edge/agreement from the other 18 layers.

TWO MODEL INSTANCES, ONE CLASS
----------------------------------------------------------------------
RiseFallWinClassifier is instantiated twice, with different window sizes
and constants -- NOT two different classes:
    tick_model   = RiseFallWinClassifier(kind="tick",   window_size=WINDOW_SIZE_TICKS)
    minute_model = RiseFallWinClassifier(kind="minute", window_size=WINDOW_SIZE_MINUTES)
Each is trained, persisted (separate Supabase rows -- see
risefall_lstm_train.py), and hot-reloaded independently. `kind` is stored on
the instance and included in the serialized meta purely so a state_dict can
never be silently loaded into the wrong slot.

DEPTH, vs the v9 EXPIRYRANGE encoder
----------------------------------------------------------------------
  - 2-layer dilated causal Conv1d front end (kernel sizes widen the
    receptive field cheaply before the LSTM ever sees the sequence -- lets
    the LSTM start from locally-smoothed multi-scale features instead of
    raw single-tick noise) -> stacked LSTM (NUM_LSTM_LAYERS, default 3,
    vs 2 in EXPIRYRANGE) -> additive attention pooling over every timestep's
    LSTM output (vs EXPIRYRANGE's plain "take the final hidden state").
    Attention pooling matters more here than it would for a barrier
    classifier: recent-vs-early-window relevance genuinely shifts with
    regime (e.g. a jump 150 ticks back should matter less once the
    HMM/ADX/vol regime has visibly changed since), and a fixed final
    hidden state can't reweight that per-example the way attention can.
  - N_ENSEMBLE_HEADS (default 5) independent small MLP heads on top of the
    shared encoder, instead of EXPIRYRANGE's single head.

INTERFACE PARITY WITH expiryrange_lstm_model.py (deliberate)
----------------------------------------------------------------------
compute_hidden() / predict_probs_batch() mirror EXPIRYRANGE's
compute_hidden() / predict_win_probs_batch() split for the same reason: the
encoder pass is the expensive part and does NOT depend on which candidate
duration is being scored, so the live bot encodes once per symbol per cycle
and batches every candidate duration in CANDIDATE_DURATIONS /
CANDIDATE_DURATIONS_MINUTES through the cheap head in one forward pass.

NORMALIZATION IS PER-WINDOW AND LOCAL, NOT A GLOBAL BAKED-IN SCALAR
----------------------------------------------------------------------
See local_normalize() below. Each window is z-scored against its own
mean/std at both train and inference time -- there's no scaler object or
persisted buffer to keep in sync, and critically, it generalizes to any
symbol regardless of that symbol's native volatility scale, which matters
because the live bot's symbol universe is discovered dynamically (see
fetch_tradable_symbols()/select_top_1hz() in the bot) and isn't fully
known even at training time.
"""
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CONSTANTS
# =============================================================================
# Tick-window model: raw tick log-returns, same granularity as the bot's
# existing SymbolData tick stream.
WINDOW_SIZE_TICKS   = 200

# Minute-bar model: log-returns of last-price-per-minute bars, resampled
# from the same tick stream (see risefall_lstm_train.build_minute_bars() and
# the live-bot-side equivalent the encoder docstring below assumes exists,
# e.g. SymbolData.minute_bar_returns()). 200 one-minute bars = ~3.3 hours of
# context, deliberately much longer wall-clock lookback than the tick model
# gets, since minute contracts are themselves a much longer-horizon bet.
WINDOW_SIZE_MINUTES = 200

CONV_CHANNELS   = 16     # dilated causal conv front-end width
CONV_KERNEL     = 3
CONV_DILATIONS  = (1, 2, 4)   # receptive field ~ 1 + 2*sum((k-1)*d) = 25 ticks/bars
LSTM_HIDDEN     = 48      # vs EXPIRYRANGE's 32 -- still small/regularizing,
                          # bumped slightly to give the attention pool
                          # something worth attending over.
NUM_LSTM_LAYERS = 3
ATTN_HIDDEN     = 24
# Inter-layer dropout applied by nn.LSTM between its stacked layers (PyTorch
# doesn't expose true per-timestep recurrent dropout on nn.LSTM without a
# custom cell, but inter-layer dropout is the standard practical stand-in
# and is what most "add dropout to the LSTM" advice actually means in
# practice). Combined with HEAD_DROPOUT + the bagging masks at train time.
LSTM_DROPOUT    = 0.1

N_ENSEMBLE_HEADS = 5
HEAD_HIDDEN      = 24
HEAD_DROPOUT     = 0.15   # applied inside each head; combined with the
                          # per-head bagging mask at train time, this is
                          # what actually decorrelates the K heads --
                          # dropout alone at inference time with no bagging
                          # would just be K noisy copies of the same head.

# Candidate duration grids this model family is trained/queried against.
# Kept in loose sync with CANDIDATE_DURATIONS (ticks) in
# risefall_bot_v4_hmm_gbm.py and its not-yet-existing minute counterpart --
# exact sync isn't critical, same reasoning as EXPIRYRANGE's grid constants:
# the head is a continuous function of duration, not a lookup table.
CANDIDATE_DURATIONS_TICKS   = [1, 3, 5, 7, 10]
CANDIDATE_DURATIONS_MINUTES = [1, 2, 3, 5, 10]

# Duration is normalized on a log scale before hitting the head: a 1-tick
# vs 10-tick difference and a 1-minute vs 10-minute difference are both
# "10x", but 10 ticks (~20s) and 10 minutes (600s) are a 30x gap in real
# time -- log-scaling keeps the head's input well-conditioned across both
# regimes without needing a shared linear scale that flatters one of them.
DURATION_LOG_NORM_CAP_TICKS   = float(max(CANDIDATE_DURATIONS_TICKS) * 3)
DURATION_LOG_NORM_CAP_MINUTES = float(max(CANDIDATE_DURATIONS_MINUTES) * 3)


def normalize_duration_count(n_units: float, kind: str) -> float:
    """Log-scale-normalizes a raw duration count (ticks or minutes) into
    roughly [0, 1] for head input. `kind` selects the cap so a "10" means
    something calibrated to that model's own candidate grid, not the other
    one's."""
    cap = DURATION_LOG_NORM_CAP_TICKS if kind == "tick" else DURATION_LOG_NORM_CAP_MINUTES
    n = max(float(n_units), 0.0)
    return float(math.log1p(n) / math.log1p(cap))


def local_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-window z-score, computed independently for every example from
    its OWN window -- deliberately NOT a global scalar fit once at train
    time and baked into the model. The live bot trades a DYNAMIC,
    data-driven basket of volatility-index symbols discovered fresh at
    every deep calibration (see fetch_tradable_symbols()/select_top_1hz()
    in risefall_bot_v4_hmm_gbm.py); the exact set isn't fully known even
    at training time, and different symbols in that family have very
    different native volatility scales (a Volatility 100 index's returns
    are roughly 10x the magnitude of a Volatility 10 index's). A single
    baked-in mean/std calibrated to whichever symbols happened to be in
    the training pool would systematically mis-scale any symbol whose
    volatility differs from that pool. Local, per-window normalization
    makes every window ~N(0,1) regardless of which symbol or scale
    produced it -- genuinely scale-invariant, and it generalizes
    correctly even to a symbol the trainer never saw.

    x: (batch, seq_len, 1). risefall_lstm_train.py's diagnostic GRU/CNN
    competitors call this too, so the baseline comparison isn't confounded
    by the served model getting a normalization advantage they didn't."""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std


# =============================================================================
# ENCODER
# =============================================================================
class _CausalConvBlock(nn.Module):
    """One dilated causal Conv1d + GELU + residual. Causal padding (pad only
    on the left) so no future information leaks into a given timestep --
    matters here even though the LSTM after it is already causal, because
    otherwise this front end alone would defeat the point of an online,
    tick-by-tick live bot."""
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)   # ~LayerNorm over channels, batch-size-robust

    def forward(self, x):
        # x: (batch, channels, seq_len)
        y = F.pad(x, (self.pad, 0))
        y = self.conv(y)
        y = self.norm(y)
        y = F.gelu(y)
        return x + y


class AttentionPool(nn.Module):
    """Additive (Bahdanau-style) attention pooling over an LSTM's full
    output sequence, collapsing (batch, seq_len, hidden) -> (batch, hidden).
    Replaces EXPIRYRANGE's "just take h_n[-1]" -- lets the model learn which
    part of the window is actually informative for THIS example instead of
    always trusting the most recent timestep equally across every regime."""
    def __init__(self, hidden_size: int, attn_hidden: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (batch, seq_len, hidden)
        scores = self.score(seq).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)         # (batch, seq_len)
        pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)  # (batch, hidden)
        return pooled


class RiseFallEncoder(nn.Module):
    """Dilated causal conv front-end -> stacked LSTM -> attention pooling.
    Operates identically whether fed raw tick returns or minute-bar
    returns -- the only difference between the tick and minute MODELS is
    which window_size/data they're instantiated and trained with, not the
    encoder architecture itself."""
    def __init__(self, window_size: int, hidden_size: int = LSTM_HIDDEN,
                num_layers: int = NUM_LSTM_LAYERS):
        super().__init__()
        self.window_size = window_size
        self.in_proj = nn.Conv1d(1, CONV_CHANNELS, kernel_size=1)
        self.conv_blocks = nn.ModuleList([
            _CausalConvBlock(CONV_CHANNELS, CONV_KERNEL, d) for d in CONV_DILATIONS
        ])
        self.lstm = nn.LSTM(input_size=CONV_CHANNELS, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=(LSTM_DROPOUT if num_layers > 1 else 0.0))
        self.attn_pool = AttentionPool(hidden_size, ATTN_HIDDEN)
        self.out_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, 1)
        h = x.transpose(1, 2)               # (batch, 1, window_size)
        h = self.in_proj(h)                 # (batch, CONV_CHANNELS, window_size)
        for block in self.conv_blocks:
            h = block(h)
        h = h.transpose(1, 2)               # (batch, window_size, CONV_CHANNELS)
        seq, _ = self.lstm(h)               # (batch, window_size, hidden_size)
        return self.attn_pool(seq)          # (batch, hidden_size)


# =============================================================================
# ENSEMBLE CLASSIFIER
# =============================================================================
class RiseFallWinClassifier(nn.Module):
    """
    P(price is higher than entry at expiry | recent window, candidate
    duration), as a small deep ensemble.

    kind: "tick" or "minute" -- purely descriptive/safety metadata (also
    picks the duration log-norm cap), stored on the instance and echoed
    into the serialized meta dict so a checkpoint can't silently get loaded
    into the wrong live-bot slot.

    encode() / forward(): training-time full path, ALL heads at once.
    compute_hidden() / predict_probs_batch(): live-bot inference path --
    encode once per symbol per cycle, batch every candidate duration in
    this model's own CANDIDATE_DURATIONS_* grid through the heads cheaply.
    predict_probs_batch() returns (p_mean, p_std) per duration -- p_std is
    the ensemble's disagreement, i.e. an epistemic uncertainty estimate the
    live bot can gate on directly (see lstm_duration_scan() below).
    """
    def __init__(self, kind: str, window_size: int,
                hidden_size: int = LSTM_HIDDEN, num_layers: int = NUM_LSTM_LAYERS,
                n_heads: int = N_ENSEMBLE_HEADS):
        super().__init__()
        assert kind in ("tick", "minute"), f"kind must be 'tick' or 'minute', got {kind!r}"
        self.kind = kind
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_heads = n_heads

        self.encoder = RiseFallEncoder(window_size, hidden_size, num_layers)
        # +1 input: normalized duration. No barrier_sigma -- RISEFALL has no
        # barrier, direction alone is the whole question.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + 1, HEAD_HIDDEN),
                nn.GELU(),
                nn.Dropout(HEAD_DROPOUT),
                nn.Linear(HEAD_HIDDEN, 1),
            ) for _ in range(n_heads)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(local_normalize(x))

    def forward(self, x: torch.Tensor, duration_norm: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, 1); duration_norm: (batch, 1) already
        log-normalized (see normalize_duration_count()). Returns raw logits
        stacked per head: (n_heads, batch). Caller (training loop) applies
        BCEWithLogitsLoss per head and combines -- see risefall_lstm_train's
        per-head bagging-mask loss."""
        hidden = self.encode(x)                          # (batch, hidden_size)
        combined = torch.cat([hidden, duration_norm], dim=1)   # (batch, hidden_size+1)
        logits = torch.stack([head(combined).squeeze(-1) for head in self.heads], dim=0)
        return logits   # (n_heads, batch)

    def compute_hidden(self, recent_window) -> torch.Tensor:
        """Live-bot inference helper. `recent_window` is a 1-D array-like:
        tick log-returns for the tick model, minute-bar log-returns for the
        minute model -- caller is responsible for passing the right series
        for this instance's `kind`. Padded/truncated to window_size, same
        convention as EXPIRYRANGE's compute_hidden()."""
        arr = np.asarray(recent_window, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= self.window_size:
            arr = arr[-self.window_size:]
        else:
            arr = np.concatenate([np.zeros(self.window_size - len(arr)), arr])
        x = torch.tensor(arr, dtype=torch.float32).view(1, self.window_size, 1)
        self.eval()
        with torch.no_grad():
            return self.encode(x)

    def predict_probs_batch(self, hidden: torch.Tensor,
                            duration_counts: Sequence[float]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """hidden: (1, hidden_size) from compute_hidden(), reused across the
        whole call. duration_counts: RAW counts in this model's own unit
        (ticks for the tick model, minutes for the minute model) -- NOT
        pre-normalized, same "caller doesn't have to remember" convention
        as EXPIRYRANGE. Returns (p_mean, p_std), each shape (len(duration_
        counts),) -- p_std is ensemble disagreement across the n_heads."""
        n = len(duration_counts)
        if n == 0:
            return np.array([]), np.array([])
        hidden_rep = hidden.expand(n, -1)
        dn = torch.tensor(
            [[normalize_duration_count(c, self.kind)] for c in duration_counts],
            dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            combined = torch.cat([hidden_rep, dn], dim=1)
            per_head_logits = torch.stack(
                [head(combined).squeeze(-1) for head in self.heads], dim=0)  # (n_heads, n)
            per_head_probs = torch.sigmoid(per_head_logits)
        p_mean = per_head_probs.mean(dim=0).numpy()
        p_std = per_head_probs.std(dim=0).numpy()
        return p_mean, p_std


# =============================================================================
# LIVE-BOT INTEGRATION HELPER
# =============================================================================
# Not required to use the models -- compute_hidden()/predict_probs_batch()
# are sufficient on their own -- but this mirrors hmm_gbm_scan()'s shape
# (risefall_bot_v4_hmm_gbm.py) closely enough that it can be dropped in as
# an additional MC-blend input with minimal glue code, and it's the piece
# that actually exercises BOTH models across BOTH duration units in one
# call, which is the whole point of splitting them.
def lstm_duration_scan(tick_model: Optional["RiseFallWinClassifier"],
                       minute_model: Optional["RiseFallWinClassifier"],
                       tick_returns_window: Optional[np.ndarray],
                       minute_returns_window: Optional[np.ndarray],
                       tick_durations: List[int] = None,
                       minute_durations: List[int] = None,
                       max_uncertainty: float = 0.18) -> dict:
    """
    Sweeps both duration units through their respective models and returns
    the best (direction, duration, duration_unit) pick, same return shape
    as hmm_gbm_scan() plus `duration_unit` and `p_std`.

    MINUTE TAKES PRIORITY OVER TICK -- not a pure edge comparison. If the
    minute model produced ANY candidate that clears max_uncertainty, that
    candidate (the best-by-edge among minute candidates specifically) is
    returned, regardless of whether some tick candidate happened to have
    a numerically higher edge. Tick is only used as a fallback when the
    minute model produced nothing usable this cycle (model not loaded
    yet, minute_returns_window too short, or every minute candidate
    exceeded the uncertainty cap). This is a deliberate policy, not an
    accident of implementation -- minute-duration RISEFALL contracts are
    the priority target, ticks are the fallback.

    max_uncertainty: candidates whose ensemble p_std exceeds this are
    dropped from consideration entirely before picking the best edge --
    a confident-looking 0.58 with p_std=0.20 (heads scattered from 0.40 to
    0.80) is worth less than a 0.54 with p_std=0.02, and edge alone can't
    tell the two apart.

    Returns {"direction": +1/-1, "duration": int, "duration_unit": "t"|"m",
             "p": float, "p_std": float, "edge": float, "grid": {...}}
    or None if neither model produced any candidate under the uncertainty
    cap (caller should fall back to the existing HMM/GBM MC alone).
    """
    tick_durations = tick_durations or CANDIDATE_DURATIONS_TICKS
    minute_durations = minute_durations or CANDIDATE_DURATIONS_MINUTES

    grid = {}
    best_by_unit = {"t": None, "m": None}

    def _consider(model, window, durations, unit):
        if model is None or window is None or len(window) < 5:
            return
        hidden = model.compute_hidden(window)
        p_up, p_std = model.predict_probs_batch(hidden, durations)
        for dur, p, s in zip(durations, p_up, p_std):
            grid[(unit, dur)] = (float(p), float(s))
            if s > max_uncertainty:
                continue
            for direction, p_dir in ((1, float(p)), (-1, 1.0 - float(p))):
                edge = abs(p_dir - 0.5)
                current = best_by_unit[unit]
                if current is None or edge > current["edge"]:
                    best_by_unit[unit] = {"direction": direction, "duration": int(dur),
                                          "duration_unit": unit, "p": p_dir,
                                          "p_std": float(s), "edge": edge}

    _consider(tick_model, tick_returns_window, tick_durations, "t")
    _consider(minute_model, minute_returns_window, minute_durations, "m")

    # Minute priority: use it whenever it produced anything usable at all,
    # regardless of relative edge vs. the tick pick. Only fall back to
    # tick when minute has nothing.
    best = best_by_unit["m"] if best_by_unit["m"] is not None else best_by_unit["t"]

    if best is not None:
        best["grid"] = grid
    return best
