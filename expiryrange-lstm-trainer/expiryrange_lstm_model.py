"""
expiryrange_lstm_model.py

Shared model architecture for the EXPIRYRANGE bot's LSTM component.
Imported by BOTH the training service (expiryrange_lstm_train.py) and the
live bot (expiryrange_bot_v3_1hz10v.py) -- keeping the class definition in
one place means a state_dict trained by one always loads correctly in the
other; a drifted/duplicated class definition is a classic way to get
silent shape-mismatch bugs after a retrain.

v9 REDESIGN -- direct (duration, barrier) win-probability classifier
---------------------------------------------------------------------
The original (v8) version predicted a generic next-tick Gaussian (mu,
sigma) and let the caller extrapolate to a terminal distribution via the
same sqrt(duration)-scaling assumption GARCH and the HMM/GBM sampler
already make. That's a real limitation: the LSTM never actually saw the
thing the bot needs answered -- "given this window, will price stay
within +/-barrier for the next N ticks?" -- it just forecasted one tick
and left extrapolation to a formula.

v9 instead trains the model to answer that exact question directly:
  Input:  the last WINDOW_SIZE tick returns (via LSTM encoder) PLUS the
          specific (duration, barrier_sigma) being evaluated.
  Output: P(price stays within +/-barrier_sigma * local_vol_terminal at
          expiry) -- a single probability, via sigmoid.
No generic forecast, no extrapolation formula -- the model is trained
directly against realized historical outcomes for real (duration,
barrier) combinations (see expiryrange_lstm_train.py's label construction:
walks historical ticks, computes what ACTUALLY happened for sampled
(duration, barrier_sigma) pairs, no live trading required to generate
labels).

EFFICIENCY NOTE: the LSTM encoding of the return window does NOT depend on
duration or barrier_sigma -- only the small head does. So the live bot
computes the encoder's hidden state ONCE per symbol per cycle
(compute_hidden()), then batches every (duration, barrier_sigma) combo in
the sweep through the tiny head in a single forward pass
(predict_win_probs_batch()) -- this is actually cheaper than the v8
per-duration sampling approach, not more expensive, despite conditioning
on more information.

NOTE ON GRID CONSTANTS BELOW: DURATION_TICKS_RANGE / BARRIER_SIGMA_RANGE
are the trainer's sampling ranges for synthetic labels, kept in loose sync
with DURATION_CANDIDATES / BARRIER_SIGMAS in the main bot file. Exact sync
isn't critical -- the classifier is a continuous function of duration and
barrier_sigma, not a lookup table, so it generalizes across nearby values.
Update these if the main bot's grids move to a very different range.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

WINDOW_SIZE  = 200   # ticks of recent returns fed in as input
HIDDEN_SIZE  = 32    # kept small deliberately -- a small, regularizing
                     # choice for a near-random-walk series.
NUM_LAYERS   = 2

# Trainer's sampling ranges for synthetic (duration, barrier_sigma) labels.
# Keep loosely in sync with DURATION_CANDIDATES / BARRIER_SIGMAS in the
# main bot file (deriv_multisymbol_bot__1_.py).
DURATION_TICKS_RANGE   = (60.0, 480.0)
BARRIER_SIGMA_RANGE    = (0.85, 1.35)
DURATION_NORM_SCALE    = DURATION_TICKS_RANGE[1]   # normalize duration to ~[0.125, 1.0]


def normalize_duration(n_steps: float) -> float:
    return float(n_steps) / DURATION_NORM_SCALE


class BarrierWinClassifier(nn.Module):
    """
    encode(): (batch, window_size, 1) -> (batch, hidden_size) LSTM hidden state.
    forward(): full path for TRAINING -- (window_batch, duration_sigma_batch)
               -> raw logits (batch,). Use nn.BCEWithLogitsLoss against 0/1
               win labels.
    compute_hidden() / predict_win_probs_batch(): INFERENCE path for the
               live bot -- encode once per cycle, batch many (duration,
               barrier_sigma) evaluations through the head cheaply.
    """
    def __init__(self, window_size: int = WINDOW_SIZE,
                hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # +2 inputs: normalized duration, barrier_sigma
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, 1)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # (batch, hidden_size) -- final layer's hidden state

    def forward(self, x: torch.Tensor, duration_sigma: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, 1); duration_sigma: (batch, 2)
        [duration_norm, barrier_sigma]. Returns raw logits (batch,) --
        caller applies sigmoid (or BCEWithLogitsLoss, which does it
        internally and is more numerically stable for training)."""
        hidden = self.encode(x)
        combined = torch.cat([hidden, duration_sigma], dim=1)
        return self.head(combined).squeeze(-1)

    def compute_hidden(self, recent_returns) -> torch.Tensor:
        """Live-bot inference helper: takes a 1-D array-like of the most
        recent tick returns (any length -- padded/truncated to
        window_size), returns the encoded hidden state (1, hidden_size) as
        a detached tensor, ready to reuse across many predict_win_probs_
        batch() calls this cycle."""
        arr = np.asarray(recent_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= self.window_size:
            arr = arr[-self.window_size:]
        else:
            # Left-pad with zeros -- only matters right after a fresh
            # deploy/model swap before window_size ticks have accumulated.
            arr = np.concatenate([np.zeros(self.window_size - len(arr)), arr])
        x = torch.tensor(arr, dtype=torch.float32).view(1, self.window_size, 1)
        self.eval()
        with torch.no_grad():
            return self.encode(x)

    def predict_win_probs_batch(self, hidden: torch.Tensor,
                                duration_sigma_pairs: List[Tuple[float, float]]
                                ) -> np.ndarray:
        """hidden: (1, hidden_size) from compute_hidden(), reused across
        this whole call. duration_sigma_pairs: list of (n_steps,
        barrier_sigma) -- RAW n_steps (ticks), NOT pre-normalized; this
        function normalizes internally so callers don't have to remember
        to. Returns a numpy array of P(win), same order as the input list.
        """
        n = len(duration_sigma_pairs)
        if n == 0:
            return np.array([])
        hidden_rep = hidden.expand(n, -1)
        ds = torch.tensor(
            [[normalize_duration(d), s] for d, s in duration_sigma_pairs],
            dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            combined = torch.cat([hidden_rep, ds], dim=1)
            logits = self.head(combined).squeeze(-1)
            probs = torch.sigmoid(logits)
        return probs.numpy()
