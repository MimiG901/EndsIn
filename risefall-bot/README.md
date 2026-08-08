# RISEFALL bot — Railway deployment

This folder is one Railway service: the always-on live trading bot
(`risefall_bot_v4_hmm_gbm.py`). It runs continuously (background worker,
no HTTP port needed) and never exits on its own — Railway's restart policy
(`railway.json`) brings it back up if it ever crashes, and its own internal
watchdog re-execs the process in place if it goes idle for 5+ minutes.

`risefall_lstm_model.py` **must** ship in this same folder — it's the
shared model-architecture module the bot imports directly (`RiseFallWinClassifier`,
`lstm_duration_scan`, etc.) to run Gate 6, the LSTM ensemble second-opinion
check described below. It has to be byte-for-byte the same class definition
the trainer service used, or a state_dict trained by one won't load in the
other.

## Files

| File | Purpose |
|---|---|
| `risefall_bot_v4_hmm_gbm.py` | The bot. Entry point. |
| `risefall_lstm_model.py` | Shared LSTM architecture — same file as in the trainer repo. |
| `requirements.txt` | Python deps (CPU-only torch). |
| `.python-version` | Pins Python 3.11 for the Railpack build. |
| `railway.json` | Start command + restart policy. |
| `.env.example` | Every environment variable this service reads. |

## One-time Supabase setup

Run this **once**, in the Supabase SQL editor, before your first deploy —
covers both the bot's own persisted state and the LSTM table the trainer
service (deployed separately) writes into:

```sql
CREATE TABLE IF NOT EXISTS bot_trade_log (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ DEFAULT now(),
    symbol      TEXT,
    direction   INTEGER,
    step        INTEGER,
    stake       REAL,
    won         BOOLEAN,
    profit      REAL,
    p_up        REAL,
    confidence  REAL,
    duration    INTEGER,
    layer_votes JSONB,
    n_agree     INTEGER,
    n_disagree  INTEGER
);

CREATE TABLE IF NOT EXISTS bot_symbol_state (
    symbol         TEXT PRIMARY KEY,
    reliability    REAL,
    threshold      REAL,
    step0_wins     INTEGER DEFAULT 0,
    step0_total    INTEGER DEFAULT 0,
    layer_weights  JSONB  DEFAULT '{}',
    payout_history JSONB  DEFAULT '[]',
    updated_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bot_global_state (
    key        TEXT PRIMARY KEY,
    value      JSONB,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bot_gate_config (
    key        TEXT PRIMARY KEY,
    value      REAL,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- v7: written by the risefall-trainer service, read-only from the bot.
CREATE TABLE IF NOT EXISTS bot_risefall_lstm_model (
    key                TEXT PRIMARY KEY,   -- "current_tick" / "current_minute"
    kind               TEXT,
    state_dict_b64     TEXT,
    window_size        INTEGER,
    hidden_size        INTEGER,
    num_layers         INTEGER,
    n_heads            INTEGER,
    trained_at         TIMESTAMPTZ,
    symbol             TEXT,
    n_ticks_used       INTEGER,
    n_train_examples   INTEGER,
    n_val_examples     INTEGER,
    val_loss           REAL,
    val_accuracy       REAL,
    baseline_comparison JSONB,   -- persistence/AR(1)/GBM/GRU/CNN comparison
                                  -- report, see risefall_lstm_train.py
    updated_at         TIMESTAMPTZ DEFAULT now()
);

-- v8: written AND read by risefall-trainer only (the bot never touches
-- this table) -- persistent minute-bar archive that works around Deriv's
-- ~24h ticks_history retention limit. Every minute-mode cron run upserts
-- its freshly resampled bars here (deduped on symbol+epoch) and trains on
-- the full accumulated history instead of just that run's fresh fetch.
-- See risefall-trainer/README.md.
CREATE TABLE IF NOT EXISTS bot_risefall_minute_bars (
    symbol TEXT NOT NULL,
    epoch  BIGINT NOT NULL,
    price  REAL NOT NULL,
    PRIMARY KEY (symbol, epoch)
);
CREATE INDEX IF NOT EXISTS idx_risefall_minute_bars_symbol_epoch
    ON bot_risefall_minute_bars (symbol, epoch);
```

## Deploy

1. Push this folder as its own GitHub repo (or point Railway at a
   subfolder) — the root Railway builds from must contain
   `requirements.txt`, `railway.json`, and both `.py` files.
2. Railway → New Project → Deploy from GitHub repo → select it.
3. Add every variable from `.env.example` under Service → Variables.
   At minimum: `DERIV_APP_ID`, `DERIV_API_TOKEN`, `SUPABASE_URL`,
   `SUPABASE_KEY`. Leave `DERIV_ACCOUNT_TYPE=demo` until you've watched it
   trade successfully on demo.
4. Deploy. Watch the logs — it bootstraps ~10k ticks per symbol, runs a
   full deep calibration pass, then starts scanning. First trades usually
   land within a few minutes once calibration completes.
5. Also deploy the `risefall-trainer` service (separate Railway service,
   its own repo/folder) so `bot_risefall_lstm_model` actually gets
   populated — until then Gate 6 just logs "unavailable" and the bot
   trades exactly as it did before the LSTM was wired in (`LSTM_ENABLED`
   can also be set to `false` to disable Gate 6 outright).

## v9: the entire signal stack now runs on minutes, not just the LSTM

Previously, only Gate 6 (the trained LSTM) actually understood minute-bar
data — the ~16-18 layer intelligence stack (Markov, HMM regime, Hawkes,
OU, Hurst, ARFIMA, Kalman, copula, vol_trust, entropy, etc., all feeding
`bayesian_fusion`) and the Monte Carlo (`hmm_gbm_scan`,
`monte_carlo_duration`) only ever computed on raw tick data. Even with
"minute-first" candidate selection, whenever the LSTM didn't clear its
own bar, the fallback pipeline that fired was 100% tick-native — there
was no minute-scale version of it to fall back *into*.

That's fixed now. `MinuteBarView` is a thin adapter that presents a
symbol's resampled minute bars through the exact same interface
`SymbolData` itself exposes (`.symbol`, `.prices()`, `.epochs()`,
`.returns()`, `.mean_tick_dt()`). Every function in the tick-gate
pipeline was audited (see conversation history) and confirmed to only
touch its input data through that interface, or — for `entropy_gate_passes`,
`multi_timeframe_confluence`, `hmm_gbm_scan`, `monte_carlo_duration`,
`meta_ensemble_agrees` — to not take a `SymbolData`-like object at all,
just plain `prices`/`returns` arrays. That means the entire existing,
tested analytical stack runs unmodified against minute-bar data, just by
feeding it through this adapter instead of ticks — no rewrite of the
actual statistics.

**New per-cycle priority order** (both the normal scan loop and the
martingale recovery path):

1. **`try_minute_gates_candidate()`** — the full Gates 1-6 + Monte Carlo
   pipeline, running on minute bars via `MinuteBarView` and a parallel
   `state.minute_model_cache` (HMM/GARCH/OU/Hawkes fit on minute bars,
   via `fit_minute_models_for_symbol()`). If this qualifies, that trade
   fires directly — **nothing else below it even runs** this cycle.
2. **LSTM standalone** (`lstm_evaluate`, unchanged from v8) — only
   reached if step 1 didn't qualify.
3. **Tick-gate pipeline** (the original, all-ticks Gates 1-6) — the true
   fallback now, only reached if neither of the above produced anything.

**What this does NOT include (scoped down deliberately, see below)**:
the tick path's adaptive per-symbol thresholds and walk-forward-learned
per-layer weights come from `expanding_window_walk_forward()` — a
multi-fold backtesting routine that's expensive and needs a lot of
history. The minute path's models are fit directly (no walk-forward OOS
validation, no learned layer weights) and use `per_layer_weights=None`
(static defaults) plus the SAME `state.adaptive_threshold` /
`state.per_symbol_threshold` / `state.reliability` tracking the tick path
already maintains per symbol — shared, not duplicated, since both paths
trade the same underlying symbol just resampled differently. If the
minute path's win rate ends up systematically different from the tick
path's, those shared adaptive mechanisms will still drift toward
whatever's actually working, just without the head start a full
walk-forward fit would give it. Building a full parallel walk-forward
validator for minute bars would be a further, separate undertaking — say
the word if you want that too.

**A real practical constraint worth knowing**: `MinuteBarView` resamples
from the bot's own in-memory tick buffer (`SymbolData`, `maxlen`-bounded),
not from `risefall-trainer`'s persistent Supabase minute-bar archive. For
a 1HZ symbol at the default buffer size that's roughly ~200 minutes of
history — workable for Gates 1-5's shorter internal windows and for
fitting HMM/GARCH/OU/Hawkes, but thinner than what the trainer
accumulates over days via its archive. If richer live minute history
turns out to matter, the same archive pattern could be added to the bot
too.

## What Gate 6 (the LSTM ensemble) actually changes

- **v9: minute-duration trading is now the PRIMARY path, tick is the
  FALLBACK — architecturally, not just a priority tiebreak.** Every scan
  cycle, the LSTM ensemble is checked FIRST, before the tick-gate
  pipeline (Gates 1-5) even runs. If it already has a confident MINUTE-
  duration opinion (`LSTM_MIN_EDGE_STANDALONE`, default 0.12) on a
  symbol, that trade fires directly and the tick-gate pipeline is
  skipped entirely for that symbol that cycle — no wasted compute on 5
  gates that were never going to matter. The tick-gate pipeline only
  runs when the LSTM has nothing confident to say in minutes. This
  applies to both the normal scan loop and the martingale recovery path
  (which scans every ready symbol's LSTM ensemble independently before
  falling back to its own tick-based pick).
  **The confidence bar itself is unchanged** — this restructure changes
  *which path runs first*, not how confident the LSTM has to be. If you
  want minute trades to fire more often given the model's current edge
  levels, lower `LSTM_MIN_EDGE_STANDALONE` (and/or `LSTM_MIN_EDGE_FOR_MINUTE`
  for the in-pipeline override path below) — that's the intended lever,
  not a code change.
- Within `lstm_duration_scan()` itself (`risefall_lstm_model.py`), the
  same priority applies one level down: whenever the minute model
  produces anything usable at all, it's used, even if some tick
  candidate happened to have a higher raw edge. Tick is only considered
  when minute has nothing (model not loaded, window too short, or every
  minute candidate exceeded the uncertainty cap).
- Gate 6 is still a **hard veto**, not a diagnostic, inside the tick-gate
  fallback pipeline — if the LSTM ensemble disagrees with the layer
  stack's direction there, the trade is skipped. This is deliberate: the
  LSTM is the model `risefall-trainer` actually optimizes and re-uploads
  every cron cycle (unlike its five comparison baselines — persistence,
  AR(1)/Hurst, a GBM, a GRU, a dilated CNN — which stay diagnostic-only
  and never influence a live trade), so it gets real veto power to match.
  **Practical consequence**: this can meaningfully cut trade frequency,
  and the quality of your trades now depends on the LSTM actually being
  good, not just present. Consider leaving `LSTM_ENABLED=false` until
  `risefall-trainer`'s `baseline_comparison` logs show the LSTM reliably
  beating its persistence baseline (and ideally the richer AR(1)/GBM/GRU/
  CNN ones too) over a few cron cycles, then flip it on.
- If Deriv rejects `duration_unit="m"` for a particular symbol, the buy
  attempt fails and the bot immediately retries the same trade as a tick
  contract in the same cycle (the tick MC pick is always kept as a
  same-cycle fallback duration) — no trade is ever dropped because of
  this.
- Until the trainer has completed at least one successful run of each
  `MODEL_KIND`, both models are simply absent and every LSTM-related
  check is a no-op (same as `LSTM_ENABLED=false`) — the bot falls
  straight through to the tick-gate pipeline every cycle, same as before
  any of this existed.
- The trainer now trains **one shared model pooled across a whole basket
  of symbols** (see its README), not just `1HZ10V` — normalization is
  per-window/local (`local_normalize()` in `risefall_lstm_model.py`), so
  the same model applies sanely to any symbol regardless of that symbol's
  native volatility scale, including ones outside the trainer's basket.
- Trade summaries (`emit_sequence_summary`/`log_trade_summary`) now
  correctly report `minutes` vs `ticks` based on what was actually
  traded — previously this was hardcoded to always print "ticks"
  regardless of `duration_unit`, which would have silently misreported
  any minute trade that did fire.

## Two other real bugs fixed in this pass

**1. Crash loop on every drift-triggered recalibration.**
`check_calibration_triggers()` returns `("drift", [list of flagged
symbols])`, but `run_calibration()`'s startup print assumed the second
element was always a plain string (`':' + loss_symbol`), which raised
`TypeError: can only concatenate str (not "list") to str` on every single
drift event. The bot's watchdog catches this and restarts the process in
place, so it wasn't fatal, but it meant the bot could spend most of its
time crash-looping through `deep_startup_calibration()` (meant to run
ONCE per process lifetime) instead of ever settling into steady-state
trading. Fixed to handle every shape `trigger_reason`'s second element
can actually take (list, string, or `None`).

**2. Directional bias from `ConfidenceCalibrator`.** This one was a real
find, not something I noticed on my own — full credit for tracing it to
`expanding_window_walk_forward()`'s outcome label. The walk-forward
report was labeling each training example `won = (predicted_dir ==
actual_dir)` — a symmetric "was the prediction correct" label — and
feeding that into a calibrator whose whole job is producing a
*directional* `P(up)` estimate. A confident, genuinely-correct PUT call
and a confident, genuinely-correct CALL call both score `won=1`
identically, so the fitted isotonic table ends up mapping "this
confidence level" to a *win rate*, not to `P(up)` — and `calibrate()`
blends that win rate straight into a probability of the wrong thing.
Reproducing the exact scenario confirmed it's worse than it first looks:
the temperature-scaling stage collapses almost *all* directional signal
(a synthetic model with genuine, symmetric 70% accuracy calibrated a
confident PUT and a confident CALL to nearly the same ~0.63, since a
direction-blind label gives the temperature fit nothing informative to
distinguish them), and the isotonic stage on top of that further dragged
confident PUTs toward CALL specifically. Fixed by changing the label fed
into calibration to "did price actually go up" — symmetric with what
`p_up` itself means — while leaving the separate, legitimately-symmetric
hit-rate/accuracy reporting (`per_duration_outcomes`, `hits_fold`, the
per-fold hit rate you see in calibration logs) untouched, since "how
often is the model right" is a different, correctly-symmetric question
from "what does this confidence level imply about direction".
**This bug would have applied to both tick and minute trading equally**
— it's upstream of the duration/unit decision entirely, in the
directional confidence signal every gate consumes.

## Safety notes

- `DERIV_ACCOUNT_TYPE=real` trades real money. The bot prints a loud
  warning banner on startup if set, but does not block it — that decision
  is left to you.
- Railway's filesystem is ephemeral. Every piece of state this bot needs
  to survive a restart (thresholds, reliability scores, gate config,
  direction history, meta-learner weights) is persisted to Supabase — if
  `SUPABASE_URL`/`SUPABASE_KEY` are unset, the bot still runs but forgets
  everything on every restart.
