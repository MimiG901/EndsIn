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

## What Gate 6 (the LSTM ensemble) actually changes

- Gate 6 is a **hard veto**, not a diagnostic — different from Gate 5's
  HMM/GBM Monte Carlo, which only vetoes borderline signals. If the LSTM
  ensemble disagrees with the layer stack's direction, the trade is
  skipped, on every signal, not just weak ones. This is deliberate: the
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
- **v8: the LSTM ensemble can now originate a trade entirely on its own**
  (`LSTM_MIN_EDGE_STANDALONE`, default 0.12 -- a higher confidence bar
  than the minute-override threshold below, since a standalone LSTM trade
  has none of Gates 1-5's tick-based corroboration behind it). Previously
  a trained minute model could only ever ride along on top of an already-
  qualified tick trade, as a duration swap -- if the tick-based layer
  stack (Gates 1-5) didn't qualify a symbol that cycle, the minute
  model's opinion never got a chance to matter, no matter how confident
  it was. Now both pipelines run independently every cycle, and
  **whichever is rated higher wins** -- rating is `|p-0.5|` (edge) for
  both, since that's directly comparable regardless of which one produced
  it. Watch for `[Signal]`/`[LSTM]` log lines showing which pipeline's
  pick actually got traded. The atomic final recheck immediately before
  firing also branches correctly by source: a `tick_gates` trade still
  gets Gate 1 re-verified right before execution (unchanged); an
  `lstm_standalone` trade gets the ensemble itself re-verified instead
  (re-checking Gate 1 on a trade Gate 1 was never part of would defeat
  the whole point).
- When the **minute-bar** model (not the tick model) produces the most
  confident read in a given cycle — low ensemble disagreement, meaningful
  edge, direction already agreeing with the layer stack (Gate 6 already
  vetoed any disagreement above) — the bot will attempt a minute-duration
  Rise/Fall contract instead of its usual tick contract. If Deriv doesn't
  support minute-duration contracts for that particular symbol, the buy
  attempt fails and the bot immediately retries the same trade as a tick
  contract in the same cycle — no trade is ever dropped because of this.
- Until the trainer has completed at least one successful run of each
  `MODEL_KIND`, both models are simply absent and Gate 6 is a no-op (same
  as `LSTM_ENABLED=false`) — it never blocks trades it has no opinion on,
  and the standalone-origination path above simply never fires either.
- The trainer now trains **one shared model pooled across a whole basket
  of symbols** (see its README), not just `1HZ10V` — normalization is
  per-window/local (`local_normalize()` in `risefall_lstm_model.py`), so
  the same model applies sanely to any symbol regardless of that symbol's
  native volatility scale, including ones outside the trainer's basket.

Note: the martingale recovery path (continuing an already-lost sequence
on a specific symbol/direction) still only uses the tick-gate pipeline +
Gate 6's veto/minute-override — it doesn't yet have the standalone-
origination path, since "continue this specific losing sequence" is a
different kind of decision than "originate a fresh trade" and it wasn't
obviously right to let the LSTM redirect a recovery sequence onto a
symbol/direction the martingale logic wasn't already committed to.

## Safety notes

- `DERIV_ACCOUNT_TYPE=real` trades real money. The bot prints a loud
  warning banner on startup if set, but does not block it — that decision
  is left to you.
- Railway's filesystem is ephemeral. Every piece of state this bot needs
  to survive a restart (thresholds, reliability scores, gate config,
  direction history, meta-learner weights) is persisted to Supabase — if
  `SUPABASE_URL`/`SUPABASE_KEY` are unset, the bot still runs but forgets
  everything on every restart.
