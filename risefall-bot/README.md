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

- It's a **second opinion**, not a hard requirement — same treatment as
  Gate 5's HMM/GBM Monte Carlo. It only vetoes a trade when the layer
  stack's own signal is borderline (close to its qualifying threshold)
  AND the LSTM ensemble disagrees on direction. Strong signals fire
  regardless; the LSTM's read is logged as diagnostic-only.
- When the **minute-bar** model (not the tick model) produces the most
  confident read in a given cycle — low ensemble disagreement, meaningful
  edge, direction agreeing with the layer stack — the bot will attempt a
  minute-duration Rise/Fall contract instead of its usual tick contract.
  If Deriv doesn't support minute-duration contracts for that particular
  symbol, the buy attempt fails and the bot immediately retries the same
  trade as a tick contract in the same cycle — no trade is ever dropped
  because of this.
- Until the trainer has completed at least one successful run of each
  `MODEL_KIND`, both models are simply absent and Gate 6 is a no-op.

## Safety notes

- `DERIV_ACCOUNT_TYPE=real` trades real money. The bot prints a loud
  warning banner on startup if set, but does not block it — that decision
  is left to you.
- Railway's filesystem is ephemeral. Every piece of state this bot needs
  to survive a restart (thresholds, reliability scores, gate config,
  direction history, meta-learner weights) is persisted to Supabase — if
  `SUPABASE_URL`/`SUPABASE_KEY` are unset, the bot still runs but forgets
  everything on every restart.
