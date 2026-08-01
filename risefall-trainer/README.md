# RISEFALL LSTM trainer — Railway deployment

This folder is one Railway service: a native **Cron Job** (not a
long-running worker) that trains both the tick and the minute RISEFALL
LSTM models every 2 hours and uploads whichever ones pass their sanity
check to Supabase, for the `risefall-bot` service (deployed separately) to
pick up.

`risefall_lstm_model.py` **must** be byte-for-byte identical to the copy
in the `risefall-bot` folder — it's the shared architecture class both
services import, and a state_dict trained here has to load cleanly into
the bot's `RiseFallWinClassifier` instance.

## How the schedule works

`railway.json`'s `deploy.cronSchedule: "0 */2 * * *"` triggers a fresh
container every 2 hours (UTC). It runs `entrypoint.sh`, which trains
`MODEL_KIND=tick` then `MODEL_KIND=minute` **sequentially in the same
run** (two separate `risefall_lstm_train.py` processes, one after the
other), then the container exits. Railway does not start the next
scheduled run until the current one has finished — if training runs long,
you'll simply see fewer runs per day rather than overlapping ones.

Trigger it manually any time from the Railway dashboard/CLI to test
without waiting for the schedule.

## Files

| File | Purpose |
|---|---|
| `risefall_lstm_train.py` | Trains one model (`MODEL_KIND` env var picks tick/minute), runs baseline diagnostics, uploads to Supabase. |
| `risefall_lstm_model.py` | Shared LSTM architecture — same file as in the bot repo. |
| `entrypoint.sh` | Runs both `MODEL_KIND` values sequentially per cron trigger. |
| `requirements.txt` | Python deps — CPU-only torch + scikit-learn (for the GBM baseline). |
| `.python-version` | Pins Python 3.11 for the Railpack build. |
| `railway.json` | Cron schedule + start command. |
| `.env.example` | Every environment variable this service reads. |

## Deploy

1. Run the Supabase SQL from the `risefall-bot` README once (both
   services share the same `bot_risefall_lstm_model` table) — do this
   **before** the first cron trigger fires.
2. Push this folder as its own GitHub repo.
3. Railway → New Project → Deploy from GitHub repo → select it.
4. Add every variable from `.env.example`. At minimum: `DERIV_APP_ID`,
   `DERIV_API_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY`.
5. Deploy once to confirm it builds, then either wait for the next
   2-hour mark or trigger a manual run from the dashboard. Watch the
   logs — a full tick+minute cycle typically takes several minutes
   (history pull + example construction + training + 5 baseline
   diagnostics, twice).

## What actually gets trained and uploaded, every run

For **each** of `MODEL_KIND=tick` and `MODEL_KIND=minute`:

1. Pulls recent tick history from Deriv (`LSTM_TRAIN_HISTORY_DAYS`,
   capped at `LSTM_MAX_TICKS`), resampling into minute bars first if
   `MODEL_KIND=minute` (see `build_minute_bars()`).
2. Builds labeled direction examples, chronologically splits train/val,
   and **purges** any training example whose label horizon reaches into
   the validation window — a walk-forward split alone isn't enough here,
   since a label looks forward by up to `max(CANDIDATE_DURATIONS)`.
3. Computes return-normalization stats from the **training split only**
   and bakes them into the model as persisted buffers (`return_mean`/
   `return_std`) — the exact same transform is then applied automatically
   at live-bot inference time, no separate scaler to keep in sync.
4. Trains the served `RiseFallWinClassifier` deep ensemble (dilated causal
   conv front-end → 3-layer LSTM w/ inter-layer dropout → attention pool →
   5 bagged heads).
5. Runs it against **five baselines** on the identical purged validation
   split — see `run_baseline_diagnostics()`:
   - **Persistence** (naive "whatever just happened, keep happening")
   - **AR(1) + Hurst exponent** (linear autocorrelation baseline; Hurst is
     logged as a standalone diagnostic of how much real short-range
     memory the process has at all)
   - **HistGradientBoostingClassifier** on hand-engineered return-window
     features (multi-scale mean/std, skew, streak length)
   - **Compact GRU** and **compact dilated causal CNN** (same input, cheap
     point-estimate competitors — not the served ensemble)

   None of the five are served to the bot; this is purely a "is the extra
   complexity earning its keep" report, logged to console and uploaded
   as JSON in `baseline_comparison`.
6. **Gate**: if `LSTM_REQUIRE_BEAT_PERSISTENCE=true` (default) and the
   LSTM doesn't beat the persistence baseline on this run's validation
   split, the upload is skipped entirely and the bot keeps using whatever
   model is already live. The richer baselines (AR(1)/GBM/GRU/CNN) are
   diagnostic only and never block an upload on their own — read the
   comparison table in the logs (or the `baseline_comparison` column) and
   decide for yourself whether a given run's numbers are worth trusting.

## Safety / cost notes

- This service never places a trade — it's read-only against the Deriv
  API (`ticks_history`).
- `LSTM_TRAIN_HISTORY_DAYS` for the minute model defaults to 30 days,
  since 30 days of ticks resample down to a much smaller number of
  distinct minute bars than the tick model needs directly. That's a much
  larger `ticks_history` pull — expect the minute run to take noticeably
  longer than the tick run in the same cron cycle.
- Each run trains 3 torch models (the served ensemble + the GRU/CNN
  diagnostics) plus one sklearn GBM, twice per cron trigger (tick +
  minute) — this is meaningfully more CPU time per run than a bare
  LSTM-only trainer. If Railway cron runs start taking long enough to
  risk missing the next 2h trigger, the first things to reduce are
  `LSTM_EPOCHS` and `LSTM_COMBOS_PER_ANCHOR`.
