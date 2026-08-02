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

1. Fetches recent tick history **separately for every symbol in
   `RISEFALL_TRAIN_SYMBOLS`** (default basket covers both symbol families
   the live bot actually draws from — see `risefall-bot/README.md`'s
   "What Gate 6 actually changes"), resampling into minute bars first if
   `MODEL_KIND=minute` (`build_minute_bars()`). `LSTM_MAX_TICKS` is a
   TOTAL budget divided evenly across the basket, so adding symbols
   doesn't multiply wall-clock time.
2. Per symbol: builds labeled direction examples, chronologically splits
   train/val, and **purges** any training example whose label horizon
   reaches into that symbol's own validation window (`build_symbol_split()`
   — a walk-forward split alone isn't enough, since a label looks forward
   by up to `max(CANDIDATE_DURATIONS)`). One symbol's returns are never
   mixed into another symbol's index space.
3. Pools every symbol's train set (and separately, val set) into one
   combined training run via `torch.utils.data.ConcatDataset` — there's
   still exactly **one served state_dict per `MODEL_KIND`**, not one per
   symbol, since Gate 6 in the bot applies whichever model is current to
   every symbol it evaluates. Normalization is **per-window and local**
   (`local_normalize()` in `risefall_lstm_model.py`, z-scores each window
   against its own mean/std) rather than one global scalar baked into the
   model — that's what makes pooling symbols of very different native
   volatility (a Volatility 100 index's returns are roughly 10x a
   Volatility 10 index's) into one training set sound, and it's why this
   model generalizes even to a symbol outside the training basket.
4. Trains the served `RiseFallWinClassifier` deep ensemble (dilated causal
   conv front-end → 3-layer LSTM w/ inter-layer dropout → attention pool →
   5 bagged heads).
5. Runs it against **five baselines**, each computed per-symbol on that
   symbol's own purged validation split and then pooled in the same order
   for a fair comparison against the LSTM's pooled val predictions — see
   `run_baseline_diagnostics()`:
   - **Persistence** (naive "whatever just happened, keep happening")
   - **AR(1) + Hurst exponent** (linear autocorrelation baseline, fit
     per symbol; Hurst is logged per symbol too, as a standalone
     diagnostic of how much real short-range memory each process has)
   - **HistGradientBoostingClassifier** on hand-engineered return-window
     features (multi-scale mean/std, skew, streak length), trained on the
     same pooled multi-symbol set the LSTM sees
   - **Compact GRU** and **compact dilated causal CNN** (same pooled
     input incl. `local_normalize()`, cheap point-estimate competitors —
     not the served ensemble)

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

   **Read this before flipping `LSTM_ENABLED=true` on the bot**: Gate 6 in
   `risefall-bot` is a HARD veto on any direction disagreement, not a
   diagnostic — it can meaningfully cut trade frequency. Watch a few cron
   cycles of `baseline_comparison` first.

## Safety / cost notes

- This service never places a trade — it's read-only against the Deriv
  API (`ticks_history`).
- The default `RISEFALL_TRAIN_SYMBOLS` basket has 10 symbols. `LSTM_MAX_TICKS`
  auto-divides across however many symbols are configured, so wall-clock
  time per `MODEL_KIND` run stays roughly constant as the basket grows —
  but each symbol still needs its own `active_symbols`-style history pull
  and its own labeled-example construction pass, so more symbols still
  means more *overhead* even at a fixed total tick budget. If runs start
  taking long enough to risk missing the next 2h trigger, trim
  `RISEFALL_TRAIN_SYMBOLS` before reaching for `LSTM_EPOCHS`/
  `LSTM_COMBOS_PER_ANCHOR`.
- `LSTM_TRAIN_HISTORY_DAYS` for the minute model defaults to 30 days,
  since 30 days of ticks resample down to a much smaller number of
  distinct minute bars than the tick model needs directly. That's a much
  larger `ticks_history` pull per symbol — expect the minute run to take
  noticeably longer than the tick run in the same cron cycle.
- Each run trains 3 torch models (the served ensemble + the GRU/CNN
  diagnostics) plus one sklearn GBM, twice per cron trigger (tick +
  minute) — this is meaningfully more CPU time per run than a bare
  LSTM-only trainer. If Railway cron runs start taking long enough to
  risk missing the next 2h trigger, the first things to reduce are
  `LSTM_EPOCHS` and `LSTM_COMBOS_PER_ANCHOR`.
