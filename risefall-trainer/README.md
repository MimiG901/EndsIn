# RISEFALL LSTM trainer — Railway deployment (minute model only)

This folder is one Railway service: a native **Cron Job** that trains the
RISEFALL **minute** model every 5 hours and uploads it to Supabase, for the
`risefall-bot` service (deployed separately) to pick up. It deliberately
does **not** train a tick model — `MODEL_KIND=minute` is fixed in
`.env.example`, and there's no `entrypoint.sh` looping over both kinds
anymore. The bot handles a permanently-absent tick model gracefully:
`lstm_duration_scan()` just skips it and Gate 6 runs on the minute model
alone. **The `risefall-bot` service needs no changes at all for this** —
it already checks for each model independently.

`risefall_lstm_model.py` **must** be byte-for-byte identical to the copy
in the `risefall-bot` folder — it's the shared architecture class both
services import, and a state_dict trained here has to load cleanly into
the bot's `RiseFallWinClassifier` instance.

## How the schedule works

`railway.json`'s `deploy.cronSchedule: "0 */5 * * *"` triggers a fresh
container every 5 hours (UTC: 00:00, 05:00, 10:00, 15:00, 20:00), runs
`python risefall_lstm_train.py` directly (`MODEL_KIND=minute` from the
environment), and the container exits when done. Railway does not start
the next scheduled run until the current one has finished.

Trigger it manually any time from the Railway dashboard/CLI to test
without waiting for the schedule.

## Why wall-clock time matters here (read this if a run ever looks "stuck")

Pooling N symbols into one training run multiplies the labeled-example
count by roughly N. Two real bottlenecks this created, both fixed now:

1. **Unbounded pooled dataset size.** A full 10-symbol basket with no cap
   measured out to ~1 CPU-hour for the main model's epochs **plus**
   another ~1 hour for the GRU/dilated-CNN diagnostic competitors (which
   trained on the same full pooled set) — a run could take ~2 hours
   before even finishing the tick stage, easily outlasting a manually
   triggered run or colliding with the next redeploy. `LSTM_MAX_TRAIN_EXAMPLES`
   / `LSTM_MAX_VAL_EXAMPLES` now cap the POOLED total across every symbol
   (subsampled proportionally per symbol, see `cap_pooled_examples()`), so
   wall-clock time stays roughly constant regardless of basket size.
   Defaults keep a full 10-symbol run under ~20 minutes; lower them
   further if your Railway plan needs it faster still.
2. **Minute mode needs far more raw ticks per symbol than tick mode.**
   `WINDOW_SIZE_MINUTES=200` means every labeled example needs 200+
   distinct one-minute bars of lookback -- the tick-mode tick budget
   (~30k ticks/symbol ≈ 8 hours) only resamples down to ~500 minute bars,
   nowhere near enough to clear the 200-anchor-per-symbol minimum once
   anchor striding is applied. `LSTM_MAX_TICKS` now defaults to 800,000
   (vs 300,000 for tick mode) specifically for `MODEL_KIND=minute`.
3. **Deriv's `ticks_history` API has a real retention limit** — observed
   at ~86,400 ticks (~24h) for at least some 1-second-tick symbols. Asking
   for more doesn't error: past that boundary the API silently returns a
   page that wraps back near "latest" instead of continuing further into
   the past, which used to get spliced straight into the series and crash
   `build_minute_bars()` with an obscure `IndexError` once fetched ticks
   exceeded that boundary (this is exactly what the first minute-mode run
   hit). `fetch_full_history()` now detects a page that isn't further back
   than what's already been fetched and stops cleanly there instead — you
   get a smaller-than-requested but valid dataset, never a crash.
   `LSTM_MAX_TICKS`'s minute-mode default (800,000 total ÷ basket size)
   is set to stay under this limit per symbol in the first place.
4. **That ~24h ceiling means no single run can ever see more than ~22h of
   history on its own — nothing accumulates across cron cycles by
   default.** Two things now compensate for that (see "PERSISTENT ARCHIVE
   + WARM-START" in the module docstring for the full explanation):
   - **Persistent minute-bar archive**: every run upserts its freshly
     resampled bars into `bot_risefall_minute_bars` (Supabase, deduped on
     `symbol,epoch`) and trains on the FULL accumulated archive (up to
     `LSTM_ARCHIVE_MAX_DAYS`, default 45) instead of just that run's fresh
     ~22h window. Real longitudinal history builds up over days/weeks of
     cron cycles even though no single fetch can ever see more than ~24h.
   - **Warm-start**: every run continues training from whichever model is
     currently live in Supabase instead of reinitializing from scratch, so
     whatever a previous cycle learned carries forward rather than being
     discarded every 5 hours. Safe by construction — still gated by
     `LSTM_REQUIRE_BEAT_PERSISTENCE` before any upload, so a bad warm-start
     can only fail to improve on the live model, never silently replace it
     with something worse.
   - Set `LSTM_ARCHIVE_MINUTE_BARS=false` to disable both and go back to
     training fresh on just each run's own window (useful for isolating
     whether an odd result is coming from the archive/warm-start path).

If a run still looks stuck after this, check Railway's **Deployments**
tab for a second deployment landing mid-run (a new push or redeploy will
kill the container without any error in the logs — see if the timestamps
line up) before assuming it's a training bug.

## Files

| File | Purpose |
|---|---|
| `risefall_lstm_train.py` | Trains the minute model, runs baseline diagnostics, uploads to Supabase. |
| `risefall_lstm_model.py` | Shared LSTM architecture — same file as in the bot repo. |
| `requirements.txt` | Python deps — CPU-only torch + scikit-learn (for the GBM baseline). |
| `.python-version` | Pins Python 3.11 for the Railpack build. |
| `railway.json` | Cron schedule + start command. |
| `.env.example` | Every environment variable this service reads. |

## Deploy

1. Run the Supabase SQL from the `risefall-bot` README once (covers both
   `bot_risefall_lstm_model`, shared with the bot, and
   `bot_risefall_minute_bars`, used only by this service's archive) — do
   this **before** the first cron trigger fires.
2. Push this folder as its own GitHub repo (or point Railway at this
   subfolder if it's in the same repo as `risefall-bot`).
3. Railway → New Project (or New Service, if reusing the bot's project) →
   Deploy from GitHub repo → select it.
4. Add every variable from `.env.example`. At minimum: `DERIV_APP_ID`,
   `DERIV_API_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY`, and confirm
   `MODEL_KIND=minute` is set.
5. Deploy once to confirm it builds, then either wait for the next
   5-hour mark or trigger a manual run from the dashboard. Watch the
   logs — expect: history pull per symbol → labeled-example construction
   → pooled-example cap message → training epochs → baseline comparison
   table → upload confirmation (or the `ABORTING upload` line, a valid
   outcome, not an error).

## What actually gets trained and uploaded, every run

1. Fetches recent tick history **separately for every symbol in
   `RISEFALL_TRAIN_SYMBOLS`** (default basket covers both symbol families
   the live bot actually draws from), resampling into minute bars
   (`build_minute_bars()`). `LSTM_MAX_TICKS` is a TOTAL budget divided
   evenly across the basket.
2. Per symbol: builds labeled direction examples, chronologically splits
   train/val, and **purges** any training example whose label horizon
   reaches into that symbol's own validation window (`build_symbol_split()`
   — a walk-forward split alone isn't enough, since a label looks forward
   by up to `max(CANDIDATE_DURATIONS_MINUTES)`). One symbol's returns are
   never mixed into another symbol's index space.
3. Pools every symbol's train set (and separately, val set) into one
   combined training run via `torch.utils.data.ConcatDataset`, then
   **caps** the pooled total at `LSTM_MAX_TRAIN_EXAMPLES`/
   `LSTM_MAX_VAL_EXAMPLES` (`cap_pooled_examples()`) so wall-clock time
   stays bounded regardless of basket size. There's still exactly **one
   served state_dict**, not one per symbol, since Gate 6 in the bot
   applies whichever model is current to every symbol it evaluates.
   Normalization is **per-window and local** (`local_normalize()` in
   `risefall_lstm_model.py`, z-scores each window against its own
   mean/std) rather than one global scalar baked into the model — that's
   what makes pooling symbols of very different native volatility (a
   Volatility 100 index's returns are roughly 10x a Volatility 10 index's)
   into one training set sound, and it's why this model generalizes even
   to a symbol outside the training basket.
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
   cycles of `baseline_comparison` first. Two real runs so far both landed
   at ~50% val accuracy (chance level, `train_bce` pinned near `ln(2)`) —
   verified via a controlled test with injected synthetic signal that this
   is the training pipeline correctly reporting "no learnable signal
   found," not a bug. That's a legitimate possible outcome for near-
   random-walk synthetic indices at short horizons, not something to
   ignore — keep `LSTM_ENABLED=false` until a run's numbers actually clear
   persistence by a real margin.

## Safety / cost notes

- This service never places a trade — it's read-only against the Deriv
  API (`ticks_history`).
- Wall-clock time is now bounded by `LSTM_MAX_TRAIN_EXAMPLES`/
  `LSTM_MAX_VAL_EXAMPLES` regardless of how many symbols are in
  `RISEFALL_TRAIN_SYMBOLS` (see "Why wall-clock time matters" above) —
  the default 10-symbol basket should complete in well under 20 minutes.
  If you add more symbols and it's still too slow, lower those caps
  before reaching for `LSTM_EPOCHS`.
- `LSTM_MAX_TICKS` defaults to 800,000 total for minute mode (vs 300,000
  for tick mode) — minute mode needs far more raw ticks per symbol to
  accumulate enough distinct one-minute bars, while staying under Deriv's
  observed ~24h retention limit per symbol.
- Each run trains 3 torch models (the served ensemble + the GRU/CNN
  diagnostics) plus one sklearn GBM — this is meaningfully more CPU time
  per run than a bare LSTM-only trainer, though the pooled-example cap
  keeps it well within the 5h cron window at the defaults above.
- The persistent archive adds a handful of extra Supabase REST calls per
  symbol per run (one paginated upsert, one prune delete, one paginated
  read-back) — noticeable but not a meaningful fraction of total run time
  next to the fetch/train stages. The table itself stays bounded by
  `LSTM_ARCHIVE_MAX_DAYS` (default 45 days × 10 symbols × ~1440 bars/day
  ≈ 650k rows at steady state, pruned continuously) rather than growing
  forever.
- If you'd rather the bot pick up a fresher/staler model check faster
  than its own default `LSTM_RELOAD_INTERVAL_SECS` (2h), you can set that
  env var on the **bot** service to `18000` (5h) to match this cadence —
  optional, since reload is just a floor before the next opportunistic
  check, not a strict schedule.
