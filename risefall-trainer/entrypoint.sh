#!/usr/bin/env bash
# Runs one full training cycle for BOTH model kinds, sequentially, in a
# single Railway Cron Job invocation. Each risefall_lstm_train.py run is a
# separate process (MODEL_KIND read once at import time), same "one run,
# exits" design either way -- this script is just the loop over both kinds
# that a single 2h cron trigger needs.
#
# Deliberately does NOT use `set -e`: if the tick run fails we still want
# to attempt the minute run rather than abort immediately, and the exit
# code reflects whether EITHER run failed.

echo "=== RISEFALL LSTM trainer cron run starting: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo ""
echo "--- Training TICK model ---"
MODEL_KIND=tick python risefall_lstm_train.py
tick_status=$?

echo ""
echo "--- Training MINUTE model ---"
MODEL_KIND=minute python risefall_lstm_train.py
minute_status=$?

echo ""
if [ $tick_status -ne 0 ] || [ $minute_status -ne 0 ]; then
  echo "=== Cron run finished WITH FAILURES (tick exit=$tick_status, minute exit=$minute_status) ==="
  exit 1
fi

echo "=== Cron run finished successfully -- both models trained and evaluated ==="
