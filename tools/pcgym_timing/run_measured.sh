#!/bin/bash
# Run from the repository root after installing private peer environments.
# The oracle and truth stages can also be run independently with pilot.py.
set -euo pipefail
out=${1:?usage: run_measured.sh OUTPUT_DIRECTORY}
pilot=tools/pcgym_timing/pilot.py
run() {
  uv run --frozen --no-sync python "$pilot" "$1" --arm "$2" --output "$out" --run-name "measured/$2-$1"
}
if [[ ! -f "$out/forecast-truth.json" ]]; then
  uv run --frozen --no-sync python "$pilot" truth --output "$out"
fi
for arm in cffg rxinfer numpyro gtsam; do
  if [[ ! -f "$out/measured/$arm-oracle/result.json" ]]; then
    run oracle "$arm"
  fi
  uv run --frozen --no-sync python -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["complete"])' "$out/measured/$arm-oracle/result.json"
done
for arm in cffg rxinfer numpyro gtsam; do run short "$arm"; done
for arm in gtsam numpyro rxinfer cffg; do run long "$arm"; done
for arm in numpyro cffg gtsam rxinfer; do run forecast "$arm"; done
uv run --frozen --no-sync python tools/pcgym_timing/summarize.py --output "$out"
