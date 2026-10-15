#!/usr/bin/env bash
# E-B9 follow-up (2026-10-14): ground-truth-pose B2a maps with the isolated-echo filter (configs/map/b2a_isolated.yaml),
# eval-map with the same 0.10 m band as the unfiltered r01 maps, then the arc-height analysis on both.
# usage: bash scripts/experiments/eb9_isolated.sh [scene ...]   (default: stripe_beam_v1 slope_wall_v1 pilaster_wall_v1)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-eval/bin/python
export PYTHONPATH=team/uwbench
scenes=("${@:-stripe_beam_v1 slope_wall_v1 pilaster_wall_v1}")
read -r -a scenes <<< "${scenes[*]}"
runs=()
for s in "${scenes[@]}"; do
  ds=${s}_s01
  M=data/sim/$ds/manifest.yaml
  RUN=runs/sim_${ds}_b2a_iso_gt_r01
  mkdir -p "$RUN"
  [ -d "$RUN/map" ] || nice -n 19 $PY -m uwbench map --manifest "$M" --method b2a --depth runs/sim_${ds}_b0_depth_r01 \
    --poses gt --config configs/map/b2a_isolated.yaml --out "$RUN/map" > "$RUN/map.log" 2>&1
  [ -d "$RUN/eval-map-v2-band0.10" ] || nice -n 19 $PY -m uwbench eval-map --manifest "$M" --map "$RUN/map" --band-m 0.10 \
    --out "$RUN/eval-map-v2-band0.10" > "$RUN/eval-map.log" 2>&1
  runs+=(--run "$RUN")
done
nice -n 19 $PY scripts/experiments/eb9_arc_height.py "${runs[@]}" --out reports/w5/eb9/arc_height_isolated.json
