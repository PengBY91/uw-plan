#!/usr/bin/env bash
# E-B8 (W5-06): deferred commit D on the W3-04 A0 maps. B2a is the host (B3 not built yet). One map per call:
#   eb8_deferred.sh <dataset> <A0 run> <variant> [health status.csv | D map dir for dpre]
# variants: base (ungated), dh (D on A's health, buffered never recommitted), dhgt (D + one end-of-run correction to
# ground truth: upper bound of a perfect relocalisation), dor (oracle gate on the true pose error, no recommit),
# dpre (control: commit the same number of observations as the dh map, earliest first).
set -euo pipefail
cd "$(dirname "$0")/../.."
DS=$1; A0=$2; V=$3; STATUS=${4:-}
M=data/sim/$DS/manifest.yaml; D=runs/sim_${DS}_b0_depth_r01; RUN=runs/sim_${DS}_b2a_a0$(basename "$A0" | sed 's/.*_a0//')_${V}_r01
case $V in
  base) EXTRA=() ;;
  dh) EXTRA=(--deferred "$STATUS") ;;
  dhgt) EXTRA=(--deferred "$STATUS" --recommit-poses gt) ;;
  dor) EXTRA=(--deferred-oracle) ;;
  dpre) EXTRA=(--commit-prefix "$STATUS") ;;
  *) echo "unknown variant $V" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RUN"
PYTHONPATH=team/uwbench nice -n 19 .venv-eval/bin/python -m uwbench map --manifest "$M" --method b2a --depth "$D" --poses "$A0" "${EXTRA[@]}" --out "$RUN/map" > "$RUN/map.log" 2>&1
PYTHONPATH=team/uwbench nice -n 19 .venv-eval/bin/python -m uwbench eval-map --manifest "$M" --map "$RUN/map" --band-m 0.10 --out "$RUN/eval-map-v2-band0.10" > "$RUN/eval-map.log" 2>&1
echo "$RUN done"
