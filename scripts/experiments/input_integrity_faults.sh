#!/usr/bin/env bash
# M3.4 input-integrity monitor over the W4-05 fault-injection matrix (reports/w4/fault-injection.md): every derived
# fault is replayed from its parent bag via its fault.json (counts checked against what inject_fault.py wrote), plus
# the two clean parents and the textured checkerboard sim route as false-alarm controls. Parent bags must be present and match the hashes in fault.json.
# Usage: [FAULT_GLOB='v101_*'] bash scripts/experiments/input_integrity_faults.sh <out_root> [extra input-integrity args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?out root}; shift
PY=.venv-eval/bin/python
V101=data/public/euroc/V1_01_easy_ros2; SBEAM=data/sim/stripe_beam_v1_s01/ros2
run() {  # name bag calib [--fault f]
  local name=$1 bag=$2 calib=$3; shift 3
  [ -f "$OUT/$name/summary.json" ] && return 0
  PYTHONPATH=team/uwbench nice -n 10 $PY -m uwbench input-integrity --bag "$bag" --calib "$calib" --out "$OUT/$name" "$@" "${EXTRA[@]}" \
    > "$OUT/$name.log" 2>&1 || echo "FAILED $name" >&2
}
EXTRA=("$@")
mkdir -p "$OUT"
for d in data/derived/fault/${FAULT_GLOB:-*}/; do
  n=$(basename "$d")
  parent=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['parent']['bag'])" "$d/fault.json")
  $PY - "$d/fault.json" <<'PYEOF' || { echo "parent hash mismatch for $n" >&2; exit 1; }
import hashlib, json, sys
from pathlib import Path
f = json.load(open(sys.argv[1])); bag = Path(f['parent']['bag'])
for name, want in f['parent']['files_sha256'].items():
    h = hashlib.sha256(); fh = open(bag/name, 'rb')
    while chunk := fh.read(1 << 24): h.update(chunk)
    if h.hexdigest() != want: sys.exit(f'{bag/name}: {h.hexdigest()} != {want}')
PYEOF
  calib=$(dirname "$parent")/openvins/kalibr_imucam_chain.yaml   # sim datasets carry their own calibration
  [ -f "$calib" ] || calib=configs/euroc_v101/kalibr_imucam_chain.yaml
  echo "$n $parent $calib"
done > "$OUT/plan.txt"
run v101_control $V101 configs/euroc_v101/kalibr_imucam_chain.yaml &
run sbeam01_control $SBEAM data/sim/stripe_beam_v1_s01/openvins/kalibr_imucam_chain.yaml &
CHK=data/sim/normal_checker_v1_s01_g1   # textured sim control (W5-11 normal route), if its bag is restored
[ -d $CHK/ros2 ] && run checker_control $CHK/ros2 $CHK/openvins/kalibr_imucam_chain.yaml &
wait
while read -r n parent calib; do
  run "$n" "$parent" "$calib" --fault "data/derived/fault/$n/fault.json" &
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 2; done
done < "$OUT/plan.txt"
wait
echo done
