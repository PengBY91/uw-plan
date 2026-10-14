#!/usr/bin/env bash
# D-sim (checklist W5-11): for each generated route data/sim/<ds>/raw, derive the g1-gyro dataset <ds>_g1 (pre-Allan
# placeholder: EuRoC ADIS16448 Kalibr values, as in the IMU sweep), convert to a bag, audit, run OpenVINS A0 at 1x and
# uwbench eval-traj. Same chain and flags as scripts/experiments/imu_sweep.sh. Resumable: an existing derived dataset
# or run directory is skipped; failed runs are kept. Run OpenVINS replays one at a time (parallel replays drop samples).
# run_openvins.py refuses a dirty tree; while D-sim work is uncommitted, set OV_LAB to a clean detached worktree of
# HEAD whose ros_ws, local and runs are symlinks to this checkout (team_commit then records that HEAD).
# Usage: [OV_LAB=/path/to/clean/worktree] bash scripts/experiments/dsim_pipeline.sh <ds> [<ds> ...]   e.g. vstripe_heave_v1_s01
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-eval/bin/python
N=1.6968e-4; K=1.9393e-5; LABEL='g1 = EuRoC ADIS16448 Kalibr values; pre-Allan placeholder for the platform IMU (D-sim)'
LOG=runs/dsim/pipeline.log
mkdir -p runs/dsim
for src in "$@"; do
  d=data/sim/${src}_g1; run=sim_${src}_g1_a0_r01
  seed=$(( $(printf '%s' "$src" | cksum | cut -d' ' -f1) % 100000 ))
  if [ ! -f "$d/audit/audit.json" ]; then
    if ! { { [ -d "$d" ] || $PY scripts/data/sim_imu_grade.py --run data/sim/$src --grade g1 --noise-density $N --random-walk $K \
               --seed $seed --label "$LABEL" >/dev/null; } &&
           $PY scripts/data/sim_to_bag.py --run $d >/dev/null &&
           PYTHONPATH=team/uwbench $PY -m uwbench audit --manifest $d/manifest.yaml --out $d/audit >/dev/null; }; then
      echo "$(date -Is) $run DATASET_FAILED" | tee -a $LOG; continue
    fi
  fi
  if [ -d runs/$run ]; then echo "$(date -Is) $run exists, skipped" >> $LOG; continue; fi
  OV=${OV_LAB:-$PWD}
  bash $OV/scripts/env/ros_cyclone_run.sh /usr/bin/python3 $OV/scripts/openvins/run_openvins.py --lab "$OV" --bag "$PWD/$d/ros2" --run-id $run \
    --rate 1.0 --imu-history-depth 1000 --imu-reliability reliable --config-dir "$PWD/$d/openvins" \
    --dataset-audit "$PWD/$d/audit/audit.json" > runs/dsim/$run.out 2>&1
  rc=$?
  if [ ! -d runs/$run/estimate_bag ]; then echo "$(date -Is) $run openvins_rc=$rc NO_ESTIMATE" | tee -a $LOG; continue; fi
  $PY scripts/openvins/audit_input_trace.py --bag $d/ros2 --run $PWD/runs/$run >> runs/dsim/$run.out 2>&1
  PYTHONPATH=team/uwbench $PY -m uwbench eval-traj --manifest $d/manifest.yaml --estimate runs/$run/estimate_bag \
    --out runs/$run/eval-traj >> runs/dsim/$run.out 2>&1
  echo "$(date -Is) $run openvins_rc=$rc eval_rc=$?" | tee -a $LOG
done
