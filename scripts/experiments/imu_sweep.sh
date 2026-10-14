#!/usr/bin/env bash
# Pre-hardware gyro-grade sweep (topic A v2.2 early-exit check): the 14 detect-a pilot routes re-run with the gyro
# replaced by scripts/data/sim_imu_grade.py at five grades, then OpenVINS A0 at 1x and uwbench eval-traj. Resumable: an
# existing derived dataset or run directory is skipped; failed runs are kept. detect-a is run afterwards per grade.
# Usage: bash scripts/experiments/imu_sweep.sh [grade ...]   (default: g0 g2 g1 g3 g4)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-eval/bin/python
ROUTES=(dam_wall_v1_s01 stripe_beam_v1_s01 stripe_beam_v1_s02 stripe_beam_v1_s03 stripe_beam_fog0_v1_s01
        stripe_beam_fog1_v1_s01 stripe_beam_fog4_v1_s01 stripe_wall_h_v1_s01 stripe_wall_v_v1_s01 slope_wall_v1_s01
        pilaster_wall_v1_s01 pilaster_deep_v1_s01 pilaster_deep_fog6_v1_s01 pilaster_deep_fog8_v1_s01)
# grade: N [rad/s/sqrt(Hz)], K [rad/s^2/sqrt(Hz)], label. g0 repeats the simulator assumption with a fresh realisation.
declare -A N=([g0]=1.414e-4 [g1]=1.6968e-4 [g2]=8.7e-5 [g3]=2.9e-5 [g4]=3.5e-6)
declare -A K=([g0]=2.828e-4 [g1]=1.9393e-5 [g2]=4e-6 [g3]=1e-6 [g4]=1e-7)
declare -A LABEL=([g0]='simulator assumption (control)' [g1]='EuRoC ADIS16448 Kalibr values'
                  [g2]='industrial MEMS, ARW 0.3 deg/sqrt(h)' [g3]='high-end MEMS, ARW 0.1 deg/sqrt(h)'
                  [g4]='tactical grade, ARW 0.012 deg/sqrt(h)')
if [ $# -gt 0 ]; then GRADES=("$@"); else GRADES=(g0 g2 g1 g3 g4); fi
LOG=runs/imu-sweep/sweep.log
mkdir -p runs/imu-sweep
for g in "${GRADES[@]}"; do
  gi=${g#g}
  for i in "${!ROUTES[@]}"; do
    r=${ROUTES[$i]}; d=data/sim/${r}_$g; run=sim_${r}_${g}_a0_r01
    if [ ! -f "$d/audit/audit.json" ]; then
      if ! { $PY scripts/data/sim_imu_grade.py --run data/sim/$r --grade $g --noise-density ${N[$g]} --random-walk ${K[$g]} \
               --seed $((1000*gi + i)) --label "${LABEL[$g]}" >/dev/null &&
             $PY scripts/data/sim_to_bag.py --run $d >/dev/null &&
             PYTHONPATH=team/uwbench $PY -m uwbench audit --manifest $d/manifest.yaml --out $d/audit >/dev/null; }; then
        echo "$(date -Is) $run DATASET_FAILED" | tee -a $LOG; continue
      fi
    fi
    if [ -d runs/$run ]; then echo "$(date -Is) $run exists, skipped" >> $LOG; continue; fi
    bash scripts/env/ros_cyclone_run.sh /usr/bin/python3 scripts/openvins/run_openvins.py --lab "$PWD" --bag "$PWD/$d/ros2" --run-id $run \
      --rate 1.0 --imu-history-depth 1000 --imu-reliability reliable --config-dir "$PWD/$d/openvins" \
      --dataset-audit "$PWD/$d/audit/audit.json" > runs/imu-sweep/$run.out 2>&1
    rc=$?
    PYTHONPATH=team/uwbench $PY -m uwbench eval-traj --manifest $d/manifest.yaml --estimate runs/$run/estimate_bag \
      --out runs/$run/eval-traj >> runs/imu-sweep/$run.out 2>&1
    echo "$(date -Is) $run openvins_rc=$rc eval_rc=$?" | tee -a $LOG
  done
done
