"""Tables for the pre-hardware gyro-grade sweep (scripts/imu_sweep.sh): estimator heading behaviour and detect-a per grade.

Reads runs/sim_<route>_<grade>_a0_r01/{eval-traj/metrics.json,input-audit.json} and runs/detect-a/imu_<grade>_<variant>/
summary.json (variant: frozen = review_default, online = review_default_online_yaw). Prints markdown.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROUTES = ['dam_wall_v1_s01', 'stripe_beam_v1_s01', 'stripe_beam_v1_s02', 'stripe_beam_v1_s03', 'stripe_beam_fog0_v1_s01',
          'stripe_beam_fog1_v1_s01', 'stripe_beam_fog4_v1_s01', 'stripe_wall_h_v1_s01', 'stripe_wall_v_v1_s01',
          'slope_wall_v1_s01', 'pilaster_wall_v1_s01', 'pilaster_deep_v1_s01', 'pilaster_deep_fog6_v1_s01',
          'pilaster_deep_fog8_v1_s01']
DETECTORS = ['DET-DIR', 'DET-SCALAR', 'ORACLE-cumconflict-truebias', 'DET-NIS', 'DET-COV']


def pct(x):
    return '—' if x is None else f'{100*x:.0f}%'


def estimator_table(lab, grades):
    print('| 等级 | 航次（完成/输入零缺样） | 航向越限时长合计（s） | 航向最大误差中位（°） | x / y 越限时长合计（s） | APE RMSE 中位（m） |')
    print('| --- | --- | ---: | ---: | --- | ---: |')
    for g in grades:
        yaw_s, yaw_max, x_s, y_s, ape, ok, clean = 0.0, [], 0.0, 0.0, [], 0, 0
        for r in ROUTES:
            run = lab/'runs'/f'sim_{r}_{g}_a0_r01'
            m = run/'eval-traj/metrics.json'
            if not m.exists():
                continue
            ok += 1
            a = json.loads((run/'input-audit.json').read_text()) if (run/'input-audit.json').exists() else {}
            clean += all(v.get('missing_count') == 0 for v in a.values() if isinstance(v, dict) and 'missing_count' in v) and bool(a)
            d = json.loads(m.read_text())
            ax = d['risk']['axes']
            yaw_s += ax['yaw']['exceedance']['exceed_s']
            x_s += ax['x']['exceedance']['exceed_s']
            y_s += ax['y']['exceedance']['exceed_s']
            yaw_max.append(ax['yaw']['abs_error_max'])
            ape.append(d['public']['ape_translation_m']['rmse'])
        if ok:
            print(f'| {g} | {ok}/{clean} | {yaw_s:.0f} | {np.median(yaw_max):.1f} | {x_s:.0f} / {y_s:.0f} | {np.median(ape):.3f} |')


def detect_table(lab, grades, target, grace):
    print(f'\n目标误报 {target:.0%}，提前量 {grace:g} s；漏报按统一分母（不可评计漏报）。')
    print('| 等级 | 航向来源 | T_tol（s） | 检测器 | 误报 | 可比 | 漏报 x / z / 航向 | 误归因 航向 |')
    print('| --- | --- | ---: | --- | ---: | --- | --- | ---: |')
    for g in grades:
        for variant in ('frozen', 'online'):
            p = lab/'runs/detect-a'/f'imu_{g}_{variant}'/'summary.json'
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            horizon = [h for h in s['frozen_budget_reaches_yaw_tolerance_s'].values() if h is not None]
            t_tol = f'{np.median(horizon):.0f}' if horizon else '∞'
            for e in s['pooled']:
                if e['target'] != target or e['early_warning_s'] != grace or e['detector'] not in DETECTORS:
                    continue
                if variant == 'online' and e['detector'] not in ('DET-DIR', 'DET-SCALAR'):
                    continue   # the other detectors do not depend on the heading source
                print(f"| {g} | {'冻结零偏−预算' if variant == 'frozen' else '在线零偏'} | {t_tol} | {e['detector']} | {pct(e['false_alarm_fraction'])} | "
                      f"{'是' if e['comparable'] else '否'}（{e['runs_comparable']}） | {pct(e['r_miss_x'])} / {pct(e['r_miss_z'])} / {pct(e['r_miss_yaw'])} | {pct(e['r_offaxis_yaw'])} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--grades', nargs='+', default=['g0', 'g1', 'g2', 'g3', 'g4'])
    parser.add_argument('--target', type=float, default=0.05)
    parser.add_argument('--grace', type=float, default=0.0)
    args = parser.parse_args()
    estimator_table(args.lab, args.grades)
    detect_table(args.lab, args.grades, args.target, args.grace)


if __name__ == '__main__':
    main()
