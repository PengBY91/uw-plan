"""Defaults and detector catalogue for detect-a; every key of CONFIG can be overridden by the --config YAML."""
import numpy as np

from uwbench.eval_traj import DEFAULTS


ATMOSPHERE_PA, RHO, G = 101325.0, 1000.0, 9.81  # same constants as scripts/data/sim_to_bag.py
SIM_EPOCH_S = 1_700_000_000  # scripts/data/sim_to_bag.py maps simulator t = 0 to this stamp
CONFIG = {'nis_window_s': 1.0, 'feat_window_s': 1.0, 'info_window_s': 1.0, 'conflict_window_s': 10.0,
          'lambda_min_quantile': 0.10, 'false_alarm_targets': [0.0, 0.01, 0.05],
          'early_warning_s': [0.0, 10.0], 'budget_n_sigma': 3.0,
          'sonar_threshold': 0.25, 'sonar_inlier_m': 0.05, 'sonar_min_inliers': 20, 'sonar_gate_deg': 15.0,
          'sonar_reference_s': DEFAULTS['risk']['align_window_s'],  # reference plane over the label alignment window (pilot: 0.0)
          'sonar_max_thickness': 0.2, 'sonar_hold_s': 1.0, 'sonar_match_deg': 3.0, 'sonar_jump_m': 0.15,
          'map_voxel_m': 0.05, 'map_assoc_m': 0.3, 'map_huber_m': 0.05, 'map_min_pairs': 20,
          'map_sigma_m': 0.03, 'map_prior_m': 0.02, 'map_prior_deg': 0.02, 'map_iterations': 6,
          'map_normal_k': 12, 'map_degenerate_ratio': 0.02, 'map_lever_m': 2.0,
          'map_grow': False, 'map_rotation': False, 'map_init_s': 0.0, 'sonar_footprint': False,
          'dir_yaw_source': 'frozen_budget',
          # review fixes (2026-10-12): see the module docstring
          'conflict_reference': 'window', 'threshold_allocation': 'union', 'comparable_tolerance': 0.5,
          'health_target': None, 'health': None,  # M3.3 state machine outputs at this DET-DIR operating point (off by default)
          'gyro_noise': None}  # {'noise_density': .., 'random_walk': .., 'source': 'Allan variance ...'} overrides the run's IMU chain
DETECTORS = {'DET-NIS': 'global', 'DET-COV': 'axis', 'DET-FEAT': 'global', 'DET-SCALAR': 'global', 'DET-SCALAR-l2': 'global',
             'DIR-gap': 'axis', 'DIR-conflict': 'axis',
             'DIR-gap+conflict': 'axis', 'DIR-cumconflict': 'axis', 'DIR-gap+cumconflict': 'axis',
             'DIR-cumconflict-frozen': 'axis', 'DIR-cumconflict-frozen-budget': 'axis', 'ORACLE-cumconflict-truebias': 'axis',
             'DIR-cumconflict+sonar': 'axis', 'DIR-cumconflict+sonar-multi': 'axis', 'DIR-cumconflict+sonar-map': 'axis',
             'DIR-sonar-map+yaw': 'axis', 'DET-DIR': 'axis'}
CUMULATIVE = {'DIR-cumconflict': 'conflict_cum', 'DIR-cumconflict-frozen': 'conflict_cum_frozen',
              'DIR-cumconflict-frozen-budget': 'conflict_cum_frozen_excess', 'ORACLE-cumconflict-truebias': 'conflict_cum_true',
              'DIR-cumconflict+sonar': 'conflict_cum_sonar', 'DIR-cumconflict+sonar-multi': 'conflict_cum_sonar_multi',
              'DIR-cumconflict+sonar-map': 'conflict_cum_sonar_map', 'DIR-sonar-map+yaw': 'conflict_cum_sonar_map_yaw',
              'DET-DIR': 'conflict_dir'}
SOURCE_AXES = {'conflict_cum_sonar': (0, 1, 2, 3), 'conflict_cum_sonar_multi': (0, 1, 2, 3),
               'conflict_cum_sonar_map': (0, 1, 2, 3), 'conflict_cum_sonar_map_yaw': (0, 1, 2, 3),
               'conflict_dir': (0, 2, 3)}  # every other cumulative form: depth for z, gyro for heading only


def default_tolerances():
    return np.array([DEFAULTS['risk']['tolerance_position_m']]*3 + [DEFAULTS['risk']['tolerance_yaw_deg']])
