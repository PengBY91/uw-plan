"""Health state machine (method spec M3.3, product output; not research evidence on its own).

Combines the common base monitor C0 (initialised, finite pose, observation age, time regression) with per-axis
DET-DIR alarms into NORMAL / DEGRADED / LOST / RECOVERING plus axis_ok[x, y, z, yaw]. Everything runs on the message
sample clock (estimate stamps, observation stamps); process timeouts on a monotonic wall clock belong to the online
node and are not part of this offline replay. Inputs never include reference truth or after-the-fact errors.

Per-axis alarm latch (hysteresis): an axis turns on when its statistic exceeds its threshold and turns off only when
it drops below threshold - release * tolerance. A NaN statistic is 'unscorable' (no alarm, not ok); an axis without
an independent source is 'no_source' (y, topic A v2.3) and is never treated as ok.

Transitions (durations are continuous holding times on the sample clock, all from the config):
  any -> LOST          hard fault: time regression (new epoch), input-integrity LOST (M3.4), non-finite pose, no
                       observation for obs_timeout_s, or not yet initialised
  NORMAL -> DEGRADED   some axis alarming for t_bad_s
  DEGRADED -> NORMAL   no axis alarming for t_good_s
  DEGRADED -> LOST     every sourced position axis (x, z) alarming for t_fail_s; an unscorable axis is not alarming
  LOST -> RECOVERING   no hard fault and not every sourced position axis alarming
  RECOVERING -> NORMAL no hard fault and no alarm for recover_hold_s
  RECOVERING -> DEGRADED some axis alarming for t_bad_s
A missing sonar alone never forces LOST: x just becomes unscorable.
"""
import numpy as np

AXES = ('x', 'y', 'z', 'yaw')
POSITION_AXES = (0, 1, 2)
STATES = ('NORMAL', 'DEGRADED', 'LOST', 'RECOVERING')
HEALTH = {'t_bad_s': 1.0, 't_good_s': 3.0, 't_fail_s': 5.0, 'obs_timeout_s': 1.0, 'recover_hold_s': 2.0,
          'release_tol_fraction': 0.1}


def latch(stats, thresholds, tolerances, release, sourced=(0, 2, 3)):
    """Per-axis alarms with hysteresis. stats (n, 4); thresholds/tolerances per axis. Returns (alarm bool (n, 4),
    status (n, 4) of 'ok' / 'alarm' / 'unscorable' / 'no_source')."""
    n = len(stats)
    alarm = np.zeros((n, 4), bool)
    status = np.full((n, 4), 'no_source', dtype=object)
    for a in sourced:
        on = False
        for i in range(n):
            s = stats[i, a]
            if not np.isfinite(s):
                on = False
                status[i, a] = 'unscorable'
                continue
            on = s > thresholds[a] if not on else s >= thresholds[a] - release*tolerances[a]
            alarm[i, a] = on
            status[i, a] = 'alarm' if on else 'ok'
    return alarm, status


def run(t, alarm, axis_status, obs_t, pose_finite, cfg=None, input_lost=None):
    """State per sample. t: sample-clock stamps (may regress); alarm/axis_status from latch(); obs_t: stamps of valid
    observations (sorted); pose_finite: bool per sample; input_lost: optional (t_s, reason) of the first LOST from the
    M3.4 input-integrity monitor (uwbench.a.integrity), a hard fault from then on (that monitor latches until the
    estimator re-initialises). Returns dict of per-sample arrays and the transition list."""
    c = {**HEALTH, **(cfg or {})}
    n = len(t)
    state, epoch = 'LOST', 0
    states, reasons, epochs, last_obs = [], [], np.zeros(n, int), np.full(n, np.nan)
    transitions = []
    since = {'alarm': None, 'clean': None, 'calm': None, 'position': None}
    t_prev = None

    def move(i, new, reason):
        nonlocal state
        if new != state:
            transitions.append({'t_s': float(t[i]), 'from': state, 'to': new, 'reason': reason, 'epoch_id': epoch})
            state = new
        for key in since:
            since[key] = None

    def held(key, active, now, duration):
        """True once `active` has held continuously for `duration` (tracked in since[key])."""
        if not active:
            since[key] = None
            return False
        if since[key] is None:
            since[key] = now
        return now - since[key] >= duration - 1e-9

    for i in range(n):
        now = float(t[i])
        k = int(np.searchsorted(obs_t, now, side='right')) - 1
        last_obs[i] = obs_t[k] if k >= 0 else np.nan
        hard = None
        if t_prev is not None and now <= t_prev:
            epoch += 1
            hard = 'time_regression'
            since.update({key: None for key in since})
        elif input_lost is not None and now >= input_lost[0]:
            hard = 'input_' + input_lost[1]
        elif not pose_finite[i]:
            hard = 'pose_nonfinite'
        elif k < 0:
            hard = 'not_initialised'
        elif now - obs_t[k] > c['obs_timeout_s']:
            hard = 'observation_timeout'
        t_prev = now
        alarming = [AXES[a] for a in range(4) if alarm[i, a]]
        sourced_pos = [a for a in POSITION_AXES if axis_status[i, a] != 'no_source']
        all_pos = bool(sourced_pos) and all(alarm[i, a] for a in sourced_pos)
        reason = hard or ('alarm:' + ','.join(alarming) if alarming else 'ok')
        if hard:
            if state != 'LOST':
                move(i, 'LOST', hard)
        elif state == 'LOST':
            if not all_pos:   # leaving LOST needs its cause gone: inputs valid and not every position axis alarming
                move(i, 'RECOVERING', 'inputs_valid')
        elif state == 'NORMAL':
            if held('alarm', bool(alarming), now, c['t_bad_s']):
                move(i, 'DEGRADED', reason)
        elif state == 'DEGRADED':
            if held('position', all_pos, now, c['t_fail_s']):
                move(i, 'LOST', 'position_axes_alarm')
            elif held('calm', not alarming, now, c['t_good_s']):
                move(i, 'NORMAL', 'alarms_cleared')
        elif state == 'RECOVERING':
            if held('alarm', bool(alarming), now, c['t_bad_s']):
                move(i, 'DEGRADED', reason)
            elif held('clean', not alarming, now, c['recover_hold_s']):
                move(i, 'NORMAL', 'recovered')
        states.append(state)
        reasons.append(reason)
        epochs[i] = epoch
    return {'state': np.array(states, dtype=object), 'reason': np.array(reasons, dtype=object), 'epoch_id': epochs,
            'last_observation_s': last_obs, 'transitions': transitions}


def durations(t, states):
    """Seconds spent in each state, sample-and-hold, skipping regressions."""
    out = {s: 0.0 for s in STATES}
    for i in range(1, len(t)):
        if t[i] > t[i - 1]:
            out[states[i - 1]] += float(t[i] - t[i - 1])
    return out
