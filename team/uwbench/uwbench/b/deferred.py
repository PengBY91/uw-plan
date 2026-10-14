"""Deferred commit D (method spec M6.1, v2.4; engineering mechanism, not a contribution).

Every stereo and sonar observation enters a buffer stamped with A's health output at its own time (sample-and-hold
of the latest health sample at or before the observation, so only online state is read). It is committed to the map
only while health is NORMAL and none of the axes it depends on is alarming (sonar: x, yaw; stereo: x, y, z, yaw);
otherwise it stays in the buffer and is never dropped. When the pose is corrected (correction_id changes) the buffered
observations up to the correction are recommitted with the corrected poses. D is not per-axis gating of the map:
the axis check only holds back observations whose own geometry depends on an alarming axis, and NORMAL already
requires every axis calm for t_good_s.

Axis statuses come from uwbench.a.health: 'ok', 'alarm', 'unscorable' (no reference this sample) and 'no_source'
(y has no independent channel). Only 'alarm' holds an observation back by default; holding on 'no_source' would stop
all stereo mapping, holding on 'unscorable' is the stricter option (block=('alarm', 'unscorable')).

Recommit and replay: the map is rebuilt by replaying every committed observation in time order, each with the pose it
was committed with (the estimate for direct commits, the corrected pose for recommits). Log-odds clamping makes the
update order matter, so this replay is what 'recommit equals replay from scratch with the corrected poses' means.

Oracle gate (diagnostic upper bound, reads truth): the same commit rule with 'alarm' replaced by 'the true pose error
on that axis exceeds its tolerance', which separates the quality of A's alarms from the mechanism itself.
"""
import csv
from pathlib import Path

import numpy as np

AXES = ('x', 'y', 'z', 'yaw')
DEPENDS = {'sonar': ('x', 'yaw'), 'depth': ('x', 'y', 'z', 'yaw')}


class HealthTrack:
    """A's per-sample health output (uwbench.a.health status.csv) on the sim clock."""

    def __init__(self, t, state, axis_status, correction_id=None):
        self.t = np.asarray(t, dtype=float)
        if len(self.t) and np.any(np.diff(self.t) < 0):
            raise ValueError('Health samples must be time-ordered (one epoch per track)')
        self.state = np.asarray(state, dtype=object)
        self.axis_status = np.asarray(axis_status, dtype=object).reshape(len(self.t), 4)
        self.correction_id = np.zeros(len(self.t), int) if correction_id is None else np.asarray(correction_id, int)

    @classmethod
    def from_status_csv(cls, path, epoch_s=0.0):
        with Path(path).open() as f:
            rows = list(csv.DictReader(f))
        return cls([float(r['t_s']) - epoch_s for r in rows], [r['state'] for r in rows],
                   [[r[f'axis_ok_{a}'] for a in AXES] for r in rows],
                   [int(r.get('correction_id') or 0) for r in rows])

    def index(self, t):
        """Latest sample at or before t (causal), or None before the first sample."""
        i = int(np.searchsorted(self.t, t, side='right')) - 1
        return i if i >= 0 else None


def health_decision(track, block=('alarm',)):
    """decide(t, kind) -> (commit, reason) from A's online health output."""
    def decide(t, kind):
        i = track.index(t)
        if i is None:
            return False, 'no_health'
        if track.state[i] != 'NORMAL':
            return False, 'state:' + track.state[i]
        held = [a for a in DEPENDS[kind] if track.axis_status[i, AXES.index(a)] in block]
        return (False, 'axis:' + ','.join(held)) if held else (True, 'commit')
    return decide


def body_errors(T_est, T_true):
    """Pose error in the true body frame: x, y, z (m) and |yaw| (deg), the axes of A's tolerances."""
    dp = T_true[:3, :3].T @ (T_est[:3, 3] - T_true[:3, 3])
    dR = T_true[:3, :3].T @ T_est[:3, :3]
    yaw = np.degrees(np.arctan2(dR[1, 0], dR[0, 0]))
    return np.array([abs(dp[0]), abs(dp[1]), abs(dp[2]), abs(yaw)])


def oracle_decision(pose_est, pose_true, tolerances=(0.15, 0.15, 0.15, 3.0)):
    """decide(t, kind) committing only while the true error of every axis the observation depends on is within
    tolerance. Reads truth: a diagnostic upper bound on what a perfect detector would give D, never a method."""
    tol = np.asarray(tolerances, dtype=float)

    def decide(t, kind):
        T_est, T_true = pose_est(t), pose_true(t)
        if T_est is None:
            return False, 'no_pose'
        bad = [a for a in DEPENDS[kind] if body_errors(T_est, T_true)[AXES.index(a)] > tol[AXES.index(a)]]
        return (False, 'error:' + ','.join(bad)) if bad else (True, 'commit')
    return decide


def prefix_decision(t_end):
    """Control for D: commit every observation up to t_end and buffer the rest ('map only the start of the run').
    With t_end chosen so the committed count equals D's, it separates D's choice of segments from mapping less."""
    return lambda t, kind: (True, 'commit') if t <= t_end else (False, 'after_prefix')


class DeferredCommit:
    """Buffer of observations with commit decisions; never drops one. Events are (t, kind, index)."""

    def __init__(self, decide):
        self.decide = decide
        self.committed = []     # (t, kind, index, pose_source)
        self.buffer = []        # (t, kind, index, reason)
        self.recommits = []     # (correction_t, count)

    def offer(self, t, kind, index):
        ok, reason = self.decide(t, kind)
        if ok:
            self.committed.append((t, kind, index, 'estimate'))
        else:
            self.buffer.append((t, kind, index, reason))
        return ok, reason

    def correct(self, t_correction):
        """Pose correction at t_correction: buffered observations up to it are recommitted with corrected poses."""
        moved = [b for b in self.buffer if b[0] <= t_correction]
        self.buffer = [b for b in self.buffer if b[0] > t_correction]
        self.committed += [(t, kind, i, 'corrected') for t, kind, i, _ in moved]
        self.recommits.append((float(t_correction), len(moved)))
        return len(moved)

    def replay_order(self):
        """Committed observations in time order (the map is rebuilt from scratch in this order)."""
        return sorted(self.committed, key=lambda e: (e[0], e[1], e[2]))

    def summary(self):
        reasons = {}
        for b in self.buffer:
            reasons[b[3]] = reasons.get(b[3], 0) + 1
        count = lambda items, kind: sum(1 for e in items if e[1] == kind)
        return {'committed': len(self.committed), 'committed_direct': sum(1 for e in self.committed if e[3] == 'estimate'),
                'recommitted': sum(1 for e in self.committed if e[3] == 'corrected'), 'still_buffered': len(self.buffer),
                'committed_depth': count(self.committed, 'depth'), 'committed_sonar': count(self.committed, 'sonar'),
                'buffered_depth': count(self.buffer, 'depth'), 'buffered_sonar': count(self.buffer, 'sonar'),
                'buffered_reasons': reasons, 'recommits': self.recommits}
