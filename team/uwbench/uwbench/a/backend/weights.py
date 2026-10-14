"""A2 bounded scalar weight for sonar factors (W4-01, P10), decided causally when a factor is added.

The weight w in [w_min, 1] scales the information of the fixed-weight model: sigma_eff = sigma / sqrt(w).
Signals (one per policy, never mixed):
  innovation  whitened norm of the new measurement against the current pose and landmark estimates, smoothed per
              track with an exponential moving average; a track switches to w_min when the average exceeds `high`
              and back to 1 only when it falls below `low` (hysteresis).
  support     w rises linearly from w_min at the first observation of a track to 1 at `n_full` observations.
'none' leaves every weight at 1 (identical to A1). Fixed weight and robust kernel stay separate switches.
"""

POLICIES = ('none', 'innovation', 'support')
DEFAULTS = {'kind': 'none', 'w_min': 0.1, 'ema_alpha': 0.3, 'high': 3.0, 'low': 1.5, 'n_full': 10}


class ScalarWeight:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        c = self.cfg
        if c['kind'] not in POLICIES:
            raise ValueError('Unknown weight policy ' + str(c['kind']))
        if not 0 < c['w_min'] <= 1:
            raise ValueError('w_min must be in (0, 1]')
        if c['kind'] == 'innovation' and not 0 < c['low'] < c['high']:
            raise ValueError('Hysteresis needs 0 < low < high')
        if c['kind'] == 'support' and c['n_full'] < 2:
            raise ValueError('n_full must be at least 2')
        self.ema, self.low_state, self.count = {}, {}, {}

    def weight(self, track, innovation=None):
        """Weight for the next factor of `track`; innovation is the whitened residual norm (needed for 'innovation')."""
        c = self.cfg
        self.count[track] = self.count.get(track, 0) + 1
        if c['kind'] == 'none':
            return 1.0
        if c['kind'] == 'support':
            n = self.count[track]
            return c['w_min'] + (1 - c['w_min'])*min(1.0, (n - 1)/(c['n_full'] - 1))
        if innovation is None:
            raise ValueError('innovation policy needs the innovation')
        m = innovation if track not in self.ema else c['ema_alpha']*innovation + (1 - c['ema_alpha'])*self.ema[track]
        self.ema[track] = m
        low = self.low_state.get(track, False)
        if not low and m > c['high']:
            low = True
        elif low and m < c['low']:
            low = False
        self.low_state[track] = low
        return c['w_min'] if low else 1.0
