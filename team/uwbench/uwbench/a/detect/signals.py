"""Causal time-series helpers: trailing windows, gyro integration, the frozen-bias drift budget, holding and
re-referencing cumulative statistics, and the information gap tau."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation



def windowed(update_t, values, t, window, reduce):
    """reduce(values in (t - window, t]) for each t; NaN where the window is empty."""
    out = np.full(len(t), np.nan)
    lo = np.searchsorted(update_t, t - window, side='right')
    hi = np.searchsorted(update_t, t, side='right')
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = reduce(values[a:b])
    return out


def gyro_heading(imu, bias_t, bias):
    """Body-to-world rotation from integrating gyro minus the latest causal bias (bias_t, bias), starting at identity."""
    k = np.clip(np.searchsorted(bias_t, imu[:, 0], side='right') - 1, 0, None)
    w = imu[:, 1:] - bias[k]
    w[imu[:, 0] < bias_t[0]] = 0.0  # before the estimator exists there is no bias and nothing to compare
    dt = np.diff(imu[:, 0], prepend=imu[0, 0])
    steps = Rotation.from_rotvec(w*dt[:, None]).as_matrix()
    Q = np.empty_like(steps)
    current = np.eye(3)
    for i, step in enumerate(steps):
        current = current @ step
        Q[i] = current
    return Q


def drift_budget_deg(T, noise_density, random_walk, n_sigma):
    """n_sigma bound on heading error from integrating a gyro whose bias was frozen T seconds ago (bias error excluded)."""
    T = np.clip(T, 0, None)
    return np.degrees(n_sigma*np.sqrt(noise_density**2*T + random_walk**2*T**3/3))


def budget_horizon_s(gyro_noise, n_sigma, tol_yaw):
    """Seconds until the frozen-bias drift budget reaches the heading tolerance; None when it never does."""
    if not any(gyro_noise) or drift_budget_deg(1e7, *gyro_noise, n_sigma) < tol_yaw:
        return None
    return float(brentq(lambda T: drift_budget_deg(T, *gyro_noise, n_sigma) - tol_yaw, 1e-6, 1e7))


def hold(value_t, values, t, hold_s, width):
    """Latest value at or before each t, NaN (not scorable) before the first one or when it is older than hold_s."""
    out = np.full((len(t), width), np.nan)
    if values:
        value_t = np.asarray(value_t)
        k = np.searchsorted(value_t, t, side='right') - 1
        ok = k >= 0
        ok[ok] = t[ok] - value_t[k[ok]] <= hold_s
        out[ok] = np.asarray(values)[k[ok]]
    return out


def rereference(signed, window):
    """Signed cumulative statistic minus its mean over the alignment window, computed causally.

    eval-traj's risk mode aligns the estimate by yaw and translation over a predeclared initial window, so a label error
    of zero means "equal to the window mean", whereas a cumulative conflict is zero at t_0, the first estimate.
    Subtracting the window mean puts both on one reference. Inside the window the running mean of the window samples so
    far is used (those samples are never scored), after it the full window mean; a column without a finite window
    sample keeps the t_0 reference. window=None leaves the t_0 reference.
    """
    s = np.asarray(signed, dtype=float)
    if window is None:
        return s
    flat = s.ndim == 1
    s2 = s[:, None] if flat else s
    use = np.asarray(window, bool)[:, None] & np.isfinite(s2)
    count = np.cumsum(use, axis=0)
    total = np.cumsum(np.where(use, s2, 0.0), axis=0)
    out = s2 - np.where(count > 0, total/np.maximum(count, 1), 0.0)
    return out[:, 0] if flat else out


def information_gap(t, lam, lam_min):
    """tau_k: time since the windowed information on axis k last reached lam_min (from the first stamp if never)."""
    tau = np.zeros_like(lam)
    for a in range(lam.shape[1]):
        last = t[0]
        for i, (ti, v) in enumerate(zip(t, lam[:, a])):
            if np.isfinite(v) and v >= lam_min[a]:
                last = ti
            tau[i, a] = ti - last
    return tau
