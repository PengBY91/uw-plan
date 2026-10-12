"""Sonar range-bearing factor for GTSAM (A plan section 2, form 1: point landmark with known correspondence).

State: X = T_world_body (Pose3, body = IMU), landmark p_W (Point3). Fixed extrinsic T_body_sonar.
p_S = T_body_sonar^-1 (X^-1 p_W); residual e = [rho(p_S) - rho_meas, wrap(theta(p_S) - theta_meas)] in metres and radians.
Elevation is not part of the measurement and is never set to zero; it is whatever the landmark position implies.
"""
import gtsam
import numpy as np

from uwbench.sonar import range_bearing, range_bearing_jacobian, wrap_angle

KERNELS = ('none', 'huber', 'cauchy')


def noise_model(sigma_range_m, sigma_bearing_rad, kernel='none', k=1.345):
    """Diagonal Gaussian in (m, rad), optionally wrapped in a fixed robust kernel (k in whitened units)."""
    if sigma_range_m <= 0 or sigma_bearing_rad <= 0:
        raise ValueError('Sonar sigmas must be positive')
    base = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_range_m, sigma_bearing_rad], dtype=float))
    if kernel == 'none':
        return base
    if kernel not in KERNELS:
        raise ValueError('Unknown robust kernel: ' + str(kernel))
    estimator = (gtsam.noiseModel.mEstimator.Huber.Create(k) if kernel == 'huber'
                 else gtsam.noiseModel.mEstimator.Cauchy.Create(k))
    return gtsam.noiseModel.Robust.Create(estimator, base)


def predict(T_world_body, T_body_sonar, p_world):
    """Sonar-frame point, its (rho, theta) and Jacobians of (rho, theta) w.r.t. the body pose (GTSAM right
    perturbation [omega, v]) and the world point."""
    H_pose, H_point = np.zeros((3, 6), order='F'), np.zeros((3, 3), order='F')
    p_body = T_world_body.transformTo(np.asarray(p_world, dtype=float), H_pose, H_point)
    R_bs, t_bs = T_body_sonar.rotation().matrix(), T_body_sonar.translation()
    p_sonar = R_bs.T @ (p_body - t_bs)
    J = range_bearing_jacobian(p_sonar) @ R_bs.T
    rho, theta = range_bearing(p_sonar)
    return p_sonar, np.array([rho, theta]), J @ H_pose, J @ H_point


def residual(T_world_body, T_body_sonar, p_world, measured):
    _, z, Hx, Hp = predict(T_world_body, T_body_sonar, p_world)
    e = z - np.asarray(measured, dtype=float)
    e[1] = wrap_angle(e[1])
    return e, Hx, Hp


def sonar_factor(pose_key, point_key, measured, T_body_sonar, model):
    measured = np.asarray(measured, dtype=float).copy()

    def error(this, values, jacobians):
        e, Hx, Hp = residual(values.atPose3(pose_key), T_body_sonar, values.atPoint3(point_key), measured)
        if jacobians is not None:
            jacobians[0] = Hx
            jacobians[1] = Hp
        return e

    return gtsam.CustomFactor(model, [pose_key, point_key], error)


def whitened_norm(factor, values):
    """sqrt(2 * error): the whitened residual norm before any robust kernel (factor.error applies the kernel)."""
    f = factor
    if isinstance(f, gtsam.CustomFactor):
        e = f.unwhitenedError(values)
        model = f.noiseModel()
        if isinstance(model, gtsam.noiseModel.Robust):
            model = model.noise()
        return float(np.linalg.norm(model.whiten(e)))
    return float(np.sqrt(2*f.error(values)))


def elevation_search(T_first, T_body_sonar, measured_first, others, aperture_rad, sigma, steps=41):
    """Bounded non-parametric elevation initialisation (Westman 2020 style): try phi over the vertical aperture
    for the first echo, place the landmark on that arc, keep the phi whose arc point best explains the other
    (pose, measurement) pairs. Returns (p_world, phi, mean squared whitened residual) or None if nothing usable."""
    from uwbench.sonar import arc_points
    if aperture_rad <= 0:
        raise ValueError('Vertical aperture must be positive')
    if not others:
        return None
    phis = np.linspace(-aperture_rad/2, aperture_rad/2, steps)
    arc = arc_points(measured_first[0], measured_first[1], phis)
    T_ws = T_first.compose(T_body_sonar)
    best = None
    for phi, q in zip(phis, arc):
        p_w = T_ws.transformFrom(q)
        cost = 0.0
        for T, z in others:
            e, _, _ = residual(T, T_body_sonar, p_w, z)
            cost += float(np.sum((e/sigma)**2))
        cost /= len(others)
        if best is None or cost < best[2]:
            best = (p_w, float(phi), cost)
    return best
