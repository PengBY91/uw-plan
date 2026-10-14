"""uwbench detect-a, W2 pilot (P18 contract): causal replay of detector statistics over finished estimator runs.

Modules, in dependency order: config (defaults, detector catalogue) -> signals (causal time-series helpers, drift
budget) -> sonar (sonar conflict sources) and inputs (what is read from a run) -> statistics (per-run detector
statistics) -> scoring (leave-one-run-out thresholds and miss / false-alarm rows) -> cli (`uwbench detect-a`, outputs).

Pilot scope, stated so nobody reads it as the M3 protocol:
- detectors: DET-NIS, DET-COV, DET-FEAT, and two DET-DIR parts: information gap tau_k (equivalent to thresholding a
  monotone drift envelope) and cross-sensor conflict (gyro-only vs estimated heading, depth gauge vs estimated z).
  DET-SCALAR is not implemented (A2 score undefined). Sonar is not used by A0, so x/y have no conflict source.
- DET-DIR (added after pilot round 15) combines one source per axis: single-plane sonar for x, none for y (scored
  as unscorable), depth for z, frozen-bias gyro minus its drift budget for heading. It was assembled from results
  on the same pilot runs, so its pilot scores are optimistic.
- DET-SCALAR (added after the 2026-10-12 review) folds the same per-axis sources as DET-DIR into one number
  (max, or l2, of the tolerance-normalised conflicts) with one threshold, so DET-DIR against DET-SCALAR isolates what
  splitting by axis contributes; the earlier scalar baselines (NIS, feature count) do not see those sensors at all.
- operating points: leave-one-run-out; thresholds from the other runs' normal samples, false alarms counted as the
  fraction of normal samples with any alarm (too little normal data for alarms per 10 min). With early_warning_s > 0,
  samples that far before an exceedance onset are neither normal nor exceedance (alarms there are lead time).
  threshold_allocation 'union' (default since the review) gives every detector the same nominal any-axis budget on
  the training normals by bisection over a common per-axis quantile level; 'fixed' is the pilot's allocation (target/4
  per axis, target/8 for the conflict families), whose union budget depended on the detector. Operating points whose
  realised false-alarm fraction is more than comparable_tolerance (50 %) away from the target are flagged comparable:
  false, as the ablation runbook requires; before the review that rule was only applied by hand.
- scoring (since the review): r_miss counts an exceedance sample as missed whenever the detector did not alarm,
  including samples where its statistic was not scorable (NaN), so every detector is scored on the same samples.
  r_miss_scorable is the pilot's definition (unscorable samples left out), reported next to r_unscorable.
- reference (since the review): the labels come from eval-traj's risk mode, which aligns over the first
  align_window_s, so an error of zero means "equal to the window mean"; with conflict_reference 'window' every
  cumulative conflict (depth, gyro, single-plane sonar) is likewise referenced to its causal mean over that window
  instead of to t_0 ('t0', the pilot's form). The multi-plane and scan-to-map variants keep the t_0 reference.
- conflicts come in two forms: over a trailing window (injection rate) and since the first estimate (accumulated);
  the second was added after the first pilot pass showed the windowed form peaks before onset (candidate 2 of 10).
- the accumulated heading conflict integrates the gyro with three bias sources (method spec v2.2 M3.1): the
  estimator's online bias (absorbs part of the conflict), the online bias frozen at the first estimate, and, in
  simulation only, the simulator's true bias (an oracle bound, never a detector). The frozen form is also scored
  against its drift budget e(T) = n_sigma * sqrt(N^2 T + K^2 T^3 / 3) from the run's IMU noise density N and random
  walk K; the budget leaves out the error of the frozen bias itself, so it is optimistic.
- sonar (when the run's dataset has sonar images): per frame, the first return above a threshold in each azimuth
  column (range and bearing only, as an imaging sonar gives) and a RANSAC line through them. The lines of the first
  sonar_reference_s after t_0 fix a world plane through the estimated poses (default: the label alignment window,
  align_window_s = 10 s; 0 uses the first line alone, the pilot's choice). An error d in the reference normal adds d
  times the distance travelled along the structure; the labels fix their yaw by the mean over the alignment window,
  so a plane fixed from one frame at t_0 differs from the label frame by the heading error at t_0 and the conflict
  then drifts away from the label error with travel (round 18: on the vertical-stripe wall the conflict read 0.46 m
  where the label read 0.18 m). The pilot judged the single frame better against t_0-aligned truth, which is not the
  label reference. Afterwards the measured distance to the line is compared with the distance the estimated position
  predicts. The difference is a cumulative conflict along the plane normal, projected on the estimated body x and y. Frames whose line normal is more than gate_deg from the reference plane
  (seen through the estimated attitude) are skipped, so another structure does not count as conflict.
  The multi-plane form (sonar_conflict_multi) keeps every surface it has seen: a line is matched to a known plane by
  normal (sonar_match_deg) and by a small change of that plane's conflict since it was last seen (sonar_jump_m);
  unmatched lines start a new plane through the current estimated pose that inherits the conflict accumulated so far.
  The scan-to-map form (sonar_conflict_map) keeps the first-return points themselves: the first scan after t_0 is
  the map, every later scan is registered to it by 2-D point-to-plane ICP over a horizontal translation and a
  rotation (points weighted by 1/map_sigma_m^2, plus a random-walk prior on the change of the correction between
  scans, so directions the geometry does not constrain stay put and single-scan noise does not bend the map),
  and the scan's new points are added where the registration puts them, not where the estimate does. The map
  therefore stays in the t_0 frame as it grows; the conflict is the registered minus the estimated sonar position,
  and the registered rotation is a sonar heading conflict.
- every statistic at time t uses only data stamped <= t; reference errors are read only to score.
Inputs per run: estimator.log with the b64db17 [DIAG] lines, estimate_bag (poseimu + odomimu), the input bag (imu0,
depth/pressure) recorded in run.json, and eval-traj/axis_errors.csv for labels.
"""
