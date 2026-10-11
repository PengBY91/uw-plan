# W1 1x replay repair protocol (declared before depth experiments)

Keep data, estimator YAML, calibration, middleware, evaluation and corrected/original references unchanged. Instrument actual estimator callbacks and camera update entry. Use one patched binary for both IMU SensorDataQoS history depths 5 and 1000. Both arms include capture-by-value and owned-worker lifetime fixes, so the queue comparison isolates history depth but does not quantify each lifetime fix independently.

Run r05 at 1x/depth 5 as diagnostic control, r06 at 1x/depth 1000, then repeat the passing repair as r07 with exactly the same settings. Preserve every run. No build or other full replay should overlap these measurements.

Engineering regression acceptance, not product accuracy targets: all 29120 primary IMU samples and all 2912 synchronized stereo pairs received in source order, no duplicates/mismatched stereo pairs, all received pairs processed; pose coverage >=95%, no post-init output gaps >75ms; corrected-reference SE3 APE RMSE <=0.20 m and 1s translation RPE RMSE <=0.08 m (bounds based on prior half-speed 0.177510/0.067166 m); no unexpected child exit or class_loader warning. Require two complete 1x repaired runs. Always report original-reference scores too.

If repaired runs fail a criterion, diagnose before changing the criterion; record any later method or protocol change explicitly. Report callback-to-update queue wait separately from sensor-to-output latency. Callback tracing can alter scheduling slightly; results characterize this audited configuration.
