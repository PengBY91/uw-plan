# W1 1x replay repair protocol (declared before depth experiments)

Keep data, estimator YAML, calibration, middleware, evaluation and corrected/original references unchanged. Instrument actual estimator callbacks and camera update entry. Use one patched binary for both IMU SensorDataQoS history depths 5 and 1000. Both arms include capture-by-value and owned-worker lifetime fixes, so the queue comparison isolates history depth but does not quantify each lifetime fix independently.

Run r05 at 1x/depth 5 as diagnostic control, r06 at 1x/depth 1000, then repeat the passing repair as r07 with exactly the same settings. Preserve every run. No build or other full replay should overlap these measurements.

Engineering regression acceptance, not product accuracy targets: all 29120 primary IMU samples and all 2912 synchronized stereo pairs received in source order, no duplicates/mismatched stereo pairs, all received pairs processed; pose coverage >=95%, no post-init output gaps >75ms; corrected-reference SE3 APE RMSE <=0.20 m and 1s translation RPE RMSE <=0.08 m (bounds based on prior half-speed 0.177510/0.067166 m); no unexpected child exit or class_loader warning. Require two complete 1x repaired runs. Always report original-reference scores too.

If repaired runs fail a criterion, diagnose before changing the criterion; record any later method or protocol change explicitly. Report callback-to-update queue wait separately from sensor-to-output latency. Callback tracing can alter scheduling slightly; results characterize this audited configuration.

## Amendment after r06, before r07

r06 at best-effort depth 1000 recovered accuracy (APE 0.044129994 m, 1s RPE 0.013804897 m) but missed 45/29120 IMUs across the sequence, so it fails input acceptance and is NOT counted as a passing repair. Test optional reliable IMU subscription with depth 1000 in r07, followed by r08 if it passes. The bag publisher already offers reliable delivery; no estimator YAML, calibration, data, or acceptance bound changes. This isolates reliability from depth at configuration level, although the binary is rebuilt for the new option. Bare OpenVINS launch retains best-effort default for live-driver compatibility; our versioned bag runner explicitly defaults to reliable. Verify actual endpoints during replay.

## Amendment after r07, before r08

r07 reliable/depth1000 received all IMUs and stereo pairs but updated only 2911/2912 pairs. It fails complete-input acceptance despite APE 0.051242766 m. Trace shows the final worker was triggered at the last camera stamp, then all nine later IMUs arrived while the preceding camera update was running; the worker used its stale trigger timestamp and left the last camera queued. Refresh an atomic watermark of IMUs already fed before each queued camera update. It does not synthesize IMU data or drain using future truth. Use r08 and r09 as the two new complete 1x regression runs; all acceptance bounds unchanged. Preserve r07 as a failed intermediate run.

## Amendment after r08, before r09

r08 still fails one final camera update: trace shows the final stereo callback occurs AFTER all IMU callbacks, so refreshing the watermark alone is insufficient. Replace per-IMU ephemeral update threads with one owned worker and condition-variable pending predicate. Signal after either IMU feed or camera queue append, retaining notifications during processing; keep the latest fed IMU watermark. No input, estimator parameter or acceptance change. Run r09 and r10 as the two final complete 1x regressions. r08 remains failed (APE 0.044367362 m, complete receipt but 2911 updates).
