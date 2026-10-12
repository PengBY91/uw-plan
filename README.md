# Underwater acoustic-optic lab (topics A and B)

One repository serves topic A (direction-level failure detection, plan v2) and topic B (single-FLS + stereo mapping, plan v2), following the shared W1, month-1 and month-2 plans.
Team-maintained acquisition, conversion, audit, baseline and evaluation code is versioned here.

- scripts/: W1 public-baseline tooling (EuRoC import, OpenVINS replay, audits, acceptance).
- team/uwbench/: shared package and the uwbench CLI contract (runbook E7/R5). Run with PYTHONPATH=team/uwbench; `python -m uwbench list` shows which contracted subcommands exist. Unimplemented ones exit non-zero.
  - uwbench.sonar: range/bearing model and elevation arcs shared by A's sonar factor and B's candidates; B 5.3 effect-size estimate.
  - uwbench.metrics.axes: A v2 scoring, body-axis x/y/z/yaw error, estimator sigma and time-integrated exceedance with unavailable gaps.
  - uwbench.metrics.occupancy: B section 6 map ratios with undefined (None) empty denominators.
  - uwbench.mapping: B0 dense tri-state voxel grid (OctoMap log-odds defaults) with per-voxel source and update time; invalid or out-of-range depth never clears, mm depth is rejected.
  - `uwbench audit` (W1-05/P03): manifest schema from runbook E6; hashes, monotonic time, back-jumps, drop estimate, stereo pairing, image shape, IMU units/gravity axis/rate, pressure units, empty sonar. Example: manifests/euroc_v101_run.yaml, evidence in reports/w1/euroc_v101_audit/.
  - `uwbench eval-traj` (W1-07/P04): APE/RPE, time coverage, lost intervals, reference-invalid vs unmatched counts, v2 per-axis exceedance; empty estimates fail.
- reports/w1/backend-decision.md (W1-09): SVIn not reused (zero-elevation single-beam sonar model, sonar/depth disabled on ROS2 main); A1 uses GTSAM 4.3.0 in a separate venv.
- Tests: `.venv-eval/bin/python -m unittest discover -s team/uwbench/tests` with PYTHONPATH=team/uwbench, plus scripts/test_*.py from scripts/.
The Git branch is codex/w1-public-baseline. No remote is configured by this setup.

- Runtime: Ubuntu 24.04, ROS 2 Jazzy, system Python 3.12.
- Run ROS tools through bash scripts/ros-run.sh.
- Build OpenVINS with bash scripts/build-openvins.sh.
- Offline evaluation uses .venv-eval; locked packages are in locks/eval-requirements.txt.
- Official upstream SHAs and URLs are in locks/repositories.json.
- Apply locks/openvins-jazzy.patch then locks/openvins-ros2-replay.patch to the locked OpenVINS checkout; commit the patched source before building. Order and hashes: locks/openvins-replay-fix.json.
- vendor/ and ros_ws/ remain separate upstream working copies and are ignored by this repository.
- data/, runs/, cache/, SDK packages, virtual environments, compiled binaries and build products are not committed.
- Small manifests preserve paths, sizes, checksums and provenance without copying dataset files.

Public-dataset runs verify the software pipeline, not underwater performance.
Execution reports and outstanding physical-platform inputs are in reports/w1/.

## Simulation (W2-11)

HoloOcean cannot start inside WSL (UE5 Vulkan fails on Dozen and rejects lavapipe), so generation runs in the Windows Python that has holoocean 2.3.0; conversion, audit, replay and scoring run here. Route JSONs are in configs/sim/; assumptions are marked ASSUMPTION. Results and diagnoses: reports/w2/sim-w2-11.md.

    PYW=/mnt/c/Users/pengb/AppData/Local/Programs/Python/Python311/python.exe
    $PYW "$(wslpath -w team/sim/holoocean_generate.py)" "$(wslpath -w configs/sim/<route>.json)" "$(wslpath -w $PWD/data/sim)\\<route>_s01\\raw"
    .venv-eval/bin/python scripts/sim_to_bag.py --run data/sim/<route>_s01
    PYTHONPATH=team/uwbench .venv-eval/bin/python -m uwbench audit --manifest data/sim/<route>_s01/manifest.yaml --out data/sim/<route>_s01/audit
    # then scripts/run_openvins.py with --config-dir data/sim/<route>_s01/openvins --dataset-audit .../audit/audit.json, and uwbench eval-traj

raw/ is the only irreplaceable part (HoloOcean noise is not seedable). ros2/ bags are byte-reproducible and may be deleted; restore with `scripts/sim_to_bag.py --run data/sim/<route>_s01 --restore-bag`, which verifies checksums.sha256. Sim A0 keeps configs/euroc_v101 except gravity and fixed camera calibration (SIM_BASELINE in scripts/sim_to_bag.py), plus per-route overrides recorded in the derived config header. `scripts/sim_track_check.py` checks visual tracks against ground truth (needs the ROS system python3 with cv2).

## Verified public baseline (2026-10-10)

The accepted 1x engineering regressions are r09/r10 (execution commit f8116d608064745d3148f3b16e15b05118028afa; OpenVINS ac543cd427ba880b00582abac45e01dbc51cc23b). Both received all 29120 IMUs and 2912 stereo pairs in source order and processed every pair. Both corrected-reference SE3 APE RMSE: 0.044367 m; 1s translation RPE RMSE: 0.012527 m, with clean shutdown. This does not establish underwater or hardware end-to-end real-time performance. See reports/w1/replay-fix.md and replay-fix-summary.json, including failed r05-r08. Historical r01-r04 remain in v101-baseline.md.

Runtime precondition (temporary until WSL restart):

    sudo /usr/sbin/sysctl -w net.core.rmem_max=16777216

Original limit was 212992; no persistent sysctl file was changed. The Cyclone wrapper refuses a limit below 4 MiB.

To restore the local Cyclone installation on this same Linux workspace layout, download the five exact files from locks/cyclone-debs.json on a reachable host and verify their SHA256 values. For each verified file run:

    dpkg-deb -x /path/to/verified-package.deb /home/steve/uw-ao-lab/local/cyclone

This is a project-local extraction, not a system package upgrade. The wrapper supplies both lib and lib/x86_64-linux-gnu paths plus the local ament prefix. Cyclone XML is tracked in configs/cyclonedds-local.xml. No ROS daemon is required for the launch guard.

From clean team and OpenVINS Git checkouts, choose an unused run ID (r01-r10 already exist):

    RUN=w1_v101_a0_r11
    bash scripts/ros-cyclone-run.sh /usr/bin/python3 scripts/run_openvins.py --lab "$PWD" --bag "$PWD/data/public/euroc/V1_01_easy_ros2" --run-id "$RUN" --rate 1.0 --imu-history-depth 1000 --imu-reliability reliable
    .venv-eval/bin/python scripts/audit_input_trace.py --bag "$PWD/data/public/euroc/V1_01_easy_ros2" --run "$PWD/runs/$RUN"
    .venv-eval/bin/python scripts/evaluate_openvins.py --run "$PWD/runs/$RUN" --audit "$PWD/reports/w1/v101-data-audit/audit.json" --gt-original "$PWD/data/public/euroc/V1_01_easy_asl/mav0/state_groundtruth_estimate0/data.csv" --gt-corrected "$PWD/ros_ws/src/open_vins/ov_data/euroc_mav/V1_01_easy.csv"
    .venv-eval/bin/python scripts/check_replay_acceptance.py --run "$PWD/runs/$RUN"

Shared-tool equivalent (manifest-based; public SE(3) and v2 risk alignment reported side by side):

    PYTHONPATH=team/uwbench .venv-eval/bin/python -m uwbench eval-traj --manifest manifests/euroc_v101_run.yaml --estimate "$PWD/runs/$RUN/estimate_bag" --out "$PWD/runs/$RUN/eval-traj"

On r10 it reproduces the archived APE (0.044362 vs 0.044367 m; one estimate past the reference end is counted as outside-valid instead of matched). Risk mode max |error|: x 0.050 m, y 0.128 m, z 0.104 m, yaw 1.49 deg; no exceedance at the provisional 0.15 m / 3 deg, so V1_01 does not exercise the detection track.

r11 is an example new ID, not a completed run. The bag runner explicitly uses reliable IMU by default. Bare OpenVINS launch defaults to best-effort for live-driver compatibility; reliable subscription requires a compatible publisher. Input trace is written after worker shutdown. Acceptance checks input and update completeness, output validity/coverage, accuracy and shutdown, not just playback. RPE uses actual timestamp pairs; evo delta_unit=s is unsupported.

First-time data acquisition: copy locks/archive-entries.json, locks/eth-item.json and locks/eth-files.json into the staging directory before running scripts/download_euroc.py --stage <staging> [--sequence V1_01_easy]. On this machine Windows networking was used, followed by scripts/prepare_euroc.py in the evaluation venv. Existing verified data can be re-audited with prepare_euroc.py --resume; the downloader refuses overwriting final files. All data fingerprints are in manifests/euroc_v101.yaml. Other sequences: run download_euroc.py --sequence MH_01_easy --list-entries once to read the archive central directory into archive-entries-machine_hall.json, then download, then prepare_euroc.py --sequence MH_01_easy --report reports/w2/euroc-mh --manifest manifests/euroc_mh01.yaml. The server rate-limits Range requests (HTTP 429), so the downloader backs off for minutes.

The raw data, binaries and full run logs stay local. Curated metadata, hash manifests and the report are committed. No remote has been configured or pushed.
