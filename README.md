# Underwater acoustic-optic W1 lab

Team-maintained acquisition, conversion, audit and baseline scripts are versioned here.
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

r11 is an example new ID, not a completed run. The bag runner explicitly uses reliable IMU by default. Bare OpenVINS launch defaults to best-effort for live-driver compatibility; reliable subscription requires a compatible publisher. Input trace is written after worker shutdown. Acceptance checks input and update completeness, output validity/coverage, accuracy and shutdown, not just playback. RPE uses actual timestamp pairs; evo delta_unit=s is unsupported.

First-time data acquisition: copy locks/archive-entries.json, locks/eth-item.json and locks/eth-files.json into the staging directory before running scripts/download_v101.py --stage <staging>. On this machine Windows networking was used, followed by scripts/prepare_euroc.py in the evaluation venv. Existing verified data can be re-audited with prepare_euroc.py --resume; the downloader refuses overwriting final files. All data fingerprints are in manifests/euroc_v101.yaml.

The raw data, binaries and full run logs stay local. Curated metadata, hash manifests and the report are committed. No remote has been configured or pushed.
