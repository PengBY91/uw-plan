# Underwater acoustic-optic W1 lab

Team-maintained acquisition, conversion, audit and baseline scripts are versioned here.
The Git branch is codex/w1-public-baseline. No remote is configured by this setup.

- Runtime: Ubuntu 24.04, ROS 2 Jazzy, system Python 3.12.
- Run ROS tools through bash scripts/ros-run.sh.
- Build OpenVINS with bash scripts/build-openvins.sh.
- Offline evaluation uses .venv-eval; locked packages are in locks/eval-requirements.txt.
- Official upstream SHAs and URLs are in locks/repositories.json.
- Apply locks/openvins-jazzy.patch to the locked OpenVINS checkout before building.
- vendor/ and ros_ws/ remain separate upstream working copies and are ignored by this repository.
- data/, runs/, cache/, SDK packages, virtual environments, compiled binaries and build products are not committed.
- Small manifests preserve paths, sizes, checksums and provenance without copying dataset files.

Public-dataset runs verify the software pipeline, not underwater performance.
Execution reports and outstanding physical-platform inputs are in reports/w1/.

## Verified public baseline (2026-10-10)

The accepted W1 R1 run is runs/w1_v101_a0_r04, team commit ca5f5549a73acb6063da9b8260597bd18ccba65e, at 0.5x replay. Corrected-reference SE3 APE RMSE: 0.177510 m; 1-second translation RPE RMSE: 0.067166 m. The 1x run diverged; real-time operation is not accepted. Shutdown emitted class_loader warnings; consult reports/w1/v101-baseline.md and baseline-summary.json, including all failed runs.

Runtime precondition (temporary until WSL restart):

    sudo /usr/sbin/sysctl -w net.core.rmem_max=16777216

Original limit was 212992; no persistent sysctl file was changed. The Cyclone wrapper refuses a limit below 4 MiB.

To restore the local Cyclone installation on this same Linux workspace layout, download the five exact files from locks/cyclone-debs.json on a reachable host and verify their SHA256 values. For each verified file run:

    dpkg-deb -x /path/to/verified-package.deb /home/steve/uw-ao-lab/local/cyclone

This is a project-local extraction, not a system package upgrade. The wrapper supplies both lib and lib/x86_64-linux-gnu paths plus the local ament prefix. Cyclone XML is tracked in configs/cyclonedds-local.xml. No ROS daemon is required for the launch guard.

From a clean Git checkout, choose a new run ID (r01-r04 already exist):

    bash scripts/ros-cyclone-run.sh /usr/bin/python3 scripts/run_openvins.py --lab "$PWD" --bag "$PWD/data/public/euroc/V1_01_easy_ros2" --run-id w1_v101_a0_r05 --rate 0.5
    .venv-eval/bin/python scripts/evaluate_openvins.py --run "$PWD/runs/w1_v101_a0_r05" --audit "$PWD/reports/w1/v101-data-audit/audit.json" --gt-original "$PWD/data/public/euroc/V1_01_easy_asl/mav0/state_groundtruth_estimate0/data.csv" --gt-corrected "$PWD/ros_ws/src/open_vins/ov_data/euroc_mav/V1_01_easy.csv"

r05 is an example new ID, not a claimed completed run. Evaluation uses actual timestamps for 1s/5s RPE; evo delta_unit=s is unsupported. A scored metrics file is not automatically an accuracy pass. Keep initialization and unmatched/failed intervals visible.

First-time data acquisition: copy locks/archive-entries.json, locks/eth-item.json and locks/eth-files.json into the staging directory before running scripts/download_v101.py --stage <staging>. On this machine Windows networking was used, followed by scripts/prepare_euroc.py in the evaluation venv. Existing verified data can be re-audited with prepare_euroc.py --resume; the downloader refuses overwriting final files. All data fingerprints are in manifests/euroc_v101.yaml.

The raw data, binaries and full run logs stay local. Curated metadata, hash manifests and the report are committed. No remote has been configured or pushed.
