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
