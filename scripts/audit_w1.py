#!/usr/bin/env python3
"""Read-only W1 probes; write evidence under the explicitly selected lab root."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def probe(argv, timeout=30):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"command": argv, "exit_code": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "exit_code": None, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=Path, required=True)
    args = parser.parse_args()
    lab = args.lab.resolve()
    if not (lab / "ros_ws/src/open_vins/.git").is_dir():
        parser.error("Expected the prepared W1 workspace with OpenVINS")
    out = lab / "reports/w1"
    out.mkdir(parents=True, exist_ok=True)
    commands = {
        "os": ["cat", "/etc/os-release"],
        "kernel": ["uname", "-r"],
        "cpu": ["lscpu", "-J"],
        "memory": ["free", "-h"],
        "disk": ["df", "-h", str(lab), "/mnt/c"],
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "git": ["git", "--version"],
        "cmake": ["cmake", "--version"],
        "compiler": ["g++", "--version"],
        "python": ["/usr/bin/python3", "--version"],
        "packages": ["dpkg-query", "-W", "libeigen3-dev", "libceres-dev", "libopencv-dev", "libboost-all-dev", "ros-jazzy-ros-base"],
        "eval_check": [str(lab / ".venv-eval/bin/python"), "-m", "pip", "check"],
        "eval_imports": [str(lab / ".venv-eval/bin/python"), "-c", "import evo,rosbags,numpy,scipy,pandas,yaml; print('imports OK')"],
        "evo_cli": [str(lab / ".venv-eval/bin/evo_ape"), "--help"],
        "rosbags_cli": [str(lab / ".venv-eval/bin/rosbags-convert"), "--help"],
    }
    results = {name: probe(cmd) for name, cmd in commands.items()}
    clean = ["env", "-i", "HOME=" + str(Path.home()), "PATH=/usr/bin:/bin:/usr/sbin:/sbin", "LANG=C.UTF-8", "bash", "--noprofile", "--norc", "-c"]
    results["ros"] = probe(clean + ["source /opt/ros/jazzy/setup.bash && command -v ros2 && python3 -c 'import rclpy;print(rclpy.__file__)' && ros2 pkg list"])
    repos = []
    for rel in ["ros_ws/src/open_vins", "vendor/octomap", "vendor/SVIn", "vendor/SonarSweep", "vendor/SonarSweep_dataset"]:
        p = lab / rel
        repos.append({"path": rel, "head": probe(["git", "-C", str(p), "rev-parse", "HEAD"]),
                      "origin": probe(["git", "-C", str(p), "remote", "get-url", "origin"]),
                      "status": probe(["git", "-C", str(p), "status", "--short"]),
                      "submodules": probe(["git", "-C", str(p), "submodule", "status"])})
    result = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "lab": str(lab), "probes": results, "repositories": repos,
              "device_nodes": [str(p) for pattern in ["video*", "ttyUSB*", "ttyACM*", "dxg"] for p in Path("/dev").glob(pattern)]}
    (out / "environment-probes.json").write_text(json.dumps(result, indent=2) + "\n")
    (lab / "locks/repositories.json").write_text(json.dumps(repos, indent=2) + "\n")
    print(json.dumps({name: value.get("exit_code") for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
