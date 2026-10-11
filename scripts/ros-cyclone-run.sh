#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$(cat /proc/sys/net/core/rmem_max)" -lt 4194304 ]; then
  echo "UDP receive limit too small. Run: sudo /usr/sbin/sysctl -w net.core.rmem_max=16777216" >&2
  exit 2
fi
exec bash "$ROOT/scripts/ros-run.sh" bash -c '
P="$AO_LAB/local/cyclone/opt/ros/jazzy"
export AMENT_PREFIX_PATH="$P:$AMENT_PREFIX_PATH"
export LD_LIBRARY_PATH="$P/lib:$P/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$AO_LAB/configs/cyclonedds-local.xml"
export ROS_DOMAIN_ID=76
export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT
exec "$@"
' bash "$@"
