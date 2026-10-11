#!/usr/bin/env bash
set -euo pipefail
AO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec env -i HOME="$HOME" USER="$(id -un)" PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=C.UTF-8 AO_LAB="$AO_ROOT" bash --noprofile --norc -c '
source /opt/ros/jazzy/setup.bash
if [ -f "$AO_LAB/ros_ws/install/setup.bash" ]; then source "$AO_LAB/ros_ws/install/setup.bash"; fi
export CMAKE_PREFIX_PATH="$AO_LAB/local:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$AO_LAB/local/lib:$LD_LIBRARY_PATH"
exec "$@"
' bash "$@"
