#!/usr/bin/env bash
set -euo pipefail
AO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec env -i HOME="$HOME" USER="$(id -un)" PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=C.UTF-8 AO_LAB="$AO_ROOT" MAKEFLAGS=-j2 CMAKE_BUILD_PARALLEL_LEVEL=2 bash --noprofile --norc -c '
source /opt/ros/jazzy/setup.bash
cd "$AO_LAB/ros_ws"
colcon build --executor sequential --packages-select ov_core ov_init ov_msckf ov_eval --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDISABLE_MATPLOTLIB=ON
'
