#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /opt/ros/noetic/setup.bash || ! -f /code/devel/setup.bash ]]; then
    echo "error: ROS Noetic/Duckietown setup not found; run this inside dts start_gui_tools." >&2
    exit 2
fi

# Old catkin setup scripts consume the caller's positional parameters.
TELEOP_ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/physical_duckiebot_teleop.py" "${TELEOP_ARGS[@]}"
