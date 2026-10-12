#!/usr/bin/env bash
# Build the onboard dependencies; never launch a robot or camera process.
set -eo pipefail

case "${1:-}" in
    --help|-h)
        echo "Usage: bash setup/robot/install_robot.sh [--check]"
        echo "Ubuntu 20.04 / ROS 2 Foxy / ARM64. --check only checks prerequisites."
        exit 0 ;;
    ""|--check) ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo "Expected at most one option." >&2; exit 2; }
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] ||
    fail "The onboard installer requires ARM64 Linux (the supplied hand SDK is ARM64)."
ros_setup="${EGH_ROS_SETUP:-/opt/ros/foxy/setup.bash}"
[[ -f "$ros_setup" ]] || fail "Install ROS 2 Foxy first: missing $ros_setup"
for tool in python3 git cmake make g++ colcon; do
    command -v "$tool" >/dev/null || fail "Missing prerequisite: $tool"
done
python3 -c 'import ensurepip, sys; assert sys.version_info[:2] == (3, 8), "Foxy requires Python 3.8"'
if [[ "${1:-}" == --check ]]; then
    echo "Prerequisite checks passed; no installation or hardware access performed."
    exit 0
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="$project_root/.venv-robot"
if [[ ! -e "$venv" ]]; then
    python3 -m venv "$venv"
else
    [[ -x "$venv/bin/python" ]] || fail "Existing $venv is not a usable environment; left unchanged."
fi
python="$venv/bin/python"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 8), "Existing environment must use Python 3.8"'
"$python" -m pip install --upgrade pip

checkout() {
    local repository="$1" revision="$2" destination="$3"
    if [[ ! -e "$destination" ]]; then
        git init "$destination"
        git -C "$destination" remote add origin "$repository"
        git -C "$destination" fetch --depth 1 origin "$revision"
        git -C "$destination" checkout --detach FETCH_HEAD
    fi
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] ||
        fail "Revision mismatch in $destination; existing files were not replaced."
    git -C "$destination" diff --quiet HEAD -- ||
        fail "Modified upstream source in $destination; left unchanged."
}
sources="$venv/sources"
mkdir -p "$sources"
checkout https://github.com/eclipse-cyclonedds/cyclonedds.git 9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd "$sources/cyclonedds"
checkout https://github.com/ros2/rmw_cyclonedds.git c12abc56983204f1d91f2d839d394528c7b29b42 "$sources/rmw_cyclonedds"
checkout https://github.com/unitreerobotics/unitree_ros2.git 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88 "$sources/unitree_ros2"
checkout https://github.com/BrainCoTech/unitree-g1-brainco-hand.git 1866fe5042f796560164733f2cb289c11b57c058 "$sources/brainco"
checkout https://github.com/unitreerobotics/unitree_sdk2_python.git 814556d15970dd2ecf1c9984e845ca02ab07e206 "$sources/sdk"

# This generated build workspace is separate from the pinned upstream checkouts.
workspace="$venv/ros_ws"
stage() { mkdir -p "$2"; cp -R "$1/." "$2/"; }
stage "$sources/cyclonedds" "$workspace/src/cyclonedds"
stage "$sources/rmw_cyclonedds" "$workspace/src/rmw_cyclonedds"
for package in unitree_api unitree_go unitree_hg; do
    stage "$sources/unitree_ros2/cyclonedds_ws/src/unitree/$package" "$workspace/src/$package"
done
for package in ros2_stark_interfaces ros2_stark_controller; do
    stage "$sources/brainco/ros2_stark_ws/src/$package" "$workspace/src/$package"
done
patches="$project_root/robot/hands"
cp "$patches/MotorStatus.msg" "$workspace/src/ros2_stark_interfaces/msg/"
cp "$patches/stark_node.cpp" "$workspace/src/ros2_stark_controller/src/"
cp "$patches/stark_node.hpp" "$workspace/src/ros2_stark_controller/include/ros2_stark_controller/"

cd "$workspace"
export CMAKE_BUILD_PARALLEL_LEVEL=2
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
colcon build --executor sequential --packages-select cyclonedds --cmake-args -DBUILD_TESTING=OFF
# ROS setup scripts are not compatible with bash nounset.
source "$ros_setup"
source "$workspace/install/setup.bash"
colcon build --executor sequential --packages-select rmw_cyclonedds_cpp unitree_api unitree_go unitree_hg ros2_stark_interfaces ros2_stark_controller --cmake-args -DBUILD_TESTING=OFF
source "$workspace/install/setup.bash"
[[ -x "$workspace/install/ros2_stark_controller/lib/ros2_stark_controller/stark_node" ]] ||
    fail "Hand driver did not build."

export CYCLONEDDS_HOME="$workspace/install/cyclonedds"
# Editable SDK installation retains the native CRC libraries omitted by its wheel.
"$python" -m pip install -e "$sources/sdk" "$project_root/robot[bridge]"
"$python" -m pip check
"$venv/bin/eh2-robot-bridge" --check
"$venv/bin/eh2-gopro-relay" --help >/dev/null
echo "Installation complete. No services or hardware were started."
echo "Source $ros_setup, $workspace/install/setup.bash and $venv/bin/activate before use."
