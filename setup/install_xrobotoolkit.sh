#!/usr/bin/env bash
# Install the pinned native PICO binding into an existing Python 3.10 venv.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: bash setup/install_xrobotoolkit.sh VENV"
    exit 0
fi
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ $# == 1 && -x "$1/bin/python" ]] || fail "Provide an existing Python 3.10 environment."
[[ "$(uname -s)" == Linux ]] || fail "This installer requires Linux."
venv="$(cd "$1" && pwd)"
python="$venv/bin/python"
"$python" -c 'import sys, sysconfig; from pathlib import Path; assert sys.prefix != sys.base_prefix, "Use a virtual environment"; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"; assert Path(sysconfig.get_path("include"), "Python.h").is_file(), "Python development headers required"'
for tool in git cmake make g++; do
    command -v "$tool" >/dev/null || fail "Missing build prerequisite: $tool"
done
"$python" -m ensurepip --upgrade
"$python" -m pip install --upgrade pip 'setuptools<81' wheel 'pybind11>=2.13,<3'

# Pin both sides of the native interface; never update an existing checkout.
service_rev=85bac4dbc1fd5cef42c74a160d9c30aa3491f122
binding_rev=c64ccf6acd577a333e03b66fafe8efeeceb511b1
sdk_root="$venv/xrobotoolkit"
service="$sdk_root/service-$service_rev"
binding="$sdk_root/binding-$binding_rev"
checkout() {
    local repository="$1" revision="$2" destination="$3"
    if [[ ! -e "$destination" ]]; then
        git init "$destination"
        git -C "$destination" remote add origin "$repository"
        git -C "$destination" fetch --depth 1 origin "$revision"
        git -C "$destination" checkout --detach FETCH_HEAD
    fi
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] ||
        fail "SDK revision mismatch in $destination; existing files were not replaced."
}
mkdir -p "$sdk_root"
checkout https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "$service_rev" "$service"
checkout https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git "$binding_rev" "$binding"

sdk_source="$service/RoboticsService/PXREARobotSDK"
cmake -S "$sdk_source" -B "$sdk_source/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$sdk_source/build" --config Release --parallel 2
native_suffix=""
if [[ "$(uname -m)" == aarch64 ]]; then native_suffix=/aarch64; fi
mkdir -p "$binding/lib$native_suffix" "$binding/include$native_suffix"
cp "$sdk_source/build/libPXREARobotSDK.so" "$binding/lib$native_suffix/"
cp "$sdk_source/PXREARobotSDK.h" "$binding/include$native_suffix/"
cp -R "$sdk_source/nlohmann" "$binding/include$native_suffix/"
CMAKE_PREFIX_PATH="$("$python" -m pybind11 --cmakedir)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" \
    "$python" -m pip install --no-build-isolation "$binding"

"$python" -c 'import xrobotoolkit_sdk as xrt; required = ("init", "close", "is_body_data_available", "get_body_joints_pose", "get_left_controller_pose", "get_right_controller_pose", "get_left_hand_tracking_state", "get_right_hand_tracking_state", "get_left_hand_is_active", "get_right_hand_is_active"); missing = [name for name in required if not hasattr(xrt, name)]; assert not missing, f"Missing PICO SDK APIs: {missing}"'
echo "XRoboToolkit installed; hardware was not initialized."
