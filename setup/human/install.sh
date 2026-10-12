#!/usr/bin/env bash
# Native Linux setup; SDK source/builds stay inside the collection environment.
set -euo pipefail

case "${1:-}" in
    --help|-h)
        echo "Usage: bash setup/human/install.sh [--check]"
        echo "--check: check system prerequisites without installing anything."
        exit 0 ;;
    ""|--check) ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Expected at most one option." >&2
    exit 2
fi

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || fail "This installer requires Linux."
missing=()
for tool in python3.10 git cmake make g++ ffmpeg ffprobe bluetoothctl nmcli; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if (( ${#missing[@]} )); then
    echo "Missing tools: ${missing[*]}" >&2
    echo "On Ubuntu 22.04, install prerequisites with:" >&2
    echo "sudo apt-get install python3.10-dev python3.10-venv build-essential cmake git ffmpeg bluez network-manager" >&2
    exit 1
fi
python3.10 -c 'import ensurepip, sys, sysconfig; from pathlib import Path; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"; assert Path(sysconfig.get_path("include"), "Python.h").is_file(), "Install Python 3.10 development headers"'
if [[ "${1:-}" == --check ]]; then
    echo "System prerequisite checks passed (device connectivity not tested)."
    exit 0
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="$project_root/.venv-human"
if [[ ! -e "$venv" ]]; then
    python3.10 -m venv "$venv"
else
    [[ -x "$venv/bin/python" ]] || fail "Existing $venv is not a usable environment; left unchanged."
fi
python="$venv/bin/python"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing environment must use Python 3.10"'
bash "$project_root/setup/install_xrobotoolkit.sh" "$venv"
"$python" -m pip install -e "$project_root/data_collection/human"

# Imports and CLI checks only: never initialize hardware during installation.
"$venv/bin/eh2-collect-human" --help >/dev/null
"$venv/bin/eh2-sync-human" --help >/dev/null
echo "Installed. Activate with: source \"$venv/bin/activate\""
