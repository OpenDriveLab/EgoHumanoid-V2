#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")/../.." && pwd)
runtime=${1:-"$project_root/runtime/vision"}
mkdir -p "$runtime"
runtime=$(cd "$runtime" && pwd)
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg before running this script.' >&2; exit 1; }

checkout() {
    local name=$1 url=$2 revision=$3 sparse=${4:-}
    local destination="$runtime/$name"
    if [[ ! -e "$destination" ]]; then
        mkdir -p "$destination"
        git -C "$destination" init -q
        git -C "$destination" remote add origin "$url"
        if [[ -n "$sparse" ]]; then
            git -C "$destination" sparse-checkout init --cone
            git -C "$destination" sparse-checkout set "$sparse"
        fi
        git -C "$destination" fetch --depth 1 --filter=blob:none origin "$revision"
        git -C "$destination" checkout --detach FETCH_HEAD
    fi
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || {
        echo "Existing $name revision differs; left unchanged." >&2; exit 1;
    }
    if [[ "$name" != sam3 ]]; then
        git -C "$destination" diff --quiet HEAD -- || {
            echo "Existing $name source has edits; left unchanged." >&2; exit 1;
        }
    fi
}
checkout sam3 https://github.com/facebookresearch/sam3.git f1647e8d0c1d280991f4d1b7d974c2704537fe66
checkout diffueraser https://github.com/lixiaowen-xw/DiffuEraser.git 8e6f279ac7531e27ad1849c6f8dab5372a8597e7
checkout unitree_ros https://github.com/unitreerobotics/unitree_ros.git 43e9f3e34265f52a19b2c4b264dc92c4241dde3b robots/g1_with_brainco_hand
patch="$project_root/setup/vision/sam3.patch"
if ! git -C "$runtime/sam3" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$runtime/sam3" apply --check "$patch"
    git -C "$runtime/sam3" apply "$patch"
fi
[[ -d "$runtime/.venv-sam" ]] || uv venv --python 3.12 "$runtime/.venv-sam"
[[ -d "$runtime/.venv-inpaint" ]] || uv venv --python 3.10 "$runtime/.venv-inpaint"
uv pip install --python "$runtime/.venv-sam/bin/python" torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$runtime/.venv-sam/bin/python" -r "$project_root/setup/vision/requirements-sam.txt"
uv pip install --python "$runtime/.venv-sam/bin/python" --no-deps -e "$runtime/sam3" -e "$project_root/data_alignment/vision_alignment"
uv pip install --python "$runtime/.venv-inpaint/bin/python" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$runtime/.venv-inpaint/bin/python" -r "$project_root/setup/vision/requirements-inpaint.txt"
mkdir -p "$runtime/models/g1"
cp "$project_root/setup/robot/model/"{g1_revo2.xml,LICENSE} "$runtime/models/g1/"
if [[ ! -e "$runtime/models/g1/meshes" ]]; then
    ln -s "$runtime/unitree_ros/robots/g1_with_brainco_hand/meshes" "$runtime/models/g1/meshes"
fi
env -u ALL_PROXY -u all_proxy "$runtime/.venv-sam/bin/python" "$project_root/setup/vision/download_models.py" --runtime "$runtime"
echo "Installed. Run $runtime/.venv-sam/bin/eh2-visual-align --help"
