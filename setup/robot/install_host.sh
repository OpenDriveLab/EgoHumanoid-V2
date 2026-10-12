#!/usr/bin/env bash
# Install/build only. Never launch a controller, bridge, camera or PICO service.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: bash setup/robot/install_host.sh [--check | SONIC_DIRECTORY]"
    echo "Default: sonic; --check checks tools without installing."
    exit 0
fi
fail() { echo "ERROR: $*" >&2; exit 1; }
(( $# <= 1 )) || fail "Expected at most one argument."
[[ "${1:-}" != --* || "${1:-}" == --check ]] || fail "Unknown option: $1"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "Host setup requires x86_64 Linux."
for tool in git uv cmake make g++ ffmpeg ffprobe; do
    command -v "$tool" >/dev/null || fail "Missing prerequisite: $tool"
done
git lfs version >/dev/null
if [[ "${1:-}" == --check ]]; then
    echo "Tool checks passed. CMake will verify CUDA, TensorRT and ONNX Runtime during installation."
    exit 0
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sonic="${1:-$project_root/sonic}"
revision=b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
if [[ ! -e "$sonic" ]]; then
    mkdir -p "$(dirname "$sonic")"
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$sonic"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$sonic" checkout "$revision"
fi
[[ -e "$sonic/.git" ]] || fail "Existing target is not a Git checkout; left unchanged."
[[ "$(git -C "$sonic" rev-parse HEAD)" == "$revision" ]] || fail "Existing SONIC revision differs; left unchanged."
cd "$sonic"
patch="$project_root/setup/robot/sonic.patch"
if ! git apply --reverse --check "$patch" 2>/dev/null; then
    git apply --check "$patch" || fail "Patch conflicts with existing SONIC source."
    git apply "$patch"
fi
git lfs pull --include='gear_sonic_deploy/thirdparty/**,gear_sonic_deploy/reference/example/**,gear_sonic/data/robot_model/model_data/g1/**,gear_sonic/data/human/human_joints_info.pkl'
# Pin the same Revo2 geometry for optimization and simulation.
model_source="$PWD/models/unitree_ros"
model_revision=43e9f3e34265f52a19b2c4b264dc92c4241dde3b
if [[ ! -e "$model_source" ]]; then
    mkdir -p "$model_source"
    git -C "$model_source" init
    git -C "$model_source" remote add origin https://github.com/unitreerobotics/unitree_ros.git
    git -C "$model_source" sparse-checkout init --cone
    git -C "$model_source" sparse-checkout set robots/g1_with_brainco_hand
    git -C "$model_source" fetch --depth 1 --filter=blob:none origin "$model_revision"
    git -C "$model_source" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$model_source" rev-parse HEAD)" == "$model_revision" ]] || fail "Existing Revo2 model revision differs; left unchanged."
[[ -z "$(git -C "$model_source" status --porcelain --untracked-files=no)" ]] || fail "Existing Revo2 model source has edits; left unchanged."
model="$PWD/models/g1_revo2"
mkdir -p "$model"
cp "$project_root/setup/robot/model/"{g1_revo2.xml,scene.xml,LICENSE} "$model/"
cp "$model_source/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf" "$model/g1_revo2.urdf"
if [[ ! -e "$model/meshes" && ! -L "$model/meshes" ]]; then
    ln -s ../unitree_ros/robots/g1_with_brainco_hand/meshes "$model/meshes"
fi
[[ "$(readlink -f "$model/meshes")" == "$model_source/robots/g1_with_brainco_hand/meshes" ]] || fail "Existing Revo2 mesh directory differs; left unchanged."
if [[ ! -e .venv ]]; then
    uv venv --python 3.10 .venv
fi
[[ -x .venv/bin/python ]] || fail "Existing .venv is not usable; left unchanged."
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e 'gear_sonic[sim,data_collection]' \
    -e external_dependencies/unitree_sdk2_python 'torch==2.6.0+cpu' \
    'torchvision==0.21.0+cpu' 'huggingface-hub<1' 'opencv-python>=4.8,<4.12'
uv pip install --python .venv/bin/python -e "$project_root/data_collection/human" \
    -e "$project_root/data_alignment/action_alignment"
bash "$project_root/setup/install_xrobotoolkit.sh" "$PWD/.venv"
# Install frozen endpoint response weights.
.venv/bin/python - "$project_root" "$PWD" <<'RESPONSE_MODELS'
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen

project, sonic = map(Path, sys.argv[1:])
checksums = {
    'hands': '2cc914b6eea15abad20baa38631c9852c990a5a90a674f501229bd27a566060e',
    'feet': '771700333b10a753cda4e2020bf0dac05aa1974ff5226cff909fc14d3d71a692',
}
for name, checksum in checksums.items():
    entry = dict(path=f'models/response/{name}.pt', sha256=checksum,
                 url=f'https://github.com/OpenDriveLab/EgoHumanoid-V2/releases/download/response-models/{name}.pt')
    target = sonic / entry['path']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
            sys.exit(f'Existing response model differs; left unchanged: {target}')
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(entry['url'], timeout=60) as source:
                shutil.copyfileobj(source, stream)
            stream.flush()
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != entry['sha256']:
                sys.exit(f'Response model checksum mismatch: {target.name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
RESPONSE_MODELS
.venv/bin/python - "$project_root" <<'SONIC_MODELS'
import hashlib
from pathlib import Path
import subprocess
import sys

checksums = {
    'model_encoder.onnx': '013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3',
    'model_decoder.onnx': 'c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed',
    'observation_config.yaml': '466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b',
}
models = [(Path('gear_sonic_deploy/policy/release') / name, checksum)
          for name, checksum in checksums.items()]
models.append((Path('gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx'),
               '39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea'))

def matches(path, expected):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected

if not all(matches(path, expected) for path, expected in models):
    subprocess.run([sys.executable, 'download_from_hf.py'], check=True)
for path, expected in models:
    if not matches(path, expected):
        sys.exit(f'Missing or incompatible SONIC model: {path}')
SONIC_MODELS
cmake -S gear_sonic_deploy -B gear_sonic_deploy/build -DCMAKE_BUILD_TYPE=Release
cmake --build gear_sonic_deploy/build --parallel 4
uv pip check --python .venv/bin/python
.venv/bin/python gear_sonic/scripts/pico_manager_thread_server.py --help >/dev/null
.venv/bin/python gear_sonic/scripts/run_data_exporter.py --help >/dev/null
.venv/bin/python gear_sonic/scripts/run_gopro_receiver.py --help >/dev/null
.venv/bin/python -c 'from gear_sonic.utils.inference.openpi_http_policy import OpenPIHttpPolicy'
.venv/bin/python gear_sonic/scripts/run_vla_inference.py --help >/dev/null
echo "Installed host environment. Activate with: source \"$PWD/.venv/bin/activate\""
