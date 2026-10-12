#!/usr/bin/env bash
# Install OpenPI; do not start training or a policy server.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: bash setup/policy/install.sh"
    exit 0
fi
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ $# == 0 ]] || fail "No arguments expected."
for tool in git uv; do
    command -v "$tool" >/dev/null || fail "Missing prerequisite: $tool"
done
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
openpi="$project_root/openpi"
revision=215abfb217dbac7d5f1273282331b9b1866c0479
export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -e "$openpi" ]]; then
    git clone https://github.com/Physical-Intelligence/openpi.git "$openpi"
    git -C "$openpi" checkout "$revision"
fi
[[ -e "$openpi/.git" ]] || fail "Existing target is not a Git checkout; left unchanged."
[[ "$(git -C "$openpi" rev-parse HEAD)" == "$revision" ]] || fail "Existing OpenPI revision differs; left unchanged."
cd "$openpi"
git submodule update --init --recursive
uv sync --frozen --no-dev
uv pip install --reinstall-package transformers --link-mode copy transformers==4.53.2
cp -r src/openpi/models_pytorch/transformers_replace/. .venv/lib/python3.11/site-packages/transformers/
uv pip check
echo "Installed OpenPI. Activate with: source \"$openpi/.venv/bin/activate\""
