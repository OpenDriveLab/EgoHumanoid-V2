"""Render captured simulation configurations without running the controller again."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def render_capture(scene, capture, output, *, fps=50, width=960, height=720):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if 'simulation_qpos' not in capture:
        raise ValueError('capture must contain exact simulation_qpos measurements')
    qpos = np.asarray(capture['simulation_qpos'], dtype=float)
    if (qpos.ndim != 2 or not len(qpos) or not np.isfinite(qpos).all()
            or not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0
            or width % 2 or height % 2):
        raise ValueError('invalid configurations, frame rate or image dimensions')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
    if qpos.shape[1] != model.nq:
        raise ValueError('simulation_qpos does not match the supplied scene')
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [qpos[0, 0], qpos[0, 1], .7]
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -15
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for configuration in qpos:
                data.qpos[:] = configuration
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                process.stdin.write(renderer.render().tobytes())
            process.stdin.close()
            error = process.stderr.read().decode(errors='replace')
            if process.wait():
                raise RuntimeError(f'video encoding failed: {error}')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            process.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sonic-root', type=Path, default=Path('sonic'))
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    table = pq.read_table(args.input, columns=['debug.runtime_simulation_qpos'])
    metadata = json.loads((table.schema.metadata or {}).get(b'eh2.reference', b'{}'))
    if metadata.get('version') != 1 or 'fps' not in metadata:
        raise ValueError('input must be an exported alignment episode with a recorded frame rate')
    capture = {'simulation_qpos': np.asarray(table['debug.runtime_simulation_qpos'].to_pylist())}
    scene = args.scene or args.sonic_root / 'models/g1_revo2/scene.xml'
    render_capture(scene, capture, args.output, fps=metadata['fps'])


if __name__ == '__main__':
    main()
