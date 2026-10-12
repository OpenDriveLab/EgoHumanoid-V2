"""Simulate human SMPL or aligned motion tokens and render the measured motion."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data.dataset import read_reference
from .render import render_capture
from .simulation.rollout import encode_poses, replay_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mode', choices=('smpl', 'token'), required=True)
    parser.add_argument('--sonic-root', type=Path, default=Path('sonic'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = read_reference(args.input)
    scene = args.sonic_root / 'models/g1_revo2/scene.xml'
    options = dict(scene=scene, hands=reference['hands'], fps=50)
    if args.mode == 'smpl':
        capture = encode_poses(args.sonic_root, reference['joints'], reference['poses'],
                               reference['quaternions'], args.output, **options)
    else:
        table = pq.read_table(args.input, columns=['action.motion_token'])
        tokens = np.asarray(table['action.motion_token'].to_pylist(), dtype=np.float32)
        capture = replay_tokens(args.sonic_root, tokens, args.output, **options)
    render_capture(scene, capture, args.output / 'replay.mp4', fps=50)


if __name__ == '__main__':
    main()
