"""Join controller measurements and simulator odometry by their original tick."""

from collections import OrderedDict

import numpy as np


def _vector(value, width, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {width} finite values")
    return array.copy()


class FrameMatcher:
    """Keep a bounded odometry window and consume each debug index once."""

    def __init__(self):
        self.odometry = OrderedDict()
        self.last_index = -1

    def add_odometry(self, message):
        tick = int(message['tick'])
        position = _vector(message['position'], 3, 'position')
        orientation = _vector(message['orientation'], 4, 'orientation')
        value = dict(odometry_world=np.concatenate((position, orientation)))
        if 'simulation_qpos' in message:
            qpos = np.asarray(message['simulation_qpos'], dtype=float)
            if qpos.ndim != 1 or len(qpos) < 7 or not np.isfinite(qpos).all():
                raise ValueError('simulation_qpos must be a finite free-base configuration')
            value['simulation_qpos'] = qpos.copy()
        self.odometry[tick] = value
        while len(self.odometry) > 4096:
            self.odometry.popitem(last=False)

    def capture(self, state, expected_token=None):
        if 'state_tick' not in state:
            raise ValueError('debug state is missing state_tick')
        result = {
            name: _vector(state[name], width, name)
            for name, width in (
                ('body_q', 29), ('body_dq', 29), ('last_action', 29),
                ('base_quat', 4), ('base_ang_vel', 3), ('token_state', 64)
            )
        }
        index, tick = int(state['index']), int(state['state_tick'])
        if index <= self.last_index or tick not in self.odometry:
            return None
        if expected_token is not None:
            expected = _vector(expected_token, 64, 'expected_token').astype(np.float32)
            if not np.array_equal(result['token_state'], expected):
                return None
        result.update(index=index, state_tick=tick,
                      **{name: value.copy() for name, value in self.odometry[tick].items()})
        self.last_index = index
        return result
