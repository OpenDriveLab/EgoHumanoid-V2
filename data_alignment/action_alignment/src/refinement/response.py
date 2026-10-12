"""Pretrained endpoint-history MLP inference for motion refinement."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def measured_pair_delta(current, reference) -> np.ndarray:
    """Return corresponding-frame xyz/rotation-vector changes from wxyz poses."""
    current, reference = np.asarray(current, dtype=float), np.asarray(reference, dtype=float)
    if (current.ndim != 3 or current.shape[1:] != (4, 7) or len(current) == 0
            or reference.shape != current.shape
            or not np.isfinite(current).all() or not np.isfinite(reference).all()):
        raise ValueError('expected finite aligned (N, 4, 7) endpoint poses')
    rotations = []
    for poses in (current, reference):
        quat = poses[..., 3:].reshape(-1, 4)
        rotations.append(Rotation.from_quat(quat[:, [1, 2, 3, 0]]))
    angular = (rotations[0] * rotations[1].inv()).as_rotvec().reshape(-1, 4, 3)
    return np.concatenate((current[..., :3] - reference[..., :3], angular), axis=2).reshape(-1, 24)


def causal_history(prediction):
    n = len(prediction)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = torch.fft.rfft(prediction, n=size, dim=0)
    lag = torch.arange(n, device=prediction.device, dtype=prediction.dtype)
    history = [prediction]
    for tau in (.02, .1, .5, 2., 5.):
        alpha = np.exp(-.02 / tau)
        kernel = (1-alpha) * torch.exp(-.02 * lag / tau)
        history.append(torch.fft.irfft(
            spectrum * torch.fft.rfft(kernel, n=size)[:, None], n=size, dim=0
        )[:n])
    return torch.cat(history, dim=1)


class EndpointResponse:
    """Predict four-endpoint response from candidate history and fixed context.

    Endpoint order is left palm, right palm, left sole, right sole, with xyz
    and rotation-vector components. Candidate histories retain gradients.
    """

    def __init__(self, path, observations, *, device, fps):
        if fps != 50:
            raise ValueError('endpoint response requires 50 Hz observations')
        saved = torch.load(path, map_location=device, weights_only=True)
        state = saved['state_dict']
        self.baseline = state['baseline'].detach()
        self.xs, self.ys = saved['x_scale'], saved['y_scale']
        if self.baseline.shape != (144, 24) or self.xs.shape != (144,) or self.ys.shape != (24,):
            raise ValueError('expected 144-input, 24-output endpoint response')
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        if obs.ndim != 2 or obs.shape[1] != 994 or not len(obs):
            raise ValueError('expected nonempty 994D baseline observations')
        token, history = obs[:, :64], obs[:, 64:]
        raw = torch.cat([token, history[:, 27:30], history[:, 291:320],
                         history[:, 581:610], history[:, 871:900], history[:, -3:]], dim=1)
        if torch.any(self.xs <= 0) or torch.any(self.ys <= 0) or torch.any(saved['std'] <= 0):
            raise ValueError('response normalization must be positive')
        self.features = ((raw-saved['mean']) / saved['std']).detach()
        self.layers = [(state[f'net.{i}.weight'].detach(), state[f'net.{i}.bias'].detach())
                       for i in (0, 2, 4)]
        for value in (self.baseline, self.xs, self.ys, self.features,
                      *(v for pair in self.layers for v in pair)):
            if not torch.isfinite(value).all():
                raise ValueError('response weights and observations must be finite')

    def net(self, value):
        for i, (weight, bias) in enumerate(self.layers):
            value = torch.nn.functional.linear(value, weight, bias)
            if i < 2:
                value = torch.nn.functional.silu(value)
        return value

    def __call__(self, prediction):
        if prediction.shape != (len(self.features), 24):
            raise ValueError('candidate and baseline frames must align')
        x = causal_history(prediction) / self.xs
        residual = self.net(torch.cat([self.features, x], dim=1)) - self.net(
            torch.cat([self.features, torch.zeros_like(x)], dim=1))
        return (x @ self.baseline + residual) * self.ys


class ResponseTask:
    """Apply one-step response delay before weighted endpoint projection."""

    def __init__(self, response, pose_jacobians, groups, *, blend=1., rotation_blend=None):
        rotation_blend = blend if rotation_blend is None else rotation_blend
        if not 0 <= blend <= 1 or not 0 <= rotation_blend <= 1:
            raise ValueError('rotation response blend must be in [0, 1]')
        self.response = response
        device = response.baseline.device
        self.jacobian = torch.as_tensor(pose_jacobians, dtype=torch.float32,
                                        device=device).reshape(-1, 24, 29)
        self.groups = [(torch.as_tensor(p, dtype=torch.float32, device=device),
                        torch.as_tensor(m, dtype=torch.float32, device=device))
                       for p, m in groups]
        self.blend = torch.full((24,), float(blend), device=device)
        for offset in range(0, 24, 6):
            self.blend[offset+3:offset+6] = rotation_blend

    def __call__(self, joint_delta):
        outputs = []
        for projection, mask in self.groups:
            physical = torch.einsum('nej,nj->ne', self.jacobian, joint_delta * mask)
            dynamic = self.response(physical)
            aligned = torch.cat([torch.zeros_like(dynamic[:1]), dynamic[:-1]], dim=0)
            predicted = physical + self.blend * (aligned-physical)
            outputs.append(torch.einsum('nre,ne->nr', projection, predicted))
        return torch.cat(outputs, dim=1)
