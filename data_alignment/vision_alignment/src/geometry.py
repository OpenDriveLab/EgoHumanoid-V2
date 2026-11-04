"""G1 forward kinematics, virtual-camera projection, and arm mesh rendering.

prepare.py uses projection and alignment helpers to construct image-space priors.
render.py uses the mesh renderer and prepared per-arm transforms. Visual anchor
detection and mask fitting live in tracking.py, image_geometry.py, and axes.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_ARM_BODY_NAMES = (
    ("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"),
    ("right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link"),
)
_BRAINCO_BASE_OFFSET_M = np.asarray([0.0591, 0.0, 0.0], dtype=np.float64)
DEFAULT_G1_ROOT_BODYWARD_EXTENSION = 1.25
DEFAULT_G1_ROOT_BOUNDARY_DISTANCE_PX = 300.0


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera pose expressed in its rigid parent coordinate frame."""

    position: np.ndarray
    rotation: np.ndarray
    vertical_fov_degrees: float
    width: int
    height: int

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        fov = float(self.vertical_fov_degrees)
        width = int(self.width)
        height = int(self.height)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("camera position must be a finite 3-vector")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("camera rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-05):
            raise ValueError("camera rotation must be orthonormal")
        if np.linalg.det(rotation) <= 0.0:
            raise ValueError("camera rotation must be right-handed")
        if not 1.0 < fov < 179.0:
            raise ValueError("vertical camera FOV must lie between 1 and 179 degrees")
        if width <= 0 or height <= 0:
            raise ValueError("camera width and height must be positive")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "vertical_fov_degrees", fov)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)


@dataclass(frozen=True)
class G1ArmAppearance:
    """Camera-space tone calibration for the isolated G1 arm render."""

    camera_rgb_gain: tuple[float, float, float] = (1.35, 1.36, 1.4)
    highlight_knee: float = 185.0
    camera_gamma: float = 1.0
    material_profile: str = "g1_physical"
    headlight_ambient: float = 0.4
    headlight_diffuse: float = 0.35
    headlight_specular: float = 0.6
    silver_pbr_enabled: bool = False
    silver_pbr_specular: float = 0.8
    silver_pbr_shininess: float = 0.35
    silver_pbr_metallic: float = 0.65
    silver_pbr_roughness: float = 0.3
    silver_pbr_reflectance: float = 0.0
    silver_environment_enabled: bool = False
    silver_environment_rgb_top: tuple[float, float, float] = (0.65, 0.52, 0.4)
    silver_environment_rgb_bottom: tuple[float, float, float] = (0.06, 0.08, 0.12)
    wrist_pbr_specular: float = 0.8
    wrist_pbr_shininess: float = 0.35
    wrist_pbr_metallic: float = 0.65
    wrist_pbr_roughness: float = 0.3
    wrist_pbr_reflectance: float = 0.0
    capture_silver_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    capture_wrist_yaw_scale: float = 1.0
    capture_wrist_yaw_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    capture_dark_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        gain = np.asarray(self.camera_rgb_gain, dtype=np.float64)
        if gain.shape != (3,) or not np.isfinite(gain).all() or np.any(gain <= 0.0):
            raise ValueError("camera_rgb_gain must contain three finite positive values")
        knee = float(self.highlight_knee)
        if not np.isfinite(knee) or not 0.0 < knee < 255.0:
            raise ValueError("highlight_knee must be finite and lie between 0 and 255")
        gamma = float(self.camera_gamma)
        if not np.isfinite(gamma) or not 0.25 <= gamma <= 4.0:
            raise ValueError("camera_gamma must be finite and lie in [0.25, 4]")
        profile = str(self.material_profile)
        if profile not in {
            "g1_physical",
            "g1_brainco_capture",
            "g1_capture",
            "g1_arm_only",
            "uniform_gray",
        }:
            raise ValueError(
                "material_profile must be 'g1_physical', 'g1_brainco_capture', 'g1_capture', 'g1_arm_only', or 'uniform_gray'"
            )
        lighting = np.asarray(
            [self.headlight_ambient, self.headlight_diffuse, self.headlight_specular],
            dtype=np.float64,
        )
        if not np.isfinite(lighting).all() or np.any(lighting < 0.0) or np.any(lighting > 1.0):
            raise ValueError("headlight terms must be finite values in [0, 1]")
        pbr = np.asarray(
            [
                self.silver_pbr_specular,
                self.silver_pbr_shininess,
                self.silver_pbr_metallic,
                self.silver_pbr_roughness,
                self.silver_pbr_reflectance,
                self.wrist_pbr_specular,
                self.wrist_pbr_shininess,
                self.wrist_pbr_metallic,
                self.wrist_pbr_roughness,
                self.wrist_pbr_reflectance,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(pbr).all() or np.any(pbr < 0.0) or np.any(pbr > 1.0):
            raise ValueError("silver PBR terms must be finite values in [0, 1]")
        environment_rgb_top = np.asarray(self.silver_environment_rgb_top, dtype=np.float64)
        environment_rgb_bottom = np.asarray(self.silver_environment_rgb_bottom, dtype=np.float64)
        for name, color in (
            ("silver_environment_rgb_top", environment_rgb_top),
            ("silver_environment_rgb_bottom", environment_rgb_bottom),
        ):
            if (
                color.shape != (3,)
                or not np.isfinite(color).all()
                or np.any(color < 0.0)
                or np.any(color > 1.0)
            ):
                raise ValueError(f"{name} must contain three finite values in [0, 1]")
        wrist_scale = float(self.capture_wrist_yaw_scale)
        if not np.isfinite(wrist_scale) or not 0.0 < wrist_scale <= 1.5:
            raise ValueError("capture_wrist_yaw_scale must be finite and lie in (0, 1.5]")
        silver_rgb_scale = np.asarray(self.capture_silver_rgb_scale, dtype=np.float64)
        if (
            silver_rgb_scale.shape != (3,)
            or not np.isfinite(silver_rgb_scale).all()
            or np.any(silver_rgb_scale <= 0.0)
            or np.any(silver_rgb_scale > 1.25)
        ):
            raise ValueError(
                "capture_silver_rgb_scale must contain three finite values in (0, 1.25]"
            )
        wrist_rgb_scale = np.asarray(self.capture_wrist_yaw_rgb_scale, dtype=np.float64)
        if (
            wrist_rgb_scale.shape != (3,)
            or not np.isfinite(wrist_rgb_scale).all()
            or np.any(wrist_rgb_scale <= 0.0)
            or np.any(wrist_rgb_scale > 2.0)
        ):
            raise ValueError(
                "capture_wrist_yaw_rgb_scale must contain three finite values in (0, 2]"
            )
        dark_rgb_scale = np.asarray(self.capture_dark_rgb_scale, dtype=np.float64)
        if (
            dark_rgb_scale.shape != (3,)
            or not np.isfinite(dark_rgb_scale).all()
            or np.any(dark_rgb_scale <= 0.0)
            or np.any(dark_rgb_scale > 4.0)
        ):
            raise ValueError("capture_dark_rgb_scale must contain three finite values in (0, 4]")
        object.__setattr__(self, "camera_rgb_gain", tuple(float(x) for x in gain))
        object.__setattr__(self, "highlight_knee", knee)
        object.__setattr__(self, "camera_gamma", gamma)
        object.__setattr__(self, "material_profile", profile)
        object.__setattr__(self, "headlight_ambient", float(lighting[0]))
        object.__setattr__(self, "headlight_diffuse", float(lighting[1]))
        object.__setattr__(self, "headlight_specular", float(lighting[2]))
        object.__setattr__(self, "silver_pbr_enabled", bool(self.silver_pbr_enabled))
        object.__setattr__(self, "silver_pbr_specular", float(pbr[0]))
        object.__setattr__(self, "silver_pbr_shininess", float(pbr[1]))
        object.__setattr__(self, "silver_pbr_metallic", float(pbr[2]))
        object.__setattr__(self, "silver_pbr_roughness", float(pbr[3]))
        object.__setattr__(self, "silver_pbr_reflectance", float(pbr[4]))
        object.__setattr__(
            self, "silver_environment_enabled", bool(self.silver_environment_enabled)
        )
        object.__setattr__(
            self,
            "silver_environment_rgb_top",
            tuple(float(x) for x in environment_rgb_top),
        )
        object.__setattr__(
            self,
            "silver_environment_rgb_bottom",
            tuple(float(x) for x in environment_rgb_bottom),
        )
        object.__setattr__(self, "wrist_pbr_specular", float(pbr[5]))
        object.__setattr__(self, "wrist_pbr_shininess", float(pbr[6]))
        object.__setattr__(self, "wrist_pbr_metallic", float(pbr[7]))
        object.__setattr__(self, "wrist_pbr_roughness", float(pbr[8]))
        object.__setattr__(self, "wrist_pbr_reflectance", float(pbr[9]))
        object.__setattr__(
            self, "capture_silver_rgb_scale", tuple(float(x) for x in silver_rgb_scale)
        )
        object.__setattr__(self, "capture_wrist_yaw_scale", wrist_scale)
        object.__setattr__(
            self,
            "capture_wrist_yaw_rgb_scale",
            tuple(float(x) for x in wrist_rgb_scale),
        )
        object.__setattr__(self, "capture_dark_rgb_scale", tuple(float(x) for x in dark_rgb_scale))


@dataclass(frozen=True)
class G1ArmPbrMaterialNames:
    """Names of the two compiled silver materials used by the arm renderer."""

    silver: str
    wrist_yaw: str


@dataclass(frozen=True)
class G1CaptureSleeve:
    """Procedural cover fitted over the proximal G1 arm in robot captures."""

    core_radius_m: float = 0.032
    rib_radius_m: float = 0.033
    start_fraction: float = 0.08
    end_fraction: float = 1.08
    rib_count: int = 7
    rib_half_width_fraction: float = 0.025
    rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    material_profile: str = "dark"

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.core_radius_m,
                self.rib_radius_m,
                self.start_fraction,
                self.end_fraction,
                self.rib_half_width_fraction,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("capture sleeve parameters must be finite")
        if self.core_radius_m <= 0.0:
            raise ValueError("core_radius_m must be positive")
        if self.rib_radius_m < self.core_radius_m:
            raise ValueError("rib_radius_m cannot be smaller than core_radius_m")
        if self.start_fraction < 0.0 or self.end_fraction <= self.start_fraction:
            raise ValueError("capture sleeve start/end fractions are invalid")
        if int(self.rib_count) != self.rib_count or self.rib_count < 0:
            raise ValueError("rib_count must be a non-negative integer")
        if self.rib_half_width_fraction <= 0.0:
            raise ValueError("rib_half_width_fraction must be positive")
        rgb_scale = np.asarray(self.rgb_scale, dtype=np.float64)
        if (
            rgb_scale.shape != (3,)
            or not np.isfinite(rgb_scale).all()
            or np.any(rgb_scale <= 0.0)
            or np.any(rgb_scale > 4.0)
        ):
            raise ValueError("rgb_scale must contain three finite values in (0, 4]")
        object.__setattr__(self, "rib_count", int(self.rib_count))
        object.__setattr__(self, "rgb_scale", tuple(float(x) for x in rgb_scale))
        if self.material_profile not in {"dark", "silver"}:
            raise ValueError("material_profile must be 'dark' or 'silver'")


def add_g1_capture_sleeve_geoms(
    spec: mujoco.MjSpec, sleeve: G1CaptureSleeve
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Add a joint-locked core and shallow ribs along each proximal arm."""
    names_by_side: list[tuple[str, ...]] = []
    material_rgba = (
        _G1_SILVER_RGBA if sleeve.material_profile == "silver" else _G1_CAPTURE_SLEEVE_RGBA
    ).astype(np.float64)
    material_rgba[:3] *= np.asarray(sleeve.rgb_scale, dtype=np.float64)
    for side in ("left", "right"):
        parent = spec.body(f"{side}_shoulder_yaw_link")
        elbow = spec.body(f"{side}_elbow_link")
        if parent is None or elbow is None:
            raise ValueError(f"G1 model is missing the {side} elbow chain")
        endpoint = np.asarray(elbow.pos, dtype=np.float64)
        length = float(np.linalg.norm(endpoint))
        if not np.isfinite(length) or length <= 1e-06:
            raise ValueError(f"invalid {side} elbow offset")
        unit = endpoint / length
        start = sleeve.start_fraction * endpoint
        end = sleeve.end_fraction * endpoint
        names: list[str] = []
        core_name = f"{side}_capture_sleeve_core"
        parent.add_geom(
            name=core_name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=np.concatenate([start, end]),
            size=[sleeve.core_radius_m, 0.0, 0.0],
            rgba=material_rgba,
            group=1,
            contype=0,
            conaffinity=0,
            density=0.0,
        )
        names.append(core_name)
        centers = np.linspace(sleeve.start_fraction, sleeve.end_fraction, sleeve.rib_count)
        half_width = sleeve.rib_half_width_fraction * length
        for index, fraction in enumerate(centers):
            center = fraction * endpoint
            rib_name = f"{side}_capture_sleeve_rib_{index:02d}"
            parent.add_geom(
                name=rib_name,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=np.concatenate([center - half_width * unit, center + half_width * unit]),
                size=[sleeve.rib_radius_m, 0.0, 0.0],
                rgba=material_rgba,
                group=1,
                contype=0,
                conaffinity=0,
                density=0.0,
            )
            names.append(rib_name)
        names_by_side.append(tuple(names))
    return (names_by_side[0], names_by_side[1])


def tone_map_premultiplied_rgb(
    rgb: np.ndarray, appearance: G1ArmAppearance, *, alpha: np.ndarray | None = None
) -> np.ndarray:
    """Apply a deterministic capture response to a black-backed render."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 with shape (H, W, 3)")
    gain = np.asarray(appearance.camera_rgb_gain, dtype=np.float32)
    gamma = float(appearance.camera_gamma)
    coverage = None
    if alpha is not None:
        coverage = np.asarray(alpha, dtype=np.float32)
        if coverage.shape != image.shape[:2] or not np.isfinite(coverage).all():
            raise ValueError("alpha must be finite and match the RGB raster")
        coverage = np.clip(coverage, 0.0, 1.0)
    use_lookup = (
        np.array_equal(image[..., 0], image[..., 1])
        and np.array_equal(image[..., 0], image[..., 2])
        and (not (coverage is not None and gamma != 1.0))
    )
    if use_lookup:
        levels = np.arange(256, dtype=np.float32)[:, None] * gain[None]
        peak = np.max(levels, axis=1)
        above_knee = peak > appearance.highlight_knee
        headroom = 255.0 - appearance.highlight_knee
        rolled_peak = appearance.highlight_knee + headroom * (
            1.0 - np.exp(-(peak[above_knee] - appearance.highlight_knee) / headroom)
        )
        levels[above_knee] *= (rolled_peak / peak[above_knee])[:, None]
        if gamma != 1.0:
            levels = 255.0 * np.power(np.clip(levels, 0.0, 255.0) / 255.0, gamma)
        lookup = np.clip(np.rint(levels), 0, 255).astype(np.uint8)
        gray = image[..., 0]
        return cv2.merge([cv2.LUT(gray, lookup[:, channel]) for channel in range(3)])
    support = np.any(image != 0, axis=2)
    output = np.zeros_like(image)
    if not np.any(support):
        return output
    source = image[support].astype(np.float32)
    if coverage is not None and gamma != 1.0:
        source /= np.maximum(coverage[support][:, None], 1e-06)
        np.clip(source, 0.0, 255.0, out=source)
    scaled = source * gain[None]
    peak = np.max(scaled, axis=1)
    above_knee = peak > appearance.highlight_knee
    if np.any(above_knee):
        headroom = 255.0 - appearance.highlight_knee
        rolled_peak = appearance.highlight_knee + headroom * (
            1.0 - np.exp(-(peak[above_knee] - appearance.highlight_knee) / headroom)
        )
        scaled[above_knee] *= (rolled_peak / peak[above_knee])[:, None]
    if gamma != 1.0:
        scaled = 255.0 * np.power(np.clip(scaled, 0.0, 255.0) / 255.0, gamma)
        if coverage is not None:
            scaled *= coverage[support][:, None]
    output[support] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return output


@dataclass(frozen=True)
class ArmProjection:
    """Projected shoulder, elbow, wrist-yaw, and hand-base landmarks."""

    pixels: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True)
class AlignedArmGeometry:
    """Projected arms after a shared two-wrist similarity lock."""

    landmarks: np.ndarray
    root_points: np.ndarray
    transforms: np.ndarray


@dataclass(frozen=True)
class ExtendedRenderCanvas:
    """Centered offscreen canvas large enough for every post-render warp."""

    width: int
    height: int
    offset: np.ndarray


@dataclass(frozen=True)
class G1ArmMeshLayer:
    """Premultiplied G1 arm RGB and per-side coverage at video resolution."""

    rgb: np.ndarray
    alpha: np.ndarray
    geom_id: np.ndarray


_ARM_RENDER_MESH_SUFFIXES = (
    "shoulder_pitch_link",
    "shoulder_roll_link",
    "shoulder_yaw_link",
    "elbow_link",
    "wrist_roll_link",
    "wrist_pitch_link",
    "wrist_yaw_link",
    "base2_link",
)
_G1_SILVER_SHELL_SUFFIXES = frozenset(
    {
        "shoulder_pitch_link",
        "shoulder_roll_link",
        "shoulder_yaw_link",
        "elbow_link",
        "wrist_yaw_link",
    }
)
_G1_DARK_JOINT_SUFFIXES = frozenset({"wrist_roll_link", "wrist_pitch_link", "base2_link"})
_G1_BRAINCO_CAPTURE_SILVER_SHELL_SUFFIXES = _G1_SILVER_SHELL_SUFFIXES | {"wrist_pitch_link"}
_G1_BRAINCO_CAPTURE_DARK_JOINT_SUFFIXES = _G1_DARK_JOINT_SUFFIXES - {"wrist_pitch_link"}
_G1_CAPTURE_SILVER_SHELL_SUFFIXES = _G1_SILVER_SHELL_SUFFIXES - {"shoulder_yaw_link"}
_G1_CAPTURE_DARK_JOINT_SUFFIXES = _G1_DARK_JOINT_SUFFIXES | {"shoulder_yaw_link"}
_G1_ARM_ONLY_SILVER_SHELL_SUFFIXES = frozenset(
    {
        "shoulder_pitch_link",
        "shoulder_roll_link",
        "shoulder_yaw_link",
        "elbow_link",
        "wrist_pitch_link",
    }
)
_G1_ARM_ONLY_DARK_JOINT_SUFFIXES = frozenset({"wrist_roll_link", "wrist_yaw_link", "base2_link"})
_G1_SILVER_RGBA = np.asarray([0.62, 0.68, 0.78, 1.0], dtype=np.float32)
_G1_DARK_RGBA = np.asarray([0.1, 0.12, 0.16, 1.0], dtype=np.float32)
_G1_CAPTURE_SLEEVE_RGBA = np.asarray([0.035, 0.04, 0.05, 1.0], dtype=np.float32)
_G1_SILVER_PBR_MATERIAL_NAME = "visual_alignment_g1_silver"
_G1_WRIST_YAW_PBR_MATERIAL_NAME = "visual_alignment_g1_wrist_yaw_silver"
_G1_REFLECTION_ENVIRONMENT_NAME = "visual_alignment_g1_reflection_environment"


def g1_arm_material_suffixes(profile: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return the silver-shell and dark-joint mesh suffixes for a profile."""
    profile = str(profile)
    if profile == "g1_brainco_capture":
        return (
            _G1_BRAINCO_CAPTURE_SILVER_SHELL_SUFFIXES,
            _G1_BRAINCO_CAPTURE_DARK_JOINT_SUFFIXES,
        )
    if profile == "g1_capture":
        return (_G1_CAPTURE_SILVER_SHELL_SUFFIXES, _G1_CAPTURE_DARK_JOINT_SUFFIXES)
    if profile == "g1_arm_only":
        return (_G1_ARM_ONLY_SILVER_SHELL_SUFFIXES, _G1_ARM_ONLY_DARK_JOINT_SUFFIXES)
    if profile == "g1_physical":
        return (_G1_SILVER_SHELL_SUFFIXES, _G1_DARK_JOINT_SUFFIXES)
    if profile == "uniform_gray":
        return (frozenset(_ARM_RENDER_MESH_SUFFIXES), frozenset())
    raise ValueError(f"unsupported G1 arm material profile: {profile}")


def add_g1_arm_reflection_environment(
    spec: mujoco.MjSpec, appearance: G1ArmAppearance
) -> str | None:
    """Add a deterministic gradient skybox used only by metal reflections."""
    if not appearance.silver_environment_enabled:
        return None
    if spec.texture(_G1_REFLECTION_ENVIRONMENT_NAME) is not None:
        raise ValueError("G1 reflection environment already exists in source model")
    spec.add_texture(
        name=_G1_REFLECTION_ENVIRONMENT_NAME,
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=appearance.silver_environment_rgb_top,
        rgb2=appearance.silver_environment_rgb_bottom,
        width=256,
        height=1536,
    )
    return _G1_REFLECTION_ENVIRONMENT_NAME


def add_g1_arm_pbr_materials(
    spec: mujoco.MjSpec, appearance: G1ArmAppearance
) -> G1ArmPbrMaterialNames | None:
    """Add optional metal/roughness materials before compiling the G1 model."""
    if not appearance.silver_pbr_enabled:
        return None
    names = G1ArmPbrMaterialNames(
        silver=_G1_SILVER_PBR_MATERIAL_NAME, wrist_yaw=_G1_WRIST_YAW_PBR_MATERIAL_NAME
    )
    if spec.material(names.silver) is not None or spec.material(names.wrist_yaw) is not None:
        raise ValueError("G1 PBR material names already exist in the source model")
    common = {
        "specular": appearance.silver_pbr_specular,
        "shininess": appearance.silver_pbr_shininess,
        "reflectance": appearance.silver_pbr_reflectance,
        "metallic": appearance.silver_pbr_metallic,
        "roughness": appearance.silver_pbr_roughness,
    }
    silver_rgba = _G1_SILVER_RGBA.copy()
    silver_rgba[:3] *= np.asarray(appearance.capture_silver_rgb_scale, dtype=np.float32)
    spec.add_material(name=names.silver, rgba=silver_rgba, **common)
    wrist_rgba = silver_rgba.copy()
    wrist_rgba[:3] *= appearance.capture_wrist_yaw_scale * np.asarray(
        appearance.capture_wrist_yaw_rgb_scale, dtype=np.float32
    )
    spec.add_material(
        name=names.wrist_yaw,
        rgba=wrist_rgba,
        specular=appearance.wrist_pbr_specular,
        shininess=appearance.wrist_pbr_shininess,
        reflectance=appearance.wrist_pbr_reflectance,
        metallic=appearance.wrist_pbr_metallic,
        roughness=appearance.wrist_pbr_roughness,
    )
    return names


def assign_g1_arm_pbr_materials(
    model: mujoco.MjModel,
    geom_ids: tuple[np.ndarray, np.ndarray],
    *,
    profile: str,
    material_names: G1ArmPbrMaterialNames | None,
) -> None:
    """Assign compiled PBR materials to silver shells, never dark joints."""
    if material_names is None:
        return
    silver_suffixes, _ = g1_arm_material_suffixes(profile)
    main_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_names.silver)
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_names.wrist_yaw)
    if main_id < 0 or wrist_id < 0:
        raise ValueError("compiled G1 PBR materials are missing")
    for side_ids in geom_ids:
        for geom_id in np.asarray(side_ids, dtype=np.int64):
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name is None:
                raise ValueError(f"arm geom {geom_id} has no named visual mesh")
            suffix = str(mesh_name).removeprefix("left_").removeprefix("right_")
            if suffix == "wrist_yaw_link" and suffix in silver_suffixes:
                model.geom_matid[geom_id] = wrist_id
            elif suffix in silver_suffixes:
                model.geom_matid[geom_id] = main_id


def apply_g1_arm_material_profile(
    model: mujoco.MjModel,
    geom_ids: tuple[np.ndarray, np.ndarray],
    *,
    profile: str,
    capture_silver_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    capture_wrist_yaw_scale: float = 1.0,
    capture_wrist_yaw_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    capture_dark_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Restore the physical G1 shell/joint contrast lost by the gray MJCF."""
    profile = str(profile)
    if profile == "uniform_gray":
        return
    silver_suffixes, dark_suffixes = g1_arm_material_suffixes(profile)
    if len(geom_ids) != 2:
        raise ValueError("geom_ids must contain left and right arm selections")
    wrist_scale = float(capture_wrist_yaw_scale)
    if not np.isfinite(wrist_scale) or not 0.0 < wrist_scale <= 1.5:
        raise ValueError("capture_wrist_yaw_scale must be finite and lie in (0, 1.5]")
    silver_rgb_scale = np.asarray(capture_silver_rgb_scale, dtype=np.float64)
    if (
        silver_rgb_scale.shape != (3,)
        or not np.isfinite(silver_rgb_scale).all()
        or np.any(silver_rgb_scale <= 0.0)
        or np.any(silver_rgb_scale > 1.25)
    ):
        raise ValueError("capture_silver_rgb_scale must contain three finite values in (0, 1.25]")
    wrist_rgb_scale = np.asarray(capture_wrist_yaw_rgb_scale, dtype=np.float64)
    if (
        wrist_rgb_scale.shape != (3,)
        or not np.isfinite(wrist_rgb_scale).all()
        or np.any(wrist_rgb_scale <= 0.0)
        or np.any(wrist_rgb_scale > 2.0)
    ):
        raise ValueError("capture_wrist_yaw_rgb_scale must contain three finite values in (0, 2]")
    dark_rgb_scale = np.asarray(capture_dark_rgb_scale, dtype=np.float64)
    if (
        dark_rgb_scale.shape != (3,)
        or not np.isfinite(dark_rgb_scale).all()
        or np.any(dark_rgb_scale <= 0.0)
        or np.any(dark_rgb_scale > 4.0)
    ):
        raise ValueError("capture_dark_rgb_scale must contain three finite values in (0, 4]")
    dark_rgba = _G1_DARK_RGBA.copy()
    dark_rgba[:3] *= dark_rgb_scale
    silver_rgba = _G1_SILVER_RGBA.copy()
    silver_rgba[:3] *= silver_rgb_scale
    classified: set[str] = set()
    for side_ids in geom_ids:
        ids = np.asarray(side_ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= model.ngeom):
            raise ValueError("arm geom IDs must be a valid one-dimensional array")
        for geom_id in ids:
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name is None:
                raise ValueError(f"arm geom {geom_id} has no named visual mesh")
            suffix = str(mesh_name)
            for side in ("left_", "right_"):
                if suffix.startswith(side):
                    suffix = suffix[len(side) :]
                    break
            if suffix == "wrist_yaw_link" and suffix in silver_suffixes:
                wrist_rgba = silver_rgba.copy()
                wrist_rgba[:3] *= wrist_scale * wrist_rgb_scale
                model.geom_rgba[geom_id] = wrist_rgba
            elif suffix in silver_suffixes:
                model.geom_rgba[geom_id] = silver_rgba
            elif suffix in dark_suffixes:
                model.geom_rgba[geom_id] = dark_rgba
            else:
                raise ValueError(f"unclassified G1 arm mesh: {mesh_name}")
            classified.add(suffix)
    expected = silver_suffixes | dark_suffixes
    if classified != expected:
        raise ValueError(f"incomplete G1 arm material assignment: {sorted(expected - classified)}")


def apply_g1_arm_capture_lighting(model: mujoco.MjModel, appearance: G1ArmAppearance) -> None:
    """Set deterministic GoPro-domain light balance for isolated arm renders."""
    model.vis.headlight.ambient[:] = appearance.headlight_ambient
    model.vis.headlight.diffuse[:] = appearance.headlight_diffuse
    model.vis.headlight.specular[:] = appearance.headlight_specular


def select_g1_arm_mesh_geoms(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Select visual shoulder-through-adapter meshes, excluding every hand mesh."""
    selected: list[np.ndarray] = []
    for side in ("left", "right"):
        wanted = {f"{side}_{suffix}" for suffix in _ARM_RENDER_MESH_SUFFIXES}
        geom_ids: list[int] = []
        found: set[str] = set()
        for geom_id in range(model.ngeom):
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0 or int(model.geom_group[geom_id]) != 1:
                continue
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name in wanted:
                geom_ids.append(geom_id)
                found.add(mesh_name)
        missing = wanted - found
        if missing:
            raise ValueError(f"G1 model is missing {side} visual arm meshes: {sorted(missing)}")
        selected.append(np.asarray(geom_ids, dtype=np.int32))
    return (selected[0], selected[1])


def warp_extended_g1_arm_render(
    extended_rgb: np.ndarray,
    segmentation: np.ndarray,
    geom_ids: tuple[np.ndarray, np.ndarray],
    image_transforms: np.ndarray,
    *,
    canvas_offset: np.ndarray,
    output_shape: tuple[int, int],
    appearance: G1ArmAppearance,
) -> G1ArmMeshLayer:
    """Warp both arms directly from the extended render into the video frame."""
    image = np.asarray(extended_rgb)
    labels = np.asarray(segmentation)
    transforms = np.asarray(image_transforms, dtype=np.float64)
    offset = np.asarray(canvas_offset, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("extended_rgb must be uint8 with shape (H, W, 3)")
    if labels.shape != image.shape[:2] + (2,):
        raise ValueError("segmentation must have shape (H, W, 2)")
    if transforms.shape == (2, 3):
        transforms = np.repeat(transforms[None], 2, axis=0)
    if transforms.shape != (2, 2, 3) or not np.isfinite(transforms).all():
        raise ValueError("image_transforms must have shape (2, 3) or (2, 2, 3)")
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("canvas_offset must be a finite XY vector")
    if len(geom_ids) != 2:
        raise ValueError("geom_ids must contain left and right arm selections")
    height, width = (int(output_shape[0]), int(output_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("output_shape must be positive")
    output_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    output_alpha = np.zeros((2, height, width), dtype=np.float32)
    output_geom_id = np.full((height, width), -1, dtype=np.int32)
    visible_coverage = np.zeros((height, width), dtype=np.float32)
    object_is_geom = labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    for side, side_ids in enumerate(geom_ids):
        ids = np.asarray(side_ids, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError("each arm geom selection must be one-dimensional")
        extended_mask = object_is_geom & np.isin(labels[..., 0], ids)
        side_rgb = np.where(extended_mask[..., None], image, 0)
        adjusted = transforms[side].copy()
        adjusted[:, 2] -= adjusted[:, :2] @ offset
        affine = adjusted.astype(np.float32)
        warped_rgb = cv2.warpAffine(
            side_rgb,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_alpha = cv2.warpAffine(
            extended_mask.astype(np.float32),
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        np.clip(warped_alpha, 0.0, 1.0, out=warped_alpha)
        extended_geom_id = np.where(extended_mask, labels[..., 0], -1).astype(np.float32)
        warped_geom_id = np.rint(
            cv2.warpAffine(
                extended_geom_id,
                affine,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=-1,
            )
        ).astype(np.int32)
        output_alpha[side] = warped_alpha
        take = warped_alpha > visible_coverage
        output_rgb[take] = warped_rgb[take]
        output_geom_id[take] = warped_geom_id[take]
        visible_coverage[take] = warped_alpha[take]
    return G1ArmMeshLayer(
        rgb=tone_map_premultiplied_rgb(output_rgb, appearance, alpha=visible_coverage),
        alpha=output_alpha,
        geom_id=output_geom_id,
    )


def compute_extended_render_canvas(
    similarity_transforms: np.ndarray,
    *,
    width: int,
    height: int,
    padding_px: int = 24,
    dimension_alignment: int = 32,
    maximum_dimension_px: int | None = None,
) -> ExtendedRenderCanvas:
    """Size a centered render target so affine alignment cannot crop an arm."""
    transforms = np.asarray(similarity_transforms, dtype=np.float64)
    if transforms.shape == (2, 3):
        transforms = transforms[None]
    if transforms.ndim != 3 or transforms.shape[1:] != (2, 3):
        raise ValueError("similarity_transforms must have shape (frames, 2, 3)")
    if not len(transforms) or not np.isfinite(transforms).all():
        raise ValueError("similarity transforms must be non-empty and finite")
    width = int(width)
    height = int(height)
    padding_px = int(padding_px)
    dimension_alignment = int(dimension_alignment)
    maximum_dimension = None if maximum_dimension_px is None else int(maximum_dimension_px)
    if width <= 0 or height <= 0:
        raise ValueError("render width and height must be positive")
    if padding_px < 0 or dimension_alignment < 1:
        raise ValueError("padding must be non-negative and alignment positive")
    if maximum_dimension is not None:
        aligned_maximum = maximum_dimension // dimension_alignment * dimension_alignment
        if aligned_maximum < max(width, height):
            raise ValueError("maximum canvas dimension must contain the nominal image")
    corners = np.asarray(
        [[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]], dtype=np.float64
    )
    source_corners = []
    for transform in transforms:
        linear = transform[:, :2]
        determinant = float(np.linalg.det(linear))
        if abs(determinant) < 1e-09:
            raise ValueError("similarity transform is singular")
        source_corners.append((corners - transform[:, 2]) @ np.linalg.inv(linear).T)
    source = np.concatenate(source_corners, axis=0)
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    radius = np.max(np.abs(source - center), axis=0) + padding_px

    def aligned_dimension(required: float, minimum: int) -> int:
        pixels = max(float(minimum), 2.0 * float(required))
        return int(np.ceil(pixels / dimension_alignment) * dimension_alignment)

    canvas_width = aligned_dimension(radius[0], width)
    canvas_height = aligned_dimension(radius[1], height)
    if maximum_dimension is not None:
        canvas_width = min(canvas_width, aligned_maximum)
        canvas_height = min(canvas_height, aligned_maximum)
    offset = np.asarray(
        [(canvas_width - width) / 2.0, (canvas_height - height) / 2.0], dtype=np.float64
    )
    return ExtendedRenderCanvas(width=canvas_width, height=canvas_height, offset=offset)


class G1ArmMeshRenderer:
    """Reusable MuJoCo renderer for calibrated shoulder-to-wrist arm meshes."""

    _CAMERA_NAME = "visual_alignment_camera"

    def __init__(
        self,
        model_path: str | Path,
        calibration: CameraCalibration,
        similarity_transforms: np.ndarray,
        *,
        appearance: G1ArmAppearance | None = None,
        capture_sleeve: G1CaptureSleeve | None = None,
        canvas_padding_px: int = 24,
        canvas_alignment: int = 32,
        maximum_canvas_dimension_px: int | None = None,
    ) -> None:
        source_path = Path(model_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        self.calibration = calibration
        self.appearance = appearance or G1ArmAppearance()
        self.canvas = compute_extended_render_canvas(
            similarity_transforms,
            width=calibration.width,
            height=calibration.height,
            padding_px=canvas_padding_px,
            dimension_alignment=canvas_alignment,
            maximum_dimension_px=maximum_canvas_dimension_px,
        )
        focal_px = calibration.height / (
            2.0 * np.tan(np.deg2rad(calibration.vertical_fov_degrees) / 2.0)
        )
        extended_fovy = float(np.rad2deg(2.0 * np.arctan(self.canvas.height / (2.0 * focal_px))))
        if extended_fovy >= 179.0:
            raise ValueError("required render canvas exceeds MuJoCo camera FOV")
        camera_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(camera_quaternion, calibration.rotation.reshape(-1))
        spec = mujoco.MjSpec.from_file(str(source_path))
        reflection_environment_name = add_g1_arm_reflection_environment(spec, self.appearance)
        pbr_material_names = add_g1_arm_pbr_materials(spec, self.appearance)
        torso = spec.body("torso_link")
        if torso is None:
            raise ValueError("G1 model is missing torso_link")
        torso.add_camera(
            name=self._CAMERA_NAME,
            pos=calibration.position,
            quat=camera_quaternion,
            fovy=extended_fovy,
        )
        sleeve_names = (
            add_g1_capture_sleeve_geoms(spec, capture_sleeve)
            if capture_sleeve is not None
            else ((), ())
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = self.canvas.width
        self.model.vis.global_.offheight = self.canvas.height
        mesh_geom_ids = select_g1_arm_mesh_geoms(self.model)
        apply_g1_arm_material_profile(
            self.model,
            mesh_geom_ids,
            profile=self.appearance.material_profile,
            capture_silver_rgb_scale=self.appearance.capture_silver_rgb_scale,
            capture_wrist_yaw_scale=self.appearance.capture_wrist_yaw_scale,
            capture_wrist_yaw_rgb_scale=self.appearance.capture_wrist_yaw_rgb_scale,
            capture_dark_rgb_scale=self.appearance.capture_dark_rgb_scale,
        )
        assign_g1_arm_pbr_materials(
            self.model,
            mesh_geom_ids,
            profile=self.appearance.material_profile,
            material_names=pbr_material_names,
        )
        sleeve_geom_ids = tuple(
            np.asarray(
                [
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    for name in side_names
                ],
                dtype=np.int32,
            )
            for side_names in sleeve_names
        )
        self.geom_ids = tuple(
            (
                np.concatenate([mesh_ids, extra_ids])
                for mesh_ids, extra_ids in zip(mesh_geom_ids, sleeve_geom_ids, strict=True)
            )
        )
        self.capture_sleeve = capture_sleeve
        self.capture_sleeve_geom_names = sleeve_names
        self.reflection_environment_name = reflection_environment_name
        apply_g1_arm_capture_lighting(self.model, self.appearance)
        selected = np.concatenate(self.geom_ids)
        self.model.geom_group[:] = 5
        self.model.geom_group[selected] = 1
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        self.scene_option.geomgroup[1] = 1
        self.renderer = mujoco.Renderer(
            self.model, height=self.canvas.height, width=self.canvas.width
        )
        self.extended_vertical_fov_degrees = extended_fovy
        self.mesh_names = tuple(
            tuple(
                str(
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_MESH,
                        int(self.model.geom_dataid[geom_id]),
                    )
                )
                if int(self.model.geom_dataid[geom_id]) >= 0
                else str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)))
                for geom_id in side_ids
            )
            for side_ids in self.geom_ids
        )

    def close(self) -> None:
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            renderer.close()
            self.renderer = None

    def exclude_mesh_suffixes(self, suffixes: Sequence[str]) -> tuple[str, ...]:
        """Remove matching arm geoms from the MuJoCo render scene."""
        normalized = tuple(str(value) for value in suffixes if str(value))
        if not normalized:
            return ()
        excluded_ids: list[int] = []
        excluded_names: list[str] = []
        for ids, names in zip(self.geom_ids, self.mesh_names, strict=True):
            for geom_id, name in zip(ids, names, strict=True):
                if str(name).endswith(normalized):
                    excluded_ids.append(int(geom_id))
                    excluded_names.append(str(name))
        if excluded_ids:
            self.model.geom_group[np.asarray(excluded_ids, dtype=np.int32)] = 5
        return tuple(excluded_names)

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def render(self, body_q: np.ndarray, image_transforms: np.ndarray) -> G1ArmMeshLayer:
        """Render one exact 29-DOF pose and apply final per-arm alignment."""
        configuration = np.asarray(body_q, dtype=np.float64)
        transforms = np.asarray(image_transforms, dtype=np.float64)
        if configuration.shape != (len(BODY_JOINT_NAMES),):
            raise ValueError("body_q must have shape (29,)")
        if transforms.shape not in {(2, 3), (2, 2, 3)}:
            raise ValueError("image_transforms must have shape (2, 3) or (2, 2, 3)")
        if not np.isfinite(configuration).all() or not np.isfinite(transforms).all():
            raise ValueError("pose and transform must be finite")
        if self.renderer is None:
            raise RuntimeError("G1 arm mesh renderer is closed")
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nq >= 7:
            self.data.qpos[:3] = 0.0
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for joint_name, value in zip(BODY_JOINT_NAMES, configuration, strict=True):
            _set_hinge_qpos(self.model, self.data, joint_name, float(value))
        mujoco.mj_forward(self.model, self.data)
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(
            self.data, camera=self._CAMERA_NAME, scene_option=self.scene_option
        )
        extended_rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(
            self.data, camera=self._CAMERA_NAME, scene_option=self.scene_option
        )
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        return warp_extended_g1_arm_render(
            extended_rgb,
            segmentation,
            self.geom_ids,
            transforms,
            canvas_offset=self.canvas.offset,
            output_shape=(self.calibration.height, self.calibration.width),
            appearance=self.appearance,
        )


def project_camera_points(
    points: np.ndarray, calibration: CameraCalibration
) -> tuple[np.ndarray, np.ndarray]:
    """Project parent-frame points using MuJoCo camera-axis conventions."""
    source = np.asarray(points, dtype=np.float64)
    if source.ndim < 2 or source.shape[-1] != 3:
        raise ValueError("points must end in a three-dimensional coordinate")
    if not np.isfinite(source).all():
        raise ValueError("points must be finite")
    camera = (source - calibration.position) @ calibration.rotation
    depth = -camera[..., 2]
    focal = calibration.height / (2.0 * np.tan(np.deg2rad(calibration.vertical_fov_degrees) / 2.0))
    safe_depth = np.where(np.abs(depth) < 1e-12, np.nan, depth)
    pixels = np.empty(source.shape[:-1] + (2,), dtype=np.float64)
    pixels[..., 0] = calibration.width / 2.0 + focal * camera[..., 0] / safe_depth
    pixels[..., 1] = calibration.height / 2.0 - focal * camera[..., 1] / safe_depth
    return (pixels, depth)


def estimate_similarity_2d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the unique orientation-preserving similarity for two point pairs."""
    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if source_points.shape != (2, 2) or target_points.shape != (2, 2):
        raise ValueError("source and target anchors must both have shape (2, 2)")
    if not np.isfinite(source_points).all() or not np.isfinite(target_points).all():
        raise ValueError("source and target anchors must be finite")
    source_vector = source_points[1] - source_points[0]
    target_vector = target_points[1] - target_points[0]
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length < 1e-09 or target_length < 1e-09:
        raise ValueError("source and target anchor points must be distinct")
    denominator = source_length * target_length
    cosine = float(np.dot(source_vector, target_vector) / denominator)
    sine = float(
        (source_vector[0] * target_vector[1] - source_vector[1] * target_vector[0]) / denominator
    )
    scale = target_length / source_length
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target_points[0] - linear @ source_points[0]
    return np.column_stack([linear, translation])


def apply_affine_2d(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to any array ending in XY coordinates."""
    source = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if source.shape[-1:] != (2,):
        raise ValueError("points must end in a two-dimensional coordinate")
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 2x3 matrix")
    return source @ matrix[:, :2].T + matrix[:, 2]


def compose_affine_2d(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Compose two 2-D affine transforms as ``outer(inner(points))``."""
    outer_matrix = np.asarray(outer, dtype=np.float64)
    inner_matrix = np.asarray(inner, dtype=np.float64)
    if outer_matrix.shape != (2, 3) or inner_matrix.shape != (2, 3):
        raise ValueError("outer and inner transforms must both have shape (2, 3)")
    if not np.isfinite(outer_matrix).all() or not np.isfinite(inner_matrix).all():
        raise ValueError("affine transforms must be finite")
    linear = outer_matrix[:, :2] @ inner_matrix[:, :2]
    translation = outer_matrix[:, :2] @ inner_matrix[:, 2] + outer_matrix[:, 2]
    return np.column_stack([linear, translation])


def compute_g1_arm_root_transforms(
    aligned_landmarks: np.ndarray,
    wrist_points: np.ndarray,
    root_points: np.ndarray,
    *,
    bodyward_extension: float = DEFAULT_G1_ROOT_BODYWARD_EXTENSION,
    max_boundary_distance_px: float = DEFAULT_G1_ROOT_BOUNDARY_DISTANCE_PX,
) -> np.ndarray:
    """Return per-arm similarities from projected base/elbow to cuff/root."""
    landmarks = np.asarray(aligned_landmarks, dtype=np.float64)
    wrists = np.asarray(wrist_points, dtype=np.float64)
    roots = np.asarray(root_points, dtype=np.float64)
    extension = float(bodyward_extension)
    maximum_distance = float(max_boundary_distance_px)
    if landmarks.ndim != 3 or landmarks.shape[0] != 2 or landmarks.shape[1] < 4:
        raise ValueError("aligned_landmarks must have shape (2, >=4, 2)")
    if landmarks.shape[2] != 2 or wrists.shape != (2, 2) or roots.shape != (2, 2):
        raise ValueError("arm landmarks, wrists, and roots must contain XY points")
    if not (
        np.isfinite(landmarks).all() and np.isfinite(wrists).all() and np.isfinite(roots).all()
    ):
        raise ValueError("arm alignment anchors must be finite")
    if not np.isfinite(extension) or extension <= 0.0:
        raise ValueError("bodyward_extension must be finite and positive")
    if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise ValueError("max_boundary_distance_px must be finite and positive")
    transforms = np.empty((2, 2, 3), dtype=np.float64)
    for side in range(2):
        source_anchors = np.stack([landmarks[side, 3], landmarks[side, 1]])
        boundary_vector = roots[side] - wrists[side]
        boundary_distance = float(np.linalg.norm(boundary_vector))
        if boundary_distance < 1e-09:
            raise ValueError("wrist and root points must be distinct")
        target_distance = extension * min(boundary_distance, maximum_distance)
        extended_root = wrists[side] + boundary_vector / boundary_distance * target_distance
        target_anchors = np.stack([wrists[side], extended_root])
        transforms[side] = estimate_similarity_2d(source_anchors, target_anchors)
    return transforms


def compose_g1_arm_render_transforms(
    shared_transform: np.ndarray, root_transforms: np.ndarray
) -> np.ndarray:
    """Compose the shared camera lock with each arm's root similarity."""
    shared = np.asarray(shared_transform, dtype=np.float64)
    roots = np.asarray(root_transforms, dtype=np.float64)
    if shared.shape != (2, 3) or roots.shape != (2, 2, 3):
        raise ValueError("shared_transform must be (2, 3) and root_transforms (2, 2, 3)")
    return np.stack([compose_affine_2d(roots[side], shared) for side in range(2)])


def ray_to_image_boundary(
    origin: np.ndarray,
    toward: np.ndarray,
    *,
    width: int,
    height: int,
    fallback_to_reverse: bool = False,
) -> np.ndarray:
    """Extend ``origin -> toward`` to the first image boundary it reaches."""
    start = np.asarray(origin, dtype=np.float64)
    next_point = np.asarray(toward, dtype=np.float64)
    if start.shape != (2,) or next_point.shape != (2,):
        raise ValueError("origin and toward must both be two-dimensional points")
    if not np.isfinite(start).all() or not np.isfinite(next_point).all():
        raise ValueError("origin and toward must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    direction = next_point - start
    if float(np.linalg.norm(direction)) < 1e-09:
        raise ValueError("origin and toward must be distinct")
    candidates: list[float] = []
    boundaries = ((0.0, width - 1.0), (0.0, height - 1.0))
    for axis, limits in enumerate(boundaries):
        if abs(float(direction[axis])) < 1e-12:
            continue
        for boundary in limits:
            distance = (boundary - start[axis]) / direction[axis]
            if distance <= 1e-09:
                continue
            hit = start + distance * direction
            if -1e-07 <= hit[0] <= width - 1.0 + 1e-07 and -1e-07 <= hit[1] <= height - 1.0 + 1e-07:
                candidates.append(float(distance))
    if not candidates and fallback_to_reverse:
        direction = -direction
        for axis, limits in enumerate(boundaries):
            if abs(float(direction[axis])) < 1e-12:
                continue
            for boundary in limits:
                distance = (boundary - start[axis]) / direction[axis]
                if distance <= 1e-09:
                    continue
                hit = start + distance * direction
                if (
                    -1e-07 <= hit[0] <= width - 1.0 + 1e-07
                    and -1e-07 <= hit[1] <= height - 1.0 + 1e-07
                ):
                    candidates.append(float(distance))
    if not candidates:
        raise ValueError("ray does not intersect the image in either direction")
    hit = start + min(candidates) * direction
    return np.clip(hit, [0.0, 0.0], [width - 1.0, height - 1.0])


def nearest_image_boundary_point(point: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Project an unusable off-image anchor to its nearest image boundary."""
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("point must be a finite two-dimensional value")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    clipped = np.clip(value, [0.0, 0.0], [width - 1.0, height - 1.0])
    distances = np.asarray(
        [clipped[0], width - 1.0 - clipped[0], clipped[1], height - 1.0 - clipped[1]]
    )
    edge = int(np.argmin(distances))
    if edge == 0:
        clipped[0] = 0.0
    elif edge == 1:
        clipped[0] = width - 1.0
    elif edge == 2:
        clipped[1] = 0.0
    else:
        clipped[1] = height - 1.0
    return clipped


def _valid_savgol_window(frames: int, requested: int, order: int) -> int | None:
    if requested < 3 or requested % 2 == 0:
        raise ValueError("window_length must be an odd integer of at least 3")
    if order < 0:
        raise ValueError("polynomial_order must be non-negative")
    window = min(requested, frames if frames % 2 else frames - 1)
    if window <= order or window < 3:
        return None
    return window


def robust_smooth_keypoints(
    keypoints: np.ndarray,
    *,
    window_length: int = 9,
    polynomial_order: int = 2,
    outlier_sigma: float = 4.0,
    minimum_outlier_distance: float = 1.0,
) -> np.ndarray:
    """Hampel-repair isolated jumps, then apply zero-phase Savitzky-Golay."""
    values = np.asarray(keypoints, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 2 or values.shape[0] < 1:
        raise ValueError("keypoints must have shape (frames, ..., 2)")
    if outlier_sigma <= 0.0 or minimum_outlier_distance < 0.0:
        raise ValueError("outlier thresholds must be positive")
    window = _valid_savgol_window(values.shape[0], int(window_length), int(polynomial_order))
    if window is None:
        if not np.isfinite(values).all():
            raise ValueError("too few frames to repair non-finite keypoints")
        return values.copy()
    flattened = values.reshape(values.shape[0], -1).copy()
    index = np.arange(len(flattened), dtype=np.float64)
    for column_index in range(flattened.shape[1]):
        series = flattened[:, column_index]
        finite = np.isfinite(series)
        if not finite.any():
            raise ValueError("a keypoint coordinate contains no finite samples")
        if not finite.all():
            series[~finite] = np.interp(index[~finite], index[finite], series[finite])
        local_median = median_filter(series, size=window, mode="nearest")
        deviation = np.abs(series - local_median)
        local_mad = median_filter(deviation, size=window, mode="nearest")
        threshold = np.maximum(outlier_sigma * 1.4826 * local_mad, float(minimum_outlier_distance))
        good = deviation <= threshold
        if not good.any():
            raise ValueError("outlier rejection removed every keypoint sample")
        if not good.all():
            series[~good] = np.interp(index[~good], index[good], series[good])
        flattened[:, column_index] = savgol_filter(
            series, window_length=window, polyorder=int(polynomial_order), mode="interp"
        )
    return flattened.reshape(values.shape)


def align_projected_arms(
    projected_landmarks: np.ndarray,
    target_wrists: np.ndarray,
    *,
    width: int,
    height: int,
    elbow_index: int = 1,
    wrist_index: int = 3,
) -> AlignedArmGeometry:
    """Lock both projected wrists with one similarity and extend elbow rays."""
    projected = np.asarray(projected_landmarks, dtype=np.float64)
    targets = np.asarray(target_wrists, dtype=np.float64)
    if projected.ndim != 4 or projected.shape[1] != 2 or projected.shape[-1] != 2:
        raise ValueError("projected_landmarks must have shape (frames, 2, points, 2)")
    if targets.shape != (len(projected), 2, 2):
        raise ValueError("target_wrists must have shape (frames, 2, 2)")
    if not np.isfinite(projected).all() or not np.isfinite(targets).all():
        raise ValueError("projected landmarks and target wrists must be finite")
    point_count = projected.shape[2]
    if not -point_count <= elbow_index < point_count:
        raise ValueError("elbow_index is outside the landmark array")
    if not -point_count <= wrist_index < point_count:
        raise ValueError("wrist_index is outside the landmark array")
    landmarks = np.empty_like(projected)
    roots = np.empty((len(projected), 2, 2), dtype=np.float64)
    transforms = np.empty((len(projected), 2, 3), dtype=np.float64)
    for frame_index in range(len(projected)):
        transform = estimate_similarity_2d(
            projected[frame_index, :, wrist_index], targets[frame_index]
        )
        aligned = apply_affine_2d(projected[frame_index], transform)
        landmarks[frame_index] = aligned
        transforms[frame_index] = transform
        for side_index in range(2):
            try:
                roots[frame_index, side_index] = ray_to_image_boundary(
                    aligned[side_index, wrist_index],
                    aligned[side_index, elbow_index],
                    width=width,
                    height=height,
                    fallback_to_reverse=True,
                )
            except ValueError:
                roots[frame_index, side_index] = nearest_image_boundary_point(
                    aligned[side_index, wrist_index], width=width, height=height
                )
    return AlignedArmGeometry(landmarks=landmarks, root_points=roots, transforms=transforms)


def _set_hinge_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"G1 model is missing joint {joint_name!r}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"G1 joint {joint_name!r} is not a hinge")
    data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def project_g1_arm_landmarks(
    model_path: str | Path, body_q: np.ndarray, calibration: CameraCalibration
) -> ArmProjection:
    """Run G1 FK and project two shoulder/elbow/wrist/base polylines."""
    trajectory = np.asarray(body_q, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != len(BODY_JOINT_NAMES):
        raise ValueError("body_q must have shape (frames, 29) in production MuJoCo order")
    if len(trajectory) < 1 or not np.isfinite(trajectory).all():
        raise ValueError("body_q must be non-empty and finite")
    source_path = Path(model_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    model = mujoco.MjModel.from_xml_path(str(source_path))
    data = mujoco.MjData(model)
    if model.nq >= 7:
        data.qpos[:3] = 0.0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id < 0:
        raise ValueError("G1 model is missing torso_link")
    points_torso = np.empty((len(trajectory), 2, 4, 3), dtype=np.float64)
    for frame_index, configuration in enumerate(trajectory):
        for joint_name, value in zip(BODY_JOINT_NAMES, configuration, strict=True):
            _set_hinge_qpos(model, data, joint_name, float(value))
        mujoco.mj_forward(model, data)
        torso_position = data.xpos[torso_id]
        torso_rotation = data.xmat[torso_id].reshape(3, 3)
        for side_index, body_names in enumerate(_ARM_BODY_NAMES):
            world_points = []
            for body_name in body_names:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    raise ValueError(f"G1 model is missing body {body_name!r}")
                world_points.append(data.xpos[body_id].copy())
            wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_names[-1])
            wrist_base = (
                data.xpos[wrist_id] + data.xmat[wrist_id].reshape(3, 3) @ _BRAINCO_BASE_OFFSET_M
            )
            world_points.append(wrist_base)
            points_torso[frame_index, side_index] = (
                np.asarray(world_points) - torso_position
            ) @ torso_rotation
    pixels, depth = project_camera_points(points_torso, calibration)
    if not np.isfinite(pixels).all() or not np.isfinite(depth).all():
        raise RuntimeError("G1 arm projection produced non-finite coordinates")
    if np.any(depth <= 0.0):
        raise RuntimeError("one or more G1 arm landmarks lie behind the camera")
    return ArmProjection(pixels=pixels, depth=depth)
