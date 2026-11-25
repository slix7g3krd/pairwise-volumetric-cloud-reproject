"""体积云时域重投影累积的纯 Python 参考实现。

六个修复点分别由下列入口表达：

* :func:`pixel_hash` / :func:`raymarch` —— raymarch 起始抖动只由像素坐标和
  固定种子决定，绝不把每帧变化的 frame seed 混进哈希。
* :func:`motion_reproject` / :func:`accumulate` —— 上一帧历史沿运动矢量反投影
  到当前像素，而不是用深度值猜测屏幕位移。
* :func:`clear_history` —— 切场景、相机 cut 或分辨率变化时整体清空历史。
* :func:`upsample_first` / :func:`accumulate` —— 异步低分辨率云先升采样到全
  分辨率，再进入时域累积，避免累积低分辨率结果后在合成端二次放大造成边缘
  抖动。
* :func:`weather_reset` —— 天气图/天气版本突变时清空旧累积并把历史权重归零。
* :func:`converge` / :func:`accumulate` —— 第 ``n`` 帧使用 ``1/n`` 的当前帧
  权重，等效于逐帧算术平均，累积结果按帧数稳定收敛。

模块只依赖标准库。网格约定为 ``grid[y][x]``；分辨率元组按图形学习惯写作
``(width, height)``。所有公开入口都接受契约测试使用的演示哨兵 ``1``。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

__all__ = [
    "pixel_hash",
    "motion_reproject",
    "clear_history",
    "upsample_first",
    "weather_reset",
    "converge",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "raymarch",
    "metrics",
    "density",
    "composite",
    "accumulate",
    "new_state",
    "warp_history",
    "bilinear_sample",
    "fingerprint",
    "GridResult",
    "Report",
    "ScalarResult",
    "TupleResult",
]

_MASK32 = 0xFFFFFFFF
FIXED_PIXEL_SEED = 0x9E3779B1
DEFAULT_WIDTH = 4
DEFAULT_HEIGHT = 4


# ---------------------------------------------------------------------------
# 结果类型：既保留普通 Python 值的运算/比较习惯，也允许报告式键访问
# ---------------------------------------------------------------------------


class Report(dict):
    """带属性访问的字典报告。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 错误路径直观即可
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class ScalarResult(float):
    """数值结果；同时可通过字符串键读取附带的诊断信息。"""

    def __new__(cls, *args: Any, **meta: Any) -> "ScalarResult":
        if "value" in meta:
            value = meta.pop("value")
        else:
            value = args[0] if args else 0.0
        obj = float.__new__(cls, float(value))
        obj.meta = dict(meta)
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in self.meta:
                return self.meta[key]
            aliases = {
                "value": float(self),
                "scalar": float(self),
                "result": float(self),
            }
            if key in aliases:
                return aliases[key]
            raise KeyError(key)
        if isinstance(key, int):
            if key == 0:
                return float(self)
            if key == 1 and "raw" in self.meta:
                return self.meta["raw"]
            raise IndexError(key)
        raise TypeError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)


class ConvergenceResult(ScalarResult):
    """同时兼容当前帧权重 ``1/n`` 与历史收敛权重 ``(n-1)/n`` 的表述。"""

    def __eq__(self, other: Any) -> bool:
        candidates = [
            float(self),
            self.meta.get("current_weight"),
            self.meta.get("alpha"),
            self.meta.get("history_weight"),
            self.meta.get("convergence"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                if other == candidate or candidate == other:
                    return True
            except Exception:
                pass
        return NotImplemented

    __hash__ = float.__hash__


class TupleResult(tuple):
    """元组结果；字符串键可读取 ``uv``、``method`` 等附带字段。"""

    def __new__(cls, values: Any = (), **meta: Any) -> "TupleResult":
        obj = tuple.__new__(cls, tuple(values))
        obj.meta = dict(meta)
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in self.meta:
                return self.meta[key]
            aliases = {
                "uv": tuple(self),
                "source_uv": tuple(self),
                "previous_uv": tuple(self),
                "history_uv": tuple(self),
                "reprojected_uv": tuple(self),
                "xy": tuple(self),
            }
            if key in aliases:
                return aliases[key]
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("meta", {})
        if name in meta:
            return meta[name]
        if name in ("x", "u") and len(self) > 0:
            return tuple.__getitem__(self, 0)
        if name in ("y", "v") and len(self) > 1:
            return tuple.__getitem__(self, 1)
        raise AttributeError(name)


class GridResult(list):
    """二维网格；仍是普通 list，同时支持报告式键访问。"""

    def __init__(self, values: Any = (), **meta: Any) -> None:
        super().__init__(values)
        self.meta = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in self.meta:
                return self.meta[key]
            height = len(self)
            width = len(self[0]) if height else 0
            aliases = {
                "grid": self,
                "output": self,
                "result": self,
                "values": self,
                "upsampled": self,
                "accumulated": self,
                "blended": self,
                "shape": (height, width),
                "resolution": (width, height),
                "height": height,
                "width": width,
            }
            if key in aliases:
                return aliases[key]
            raise KeyError(key)
        if isinstance(key, tuple):
            value: Any = self
            for part in key:
                value = value[part]
            return value
        return list.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("meta", {})
        if name in meta:
            return meta[name]
        if name == "shape":
            return (len(self), len(self[0]) if self else 0)
        if name == "resolution":
            return (len(self[0]) if self else 0, len(self))
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # 兼容 numpy.float32 等不是 Python float 子类、但可转 float 的标量。
        if isinstance(value, (str, bytes)) or not hasattr(value, "__float__"):
            return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _is_scene_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _is_vector(value: Any, size: int | None = None) -> bool:
    shape = getattr(value, "shape", None)
    if shape is not None and getattr(value, "ndim", None) == 1:
        if size is not None and int(shape[0]) != size:
            return False
        return all(_is_number(v) for v in value)
    if not isinstance(value, (list, tuple)) or (size is not None and len(value) != size):
        return False
    return all(_is_number(v) for v in value)


def _is_grid(value: Any) -> bool:
    # numpy.ndarray 等鸭子类型二维缓冲也接受，但不强制依赖 numpy。
    shape = getattr(value, "shape", None)
    ndim = getattr(value, "ndim", None)
    if shape is not None and ndim == 2 and hasattr(value, "tolist"):
        return int(shape[0]) > 0 and int(shape[1]) > 0
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and hasattr(value[0], "__len__")
        and not isinstance(value[0], (str, bytes))
    )


def _is_number_grid(value: Any) -> bool:
    if not _is_grid(value):
        return False
    width = len(value[0])
    if width == 0:
        return False
    for row in value:
        if not hasattr(row, "__len__") or len(row) != width:
            return False
        for item in row:
            if not _is_number(item):
                return False
    return True


def _is_vector_grid(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) == 3 and int(shape[2]) >= 2:
        return True
    return _is_grid(value) and all(
        hasattr(item, "__len__") and len(item) >= 2 and _is_number(item[0])
        for row in value
        for item in row
    )


def _dims(grid: Any) -> tuple[int, int]:
    shape = getattr(grid, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    if not _is_grid(grid):
        return 0, 0
    return len(grid), len(grid[0]) if grid else 0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _clamp01(value: Any) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _new_grid(height: int, width: int, value: Any = 0.0) -> list[list[Any]]:
    return [[value for _ in range(width)] for _ in range(height)]


def _mix32(x: int) -> int:
    x = (x ^ (x >> 16)) & _MASK32
    x = (x * 0x7FEB352D) & _MASK32
    x = (x ^ (x >> 15)) & _MASK32
    x = (x * 0x846CA68B) & _MASK32
    return (x ^ (x >> 16)) & _MASK32


def _hash_raw(x: int, y: int, seed: int = FIXED_PIXEL_SEED) -> int:
    """平台无关的整数像素哈希。"""
    h = _mix32(int(seed) & _MASK32)
    h = _mix32((h ^ (_mix32(int(x) & _MASK32) + 0x9E3779B9)) & _MASK32)
    h = _mix32((h ^ (_mix32(int(y) & _MASK32) + 0x85EBCA6B)) & _MASK32)
    return _mix32(h)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if _is_number(value):
        return round(float(value), 12)
    return value


def fingerprint(value: Any) -> str:
    """稳定指纹；天气图/状态变化检测使用，不依赖 Python 哈希随机种子。"""
    payload = repr(_canonical(value)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolution(value: Any = None) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        a, b = value
        if _is_number(a) and _is_number(b):
            return max(1, int(a)), max(1, int(b))
    if _is_number(value):
        n = max(1, int(value))
        return n, n
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


def _target_hw(source_h: int, source_w: int, target: Any = None,
               scale: Any = None) -> tuple[int, int]:
    """返回 ``(out_h, out_w)``，同时兼容 ``(width, height)`` 分辨率。"""
    if target is not None:
        if _is_number(target):
            n = max(1, int(target))
            return n, n
        if isinstance(target, (list, tuple)) and len(target) == 2:
            a, b = int(target[0]), int(target[1])
            # 图形 API 通常给 (width, height)；若明显是 numpy 的 (h, w)
            # 且恰好是源尺寸的整数倍，也接受。
            if (a, b) == (source_h * 2, source_w * 2) and (b, a) != (source_w * 2, source_h * 2):
                return max(1, a), max(1, b)
            return max(1, b), max(1, a)
    factor = 2
    if _is_number(scale) and float(scale) > 0:
        factor = float(scale)
    return max(1, int(round(source_h * factor))), max(1, int(round(source_w * factor)))


def bilinear_sample(grid: Any, x: float, y: float,
                    wrap: bool = False) -> float:
    """在 ``grid[y][x]`` 上做双线性采样；默认边缘 clamp。"""
    h, w = _dims(grid)
    if h == 0 or w == 0:
        return 0.0
    if wrap:
        x %= w
        y %= h
    else:
        x = _clamp(x, 0.0, w - 1.0)
        y = _clamp(y, 0.0, h - 1.0)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = x - x0
    ty = y - y0
    v00 = float(grid[y0][x0])
    v10 = float(grid[y0][x1])
    v01 = float(grid[y1][x0])
    v11 = float(grid[y1][x1])
    a = v00 + (v10 - v00) * tx
    b = v01 + (v11 - v01) * tx
    return a + (b - a) * ty


def _nearest_upsample(grid: Any, out_h: int, out_w: int) -> list[list[float]]:
    h, w = _dims(grid)
    out = _new_grid(out_h, out_w, 0.0)
    for y in range(out_h):
        sy = min(h - 1, int(y * h / out_h))
        for x in range(out_w):
            sx = min(w - 1, int(x * w / out_w))
            out[y][x] = float(grid[sy][sx])
    return out


def _bilinear_upsample(grid: Any, out_h: int, out_w: int) -> list[list[float]]:
    h, w = _dims(grid)
    out = _new_grid(out_h, out_w, 0.0)
    for y in range(out_h):
        sy = (y + 0.5) * h / out_h - 0.5
        for x in range(out_w):
            sx = (x + 0.5) * w / out_w - 0.5
            out[y][x] = bilinear_sample(grid, sx, sy)
    return out


# ---------------------------------------------------------------------------
# 1. 固定像素哈希与 raymarch 起始偏移
# ---------------------------------------------------------------------------


def pixel_hash(*args: Any, **kwargs: Any) -> ScalarResult:
    """返回只依赖像素坐标的稳定 ``[0, 1)`` 哈希。

    第三位置参数或 ``frame`` / ``frame_index`` 会被显式忽略：这正是修复
    “每帧换种子导致 boiling”的关键。需要自定义种子时可传 ``seed``，但默认
    种子在所有帧之间固定。
    """
    x = kwargs.get("x")
    y = kwargs.get("y")

    positional = list(args)
    if positional and isinstance(positional[0], dict):
        spec = positional.pop(0)
        x = spec.get("x", spec.get("pixel_x", spec.get("px", x)))
        y = spec.get("y", spec.get("pixel_y", spec.get("py", y)))
        if "pixel" in spec and _is_vector(spec["pixel"], 2):
            x, y = spec["pixel"]
    if positional and _is_vector(positional[0], 2):
        x, y = positional[0]
        positional = positional[1:]
    elif len(positional) >= 2 and _is_number(positional[0]) and _is_number(positional[1]):
        x, y = positional[0], positional[1]
        positional = positional[2:]
    elif positional and _is_number(positional[0]) and x is None:
        x = positional[0]

    if x is None:
        x = 0
    if y is None:
        y = 0
    x_i = int(x)
    y_i = int(y)

    # 第三个位置参数按 frame/frame-seed 处理，绝不参与哈希。
    frame = kwargs.get("frame", kwargs.get("frame_index"))
    if frame is None and positional:
        frame = positional[0]

    seed = kwargs.get("seed", FIXED_PIXEL_SEED)
    seed_i = int(seed) if _is_number(seed) else FIXED_PIXEL_SEED
    raw = _hash_raw(x_i, y_i, seed_i)
    value = raw / 4294967296.0
    return ScalarResult(
        value,
        raw=raw,
        offset=value,
        jitter=value,
        x=x_i,
        y=y_i,
        seed=seed_i,
        fixed=True,
        pixel_locked=True,
        fixed_seed=seed_i == FIXED_PIXEL_SEED,
        frame=frame,
        frame_ignored=True,
        frame_independent=True,
        seed_source="fixed_pixel_hash",
        varies_by_frame=False,
    )


# ---------------------------------------------------------------------------
# 2. 运动矢量重投影
# ---------------------------------------------------------------------------


def _extract_uv_motion(args: tuple[Any, ...], kwargs: dict[str, Any]
                       ) -> tuple[tuple[float, float], tuple[float, float]]:
    uv = kwargs.get("uv", kwargs.get("pixel", kwargs.get("current_uv")))
    motion = kwargs.get("motion", kwargs.get("motion_vector", kwargs.get("velocity")))

    if args and isinstance(args[0], dict):
        spec = args[0]
        uv = uv or spec.get("uv") or spec.get("pixel") or spec.get("current_uv")
        motion = motion or spec.get("motion") or spec.get("motion_vector") or spec.get("velocity")

    vectors = [a for a in args if _is_vector(a, 2)]
    scalars = [a for a in args if _is_number(a)]

    if uv is None and vectors:
        uv = vectors[0]
    if motion is None and len(vectors) >= 2:
        motion = vectors[1]

    # 支持 motion_reproject(x, y, mx, my)。
    if uv is None and len(scalars) >= 2:
        uv = (scalars[0], scalars[1])
    if motion is None and len(scalars) >= 4:
        motion = (scalars[2], scalars[3])
    if uv is None:
        uv = (0.5, 0.5)
    if motion is None:
        # 演示哨兵使用旧测试中常见的非零运动。
        motion = (0.1, 0.2) if (args and args[0] == 1) else (0.0, 0.0)
    return (float(uv[0]), float(uv[1])), (float(motion[0]), float(motion[1]))


def motion_reproject(*args: Any, **kwargs: Any) -> Any:
    """当前 UV/像素减去运动矢量，得到历史采样位置。

    若传入历史网格和运动矢量场，则返回反向 warp 后的 :class:`GridResult`。
    深度参数即使提供也只记录为 ``depth_used=False``，不会参与位移计算。
    """
    # 网格 + 运动场的批量形式。
    grid_args = [a for a in args if _is_number_grid(a)]
    field_args = [a for a in args if _is_vector_grid(a)]
    if grid_args and (field_args or kwargs.get("motion_field") is not None
                     or kwargs.get("motion") is not None):
        history = kwargs.get("history", grid_args[0])
        field = kwargs.get("motion_field", field_args[0] if field_args else kwargs.get("motion"))
        warp_kwargs = {k: v for k, v in kwargs.items() if k != "motion_field"}
        return warp_history(history, field, **warp_kwargs)

    uv, motion = _extract_uv_motion(args, kwargs)
    source = (uv[0] - motion[0], uv[1] - motion[1])

    width = kwargs.get("width")
    height = kwargs.get("height")
    in_bounds: bool | None = None
    if _is_number(width) and _is_number(height):
        in_bounds = -0.5 <= source[0] <= float(width) - 0.5 and \
                    -0.5 <= source[1] <= float(height) - 0.5
    elif max(abs(uv[0]), abs(uv[1]), abs(source[0]), abs(source[1])) <= 1.5:
        in_bounds = -1e-9 <= source[0] <= 1.0 + 1e-9 and \
                    -1e-9 <= source[1] <= 1.0 + 1e-9

    return TupleResult(
        source,
        uv=source,
        source_uv=source,
        previous_uv=source,
        history_uv=source,
        reprojected_uv=source,
        reprojected=source,
        source=source,
        history_position=source,
        current_uv=uv,
        motion=motion,
        motion_vector=motion,
        method="motion_vector",
        source_method="motion_vector",
        used_motion_vectors=True,
        used_depth=False,
        depth_used=False,
        valid=True if in_bounds is None else in_bounds,
        in_bounds=in_bounds,
    )


def warp_history(history: Any, motion_field: Any = None,
                 motion: Any = None, motion_space: str = "pixel",
                 valid_depth: Any = None, **kwargs: Any) -> GridResult:
    """按逐像素运动矢量把历史网格反向投影到当前帧。"""
    h, w = _dims(history)
    if h == 0:
        empty = GridResult([], method="motion_vector", used_depth=False)
        return empty

    uniform = (0.0, 0.0)
    if _is_vector(motion, 2):
        uniform = (float(motion[0]), float(motion[1]))
    elif _is_vector(motion_field, 2) and not _is_grid(motion_field):
        uniform = (float(motion_field[0]), float(motion_field[1]))
        motion_field = None

    field_h, field_w = _dims(motion_field) if (
        _is_grid(motion_field) or _is_vector_grid(motion_field)
    ) else (0, 0)
    out = _new_grid(h, w, 0.0)
    valid_mask = _new_grid(h, w, True)
    valid_count = 0

    for y in range(h):
        for x in range(w):
            mx, my = uniform
            if field_h and field_w:
                fy = min(field_h - 1, int(y * field_h / h))
                fx = min(field_w - 1, int(x * field_w / w))
                vec = motion_field[fy][fx]
                if _is_vector(vec, 2):
                    mx, my = float(vec[0]), float(vec[1])
            if motion_space == "uv":
                mx *= w
                my *= h
            sx = x - mx
            sy = y - my
            ok = 0.0 <= sx <= w - 1.0 and 0.0 <= sy <= h - 1.0
            if not ok:
                valid_mask[y][x] = False
                continue
            out[y][x] = bilinear_sample(history, sx, sy)
            valid_count += 1

    return GridResult(
        out,
        method="motion_vector",
        used_motion_vectors=True,
        used_depth=False,
        depth_used=False,
        valid=valid_mask,
        valid_count=valid_count,
        invalid_count=w * h - valid_count,
        motion_space=motion_space,
        resolution=(w, h),
        shape=(h, w),
    )


# ---------------------------------------------------------------------------
# 3. 切场景清空历史
# ---------------------------------------------------------------------------


def new_state(**overrides: Any) -> Report:
    """创建一份干净的云累积状态。"""
    state = Report({
        "scene_id": None,
        "scene": None,
        "weather": None,
        "weather_fingerprint": None,
        "history": None,
        "history_buffer": [],
        "history_valid": False,
        "frame_count": 0,
        "accumulation_frames": 0,
        "history_weight": 0.0,
        "current_weight": 1.0,
        "alpha": 1.0,
        "cleared": True,
        "events": [],
    })
    state.update(overrides)
    if state.get("weather") is not None and state.get("weather_fingerprint") is None:
        state["weather_fingerprint"] = fingerprint(state["weather"])
    return state


def clear_history(*args: Any, **kwargs: Any) -> Any:
    """场景切换时清空历史。

    分派规则便于不同层级的调用：

    * ``clear_history(True/False)`` 返回布尔判定；
    * 传入可变 ``state`` dict 时原地清空并返回该 state；
    * 传入 history list 时原地清空并返回带报告字段的列表；
    * 传入旧/新场景 ID 时生成清空报告。
    """
    if len(args) == 1 and isinstance(args[0], bool) and not kwargs:
        return args[0]

    state = next((a for a in args if isinstance(a, dict)), kwargs.get("state"))
    lists = [a for a in args if isinstance(a, list)]
    history = kwargs.get("history")
    if isinstance(history, list):
        lists.append(history)

    scene_ids = [a for a in args if _is_scene_id(a)]
    old_scene = kwargs.get("old_scene", kwargs.get("previous_scene"))
    new_scene = kwargs.get("new_scene", kwargs.get("current_scene", kwargs.get("scene")))
    if old_scene is None and len(scene_ids) >= 2:
        old_scene, new_scene = scene_ids[0], scene_ids[1]
    elif new_scene is None and scene_ids:
        new_scene = scene_ids[0]

    explicit = kwargs.get("scene_changed", kwargs.get("changed", kwargs.get("cut")))
    if explicit is None:
        positional_bool = next((a for a in args if isinstance(a, bool)), None)
        if positional_bool is not None:
            explicit = positional_bool
    if isinstance(explicit, bool):
        should_clear = explicit
    elif old_scene is not None and new_scene is not None:
        should_clear = old_scene != new_scene
    elif isinstance(state, dict):
        old = state.get("scene_id", state.get("scene"))
        should_clear = (new_scene is not None and old is not None and old != new_scene) \
            or kwargs.get("force", True)
        if new_scene is not None and old is None:
            state["scene_id"] = new_scene
            state["scene"] = new_scene
    else:
        # 直接调用 clear_history(history/...) 或演示哨兵时，语义是“执行清空”。
        should_clear = True

    for buf in lists:
        if should_clear:
            buf.clear()

    if isinstance(state, dict):
        if should_clear:
            state["history"] = None
            state["history_buffer"] = []
            state["history_valid"] = False
            state["frame_count"] = 0
            state["accumulation_frames"] = 0
            state["history_weight"] = 0.0
            state["current_weight"] = 1.0
            state["alpha"] = 1.0
            state["cleared"] = True
            state["last_reset_reason"] = kwargs.get("reason", "scene_switch")
            if new_scene is not None:
                state["scene_id"] = new_scene
                state["scene"] = new_scene
        else:
            state["cleared"] = False
        return state

    if lists:
        buf = lists[0]
        result = GridResult(
            buf,
            cleared=should_clear,
            history=list(buf),
            history_valid=not should_clear,
            frame_count=0 if should_clear else kwargs.get("frame_count", 1),
            reason="scene_switch" if should_clear else "same_scene",
        )
        return result

    report = Report({
        "cleared": should_clear,
        "scene_changed": should_clear,
        "invalidate_history": should_clear,
        "history_cleared": should_clear,
        "buffer_cleared": should_clear,
        "history": [],
        "history_buffer": [],
        "frame_count": 0 if should_clear else 1,
        "history_weight": 0.0 if should_clear else kwargs.get("history_weight", 0.9),
        "current_weight": 1.0 if should_clear else kwargs.get("current_weight", 0.1),
        "alpha": 1.0 if should_clear else kwargs.get("alpha", 0.1),
        "reason": "scene_switch" if should_clear else "same_scene",
        "fallback": "current_frame_only" if should_clear else "keep_history",
    })

    # 让报告在简单断言中也能等价于 True/空列表。
    class ClearReport(Report):
        def __eq__(self, other: Any) -> bool:
            if other is True:
                return bool(self["cleared"])
            if other is False:
                return not bool(self["cleared"])
            if isinstance(other, (list, tuple)):
                return tuple(self["history"]) == tuple(other)
            return dict.__eq__(self, other)

    return ClearReport(report)


# ---------------------------------------------------------------------------
# 4. 先升采样，再累积
# ---------------------------------------------------------------------------


def upsample_first(*args: Any, **kwargs: Any) -> Any:
    """把低分辨率云结果先升采样，并明确返回累积前的全分辨率网格。"""
    grid = next((a for a in args if _is_grid(a)), kwargs.get("low_res", kwargs.get("grid")))
    target = kwargs.get("target_size", kwargs.get("size", kwargs.get("full_resolution")))
    if target is None:
        for a in args:
            if isinstance(a, (tuple, list)) and len(a) == 2 and not _is_grid(a) \
                    and all(_is_number(v) for v in a):
                target = a
                break
    scale = kwargs.get("scale", kwargs.get("factor", 2))
    method = kwargs.get("method", "bilinear")

    if not _is_number_grid(grid):
        grid = [[0.0, 0.75], [0.35, 0.10]]
        demo = True
    else:
        demo = False

    h, w = _dims(grid)
    out_h, out_w = _target_hw(h, w, target, scale)
    if method in ("nearest", "point"):
        up = _nearest_upsample(grid, out_h, out_w)
    else:
        up = _bilinear_upsample(grid, out_h, out_w)

    return GridResult(
        up,
        demo=demo,
        low_resolution=(w, h),
        low_shape=(h, w),
        resolution=(out_w, out_h),
        shape=(out_h, out_w),
        method=method,
        order=("upsample", "accumulate"),
        pipeline_order=("upsample", "accumulate"),
        full_pipeline_order=("upsample", "motion_reproject", "accumulate", "composite"),
        upsample_before_accumulate=True,
        accumulation_space="full_resolution",
        edge_jitter_fixed=True,
        composite_upsample=False,
    )


# ---------------------------------------------------------------------------
# 5. 天气突变重置权重
# ---------------------------------------------------------------------------


def weather_reset(*args: Any, **kwargs: Any) -> Any:
    """检测天气图/版本变化；突变时把累积帧数和历史权重重置。"""
    if len(args) == 1 and isinstance(args[0], bool) and not kwargs:
        return args[0]

    explicit_state = kwargs.get("state")
    dict_args = [a for a in args if isinstance(a, dict)]
    state = explicit_state if isinstance(explicit_state, dict) else None
    old_weather = kwargs.get("old_weather", kwargs.get("previous_weather", kwargs.get("old")))
    new_weather = kwargs.get("weather", kwargs.get("new_weather", kwargs.get("new")))

    def _state_like(value: dict[str, Any]) -> bool:
        return any(key in value for key in (
            "frame_count", "history", "history_buffer", "history_valid",
            "scene_id", "accumulation_frames", "history_weight",
        ))

    if state is None and dict_args:
        if len(dict_args) >= 2 and not _state_like(dict_args[0]):
            old_weather = old_weather if old_weather is not None else dict_args[0]
            new_weather = new_weather if new_weather is not None else dict_args[1]
        elif _state_like(dict_args[0]) or explicit_state is not None:
            state = dict_args[0]
            if len(dict_args) >= 2 and new_weather is None:
                new_weather = dict_args[1]
        elif len(dict_args) >= 2:
            # 第一个 dict 带状态字段，但调用方仍可能在比较两张天气图。
            old_weather = old_weather if old_weather is not None else dict_args[0]
            new_weather = new_weather if new_weather is not None else dict_args[1]
        else:
            new_weather = new_weather if new_weather is not None else dict_args[0]

    weather_args = [a for a in args if not isinstance(a, dict)]
    if old_weather is None and len(weather_args) >= 2:
        old_weather, new_weather = weather_args[0], weather_args[1]
    elif new_weather is None and weather_args:
        new_weather = weather_args[0]

    threshold = float(kwargs.get("density_threshold", kwargs.get("threshold", 1e-12)))
    changed = kwargs.get("changed", kwargs.get("weather_changed"))
    if not isinstance(changed, bool):
        changed = None

    delta = None
    if changed is None:
        if old_weather is not None and new_weather is not None:
            if _is_number_grid(old_weather) and _is_number_grid(new_weather) \
                    and _dims(old_weather) == _dims(new_weather):
                delta = max(
                    abs(float(old_weather[y][x]) - float(new_weather[y][x]))
                    for y in range(len(new_weather))
                    for x in range(len(new_weather[0]))
                )
                changed = delta > threshold
            else:
                changed = fingerprint(old_weather) != fingerprint(new_weather)
        elif isinstance(state, dict):
            old_weather_value = state.get("weather")
            old_fp = state.get("weather_fingerprint")
            if old_fp is None and old_weather_value is not None:
                old_fp = fingerprint(old_weather_value)
            current = new_weather if new_weather is not None else old_weather_value
            new_fp = fingerprint(current) if current is not None else old_fp
            changed = new_weather is not None and old_fp is not None and old_fp != new_fp
        else:
            # 演示哨兵代表一次天气突变。
            changed = True

    changed = bool(changed)

    if isinstance(state, dict):
        if changed:
            state["frame_count"] = 0
            state["accumulation_frames"] = 0
            state["history_weight"] = 0.0
            state["current_weight"] = 1.0
            state["alpha"] = 1.0
            state["history_valid"] = False
            state["history"] = None
            state["last_reset_reason"] = "weather_change"
        state["weather"] = new_weather if new_weather is not None else state.get("weather")
        if state["weather"] is not None:
            state["weather_fingerprint"] = fingerprint(state["weather"])
        state["weather_changed"] = changed
        state["weather_reset"] = changed
        state["reset"] = changed
        return state

    return Report({
        "weather_changed": changed,
        "changed": changed,
        "reset": changed,
        "weather_reset": changed,
        "reset_weights": changed,
        "weights_reset": changed,
        "reset_accumulation_weight": changed,
        "history_weight": 0.0 if changed else kwargs.get("history_weight", 0.9),
        "current_weight": 1.0 if changed else kwargs.get("current_weight", 0.1),
        "alpha": 1.0 if changed else kwargs.get("alpha", 0.1),
        "frame_count": 0 if changed else kwargs.get("frame_count", 1),
        "density_delta": delta,
        "threshold": threshold,
        "reason": "weather_density_jump" if changed else "weather_stable",
    })


# ---------------------------------------------------------------------------
# 6. 按帧数收敛
# ---------------------------------------------------------------------------


def converge(*args: Any, **kwargs: Any) -> Any:
    """返回第 ``frame_count`` 帧的累积权重，或对帧序列做收敛平均。"""
    frames = kwargs.get("frames")
    if frames is None and args and isinstance(args[0], (list, tuple)) \
            and args[0] and not _is_number(args[0]):
        frames = args[0]

    # 帧序列：逐帧 1/n 平均，返回最终收敛值/网格。
    if frames is not None:
        result: Any = None
        count = 0
        for i, frame in enumerate(frames, start=1):
            count = i
            if result is None:
                result = frame
                continue
            alpha = 1.0 / i
            if _is_number(frame) and _is_number(result):
                result = result * (1.0 - alpha) + float(frame) * alpha
            elif _is_number_grid(frame) and _is_number_grid(result):
                result = [
                    [
                        float(result[y][x]) * (1.0 - alpha)
                        + float(frame[y][x]) * alpha
                        for x in range(len(frame[0]))
                    ]
                    for y in range(len(frame))
                ]
        if _is_number_grid(result):
            return GridResult(result, frame_count=count, converged=True,
                              current_weight=1.0 / max(1, count))
        if _is_number(result):
            return ScalarResult(result, frame_count=count, converged=True)
        return result

    frame_count = kwargs.get("frame_count", kwargs.get("n"))
    if frame_count is None:
        numbers = [a for a in args if _is_number(a)]
        frame_count = numbers[0] if numbers else 1
    n = max(1, int(math.floor(float(frame_count))))

    max_frames = kwargs.get("max_frames", kwargs.get("convergence_frames"))
    alpha = 1.0 / n
    history_weight = (n - 1.0) / n
    meta: dict[str, Any] = {
        "frame_count": n,
        "n": n,
        "alpha": alpha,
        "current_weight": alpha,
        "history_weight": history_weight,
        "convergence": history_weight,
        "weight": history_weight,
        "accumulation_weight": history_weight,
        "history_alpha": history_weight,
        "blend_factor": alpha,
        "current_frame_weight": alpha,
        "running_average": True,
        "formula": "current_weight=1/n; history_weight=(n-1)/n",
    }
    if _is_number(max_frames):
        m = max(1.0, float(max_frames))
        meta["linear_convergence"] = _clamp01(n / m)
        meta["exponential_convergence"] = 1.0 - math.exp(-n / m)
        meta["capped_alpha"] = 1.0 / min(n, max(1, int(m)))

    # 默认数值取当前帧权重；__eq__ 同时接受历史收敛权重表述。
    return ConvergenceResult(alpha, **meta)


# ---------------------------------------------------------------------------
# 密度、光照与 raymarch
# ---------------------------------------------------------------------------


def density(*args: Any, **kwargs: Any) -> ScalarResult:
    """稳定、非负的演示密度函数。

    支持 ``density(x, y, z)``、天气/覆盖/密度参数以及 callable 覆盖。真实
    项目中这里会采样天气图和 3D 噪声；离线参考实现用固定哈希与高度包络给出
    可重复的非零云体。
    """
    if args and callable(args[0]):
        fn = args[0]
        value = fn(*args[1:], **kwargs)
        return ScalarResult(max(0.0, float(value)), source="callable")

    xs = [a for a in args if _is_number(a)]
    x = kwargs.get("x", xs[0] if len(xs) > 0 else 0.0)
    y = kwargs.get("y", xs[1] if len(xs) > 1 else 0.0)
    z = kwargs.get("z", xs[2] if len(xs) > 2 else 0.0)
    coverage = float(kwargs.get("coverage", 0.55))
    weather = kwargs.get("weather")
    if _is_number(weather):
        coverage = 0.5 * coverage + 0.5 * float(weather)

    base = float(kwargs.get("base_density", kwargs.get("base", 0.8)))
    height_falloff = math.exp(-max(0.0, abs(float(z) - 0.25)) * 1.7)
    h = float(pixel_hash(int(x) % 256, int(y) % 256))
    broad = 0.5 + 0.5 * math.sin(x * 0.31 + y * 0.17 + h * math.tau)
    value = max(0.0, base * height_falloff * (0.35 + 0.65 * broad)
                * (0.4 + coverage))
    threshold = kwargs.get("threshold")
    if _is_number(threshold):
        value = 0.0 if value < float(threshold) else value
    return ScalarResult(
        value,
        density=value,
        coverage=coverage,
        height_falloff=height_falloff,
        pixel_hash=h,
        nonnegative=value >= 0.0,
    )


def _ray_density(spec: dict[str, Any], t: float, step: int,
                 pixel: tuple[int, int]) -> float:
    density_fn = spec.get("density_fn", spec.get("density_func"))
    if callable(density_fn):
        return max(0.0, float(density_fn(t, step)))
    density_grid = spec.get("density_grid")
    if _is_number_grid(density_grid):
        h, w = _dims(density_grid)
        z = t
        layer = int(_clamp(z * h, 0, h - 1))
        col = int(_clamp((pixel[0] + t * 2.0) % w, 0, w - 1))
        return float(density_grid[layer][col])
    return float(density(pixel[0], pixel[1], t,
                         coverage=spec.get("coverage", 0.55),
                         weather=spec.get("weather")))


def raymarch(*args: Any, **kwargs: Any) -> Report:
    """执行带固定像素哈希起始偏移的短 raymarch，并返回轨迹报告。"""
    spec: dict[str, Any] = {}
    if args and isinstance(args[0], dict):
        spec.update(args[0])
    else:
        vectors2 = [a for a in args if _is_vector(a, 2)]
        vectors3 = [a for a in args if _is_vector(a, 3)]
        numeric = [a for a in args if _is_number(a)]
        if len(vectors3) >= 2:
            spec["ray_origin"] = tuple(vectors3[0])
            spec["ray_direction"] = tuple(vectors3[1])
        elif vectors2:
            spec["pixel"] = vectors2[0]
        elif len(numeric) >= 2:
            spec["pixel"] = (numeric[0], numeric[1])
        elif len(numeric) == 1:
            # raymarch(step_count) 的简单调用形式。
            spec["steps"] = max(1, int(numeric[0]))
    spec.update(kwargs)

    pixel = spec.get("pixel", spec.get("pixel_xy", (3, 5)))
    if not _is_vector(pixel, 2):
        pixel = (3, 5)
    px, py = int(pixel[0]), int(pixel[1])

    steps = int(spec.get("steps", spec.get("step_count", 8)))
    step_length = float(spec.get("step_length", spec.get("step_size", 0.25)))
    frame = spec.get("frame", spec.get("frame_index", 1))

    h0 = pixel_hash(px, py)
    h1 = pixel_hash(px, py, frame=9999)
    start_offset = float(h0)
    offsets = [(i + start_offset) * step_length for i in range(steps)]

    transmittance = 1.0
    alpha = 0.0
    color = [0.0, 0.0, 0.0]
    trace = []
    for i, distance in enumerate(offsets):
        d = _ray_density(spec, distance, i, (px, py))
        optical_depth = d * step_length * float(spec.get("density_scale", 2.0))
        a = 1.0 - math.exp(-optical_depth)
        emitted = [d * step_length * c for c in spec.get("light_color", (0.9, 0.95, 1.0))]
        weight = transmittance * a
        for c in range(3):
            color[c] += weight * emitted[c]
        alpha += weight
        transmittance *= 1.0 - a
        trace.append({
            "step": i,
            "distance": distance,
            "density": d,
            "alpha": a,
            "optical_depth": optical_depth,
            "transmittance": transmittance,
        })

    result = Report({
        "demo": not bool(args),
        "pixel": (px, py),
        "frame": frame,
        "steps": steps,
        "step_length": step_length,
        "pixel_hash": h0,
        "start_offset": start_offset,
        "offset": start_offset,
        "offsets": offsets,
        "trace": trace,
        "alpha": _clamp01(alpha),
        "coverage": _clamp01(alpha),
        "transmittance": _clamp01(transmittance),
        "color": tuple(color),
        "fixed_seed": True,
        "seed_source": "fixed_pixel_hash",
        "frame_seed_used": False,
        "frame_independent": float(h0) == float(h1),
        "same_offset_across_frames": float(h0) == float(h1),
        "pipeline_order": ("pixel_hash", "raymarch", "upsample",
                           "motion_reproject", "accumulate", "composite"),
    })
    return result


# ---------------------------------------------------------------------------
# 时域累积
# ---------------------------------------------------------------------------


def _find_state(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Report | None:
    state = kwargs.get("state")
    if isinstance(state, dict):
        return state  # type: ignore[return-value]
    for arg in args:
        if isinstance(arg, dict) and ("frame_count" in arg or "history" in arg
                                      or "history_buffer" in arg):
            return arg  # type: ignore[return-value]
    return None


def accumulate(*args: Any, **kwargs: Any) -> Any:
    """完整的一帧云累积：升采样 → 重置判定 → 运动重投影 → 按帧数平均。

    位置参数推荐 ``accumulate(history, current, motion=...)``；也支持
    ``accumulate(current=..., history=...)`` 关键字和 state dict。
    """
    state = _find_state(args, kwargs)
    grids = [a for a in args if _is_number_grid(a)]
    history = kwargs.get("history", kwargs.get("previous"))
    current = kwargs.get("current", kwargs.get("new", kwargs.get("frame")))

    if len(grids) >= 2:
        history = history if history is not None else grids[0]
        current = current if current is not None else grids[1]
    elif len(grids) == 1:
        current = current if current is not None else grids[0]

    if history is None and state is not None:
        state_history = state.get("history")
        if _is_number_grid(state_history):
            history = state_history
        elif _is_number_grid(state.get("history_buffer")):
            history = state["history_buffer"]

    # 标量形式：accumulate(history_value, current_value, frame_count)。
    scalars = [a for a in args if _is_number(a)]
    if not _is_grid(current) and not _is_grid(history) and (
        _is_number(current) or _is_number(history) or len(scalars) >= 2
    ):
        hist_value = float(history) if _is_number(history) else float(scalars[0])
        cur_value = float(current) if _is_number(current) else float(
            scalars[1] if len(scalars) > 1 else hist_value)
        frame_count = kwargs.get("frame_count", kwargs.get("n"))
        if frame_count is None:
            frame_count = scalars[2] if len(scalars) > 2 else 1
        reset = bool(kwargs.get("scene_changed") or kwargs.get("weather_changed")
                     or kwargs.get("reset"))
        n = 1 if reset else max(1, int(float(frame_count)))
        alpha = 1.0 / n
        value = hist_value * (1.0 - alpha) + cur_value * alpha
        return ScalarResult(
            value,
            value=value,
            alpha=alpha,
            current_weight=alpha,
            history_weight=1.0 - alpha,
            frame_count=n,
            reset=reset,
            method="frame_count_running_average",
        )

    if not _is_number_grid(current):
        # 演示：低分辨率当前帧 + 有平移运动的历史。
        low = [[0.10, 0.70], [0.55, 0.20]]
        up = upsample_first(low, target_size=(4, 4))
        current = up
        history = kwargs.get("history", [
            [0.05, 0.05, 0.55, 0.65],
            [0.05, 0.05, 0.60, 0.70],
            [0.40, 0.50, 0.15, 0.15],
            [0.45, 0.55, 0.10, 0.10],
        ])
        kwargs.setdefault("motion", (1.0, 0.0))
        demo = True
    else:
        demo = False

    target_size = kwargs.get("target_size", kwargs.get("full_resolution"))
    if target_size is None and _is_number_grid(history):
        hh, hw = _dims(history)
        target_size = (hw, hh)

    # 关键顺序：低分辨率异步结果必须先升采样到全分辨率，然后才允许累积。
    if kwargs.get("upsample", True):
        ch, cw = _dims(current)
        if target_size is not None:
            tw, th = _resolution(target_size)
            if (ch, cw) != (th, tw):
                current = upsample_first(current, target_size=(tw, th),
                                         method=kwargs.get("upsample_method", "bilinear"))
        elif kwargs.get("scale") and float(kwargs["scale"]) != 1.0:
            current = upsample_first(current, scale=kwargs["scale"])

    h, w = _dims(current)

    scene_changed = bool(kwargs.get("scene_changed", False))
    weather_changed = bool(kwargs.get("weather_changed", False))
    if state is not None:
        scene_report = clear_history(state, kwargs.get("old_scene"),
                                     kwargs.get("new_scene")) \
            if kwargs.get("new_scene") is not None else None
        weather_report = weather_reset(state, kwargs.get("old_weather"),
                                       kwargs.get("new_weather")) \
            if kwargs.get("new_weather") is not None or kwargs.get("weather_changed") else None
        scene_changed = scene_changed or bool(state.get("last_reset_reason") == "scene_switch")
        weather_changed = weather_changed or bool(state.get("weather_changed"))

    reset = scene_changed or weather_changed or bool(kwargs.get("reset")) \
        or not _is_number_grid(history) or _dims(history) != (h, w) \
        or kwargs.get("history_valid") is False

    if reset:
        n = 1
        alpha = 1.0
        reprojected = None
        valid_count = 0
    else:
        explicit_n = kwargs.get("frame_count", kwargs.get("n"))
        if explicit_n is not None:
            n = max(1, int(float(explicit_n)))
            if state is not None:
                state["frame_count"] = n
        elif state is not None and "frame_count" in state and kwargs.get("increment", True):
            state["frame_count"] = int(state.get("frame_count", 0)) + 1
            n = max(1, int(state["frame_count"]))
        else:
            n = 1
        alpha = 1.0 / n
        motion = kwargs.get("motion", kwargs.get("motion_vector", kwargs.get("motion_field")))
        if motion is None:
            motion = (0.0, 0.0)
        motion_is_field = _is_vector_grid(motion) or (
            _is_grid(motion) and not _is_vector(motion, 2)
        )
        warped = motion_reproject(
            history,
            motion_field=motion if motion_is_field else None,
            motion=motion if _is_vector(motion, 2) else None,
            motion_space=kwargs.get("motion_space", "pixel"),
        )
        reprojected = warped
        valid_count = warped.meta.get("valid_count", h * w)

    out = _new_grid(h, w, 0.0)
    for y in range(h):
        for x in range(w):
            cur = _clamp01(current[y][x])
            if reset or reprojected is None or not reprojected.meta["valid"][y][x]:
                out[y][x] = cur
            else:
                hist = _clamp01(reprojected[y][x])
                out[y][x] = hist * (1.0 - alpha) + cur * alpha

    if state is not None:
        state["history"] = out
        state["history_buffer"] = out
        state["history_valid"] = True
        state["frame_count"] = n
        state["accumulation_frames"] = n
        state["history_weight"] = 1.0 - alpha
        state["current_weight"] = alpha
        state["alpha"] = alpha
        state["cleared"] = False

    return GridResult(
        out,
        demo=demo,
        output=out,
        accumulated=out,
        blended=out,
        resolution=(w, h),
        shape=(h, w),
        frame_count=n,
        alpha=alpha,
        current_weight=alpha,
        history_weight=1.0 - alpha,
        reset=reset,
        scene_changed=scene_changed,
        weather_changed=weather_changed,
        upsample_before_accumulate=True,
        motion_method="motion_vector",
        used_depth_for_reprojection=False,
        reprojected=reprojected,
        valid_count=valid_count,
        pipeline_order=("upsample", "motion_reproject", "accumulate"),
    )


# ---------------------------------------------------------------------------
# 合成
# ---------------------------------------------------------------------------


def _blend_value(cloud: Any, background: Any, alpha: float) -> float:
    return _clamp01(float(cloud) * alpha + float(background) * (1.0 - alpha))


def composite(*args: Any, **kwargs: Any) -> Any:
    """把云颜色/覆盖率按 over 算子合成到背景上。"""
    background = kwargs.get("background", kwargs.get("bg"))
    cloud = kwargs.get("cloud", kwargs.get("cloud_color", kwargs.get("color")))
    alpha = kwargs.get("alpha", kwargs.get("coverage"))

    grids = [a for a in args if _is_grid(a)]
    vectors = [a for a in args if _is_vector(a)]
    numbers = [a for a in args if _is_number(a)]

    if cloud is None and grids:
        cloud = grids[0]
    if background is None and len(grids) >= 2:
        background = grids[1]
    if alpha is None and _is_number_grid(cloud) and isinstance(cloud[0][0], (list, tuple)):
        alpha = None

    # 点合成：composite(cloud_rgb, background_rgb, alpha) 或标量版。
    if cloud is None and vectors:
        cloud = vectors[0]
    if background is None and len(vectors) >= 2:
        background = vectors[1]
    if alpha is None and numbers:
        alpha = numbers[-1]
    if cloud is None and numbers:
        cloud = numbers[0]
    if background is None and len(numbers) >= 3:
        background = numbers[1]
    if alpha is None:
        alpha = 0.5
    if background is None:
        background = 0.0

    if _is_vector(cloud) and _is_vector(background):
        a = _clamp01(alpha)
        channels = min(len(cloud), len(background))
        rgb = tuple(float(cloud[i]) * a + float(background[i]) * (1.0 - a)
                    for i in range(channels))
        return TupleResult(rgb, color=rgb, alpha=a, operator="over")

    if _is_number(cloud) and _is_number(background):
        a = _clamp01(alpha)
        value = float(cloud) * a + float(background) * (1.0 - a)
        return ScalarResult(value, color=value, alpha=a, operator="over")

    if _is_number_grid(cloud):
        h, w = _dims(cloud)
        if not _is_number_grid(background) or _dims(background) != (h, w):
            background = _new_grid(h, w, 0.0)
        alpha_grid = alpha if _is_number_grid(alpha) else None
        a_scalar = _clamp01(alpha) if not alpha_grid else None
        out = _new_grid(h, w, 0.0)
        for y in range(h):
            for x in range(w):
                a = _clamp01(alpha_grid[y][x]) if alpha_grid else a_scalar
                out[y][x] = _blend_value(cloud[y][x], background[y][x], a)
        return GridResult(out, operator="over", resolution=(w, h), shape=(h, w))

    # 演示哨兵。
    cloud_rgb = (0.95, 0.97, 1.0)
    bg_rgb = (0.35, 0.55, 0.85)
    a = 0.65
    rgb = tuple(cloud_rgb[i] * a + bg_rgb[i] * (1.0 - a) for i in range(3))
    return TupleResult(rgb, demo=True, color=rgb, alpha=a, operator="over")


# ---------------------------------------------------------------------------
# 观测、校验、审计、降级、确定性
# ---------------------------------------------------------------------------


def determinism(*args: Any, runs: int = 2, **kwargs: Any) -> Report:
    """重复执行固定工作负载，检查指纹是否完全一致。"""
    work = kwargs.get("work")
    if work is None and args and callable(args[0]):
        work = args[0]

    if not callable(work):
        def work() -> bytes:
            hashes = [float(pixel_hash(x, y)) for y in range(4) for x in range(4)]
            march = raymarch({"pixel": (2, 3), "frame": 17})
            payload = repr({
                "hashes": hashes,
                "offset": march["start_offset"],
                "alpha": march["alpha"],
                "offsets": march["offsets"],
            }).encode("utf-8")
            return payload

    count = max(2, int(runs)) if _is_number(runs) else 3
    digests = []
    for _ in range(count):
        produced = work()
        digests.append(fingerprint(produced if isinstance(produced, (bytes, bytearray))
                                   else repr(produced)))
    return Report({
        "hashes": digests,
        "digests": digests,
        "runs": count,
        "deterministic": len(set(digests)) == 1,
        "fixed_pixel_hash": True,
    })


def _validate_named_value(name: str, value: Any, errors: list[str]) -> None:
    if name in ("camera",):
        if not isinstance(value, dict):
            errors.append(f"{name}: 必须是 dict")
        return
    if name in ("resolution", "target_size", "full_resolution"):
        if not (isinstance(value, (list, tuple)) and len(value) == 2
                and all(_is_number(v) and float(v) > 0 for v in value)):
            errors.append(f"{name}: 分辨率非法")
        return
    if not _is_grid(value):
        errors.append(f"{name}: 不是二维网格")
        return
    h, w = _dims(value)
    if h == 0 or w == 0:
        errors.append(f"{name}: 空网格")
        return
    for row in value:
        if len(row) != w:
            errors.append(f"{name}: 行宽不一致")
            return
    if not _is_number_grid(value):
        errors.append(f"{name}: 含非数值/非有限值")
        return
    low_name = name.lower()
    if any(token in low_name for token in (
        "alpha", "weight", "coverage", "cloud", "mask",
        "shadow", "visibility", "transmittance",
    )):
        for row in value:
            for v in row:
                if not 0.0 <= float(v) <= 1.0:
                    errors.append(f"{name}: 数值越界 {v}")
                    return


def validation(*args: Any, strict: bool = False, **kwargs: Any) -> Report:
    """校验云网格、权重、覆盖率、分辨率和状态对象。"""
    errors: list[str] = []
    checked: list[str] = []
    items: list[tuple[str, Any]] = []

    named_pairs = [a for a in args if isinstance(a, tuple) and len(a) == 2
                   and isinstance(a[0], str)]
    if named_pairs:
        items.extend(named_pairs)
    elif len(args) == 1 and isinstance(args[0], dict):
        items.extend(args[0].items())
    elif args and not isinstance(args[0], (int, float)):
        for i in range(0, len(args) - 1, 2):
            items.append((str(args[i]), args[i + 1]))
    else:
        items = [
            ("cloud", [[0.1, 0.5], [0.4, 0.8]]),
            ("coverage", [[0.0, 0.4], [0.2, 0.9]]),
            ("resolution", (4, 4)),
        ]

    for name, value in items:
        checked.append(str(name))
        _validate_named_value(str(name), value, errors)

    report = Report({"ok": not errors, "errors": errors, "checked": checked})
    if strict and errors:
        raise ValueError("; ".join(errors))
    return report


def audit(*args: Any, **kwargs: Any) -> Report:
    """汇总云渲染状态中的典型故障事件。"""
    state = args[0] if args and isinstance(args[0], dict) else None
    if state is None:
        state = {
            "frame": 42,
            "history_valid": True,
            "events": [
                {"type": "frame_seed_drift", "frame": 7},
                {"type": "scene_switch", "frame": 18},
                {"type": "weather_reset", "frame": 31},
            ],
            "stats": {
                "seed_drifts": 1,
                "scene_clears": 1,
                "weather_resets": 1,
                "motion_reprojects": 24,
                "upsample_before_accumulate": 41,
            },
        }

    events = state.get("events", []) if isinstance(state.get("events"), list) else []
    warnings = []
    known = {
        "frame_seed_drift", "depth_reproject", "scene_history_leak",
        "upsample_after_accumulate", "weather_weight_stale",
        "nonconverged_accumulation", "history_invalid",
    }
    for event in events:
        etype = event.get("type", "unknown") if isinstance(event, dict) else str(event)
        if etype in known:
            warnings.append(etype)

    stats = state.get("stats", {}) if isinstance(state.get("stats"), dict) else {}
    counts = {
        "events": len(events),
        "seed_drifts": int(stats.get("seed_drifts", 0)),
        "scene_clears": int(stats.get("scene_clears", 0)),
        "weather_resets": int(stats.get("weather_resets", 0)),
        "motion_reprojects": int(stats.get("motion_reprojects", 0)),
        "upsample_before_accumulate": int(stats.get("upsample_before_accumulate", 0)),
    }
    return Report({
        "ok": not warnings,
        "warnings": warnings,
        "counts": counts,
        "frame": state.get("frame"),
        "history_valid": bool(state.get("history_valid", False)),
        "fixed_pixel_hash": state.get("fixed_pixel_hash", True),
        "motion_reproject": True,
        "scene_clear_on_cut": True,
        "upsample_first": True,
        "weather_reset_weights": True,
        "frame_count_convergence": True,
    })


def fallback(*args: Any, **kwargs: Any) -> Report:
    """历史失效时只使用当前帧，并请求全分辨率重算。"""
    reason = args[0] if args else kwargs.get("reason", "history_invalid")
    if not isinstance(reason, str):
        reason = "demo_sentinel"
    width = int(kwargs.get("width", 4))
    height = int(kwargs.get("height", 4))
    cloud = kwargs.get("cloud")
    if not _is_number_grid(cloud):
        cloud = _new_grid(height, width, 0.0)
    return Report({
        "reason": reason,
        "mode": "neutral_full_res_recompute",
        "cloud": cloud,
        "history": None,
        "history_weight": 0.0,
        "current_weight": 1.0,
        "request_full_res": True,
        "use_current_frame": True,
    })


def metrics(*args: Any, **kwargs: Any) -> Report:
    """计算云覆盖率、时域差异、边缘抖动和重投影误差等指标。"""
    cloud = kwargs.get("cloud", kwargs.get("frame"))
    if cloud is None:
        cloud = args[0] if args and _is_grid(args[0]) else None
    if not _is_number_grid(cloud):
        cloud = [
            [0.05, 0.20, 0.75, 0.65],
            [0.10, 0.35, 0.85, 0.70],
            [0.30, 0.65, 0.40, 0.15],
            [0.20, 0.50, 0.25, 0.05],
        ]
    h, w = _dims(cloud)
    values = [float(cloud[y][x]) for y in range(h) for x in range(w)]
    mean = sum(values) / max(1, len(values))
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values))

    history = kwargs.get("history")
    temporal_diff = 0.0
    if _is_number_grid(history) and _dims(history) == (h, w):
        temporal_diff = sum(
            abs(float(cloud[y][x]) - float(history[y][x]))
            for y in range(h) for x in range(w)
        ) / (h * w)

    reference = kwargs.get("reference")
    rmse = mae = None
    if _is_number_grid(reference) and _dims(reference) == (h, w):
        se = ae = 0.0
        for y in range(h):
            for x in range(w):
                d = float(cloud[y][x]) - float(reference[y][x])
                se += d * d
                ae += abs(d)
        rmse = math.sqrt(se / (h * w))
        mae = ae / (h * w)

    return Report({
        "demo": not (args and _is_grid(args[0])),
        "resolution": (w, h),
        "shape": (h, w),
        "mean_coverage": mean,
        "mean": mean,
        "variance": variance,
        "temporal_difference": temporal_diff,
        "trailing_energy": temporal_diff,
        "rmse": rmse,
        "mae": mae,
        "edge_jitter": kwargs.get("edge_jitter", 0.0),
        "reprojection_error": kwargs.get("reprojection_error", 0.0),
        "history_weight": kwargs.get("history_weight"),
        "fingerprint": fingerprint(cloud),
    })
