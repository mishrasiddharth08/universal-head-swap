"""CPU filters for suppressing two-pixel Nyquist grid patterns.

The automatic 9-tap algorithm is adapted to NumPy/Pillow from ComfyUI-DeGrid
commit 5699bc33f71e1be12523fdea105cc9c2abfe1cd0 (Apache-2.0). This version is
row-tiled, accepts PIL images, and adds strength blending.
"""

import numpy as np
from PIL import Image


_B = np.asarray((-1, 6, -15, 20, -15, 6, -1), dtype=np.float32) / 64.0
_AUTO_B = np.asarray((1, -8, 28, -56, 70, -56, 28, -8, 1), dtype=np.float32) / 256.0
_AUTO_PAD = 4
_NEGLIGIBLE_AMP = 0.5 / 255.0


def suppress_nyquist_grid(image: Image.Image, strength: float = 1.0,
                          tile_rows: int = 256) -> Image.Image:
    """Return ``image`` filtered as C-Bx-By+Bxy, blended by ``strength``.

    RGB and RGBA images are supported. Processing is row-tiled to bound memory;
    edges are clamped and an RGBA image's alpha channel is preserved exactly.
    """
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if strength == 0.0:
        return image
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("image mode must be RGB or RGBA")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")

    raw = np.asarray(image)
    color = raw[..., :3].astype(np.float32)
    height, width = color.shape[:2]
    output = np.empty_like(raw)

    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        rows = np.clip(np.arange(start - 3, stop + 3), 0, height - 1)
        extended = color[rows]
        padded = np.pad(extended, ((0, 0), (3, 3), (0, 0)), mode="edge")

        bx_extended = sum(
            coefficient * padded[:, offset:offset + width]
            for offset, coefficient in enumerate(_B)
        )
        count = stop - start
        bx = bx_extended[3:3 + count]
        by = sum(
            coefficient * extended[offset:offset + count]
            for offset, coefficient in enumerate(_B)
        )
        bxy = sum(
            coefficient * bx_extended[offset:offset + count]
            for offset, coefficient in enumerate(_B)
        )
        filtered = color[start:stop] - bx - by + bxy
        blended = color[start:stop] + strength * (filtered - color[start:stop])
        output[start:stop, :, :3] = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)

    if image.mode == "RGBA":
        output[..., 3] = raw[..., 3]
    result = Image.fromarray(output, mode=image.mode)
    result.info.update(image.info)
    return result


def _reflect_indices(indices, size):
    period = 2 * (size - 1)
    folded = np.mod(indices, period)
    return np.where(folded < size, folded, period - folded)


def _grid_tile(raw, start, stop):
    """Extract Bx+By-Bxy for one row tile, using reflect boundaries."""
    height, width = raw.shape[:2]
    rows = _reflect_indices(np.arange(start - _AUTO_PAD, stop + _AUTO_PAD), height)
    extended = raw[rows, :, :3].astype(np.float32) / 255.0
    padded = np.pad(extended, ((0, 0), (_AUTO_PAD, _AUTO_PAD), (0, 0)), mode="reflect")
    bx_extended = sum(
        coefficient * padded[:, offset:offset + width]
        for offset, coefficient in enumerate(_AUTO_B)
    )
    count = stop - start
    bx = bx_extended[_AUTO_PAD:_AUTO_PAD + count]
    by = sum(
        coefficient * extended[offset:offset + count]
        for offset, coefficient in enumerate(_AUTO_B)
    )
    bxy = sum(
        coefficient * bx_extended[offset:offset + count]
        for offset, coefficient in enumerate(_AUTO_B)
    )
    return bx + by - bxy


def auto_degrid(image: Image.Image, strength: float = 0.5, tile_rows: int = 256,
                skip_when_clean: bool = True):
    """Automatically detect and reduce a phase-locked two-pixel VAE grid.

    Returns ``(image, report)``. RGB/RGBA alpha and metadata are preserved.
    """
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("image mode must be RGB or RGBA")
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")

    raw = np.asarray(image)
    height, width = raw.shape[:2]
    report = {
        "amp_255": 0.0, "texture_255": 0.0, "checker_255": 0.0,
        "vstripe_255": 0.0, "hstripe_255": 0.0, "limit": 0.004,
        "clipped_pct": 0.0, "skipped": True,
    }
    if height <= 2 * _AUTO_PAD or width <= 2 * _AUTO_PAD:
        return image, report

    even_height, even_width = height // 2 * 2, width // 2 * 2
    phase_sums = np.zeros((3, 4), dtype=np.float64)
    phase_counts = np.zeros(4, dtype=np.int64)
    total_values = height * width * 3
    stride = total_values // 1_000_000 + 1 if total_values > 1_000_000 else 1
    samples = []

    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        corr = _grid_tile(raw, start, stop)
        phase_stop = min(stop, even_height)
        if start < phase_stop:
            usable = corr[:phase_stop - start, :even_width]
            for row_phase in (0, 1):
                local_row = (row_phase - start) % 2
                for col_phase in (0, 1):
                    phase = row_phase * 2 + col_phase
                    values = usable[local_row::2, col_phase::2]
                    phase_sums[:, phase] += values.sum(axis=(0, 1), dtype=np.float64)
                    phase_counts[phase] += values.shape[0] * values.shape[1]
        for channel in range(3):
            flat = corr[:, :, channel].reshape(-1)
            global_start = channel * height * width + start * width
            first = (-global_start) % stride
            samples.append(np.abs(flat[first::stride]))

    means = phase_sums / phase_counts[None, :]
    amp = float(np.max(np.ptp(means, axis=1)))
    m00, m01, m10, m11 = (means[:, index] for index in range(4))
    checker = float(np.max(np.abs(((m00 + m11) - (m01 + m10)) / 2.0)))
    vstripe = float(np.max(np.abs(((m00 + m10) - (m01 + m11)) / 2.0)))
    hstripe = float(np.max(np.abs(((m00 + m01) - (m10 + m11)) / 2.0)))
    texture = float(np.quantile(np.concatenate(samples), 0.75))
    limit = float(np.clip(texture * 3.0, 0.004, 0.05))
    skipped = bool(skip_when_clean and amp < _NEGLIGIBLE_AMP)
    report.update({
        "amp_255": amp * 255.0, "texture_255": texture * 255.0,
        "checker_255": checker * 255.0, "vstripe_255": vstripe * 255.0,
        "hstripe_255": hstripe * 255.0, "limit": limit, "skipped": skipped,
    })
    if skipped or strength == 0.0:
        return image, report

    output = np.empty_like(raw)
    clipped_count = 0
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        corr = _grid_tile(raw, start, stop)
        clipped_count += int(np.count_nonzero(np.abs(corr) > limit))
        source = raw[start:stop, :, :3].astype(np.float32) / 255.0
        cleaned = np.clip(source - strength * np.clip(corr, -limit, limit), 0.0, 1.0)
        output[start:stop, :, :3] = np.rint(cleaned * 255.0).astype(np.uint8)
    if image.mode == "RGBA":
        output[..., 3] = raw[..., 3]
    report["clipped_pct"] = clipped_count * 100.0 / total_values
    result = Image.fromarray(output, mode=image.mode)
    result.info.update(image.info)
    return result, report
