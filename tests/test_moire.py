"""CPU-only regression tests for the two-pixel grid filter."""

from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khs.moire import auto_degrid, suppress_nyquist_grid


_B = np.asarray((-1, 6, -15, 20, -15, 6, -1), dtype=np.float64) / 64.0


def _slow_reference(array, strength):
    source = array[..., :3].astype(np.float64)
    height, width = source.shape[:2]
    bx = np.zeros_like(source)
    by = np.zeros_like(source)
    bxy = np.zeros_like(source)
    for y in range(height):
        for x in range(width):
            for k, coefficient in enumerate(_B):
                dx = k - 3
                bx[y, x] += coefficient * source[y, min(max(x + dx, 0), width - 1)]
                by[y, x] += coefficient * source[min(max(y + dx, 0), height - 1), x]
                for j, second in enumerate(_B):
                    dy = j - 3
                    bxy[y, x] += coefficient * second * source[
                        min(max(y + dy, 0), height - 1),
                        min(max(x + dx, 0), width - 1),
                    ]
    filtered = source - bx - by + bxy
    rgb = np.rint(np.clip(source + strength * (filtered - source), 0, 255)).astype(np.uint8)
    if array.shape[2] == 4:
        return np.dstack((rgb, array[..., 3]))
    return rgb


class MoireTests(unittest.TestCase):
    def test_checkerboard_and_stripes_are_reduced(self):
        y, x = np.indices((32, 38))
        for pattern in ((x + y) % 2, x % 2, y % 2):
            values = (96 + pattern * 64).astype(np.uint8)
            source = np.repeat(values[..., None], 3, axis=2)
            result = np.asarray(suppress_nyquist_grid(Image.fromarray(source)))
            self.assertLess(float(result[..., 0].std()), float(source[..., 0].std()) * 0.2)

    def test_flat_field_is_unchanged(self):
        source = np.full((13, 17, 3), 123, dtype=np.uint8)
        result = np.asarray(suppress_nyquist_grid(Image.fromarray(source), 0.73))
        np.testing.assert_array_equal(result, source)

    def test_zero_strength_is_exact_passthrough(self):
        image = Image.fromarray(np.arange(60, dtype=np.uint8).reshape(4, 5, 3))
        self.assertIs(suppress_nyquist_grid(image, 0), image)

    def test_tiny_non_square_rgba_preserves_alpha(self):
        for height, width in ((1, 1), (1, 8), (9, 1), (2, 5)):
            source = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
            image = Image.fromarray(source, "RGBA")
            result = np.asarray(suppress_nyquist_grid(image, 0.6, tile_rows=1))
            self.assertEqual(result.shape, source.shape)
            np.testing.assert_array_equal(result[..., 3], source[..., 3])

    def test_matches_slow_clamped_reference(self):
        rng = np.random.default_rng(42)
        for channels in (3, 4):
            source = rng.integers(0, 256, (5, 8, channels), dtype=np.uint8)
            image = Image.fromarray(source, "RGB" if channels == 3 else "RGBA")
            result = np.asarray(suppress_nyquist_grid(image, 0.37, tile_rows=2))
            np.testing.assert_array_equal(result, _slow_reference(source, 0.37))

    def test_auto_degrid_reduces_weak_noisy_grid(self):
        rng = np.random.default_rng(7)
        y, x = np.indices((128, 130))
        grid = np.where((x + y) % 2, 2.0, -2.0)
        base = 120.0 + x * 0.08 + y * 0.04
        values = np.clip(base + grid + rng.normal(0, 0.35, grid.shape), 0, 255)
        source = np.repeat(np.rint(values).astype(np.uint8)[..., None], 3, axis=2)
        image = Image.fromarray(source)
        result, before = auto_degrid(image, strength=1.0, tile_rows=17)
        _, after = auto_degrid(result, strength=0.0, tile_rows=19)
        self.assertFalse(before["skipped"])
        self.assertLess(after["amp_255"], before["amp_255"] * 0.25)

    def test_auto_degrid_skips_clean_texture_bit_for_bit(self):
        rng = np.random.default_rng(11)
        small = Image.fromarray(rng.integers(0, 256, (47, 48, 3), dtype=np.uint8))
        clean = small.resize((188, 192), Image.Resampling.BICUBIC)
        result, report = auto_degrid(clean, tile_rows=23)
        self.assertIs(result, clean)
        self.assertTrue(report["skipped"])
        self.assertLess(report["amp_255"], 0.5)
        self.assertEqual(
            set(report),
            {"amp_255", "texture_255", "checker_255", "vstripe_255",
             "hstripe_255", "limit", "clipped_pct", "skipped"},
        )

    def test_auto_degrid_preserves_rgba_alpha_and_tiles(self):
        y, x = np.indices((35, 51))
        rgb = np.repeat((126 + ((x + y) % 2) * 4).astype(np.uint8)[..., None], 3, axis=2)
        alpha = ((x * 7 + y * 3) % 256).astype(np.uint8)
        source = np.dstack((rgb, alpha))
        result, report = auto_degrid(Image.fromarray(source, "RGBA"), 0.5, tile_rows=3)
        np.testing.assert_array_equal(np.asarray(result)[..., 3], alpha)
        self.assertFalse(report["skipped"])


if __name__ == "__main__":
    unittest.main()
