"""Runtime placement and mask-safety tests for optional fine-grid cleanup."""

import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import ANY, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khs import core, runtime


class MoireIntegrationTests(unittest.TestCase):
    def session(self, **settings):
        model = NS(ref_latents=[], ini_latent=None, forge_objects=NS(vae=object()))
        host = NS(
            shared=NS(opts=NS(klein_do_reference=False), state=NS(textinfo="")),
            dynamic=NS(ref_latents=[], is_referencing=False, klein=True, edit=False),
            devices=NS(device="cpu", torch_gc=lambda: None),
            torch=NS(inference_mode=contextlib.nullcontext),
        )
        original = Image.new("RGB", (32, 32), (37, 73, 109))
        p = NS(sd_model=model, init_images=[original], extra_generation_params={})
        owner = NS(analyzer=NS(faces=lambda image: []), last_report={})
        cfg = core.normalize({
            "enable": True,
            "headshots": [Image.new("RGB", (16, 16))],
            "geometry_match": False,
            "match_sharpness": False,
            **settings,
        })
        with patch("khs.runtime.resolve_adapters", return_value=("adapter", "")), \
             patch("khs.runtime.registry", return_value={}):
            session = runtime.Session(p, cfg, owner, host)
        session.target_pose = {"box": (8, 8, 24, 24), "head_w": 16, "head_h": 16}
        return session

    @staticmethod
    def checkerboard(size=(32, 32)):
        y, x = np.indices(size[::-1])
        values = (80 + 96 * ((x + y) % 2)).astype(np.uint8)
        return Image.fromarray(np.repeat(values[..., None], 3, axis=2))

    def test_disabled_is_exact_passthrough(self):
        session = self.session(moire_enabled=False)
        image = self.checkerboard()
        with patch("khs.moire.auto_degrid") as cleanup:
            output = session.finish_image(image, 0)
        cleanup.assert_not_called()
        self.assertEqual(output.tobytes(), image.tobytes())

    def test_clean_auto_noop_is_exact_and_reported(self):
        session = self.session(moire_enabled=True, moire_strength=0.5)
        image = Image.new("RGB", (32, 32), (37, 73, 109))
        report = {"amp_255": 0.1, "skipped": True}
        with patch("khs.moire.auto_degrid", return_value=(image, report)) as cleanup:
            output = session.finish_image(image, 0)
        cleanup.assert_called_once_with(image, 0.5)
        self.assertEqual(output.tobytes(), image.tobytes())
        self.assertEqual(session.owner.last_report["quality"]["fine_grid_cleanup"], report)

    def test_protected_cleanup_runs_before_crop_and_preserves_outside(self):
        session = self.session(moire_enabled=True, moire_strength=1.0)
        mask = Image.new("L", session.original.size, 0)
        mask.paste(255, (8, 8, 24, 24))
        session.region = core.EditRegion(
            session.original.copy(), (8, 8, 24, 24), mask,
            session.original.crop((8, 8, 24, 24)),
        )
        session.canvas_size = (24, 24)
        session.canvas_box = (4, 4, 20, 20)
        decoded = self.checkerboard((24, 24))
        report = {"amp_255": 2.0, "skipped": False}
        with patch("khs.moire.auto_degrid", return_value=(decoded, report)) as cleanup:
            output = np.asarray(session.finish_image(decoded, 0))
        cleanup.assert_called_once_with(decoded, 1.0)
        self.assertEqual(cleanup.call_args.args[0].size, (24, 24))
        self.assertEqual(session.owner.last_report["quality"]["fine_grid_cleanup"], report)
        source = np.asarray(session.original)
        outside = np.asarray(mask) == 0
        np.testing.assert_array_equal(output[outside], source[outside])

    def test_full_frame_cleanup_runs_before_sharpening_and_grain(self):
        session = self.session(
            moire_enabled=True, moire_strength=1.0,
            sharpness=100, grain_amount=0.05,
        )
        session.plans = [NS(seed=7)]
        flat = Image.new("RGB", (32, 32), 128)
        report = {"amp_255": 2.0, "skipped": False}
        with patch("khs.moire.auto_degrid", return_value=(flat, report)) as cleanup:
            output = np.asarray(session.finish_image(self.checkerboard(), 0))
        cleanup.assert_called_once_with(ANY, 1.0)
        self.assertGreater(float(output.std()), 0.0)


if __name__ == "__main__":
    unittest.main()
