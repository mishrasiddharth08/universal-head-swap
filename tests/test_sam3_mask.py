import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from khs.sam3_mask import SAM3Masker


class FakeProcessor:
    def __init__(self):
        self.calls = 0
    def set_confidence_threshold(self, value):
        self.threshold = value
    def set_image(self, image):
        self.calls += 1
        return {"image": image}
    def set_text_prompt(self, prompt, state):
        masks = np.zeros((2, 1, 8, 8), dtype=bool)
        masks[0, 0, :3, :3] = True
        masks[1, 0, 3:7, 3:7] = True
        return {"masks": masks, "boxes": np.array([[0, 0, 3, 3], [3, 3, 7, 7]]),
                "scores": np.array([.9, .6])}


class SAM3MaskTests(unittest.TestCase):
    def test_is_lazy_selects_requested_face_and_caches_cpu_mask(self):
        made = []
        def loader():
            made.append(True)
            return NS(), FakeProcessor()
        masker = SAM3Masker(None, loader=loader, cache_bytes=1024)
        image = Image.new("RGB", (8, 8))
        self.assertEqual(made, [])
        first = masker.mask(image, face_box=(3, 3, 7, 7))
        first.putpixel((4, 4), 0)
        second = masker.mask(image, face_box=(3, 3, 7, 7))
        self.assertEqual(len(made), 1)
        self.assertEqual(second.getpixel((4, 4)), 255)
        self.assertEqual(masker.processor.calls, 1)
        self.assertLessEqual(masker.cache.bytes, masker.cache.max_bytes)

    def test_rejects_unrelated_detection_for_requested_face(self):
        masker = SAM3Masker(None, loader=lambda: (NS(), FakeProcessor()))
        self.assertIsNone(masker.mask(Image.new("RGB", (8, 8)), face_box=(20, 20, 30, 30)))

    def test_native_loader_passes_cpu_to_model_and_processor(self):
        model = NS(eval=Mock())
        builder = Mock(return_value=model)
        processor_class = Mock(return_value=FakeProcessor())
        modules = {
            "sam3": NS(),
            "sam3.model_builder": NS(build_sam3_image_model=builder),
            "sam3.model": NS(),
            "sam3.model.sam3_image_processor": NS(Sam3Processor=processor_class),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.touch()
            masker = SAM3Masker(checkpoint, device="cpu")
            with patch.dict(sys.modules, modules):
                masker._load()
        self.assertEqual(builder.call_args.kwargs["device"], "cpu")
        processor_class.assert_called_once_with(model, device="cpu")
        model.eval.assert_called_once()

    def test_missing_checkpoint_fails_closed_without_download(self):
        masker = SAM3Masker("missing-sam3.pt")
        self.assertIsNone(masker.mask(Image.new("RGB", (4, 4))))
        self.assertIn("automatic downloads are disabled", masker.error)

    def test_release_drops_model_but_keeps_bounded_cpu_cache(self):
        masker = SAM3Masker(None, loader=lambda: (NS(), FakeProcessor()), cache_bytes=1024)
        image = Image.new("RGB", (8, 8))
        masker.mask(image)
        masker.release()
        self.assertIsNone(masker.processor)
        self.assertIsNone(masker.model)
        self.assertIsNotNone(masker.mask(image))
        self.assertIsNone(masker.processor)

    def test_inference_failure_releases_loaded_model(self):
        processor = FakeProcessor()
        processor.set_image = Mock(side_effect=RuntimeError("inference failed"))
        masker = SAM3Masker(None, loader=lambda: (NS(), processor))
        self.assertIsNone(masker.mask(Image.new("RGB", (8, 8))))
        self.assertIn("inference failed", masker.error)
        self.assertIsNone(masker.processor)
        self.assertIsNone(masker.model)


if __name__ == "__main__":
    unittest.main()
