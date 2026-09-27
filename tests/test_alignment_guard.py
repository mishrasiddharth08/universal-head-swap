import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from khs.core import alignment_improves

class AlignmentGuardTests(unittest.TestCase):
    def test_rejects_wider_displaced_face_even_when_total_error_improves(self):
        before=dict(head_height_error_percent=16.67,head_center_error_px=15.9,
                    head_width_error_percent=.21,chin_error_px=14)
        after=dict(head_height_error_percent=0,head_center_error_px=20,
                   head_width_error_percent=17,chin_error_px=18)
        self.assertFalse(alignment_improves(before,after,180))

    def test_accepts_consistent_improvement(self):
        before=dict(head_height_error_percent=20,head_center_error_px=15,
                    head_width_error_percent=18,chin_error_px=14)
        after=dict(head_height_error_percent=2,head_center_error_px=3,
                   head_width_error_percent=2,chin_error_px=2)
        self.assertTrue(alignment_improves(before,after,180))

    def test_accepts_balanced_width_fix_with_bounded_height_tradeoff(self):
        before=dict(head_height_error_percent=1,head_center_error_px=10,
                    head_width_error_percent=18,chin_error_px=10)
        after=dict(head_height_error_percent=7.8,head_center_error_px=4,
                   head_width_error_percent=8.5,chin_error_px=0)
        self.assertTrue(alignment_improves(before,after,180))

class QualityDefaultsTests(unittest.TestCase):
    def test_default_swap_preserves_scene_and_focuses_head(self):
        from khs.core import normalize
        cfg=normalize({})
        self.assertEqual(cfg['edit_scope'],'Protected head edit')
        self.assertTrue(cfg['qwen_detail_crop'])
