import unittest
from types import SimpleNamespace as NS
from PIL import Image
from khs import core,runtime

class QualityEvaluation(unittest.TestCase):
 def session(self,family='klein',**flags):
  s=runtime.Session.__new__(runtime.Session)
  s.cfg=core.normalize(dict(geometry_match=False,match_sharpness=False,identity_check=False,**flags))
  s.original=Image.new('RGB',(128,128));s.target_pose=None;s.region=None;s.canvas_box=None
  s.owner=NS(analyzer=NS(faces=lambda im:[]),last_report={});s.quality_history=[];s.family=family;s.selected_ref=None
  s.p=NS(extra_generation_params={});s.host=NS(shared=NS(state=NS(textinfo='')))
  return s
 def test_disabled_checks_are_not_evaluated(self):
  s=self.session();s.finish_image(s.original,0);q=s.owner.last_report['quality']
  self.assertIsNone(q['quality_gate_passed']);self.assertEqual(q['quality_gate_status'],'Not Evaluated');self.assertEqual(q['quality_checks'],{})
 def test_enabled_missing_face_fails(self):
  s=self.session();s.cfg['geometry_match']=True;s.finish_image(s.original,0)
  self.assertIs(s.owner.last_report['quality']['quality_gate_passed'],False)
 def test_strict_disabled_rejected_before_model_access(self):
  cfg=core.normalize(dict(quality_strict=True,geometry_match=False,match_sharpness=False))
  with self.assertRaisesRegex(ValueError,'Not Evaluated'):runtime.Session(None,cfg,None,None)
 def test_z_warning_is_scoped_and_metadata_label_correct(self):
  s=self.session('zimage');s.finish_image(s.original,0)
  self.assertIn('Experimental Z-Image',s.host.shared.state.textinfo);self.assertIn('Z-Image quality',s.p.extra_generation_params)
  other=self.session();other.finish_image(other.original,0);self.assertNotIn('profile_warning',other.owner.last_report['quality'])
