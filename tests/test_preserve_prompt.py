import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core

class PreservePromptRegression(unittest.TestCase):
 def plan(self,text='',**changes):
  cfg={**{'ban_'+key:'Preserve' for key in core.APPEARANCE},'match_sharpness':False,**changes}
  return core.build_plan(text,'',cfg,17,core.choices({}),'bfs_head_v1_flux-klein_9b',host_cfg=1)
 def test_preserve_does_not_prime_absent_ornament_classes(self):
  plan=self.plan('natural portrait')
  for word in ['bindi','earrings','tattoos','piercings','cross']:
   self.assertNotIn(word,plan.positive.lower())
  self.assertIn('existing accessories and markings',plan.positive)
 def test_present_accessories_explicitly_requested_by_user_are_retained(self):
  text='Keep the existing pearl earrings and small forehead mark.'
  plan=self.plan(text)
  self.assertIn(text,plan.positive)
  self.assertIn('except explicitly requested changes',plan.positive)
 def test_explicit_removal_still_wins_over_general_preservation(self):
  plan=self.plan('portrait',ban_earrings='Remove')
  self.assertEqual(plan.appearance['earrings'],'Remove')
  self.assertIn(core.BAN_POS['earrings'],plan.positive)
  self.assertIn('except explicitly requested changes',plan.positive)
 def test_all_remove_policy_does_not_add_preservation_instruction(self):
  plan=self.plan('portrait',**{'ban_'+key:'Remove' for key in core.APPEARANCE})
  self.assertNotIn('preserve existing accessories',plan.positive)
 def test_zimage_uses_source_words_without_picture_reference_claim(self):
  cfg={'ban_'+key:'Preserve' for key in core.APPEARANCE}
  cfg['char_lora_strength']=.8
  plan=core.build_zimage_plan('natural portrait','',cfg,17,core.choices({}),'person-ZIT')
  self.assertIn('the source image',plan.positive)
  self.assertNotIn('Picture 1',plan.positive)
  self.assertNotIn('bindi',plan.positive.lower())
