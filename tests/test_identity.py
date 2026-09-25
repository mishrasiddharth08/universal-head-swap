"""Offline audit contracts. No recognition weights, Forge or GPU imports."""
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import identity,core

class ComparisonTests(unittest.TestCase):
    def test_cosine_is_scale_invariant(self):
        self.assertAlmostEqual(identity.similarity([2,0],[7,0]),1)
        self.assertAlmostEqual(identity.similarity([1,0],[0,1]),0)
    def test_invalid_descriptors_fail(self):
        for value in ([0,0],[float('nan'),1],[],[float('inf'),1]):
            with self.assertRaises(ValueError): identity.normalized(value)
        with self.assertRaises(ValueError): identity.similarity([1,0],[1])
    def test_single_high_match_does_not_hide_disagreement(self):
        result=identity.compare([1,0],[[1,0],[0,1],[0,1]],0,0.363)
        self.assertEqual(result['best_similarity'],1)
        self.assertFalse(result['identity_above_threshold'])
        self.assertTrue(result['reference_disagreement_slots'])
    def test_consistent_set_passes(self):
        result=identity.compare([1,0],[[1,0],[2,0]],0,0.363)
        self.assertTrue(result['identity_above_threshold'])
        self.assertEqual(result['agreeing_references'],2)
    def test_unusable_selected_reference_never_passes(self):
        result=identity.compare([1,0],[None,[1,0]],0,0.363)
        self.assertIsNone(result['selected_similarity']); self.assertFalse(result['identity_above_threshold'])
    def test_no_references_is_unverified(self):
        with self.assertRaises(ValueError): identity.compare([1,0],[None],0,0.363)
    def test_settings_appended_and_persistable(self):
        self.assertEqual(core.ARG_KEYS[87],'keep_original_canvas')
        self.assertEqual(core.ARG_KEYS[88:91],['identity_check','identity_threshold','geometry_correct'])
        self.assertFalse(core.DEFAULTS['geometry_correct'])
        cfg=core.normalize({'identity_threshold':float('nan'),'identity_check':'false'})
        self.assertEqual(cfg['identity_threshold'],0.363); self.assertFalse(cfg['identity_check'])
        self.assertIn('identity_threshold',core.SAVE_KEYS)
    def test_signed_geometry_is_not_absolute_error(self):
        target={'box':(0,0,100,100)}; generated={'box':(0,0,110,120)}
        report=core.geometry_report(target,generated)
        self.assertEqual(report['head_height_change_percent'],20)
        self.assertEqual(report['head_width_change_percent'],10)
        self.assertEqual(report['head_area_change_percent'],32)
    def test_bad_geometry_fails(self):
        with self.assertRaises(ValueError): core.geometry_report({'box':(0,0,0,1)},{'box':(0,0,1,1)})
    def test_large_translation_refused_to_protect_boundaries(self):
        im=Image.new('RGB',(500,500)); target={'box':(100,100,200,200)}; output={'box':(150,100,250,200)}
        with self.assertRaisesRegex(ValueError,'translation'):core.correct_head_scale(im,target,output)
    def test_large_head_resizing_refused(self):
        im=Image.new('RGB',(500,500)); target={'box':(100,100,200,200)}; output={'box':(90,80,210,220)}
        with self.assertRaisesRegex(ValueError,'seams'):core.correct_head_scale(im,target,output)
    def test_cropped_sample_coordinate_mapping(self):
        im=Image.new('RGB',(2000,1800)); pose={'box':(500,500,900,1000)}
        item=identity.sample(im,pose)
        sx,sy=item.scale; ox,oy=item.offset
        box=item.expected['box']
        self.assertEqual(item.canvas_size,im.size)
        self.assertLessEqual(max(item.image.size),768)
        self.assertAlmostEqual(box[0]/sx+ox,500)
        self.assertAlmostEqual(box[3]/sy+oy,1000)
    def test_sample_detached_from_input(self):
        im=Image.new('RGB',(10,10),'red'); item=identity.sample(im); im.paste('blue',(0,0,10,10))
        self.assertEqual(item.image.getpixel((0,0)),(255,0,0))

class ModelBoundaryTests(unittest.TestCase):
    def test_missing_models_reported_without_import(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{'cv2':None}):
            with self.assertRaisesRegex(ValueError,'missing'): identity.Engine(directory).ready()
    def test_bad_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/next(iter(identity.MODELS))).write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'checksum'): identity.Engine(directory).ready()
    def test_engine_failure_is_explicit(self):
        e=identity.Engine('.')
        with patch.object(e,'describe',side_effect=ValueError('No face')):
            result=e.audit(None,None,[],0,0.363)
        self.assertEqual(result['status'],'Could not verify'); self.assertNotIn('best_similarity',result)
    def test_incomplete_reference_set_is_flagged(self):
        e=identity.Engine('.'); pose={'box':(0,0,100,100),'head_h':100,'head_w':100}
        item=identity.sample(Image.new('RGB',(100,100)))
        with patch.object(e,'describe',side_effect=[(np.array([1,0]),pose),(np.array([1,0]),pose),ValueError('no face'),(np.array([1,0]),pose)]):
            result=e.audit(item,item,[item,item],0,0.363)
        self.assertEqual(result['status'],'Needs review')
    def test_target_geometry_is_scaled_to_output_canvas(self):
        e=identity.Engine('.'); small=identity.sample(Image.new('RGB',(100,100))); large=identity.sample(Image.new('RGB',(200,200)))
        p1={'box':(10,10,50,50),'head_h':40,'head_w':40}; p2={'box':(20,20,100,100),'head_h':80,'head_w':80}
        with patch.object(e,'describe',side_effect=[(np.array([1,0]),p2),(np.array([1,0]),p1),(np.array([1,0]),p1)]):
            result=e.audit(small,large,[small],0,0.363)
        self.assertTrue(result['geometry']['geometry_target_met'])

class WorkerTests(unittest.TestCase):
    def wait_done(self,auditor,timeout=3):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if all(r['status']!='Pending' for r in auditor.snapshot()): return
            time.sleep(0.01)
        self.fail('Audit worker did not finish')
    def test_results_associated_and_snapshot_isolated(self):
        auditor=identity.Auditor('.',engine=NS(audit=lambda *a:{'status':'Checked','warnings':['hello']}))
        try:
            a=auditor.submit(None,None,[],0,.363,{'seed':1})
            b=auditor.submit(None,None,[],1,.363,{'seed':2})
            self.wait_done(auditor); rows=auditor.snapshot()
            self.assertEqual([r['id'] for r in rows],[a,b]); self.assertEqual([r['seed'] for r in rows],[1,2])
            rows[0]['warnings'].append('mutated'); self.assertEqual(len(auditor.snapshot()[0]['warnings']),1)
        finally: auditor.close()
    def test_capacity_and_backpressure_report(self):
        release=threading.Event(); started=threading.Event()
        def slow(*a): started.set(); release.wait(3); return {'status':'Checked'}
        auditor=identity.Auditor('.',engine=NS(audit=slow))
        try:
            auditor.submit(None,None,[],0,.363,{}); self.assertTrue(started.wait(1))
            auditor.submit(None,None,[],0,.363,{})
            auditor.submit(None,None,[],0,.363,{},wait_seconds=0)
            self.assertEqual(len(auditor.futures),2)
            self.assertEqual(auditor.snapshot()[-1]['status'],'Could not verify')
        finally: release.set(); auditor.close()
    def test_worker_errors_release_capacity(self):
        def fail(*a): raise RuntimeError('worker failed')
        auditor=identity.Auditor('.',engine=NS(audit=fail))
        try:
            for _ in range(3): auditor.submit(None,None,[],0,.363,{})
            self.wait_done(auditor)
            self.assertTrue(all(r['status']=='Could not verify' for r in auditor.snapshot()))
        finally: auditor.close()
    def test_close_cancels_queued_work_and_refuses_new(self):
        release=threading.Event(); started=threading.Event()
        def slow(*a): started.set(); release.wait(3); return {'status':'Checked'}
        auditor=identity.Auditor('.',engine=NS(audit=slow))
        try:
            auditor.submit(None,None,[],0,.363,{}); self.assertTrue(started.wait(1))
            auditor.submit(None,None,[],0,.363,{})
            auditor.close(); self.assertEqual(auditor.snapshot()[-1]['status'],'Cancelled')
            self.assertIsNone(auditor.submit(None,None,[],0,.363,{}))
        finally: release.set(); auditor.close()
    def test_history_is_bounded(self):
        auditor=identity.Auditor('.',engine=NS(audit=lambda *a:{'status':'Checked'}),history_limit=2)
        try:
            for i in range(5): auditor.submit(None,None,[],0,.363,{'seed':i})
            self.wait_done(auditor); self.assertEqual([r['seed'] for r in auditor.snapshot()],[3,4])
        finally: auditor.close()

if __name__=='__main__': unittest.main()
