"""CPU-only regression tests. Run after training: python -m unittest discover -s tests -v."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core
from khs.vision import FaceAnalyzer

class PromptTests(unittest.TestCase):
    def plan(self,text='portrait',**cfg):
        return core.build_plan(text,'user negative',core.normalize(cfg),42,core.choices({}),
                               'bfs_head_v1_flux-klein_9b',host_cfg=1.0)
    def test_long_prompt_and_unicode_preserved(self):
        text=('A garden portrait with natural detail. '*100)+' UNIQUE_END नमस्ते'
        p=self.plan(text)
        self.assertIn(text,p.positive)
    def test_unrelated_lora_retained_and_owned_replaced(self):
        p=self.plan('portrait <lora:my_style:0.8> <lora:bfs_head_v1_flux-klein_9b:0.2>')
        self.assertIn('<lora:my_style:0.8>',p.positive)
        self.assertEqual(p.positive.count('<lora:bfs_head_v1_flux-klein_9b:'),1)
        self.assertIn(':1.00>',p.positive)
    def test_owned_alias_replaced(self):
        p=core.build_plan('<lora:face_alias:0.2> portrait','',{},1,core.choices({}),
                          'bfs_head_v1_flux-klein_9b',owned_aliases=['face_alias'])
        self.assertNotIn('face_alias',p.positive)
    def test_zero_disables_swap_wording(self):
        self.assertNotIn('head_swap:',self.plan(blend_slider=0).positive)
    def test_studio_does_not_lift_earrings(self):
        self.assertEqual(self.plan('studio portrait').appearance['earrings'],'Remove')
    def test_cross_substrings_do_not_lift(self):
        for phrase in ('walk across a room','cross-legged pose','crossbody bag'):
            self.assertFalse(core.affirmative_mention(phrase,'cross'))
    def test_negated_earrings_do_not_lift(self):
        for phrase in ('no earrings','without earrings','remove the earrings','earrings-free'):
            self.assertFalse(core.affirmative_mention(phrase,'earrings'))
    def test_explicit_earrings_lift_only_earrings(self):
        p=self.plan('wearing gold hoop earrings',ban_channel='Positive + Negative',removal_priority=False,ban_piercings='Use prompt / preset')
        self.assertEqual(p.appearance['earrings'],'Use prompt / preset')
        self.assertEqual(p.appearance['tattoos'],'Remove')
        self.assertNotIn('ear jewelry',p.negative)
        self.assertIn('tattoos',p.negative)
    def test_removal_priority_blocks_conflicting_user_and_preset(self):
        p=self.plan('wearing tattoos and earrings',earrings=['gold earrings'])
        self.assertEqual(p.appearance['earrings'],'Remove')
        self.assertEqual(p.appearance['tattoos'],'Remove')
        self.assertIn('prioritize all removal requirements',p.positive)
        self.assertTrue(any('Omitted earrings' in n for n in p.notes))
    def test_full_body_vocabulary_in_positive_and_negative(self):
        p=self.plan(ban_channel='Positive + Negative')
        for word in ('tikka','henna','navel','chest','dermal','legs','feet'):
            self.assertIn(word,p.positive)
        for word in ('nipple piercing','henna tattoo','sindoor in hair parting','dermal piercing','anklets'):
            self.assertIn(word,p.negative)
    def test_protected_reports_body_cleanup_limit(self):
        self.assertTrue(any('only cleans inside' in n for n in self.plan(edit_scope='Protected head edit').notes))
    def test_jewelry_removal_wins_over_earrings_preserve(self):
        p=self.plan(ban_earrings='Preserve')
        self.assertEqual(p.appearance['earrings'],'Remove')
    def test_multiselect_without_random_preserves_all_phrases(self):
        p=self.plan(hair_color=['brown','balayage highlights'])
        self.assertIn('brown hair, balayage highlights hair',p.positive)
    def test_separate_negated_and_affirmative_clauses(self):
        self.assertTrue(core.affirmative_mention('no tattoos, wearing pearl earrings','earrings'))
    def test_fast_mode_wins_over_legacy_cfg_checkbox(self):
        p=self.plan(neg_cfg_boost=True,block_tattoos=True,ban_channel='Positive-only (fast, CFG 1.0)')
        self.assertEqual(p.cfg,1.0); self.assertEqual(p.negative,'user negative')
        self.assertFalse(p.report()['negative_guidance_active'])
        self.assertTrue(any('inactive' in note for note in p.notes))
    def test_fresh_cleanup_defaults_activate_negative_guidance(self):
        p=self.plan()
        self.assertEqual(p.cfg,1.1); self.assertTrue(p.report()['negative_guidance_active'])
        self.assertIn('tattoos',p.negative)
    def test_dual_mode_keeps_base_cfg(self):
        p=core.build_plan('portrait','',{'ban_channel':'Positive + Negative'},1,core.choices({}),
                          'bfs_head_v1_flux-klein_9b',host_cfg=4)
        self.assertEqual(p.cfg,4)
    def test_ratio_off(self): self.assertNotIn('original head scale',self.plan(ratio_lock=False).positive)
    def test_boost_effective_value(self):
        p=self.plan(blend_lora_boost=True,blend_slider=100)
        self.assertEqual(p.fs_strength,1.25); self.assertIn(':1.25>',p.positive)
    def test_boost_bounded(self):
        p=self.plan(lora_strength=2,blend_lora_boost=True,blend_slider=100)
        self.assertEqual(p.fs_strength,2)
    def test_random_choice_deterministic_and_no_global_rng_effect(self):
        cfg={'hair_color':['brown','blonde','black'],'rand_hair_color':True}
        state=random.getstate(); a=self.plan(**cfg); b=self.plan(**cfg)
        self.assertEqual(a,b); self.assertEqual(random.getstate(),state)
    def test_custom_phrase_is_not_summarized(self):
        phrase='Retain the exact hairstyle from the body reference image with soft balayage'
        self.assertIn(phrase,self.plan(hairstyle=[phrase]).positive)
    def test_expression_slider_affects_prompt(self):
        self.assertIn('(subtle smile:1.50)',self.plan(expression=['subtle smile'],expression_strength=1.5).positive)
    def test_custom_earring_choices_loaded(self):
        self.assertIn('pearl earrings',core.choices({'earrings':['pearl earrings']})['earrings'])
    def test_token_counter_same_text_as_generation(self):
        captured=[]
        def count(text): captured.append(text); return 999
        p=core.build_plan('original ending','',{},42,core.choices({}),'bfs_head_v1_flux-klein_9b',token_counter=count)
        self.assertEqual(captured[0].strip(),p.positive); self.assertEqual(p.token_count,999)
        self.assertIn('original ending',p.positive)

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.im=Image.new('RGB',(600,800),'black')
        ImageDraw.Draw(self.im).rectangle((180,100,420,599),fill='red')
        self.pose=dict(box=(180,100,420,600),head_px=500,head_h=500,head_w=240,head_ratio=0.625,yaw=0,pitch=0,roll=0)
    def test_padding_centers_source(self):
        im=core.prepare_reference(self.im,self.pose,{'head_ratio':0.25},'Match target framing (experimental)')
        y,x=np.where(np.asarray(im)[:,:,0]>200)
        self.assertGreater(x.min(),180); self.assertGreater(y.min(),100)
    def test_face_metrics_actual_box(self):
        im=Image.new('RGB',(1000,1000),'black'); ImageDraw.Draw(im).rectangle((750,40,949,239),fill='white')
        face=core.face_crop(im,{'box':(750,40,950,240)},padding=0)
        self.assertAlmostEqual(core.metrics(face)['bright'],1.0)
    def test_top_cluster_rejects_bad_candidates(self):
        items=[{'index':i,'score':s,'usable':True} for i,s in enumerate([0.95,0.2,0.1])]
        self.assertEqual(len(core.top_cluster(items)),1)
    def test_fallback_pool_single_when_no_detection(self):
        items=[{'index':i,'score':s,'usable':False} for i,s in enumerate([0.7,0.6])]
        self.assertEqual(len(core.top_cluster(items)),1)
    def test_manual_slot_missing_fails(self):
        with self.assertRaises(ValueError): core.select_reference([],core.normalize({'manual_slot':'9'}),0,2)
    def test_memory_cap_is_final(self):
        cap,budget=core.memory_limits(16*1024**3,10*1024**3,2048,8)
        self.assertLessEqual(cap,1280); self.assertLessEqual(budget,5)
    def test_both_inputs_fit_pixel_budget_and_alignment(self):
        a=Image.new('RGB',(3000,5000)); b=Image.new('RGB',(4000,2200))
        pair=core.prepare_pair(a,b,1024,1.5)
        self.assertLessEqual(sum(im.width*im.height for im in pair),1_500_000)
        for im in pair:
            self.assertLessEqual(max(im.size),1024); self.assertEqual(im.width%64,0); self.assertEqual(im.height%64,0)
    def test_rgb_alpha_and_palette(self):
        rgba=Image.new('RGBA',(20,20),(0,0,0,0))
        self.assertEqual(core.rgb_image(rgba).getpixel((0,0)),(255,255,255))
        self.assertEqual(core.rgb_image(self.im.convert('P')).mode,'RGB')
    def test_invalid_gallery_reports_slot(self):
        with self.assertRaisesRegex(ValueError,'slot 2'): core.gallery_images([self.im,None])
    def test_excess_uploads_rejected_instead_of_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'at most 20'): core.gallery_images([self.im]*21)
    def test_protected_composite_preserves_zero_mask_pixels(self):
        region=core.build_region(self.im,self.pose,0.3,0.05)
        empty=core.build_region(self.im,self.pose,0.3,0.05,{'background':None,'layers':[],'composite':None})
        self.assertEqual(region.box,empty.box)
        self.assertEqual(region.mask.tobytes(),empty.mask.tobytes())
        result=core.composite_region(Image.new('RGB',(1024,1024),'green'),region)
        original=np.asarray(self.im); actual=np.asarray(result); outside=np.asarray(region.mask)==0
        self.assertTrue(outside.any()); np.testing.assert_array_equal(original[outside],actual[outside])
        self.assertEqual(result.size,self.im.size)
    def test_skin_only_boundary_match_reduces_jaw_neck_color_seam(self):
        original=Image.new('RGB',(100,100),(220,180,160))
        mask=Image.new('L',(100,100),0)
        ImageDraw.Draw(mask).ellipse((15,5,85,85),fill=255)
        mask=mask.filter(ImageFilter.GaussianBlur(8))
        region=core.EditRegion(original,(0,0,100,100),mask,original)
        generated=Image.new('RGB',(100,100),(240,140,100))
        ImageDraw.Draw(generated).rectangle((35,10,65,25),fill=(12,12,12))
        before=np.abs(np.asarray(generated,dtype=float)[60,50]-np.asarray(original,dtype=float)[60,50]).sum()
        result=core.composite_region(generated,region,.35)
        uncorrected=core.composite_region(generated,region,0)
        after=np.abs(np.asarray(result,dtype=float)[60,50]-np.asarray(original,dtype=float)[60,50]).sum()
        self.assertLess(after,before)
        # Continuous alpha blending may shift non-skin pixels by at most 1 level.
        self.assertLessEqual(sum(abs(a-b) for a,b in zip(result.getpixel((50,15)),uncorrected.getpixel((50,15)))),2)
        self.assertEqual(result.getpixel((0,0)),original.getpixel((0,0)))
    def test_automatic_mask_preserves_original_neck_and_shoulders(self):
        region=core.build_region(self.im,self.pose,0.3,0.08)
        generated=Image.new('RGB',region.crop.size,'green')
        result=core.composite_region(generated,region)
        x0,y0,x1,y1=self.pose['box'];hh=y1-y0
        neck_y=min(self.im.height-1,int(round(y1+hh*0.14)))
        original=np.asarray(self.im);actual=np.asarray(result)
        np.testing.assert_array_equal(actual[neck_y:],original[neck_y:])
        self.assertEqual(region.mask.getpixel(((x0+x1)//2,neck_y)),0)

    def test_face_contour_mode_preserves_side_hair_and_neck(self):
        broad=core.build_region(self.im,self.pose,0.3,0.08)
        contour=core.build_region(self.im,self.pose,0.3,0.08,preserve_hair=True)
        x0,y0,x1,y1=self.pose['box']; hw=x1-x0; hh=y1-y0
        side=(round(x0-hw*.24),round(y0+hh*.45))
        self.assertGreater(broad.mask.getpixel(side),0)
        self.assertLessEqual(contour.mask.getpixel(side),8)
        self.assertGreater(contour.mask.getpixel(((x0+x1)//2,(y0+y1)//2)),240)
        neck_y=min(self.im.height-1,round(y1+hh*.14))
        self.assertEqual(contour.mask.getpixel(((x0+x1)//2,neck_y)),0)

    def test_custom_mask_can_still_edit_neck_when_requested(self):
        mask=Image.new('L',self.im.size,0)
        x0,y0,x1,y1=self.pose['box'];hh=y1-y0
        neck_y=min(self.im.height-2,int(round(y1+hh*0.2)))
        ImageDraw.Draw(mask).rectangle((x0,neck_y,x1,neck_y+1),fill=255)
        region=core.build_region(self.im,self.pose,0,0,mask)
        self.assertEqual(region.mask.getpixel(((x0+x1)//2,neck_y)),255)

    def test_identity_recipe_preserves_user_inputs_and_resets_styles(self):
        recipe=core.identity_setup()
        self.assertTrue(set(recipe).issubset(core.ARG_KEYS))
        for key in ('headshots','lora_dropdown','char_lora_name','char_lora_trigger'):
            self.assertNotIn(key,recipe)
        self.assertEqual(recipe['edit_scope'],'Protected head edit')
        self.assertTrue(recipe['qwen_detail_crop'])
        self.assertTrue(all(recipe[key]==[] for key in core.CATEGORIES))

    def test_reference_pool_has_bounded_resolution(self):
        source=Image.new('RGB',(3000,2000),'blue')
        refs,labels=core.gallery_images([source],max_side=1536)
        self.assertLessEqual(max(refs[0].size),1536)
        self.assertEqual(source.size,(3000,2000))

    def test_protected_alignment_corrects_oversized_head(self):
        from PIL import ImageDraw
        baseline=Image.new('RGB',(200,200),'blue')
        generated=Image.new('RGB',(200,200),'green')
        ImageDraw.Draw(generated).rectangle((55,40,145,160),fill='red')
        target={'box':(70,60,130,140)}
        source={'box':(55,40,145,160)}
        corrected,report=core.align_protected_head(generated,baseline,target,source)
        self.assertAlmostEqual(report['scale_factor'],2/3,places=3)
        self.assertEqual(corrected.getpixel((100,100)),(255,0,0))
        self.assertEqual(corrected.getpixel((0,0)),(0,0,255))
        with self.assertRaises(ValueError):
            core.align_protected_head(generated,baseline,target,{'box':(0,0,199,199)})

    def test_zimage_alignment_preserves_facial_aspect(self):
        image=Image.new('RGB',(320,320),'green')
        target={'box':(100,70,200,230)}
        source={'box':(99.38,94.544,200.62,230)}
        _,report=core.align_protected_head(image,image,target,source,preserve_aspect=True)
        self.assertEqual(report['scale_x'],report['scale_y'])
        self.assertEqual(report['anchor'],'source chin')
        self.assertLess(report['scale_y'],1.10)

    def test_protected_alignment_restores_short_wide_face_and_chin(self):
        from PIL import ImageDraw
        baseline=Image.new('RGB',(320,320),'blue')
        generated=Image.new('RGB',(320,320),'green')
        # Mirrors the supplied failures: about 21% short and 9% wide.
        target={'box':(100,70,210,231)}
        source={'box':(95,104,214,231)}
        ImageDraw.Draw(generated).rectangle(source['box'],fill='red')
        corrected,report=core.align_protected_head(generated,baseline,target,source)
        self.assertEqual(report['mode'],'protected crop proportion alignment')
        self.assertAlmostEqual(report['scale_x'],110/119,places=3)
        self.assertAlmostEqual(report['scale_y'],161/127,places=3)
        self.assertEqual(report['anchor'],'source chin')
        self.assertEqual(corrected.getpixel((155,75)),(255,0,0))
        self.assertEqual(corrected.getpixel((0,0)),(0,0,255))

    def test_protected_alignment_caps_second_supplied_failure_safely(self):
        image=Image.new('RGB',(360,360),'green')
        target={'box':(100,70,210,231)}
        # 23.85% short, matching 02035 metadata: requested 1.313x.
        source={'box':(97,108.4,215,231)}
        _,report=core.align_protected_head(image,image,target,source)
        self.assertAlmostEqual(report['requested_scale_y'],161/122.6,places=3)
        self.assertEqual(report['scale_y'],1.3)
        self.assertLess(abs(122.6*report['scale_y']/161-1),0.02)

    def test_protected_alignment_refuses_extreme_aspect_warp(self):
        image=Image.new('RGB',(320,320),'green')
        target={'box':(100,70,210,231)}
        with self.assertRaisesRegex(ValueError,'proportions differ too much'):
            core.align_protected_head(image,image,target,{'box':(80,160,230,231)})

    def test_blank_editor_canvas_uses_detected_head(self):
        expected=core.build_region(self.im,self.pose)
        for canvas in (Image.new('RGB',(800,600),'black'),Image.new('RGBA',(800,600),(255,255,255,0))):
            with self.subTest(mode=canvas.mode):
                payload={'background':canvas,'composite':canvas,'layers':[]}
                actual=core.build_region(self.im,self.pose,mask=payload)
                self.assertEqual(actual.box,expected.box)
                self.assertEqual(actual.mask.tobytes(),expected.mask.tobytes())
                with self.assertRaisesRegex(ValueError,'detected target face'):
                    core.build_region(self.im,None,mask=payload)

    def test_painted_editor_mask_keeps_custom_area(self):
        canvas=Image.new('L',self.im.size,0)
        canvas.paste(255,(20,30,80,100))
        expected=core.build_region(self.im,None,mask=canvas)
        actual=core.build_region(self.im,None,mask={'background':None,'composite':canvas,'layers':[]})
        self.assertEqual(actual.box,expected.box)
        self.assertEqual(actual.mask.tobytes(),expected.mask.tobytes())

    def test_custom_empty_mask_refused_and_wrong_size_mask_resized(self):
        with self.assertRaises(ValueError): core.build_region(self.im,self.pose,mask=Image.new('L',self.im.size))
        region=core.build_region(self.im,self.pose,mask=Image.new('L',(4,4),'white'))
        self.assertEqual(region.mask.size,self.im.size)
    def test_protected_requires_detection_or_mask(self):
        with self.assertRaises(ValueError): core.build_region(self.im,None)
    def test_canvas_inverse_mapping_retains_dimensions(self):
        canvas,box=core.fit_canvas(self.im,(1024,1024))
        self.assertEqual(canvas.size,(1024,1024))
        self.assertAlmostEqual((box[2]-box[0])/(box[3]-box[1]),600/800,places=2)
    def test_quality_association_is_nearest_not_largest(self):
        other=dict(self.pose,box=(0,0,90,90))
        self.assertIs(core.nearest_face([other,self.pose],self.pose,self.im.size),self.pose)
    def test_quality_association_refuses_distant_person(self):
        pose=dict(self.pose,box=(10,10,30,30))
        other=dict(self.pose,box=(500,700,550,750))
        self.assertIsNone(core.nearest_face([other],pose,self.im.size))
    def test_scale_pose_crop_coordinates(self):
        pose=core.scale_pose(self.pose,(400,600),(800,1200),(100,50))
        self.assertEqual(pose['box'],(160,100,640,1100))
    def test_geometry_metric_reports_real_mismatch(self):
        other=dict(self.pose,box=(180,100,420,650))
        report=core.geometry_report(self.pose,other)
        self.assertAlmostEqual(report['head_height_error_percent'],10)
        self.assertFalse(report['geometry_target_met'])

    def test_geometry_gate_rejects_wide_face_and_chin_drift(self):
        target=dict(self.pose,box=(160,100,400,600))
        wide=dict(self.pose,box=(145,100,415,600))
        report=core.geometry_report(target,wide)
        self.assertGreater(report['head_width_error_percent'],10)
        self.assertFalse(report['geometry_target_met'])
        low=dict(self.pose,box=(160,145,400,645))
        report=core.geometry_report(target,low)
        self.assertGreater(report['chin_error_px'],10)
        self.assertFalse(report['geometry_target_met'])
    def test_geometry_rejects_extreme_scale(self):
        other=dict(self.pose,box=(180,100,200,140))
        with self.assertRaises(ValueError): core.correct_head_scale(self.im,self.pose,other)
    def test_sharpness_no_unnecessary_resampling(self):
        result,report=core.match_local_sharpness(self.im,self.im,self.pose,self.pose)
        self.assertEqual(report['automatic_unsharp_percent'],0)
        np.testing.assert_array_equal(np.asarray(result),np.asarray(self.im))
    def test_sharpness_correction_is_uniform_not_a_head_ring(self):
        # A head-shaped sharpening mask left a visible halo at the hairline and neck.
        rng=np.random.default_rng(7)
        detailed=Image.fromarray(rng.integers(0,255,(800,600,3),dtype=np.uint8))
        soft=detailed.filter(ImageFilter.GaussianBlur(2))
        result,report=core.match_local_sharpness(soft,detailed,self.pose,self.pose)
        percent=report['automatic_unsharp_percent']
        self.assertGreater(percent,0)
        expected=soft.filter(ImageFilter.UnsharpMask(radius=1.0,percent=percent,threshold=3))
        np.testing.assert_array_equal(np.asarray(result),np.asarray(expected))
        corner=lambda im:np.asarray(im.crop((0,0,24,24)),dtype=np.int16)
        self.assertGreater(int(np.abs(corner(result)-corner(soft)).max()),0,
                           'pixels outside the head must receive the same correction')
    def test_rotation_matrix_identity_and_scale(self):
        self.assertEqual(FaceAnalyzer.matrix_angles(np.eye(4)),(0,0,0))
        matrix=np.eye(4); matrix[:3,:3]*=3
        np.testing.assert_allclose(FaceAnalyzer.matrix_angles(matrix),(0,0,0))

class StorageTests(unittest.TestCase):
    def test_legacy_migration_and_full_preset_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'presets.json'
            path.write_text(json.dumps({'old':{'ban_earrings':False,'resolution_dropdown':'1024 (default)','latent_kernel_size':'99x99 (maximum)'}}))
            old=core.load_presets(path)['old']; self.assertEqual(old['ban_earrings'],'Preserve')
            self.assertEqual(old['resolution_dropdown'],'1024'); self.assertEqual(old['latent_kernel_size'],'15x15')
            core.save_preset(path,'new',{'char_lora_name':'my_character','char_lora_strength':0.8,'char_lora_trigger':'personx'})
            loaded=core.load_presets(path); self.assertIn('old',loaded)
            self.assertEqual(loaded['new']['char_lora_trigger'],'personx')
            self.assertEqual(json.loads(path.read_text())['schema_version'],2)
    def test_new_preset_does_not_change_reference_selection_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'presets.json'
            core.save_preset(path,'best',{})
            self.assertEqual(core.load_presets(path)['best']['pick_mode'],'Best match (smart, no rotation)')
    def test_custom_negative_presets_validate_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'custom.json'; path.write_text('{"negative_presets": ["wrong"]}')
            with self.assertRaises(ValueError): core.load_custom(path)
    def test_atomic_save_failure_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'presets.json'; path.write_text('{"old":{}}')
            with patch('khs.core.os.replace',side_effect=OSError('synthetic failure')):
                with self.assertRaises(OSError): core.atomic_json(path,{'new':{}})
            self.assertEqual(path.read_text(),'{"old":{}}'); self.assertEqual(len(list(Path(tmp).iterdir())),1)
    def test_cache_eviction_and_replacement(self):
        cache=core.BoundedCache(10); cache.put('a',1,6); cache.put('b',2,6)
        self.assertIsNone(cache.get('a')); self.assertEqual(cache.get('b'),2); self.assertEqual(cache.bytes,6)
        cache.put('b',3,4); self.assertEqual(cache.bytes,4)
        cache.clear(); self.assertEqual(cache.bytes,0)
    def test_model_size_matching(self):
        entries={'bfs_head_v1_flux-klein_4b':{},'bfs_head_v1_flux-klein_9b':{}}
        self.assertTrue(core.select_adapter(entries,4).endswith('4b'))
        self.assertTrue(core.select_adapter(entries,9).endswith('9b'))
        with self.assertRaises(ValueError): core.compatible_adapter('bfs_head_v1_flux-klein_9b',4)
        with self.assertRaises(ValueError): core.compatible_adapter('bfs_head_v5_qwen_image_edit',9)
        with self.assertRaises(ValueError): core.select_adapter(entries,None)
        with self.assertRaises(ValueError): core.compatible_adapter('flux2_unknown_adapter',9,strict=True)

    def test_dual_model_adapter_families(self):
        self.assertEqual(core.adapter_family('bfs_head_v1_flux-klein_9b')[0],'klein')
        self.assertEqual(core.adapter_family('bfs_head_v1_flux-klein_9b')[1],9)
        self.assertEqual(core.adapter_family('bfs_head_v5_qwen_image_edit')[0],'qwen')
        self.assertEqual(core.adapter_family('bfs_head_v6_qwen_image_edit_2.1')[0],'qwen')
        self.assertEqual(core.adapter_family('some_sdxl_lora')[0],'other')
        self.assertEqual(core.adapter_family('flux2_unknown_adapter')[0],'unknown')
        # Family-aware compatibility: a Klein adapter never matches a Qwen trunk.
        with self.assertRaises(ValueError): core.compatible_adapter('bfs_head_v1_flux-klein_9b',None,family='qwen')
        with self.assertRaises(ValueError): core.compatible_adapter('bfs_head_v5_qwen_image_edit',None,family='klein')
        core.compatible_adapter('bfs_head_v5_qwen_image_edit',None,family='qwen')
        core.compatible_adapter('bfs_head_v1_flux-klein_9b',9,family='klein')

    def test_qwen_auto_adapter_selection(self):
        entries={'bfs_head_v5_qwen_image_edit':{},'bfs_head_v1_flux-klein_9b':{}}
        self.assertEqual(core.select_adapter(entries,None,family='qwen'),'bfs_head_v5_qwen_image_edit')
        with self.assertRaises(ValueError): core.select_adapter(entries,None,family='klein')

    def test_zimage_character_lora_plan_has_no_bfs_or_picture_reference(self):
        cfg={'zimage_variant':'Turbo','char_lora_trigger':'person_z'}
        plan=core.build_zimage_plan('portrait','avoid blur',cfg,1,core.choices({}),'characters/person_z',host_cfg=7.0)
        self.assertEqual(plan.cfg,1.0)
        self.assertIn('<lora:characters/person_z:0.70>',plan.positive)
        self.assertNotIn('bfs_head',plan.positive.lower())
        self.assertNotIn('Picture ',plan.positive)
        self.assertTrue(any('reference headshots are not used' in note for note in plan.notes))

    def test_zimage_plan_requires_character_lora_and_base_keeps_cfg(self):
        with self.assertRaisesRegex(ValueError,'requires a character LoRA'):
            core.build_plan('', '', {},1,core.choices({}),'',family='zimage')
        plan=core.build_plan('', '', {'zimage_variant':'Base'},1,core.choices({}),'','person',
                             host_cfg=4.5,family='zimage')
        self.assertEqual(plan.cfg,4.5)

    def test_zimage_rejects_stale_bfs_prompt_tag(self):
        with self.assertRaisesRegex(ValueError,'does not match Z-Image'):
            core.build_zimage_plan('portrait <lora:bfs_head_v1_qwen_2.1:1>', '', {},1,
                                   core.choices({}),'person-ZIT')

    def test_zimage_preserves_compatible_and_unknown_prompt_tags(self):
        text='portrait <lora:person-ZIT:0.8> <lora:cinematic_style:0.4> <lora:turbo_detail:0.2>'
        plan=core.build_zimage_plan(text,'',{},1,core.choices({}),'character-ZIT')
        self.assertIn('<lora:person-ZIT:0.8>',plan.positive)
        self.assertIn('<lora:cinematic_style:0.4>',plan.positive)
        self.assertIn('<lora:turbo_detail:0.2>',plan.positive)

    def test_zimage_remove_uses_descriptive_clean_anatomy_without_object_names(self):
        plan=core.build_zimage_plan('', '', {},1,core.choices({}),'character-ZIT')
        prompt=plan.positive.lower()
        for term in ('bindi','sindoor','tikka','tilak','kumkum','earring','jewelry','tattoo',
                     'henna','piercing','crucifix','cross symbol','nose stud','nose pin'):
            self.assertNotIn(term,prompt)
        for phrase in ('plain unmarked forehead','bare natural earlobes','clear natural visible skin',
                       'one natural face with coherent eyes'):
            self.assertIn(phrase,prompt)

    def test_zimage_adapter_family_and_settings(self):
        self.assertEqual(core.adapter_family('people/alice',{'ss_base_model_version':'Tongyi-MAI/Z-Image-Turbo'}),('zimage',None))
        with self.assertRaises(ValueError):
            core.compatible_adapter('bfs_head_v5_qwen_image_edit',None,family='zimage')
        cfg=core.normalize({'zimage_variant':'bad','zimage_denoise':5,'zimage_mask_source':'bad'})
        self.assertEqual((cfg['zimage_variant'],cfg['zimage_denoise'],cfg['zimage_mask_source']),
                         ('Turbo',1.0,'SAM3 (recommended)'))

    def test_qwen_picture_remap(self):
        self.assertEqual(core.remap_picture_refs('use Picture 1 as the target; use Picture 2 identity',-1),
                         'use Picture 0 as the target; use Picture 1 identity')
        self.assertEqual(core.remap_picture_refs('no references here',-1),'no references here')
        self.assertEqual(core.remap_picture_refs('Picture 1 target',0),'Picture 1 target')

if __name__=='__main__': unittest.main()
