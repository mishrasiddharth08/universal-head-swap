import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from PIL import Image
import test_runtime as fixtures
from khs import core, runtime
from khs.external import BFS_QWEN21_TRIGGER, ExternalSession, fit_generated


class ExternalTests(unittest.TestCase):
    def test_qwen_bucket_inverse_mapping_preserves_face_aspect(self):
        from PIL import ImageDraw
        square=Image.new('L',(256,256),0)
        ImageDraw.Draw(square).ellipse((64,64,192,192),fill=255)
        mapped=fit_generated(square.convert('RGB'),(224,256)).convert('L')
        box=mapped.getbbox()
        self.assertEqual(mapped.size,(224,256))
        self.assertLessEqual(abs((box[2]-box[0])-(box[3]-box[1])),2)

    def test_qwen_bucket_inverse_mapping_rejects_anatomy_losing_crop(self):
        with self.assertRaisesRegex(ValueError,'refusing to crop away body or head anatomy'):
            fit_generated(Image.new('RGB',(768,512)),(512,768))

    def test_detail_finish_maps_actual_qwen_bucket_before_runtime_composite(self):
        from PIL import ImageDraw
        s=object.__new__(ExternalSession)
        s.external_canvas_size=None; s.region=object(); s.canvas_size=(224,256)
        square=Image.new('RGB',(256,256),'black')
        ImageDraw.Draw(square).ellipse((64,64,192,192),fill='white')
        with patch('khs.external.runtime.Session.finish_image',side_effect=lambda image,index:image):
            mapped=s.finish_image(square,0).convert('L')
        box=mapped.getbbox()
        self.assertEqual(mapped.size,s.canvas_size)
        self.assertLessEqual(abs((box[2]-box[0])-(box[3]-box[1])),2)

    def session(self, count=13):
        _,host,p,owner,cfg=fixtures.RuntimeTests().fixtures()
        cfg['headshots']=[Image.new('RGB',(256,256),(i*10,100,100)) for i in range(count)]
        owner.custom={}; owner.choices=core.choices({})
        p.extra_generation_params={}; p.prompt='preserve the scene'; p.negative_prompt='blur'
        model=NS(text_processing_engine_qwen=object(),ref_latents=[],ini_latent=None)
        host.dynamic.edit=True
        with patch('khs.runtime.resolve_adapters',return_value=('bfs_head_v1_qwen_2.1','character')),patch('khs.runtime.registry',return_value={}):
            s=ExternalSession(p,cfg,owner,host,model=model)
        return s,p,host

    def test_thirteen_headshots_feed_only_selected_pair_and_restore_request(self):
        s,p,host=self.session()
        original=p.init_images; dimensions=(p.width,p.height); original_dynamic=host.dynamic.ref_latents
        try:
            s.enter()
            plan,refs=s.prepare(0,'<lora:turbo:1> preserve scene','blur',17,1,turbo=True)
            self.assertEqual(len(s.refs),13)
            self.assertEqual(len(refs),2)
            self.assertIn('<image1>',plan.positive); self.assertIn('<image2>',plan.positive)
            self.assertIn(BFS_QWEN21_TRIGGER,plan.positive)
            self.assertIn('start <image1> as base image',plan.positive)
            self.assertIn('head from <image2>',plan.positive)
            self.assertNotIn('(head_swap:',plan.positive)
            self.assertIn('<lora:turbo:1>',plan.positive)
            self.assertIn('<lora:bfs_head_v1_qwen_2.1:',plan.positive)
            self.assertIn('<lora:character:',plan.positive)
            self.assertEqual(plan.cfg,1)
            output=s.finish_image(Image.new('RGB',(512,512)),0)
            self.assertEqual(output.size,original[0].size)
        finally: s.close()
        self.assertIs(p.init_images,original)
        self.assertEqual((p.width,p.height),dimensions)
        self.assertEqual(p.batch_size,2)
        self.assertEqual(host.dynamic.ref_latents,original_dynamic)

    def test_qwen_workload_caps_oversized_requests(self):
        s,p,host=self.session(1)
        p.width=p.height=2048
        s.cfg.update(resolution_dropdown='2048',reference_budget=8,auto_adapt=True)
        try:
            s.enter()
            _,refs=s.prepare(0,'swap','',7,1)
            self.assertLessEqual(p.width*p.height,786432)
            self.assertLessEqual(sum(im.width*im.height for im in refs),1250000)
            self.assertLessEqual(max(max(im.size) for im in refs),768)
        finally:s.close()
        self.assertEqual((p.width,p.height),(2048,2048))

    def test_reference_rotation_and_manual_slot(self):
        s,p,host=self.session()
        s.cfg['pick_mode']='Always rotation'
        try:
            s.enter()
            for i in (0,12,13):
                s.prepare(i,'swap','',i,1)
                self.assertEqual(s.selected_ref,i%13)
            s.cfg['manual_slot']='7'
            s.prepare(14,'swap','',14,1)
            self.assertEqual(s.selected_ref,6)
        finally: s.close()

    def test_exact_preferred_adapters_inside_subfolders(self):
        entries={'Z/bfs_head_v1_qwen_2.1':{},'A/bfs_head_v0_qwen_2.1':{},
                 'Z/bfs_head_v1_flux-klein_9b_step3500_rank128':{},'A/bfs_head_v1_flux-klein_9b':{}}
        self.assertEqual(core.select_adapter(entries,None,'qwen'),'Z/bfs_head_v1_qwen_2.1')
        self.assertEqual(core.select_adapter(entries,9),'Z/bfs_head_v1_flux-klein_9b_step3500_rank128')

    def test_auto_model_overrides_stale_dropdown_but_manual_mode_does_not(self):
        q='folder/bfs_head_v1_qwen_2.1'
        entries={q:NS(metadata={},filename=q+'.safetensors',alias=q)}
        cfg=core.normalize({'lora_dropdown':'bfs_head_v1_flux-klein_9b','auto_model_adapter':True})
        with patch('khs.runtime.registry',return_value=entries):
            self.assertEqual(runtime.resolve_adapters(cfg,NS(),'qwen'),(q,''))
            cfg['auto_model_adapter']=False
            with self.assertRaises(ValueError): runtime.resolve_adapters(cfg,NS(),'qwen')

    def test_zimage_requires_character_lora_and_skips_bfs(self):
        name='characters/alice'
        entries={name:NS(metadata={'ss_base_model_version':'Tongyi-MAI/Z-Image-Turbo'},
                         filename=name+'.safetensors',alias=name)}
        cfg=core.normalize({'char_lora_name':name})
        with patch('khs.runtime.registry',return_value=entries):
            self.assertEqual(runtime.resolve_adapters(cfg,NS(),'zimage'),('',name))
            cfg['char_lora_name']='None (skip)'
            with self.assertRaisesRegex(ValueError,'requires a Z-Image character LoRA'):
                runtime.resolve_adapters(cfg,NS(),'zimage')

    def test_protected_rgba_output_keeps_pixels_outside_mask(self):
        from PIL import ImageDraw
        s,p,host=self.session(1)
        s.cfg['edit_scope']='Protected head edit'
        mask=Image.new('L',s.original.size)
        ImageDraw.Draw(mask).rectangle((90,90,160,160),fill=255)
        s.cfg['custom_mask']=mask
        try:
            s.enter()
            s.prepare(0,'swap','',7,1)
            result=s.finish_image(Image.new('RGBA',s.external_canvas_size,(255,0,0,255)),0)
            self.assertEqual(result.size,s.original.size)
            self.assertEqual(result.getpixel((0,0)),s.original.getpixel((0,0)))
            self.assertGreater(result.getpixel((125,125))[0],200)
        finally: s.close()

    def test_detail_crop_protects_scene_and_restores_dimensions(self):
        from PIL import ImageDraw
        s,p,host=self.session(1)
        original=p.init_images; dimensions=(p.width,p.height)
        s.cfg.update(edit_scope='Protected head edit',qwen_detail_crop=True)
        mask=Image.new('L',s.original.size)
        ImageDraw.Draw(mask).rectangle((90,90,160,160),fill=255)
        s.cfg['custom_mask']=mask
        try:
            s.enter()
            self.assertIsNone(s.external_canvas_size)
            s.prepare(0,'swap','',7,1)
            result=s.finish_image(Image.new('RGB',s.canvas_size,'red'),0)
            self.assertEqual(result.size,s.original.size)
            self.assertEqual(result.getpixel((0,0)),s.original.getpixel((0,0)))
            self.assertGreater(result.getpixel((125,125))[0],200)
        finally:s.close()
        self.assertIs(p.init_images,original)
        self.assertEqual((p.width,p.height),dimensions)

    def test_turbo_tags_are_detected_without_matching_ordinary_text(self):
        self.assertTrue(core.has_speed_lora('<lora:folder/klein_9B_Turbo_r128:1>'))
        self.assertFalse(core.has_speed_lora('a turbo sports car <lora:character:0.7>'))

if __name__=='__main__': unittest.main()
