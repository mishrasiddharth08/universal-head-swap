"""Native Z-Image Session lifecycle tests with small Forge stand-ins."""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core,runtime


class ZImageRuntimeTests(unittest.TestCase):
    def fixtures(self,headshots=None,face=True):
        ZImage=type('ZImage',(),{})
        model=ZImage()
        model.model_config=NS(unet_config={'z_image_modulation':True})
        model.ref_latents=['old-ref']; model.ini_latent='old-ini'
        model.text_processing_engine_gemma=NS(tokenize=lambda texts:[[1,2,3]])
        state=NS(interrupted=False,stopping_generation=False,textinfo='')
        host=NS(shared=NS(opts=NS(),state=state),
            dynamic=NS(klein=True,edit=True,ref_latents=['old-dynamic'],is_referencing=True),
            devices=NS(device='cpu',torch_gc=lambda:None),
            memory=NS(get_total_memory=lambda device:8*1024**3,get_free_memory=lambda device:4*1024**3),
            torch=NS(inference_mode=contextlib.nullcontext,OutOfMemoryError=MemoryError))
        host.encode=lambda *args:(_ for _ in ()).throw(AssertionError('Z-Image must not reference-encode'))
        original=Image.new('RGB',(200,120),(20,40,60))
        custom=Image.new('L',original.size,0); ImageDraw.Draw(custom).rectangle((70,25,115,75),fill=255)
        p=NS(sd_model=model,init_images=[original],batch_size=2,n_iter=1,scripts=None,width=512,height=512,
             init=lambda *a:None,setup_conds=lambda:None,sample=lambda:None,image_mask=None,
             denoising_strength=0.22,cfg_scale=6.5,all_prompts=['portrait one','portrait two'],
             all_negative_prompts=['bad one','bad two'],all_seeds=[11,12],all_subseeds=[21,22],
             extra_generation_params={},clear_prompt_cache=lambda:None,_khs_session=None)
        pose={'box':(72,28,112,72),'head_w':40,'head_h':44,'head_px':44,'head_ratio':44/120,
              'yaw':0,'pitch':0,'roll':0,'sharp':1.0}
        analyzer=NS(faces=(lambda im:[pose] if face else []),mode='fake',error=None)
        owner=NS(analyzer=analyzer,last_report={},custom={},choices=core.choices({}),auditor=None)
        cfg=core.normalize({'enable':True,'edit_scope':'Protected head edit','headshots':headshots,
            'char_lora_name':'Z-Image Turbo CHAR LORA/rinmjq-woman-OriginalEdition-7E-2026-ZIT',
            'char_lora_trigger':'rinmjq woman','custom_mask':custom,'zimage_variant':'Turbo',
            'zimage_denoise':0.65,'geometry_match':False,'match_sharpness':False,
            'identity_check':False,'color_match':0,'resolution_dropdown':'512'})
        entry=NS(alias=cfg['char_lora_name'],filename=cfg['char_lora_name']+'.safetensors',
                 metadata={'ss_base_model_version':'Tongyi-MAI/Z-Image-Turbo'})
        return model,host,p,owner,cfg,{cfg['char_lora_name']:entry},custom

    def session(self,*args):
        model,host,p,owner,cfg,entries,_=args
        with patch('khs.runtime.registry',return_value=entries):
            return runtime.Session(p,cfg,owner,host)

    def test_native_session_lifecycle_mask_batch_and_restore(self):
        args=self.fixtures(headshots=[Image.new('RGB',(64,64),'gray')])
        model,host,p,owner,cfg,entries,custom=args
        session=self.session(*args)
        session.enter()
        self.assertEqual((p.batch_size,p.n_iter),(1,2))
        # Ratio boost is capped by Forge's generation size (512) and 1.5x, so the
        # small head still gets the small-head denoise boost.
        self.assertAlmostEqual(p.denoising_strength,0.7)
        self.assertEqual(max(session.canvas_size),512)
        self.assertEqual(p.image_mask.size,session.canvas_size)
        self.assertIsNotNone(p.image_mask.getbbox())
        self.assertTrue(session.refs)  # Optional: scoring only, never conditioning.
        self.assertEqual(model.ref_latents,[])
        session.process()
        self.assertEqual(p.cfg_scale,1.0)
        self.assertNotIn('bfs',p.all_prompts[0].lower())
        self.assertNotIn('picture ',p.all_prompts[0].lower())
        session.batch(0); session.batch(1)
        self.assertEqual(session.encodes,0)
        self.assertEqual(owner.last_report['identity_source'],'character LoRA')
        self.assertEqual(owner.last_report['encoded_sizes'],[])
        generated=Image.new('RGB',session.canvas_size,'red')
        result=session.finish_image(generated,1)
        source=np.asarray(session.original); output=np.asarray(result); mask=np.asarray(session.region.mask)
        self.assertTrue(np.array_equal(output[mask==0],source[mask==0]))
        self.assertTrue(np.any(output[mask>240]!=source[mask>240]))
        session.close()
        self.assertEqual((p.batch_size,p.n_iter,p.cfg_scale,p.denoising_strength),(2,1,6.5,0.22))
        self.assertIsNone(p.image_mask)
        self.assertEqual(model.ref_latents,['old-ref']); self.assertEqual(model.ini_latent,'old-ini')
        self.assertEqual(host.dynamic.ref_latents,['old-dynamic']); self.assertTrue(host.dynamic.is_referencing)
        self.assertTrue(host.dynamic.edit)

    def test_no_headshots_is_valid_and_mask_coordinates_map_to_crop_canvas(self):
        args=self.fixtures(headshots=None)
        session=self.session(*args); session.enter()
        self.assertEqual(session.refs,[])
        bx=session.canvas_box; mx=p_bbox=session.p.image_mask.getbbox()
        self.assertGreaterEqual(mx[0],bx[0]); self.assertGreaterEqual(mx[1],bx[1])
        self.assertLessEqual(mx[2],bx[2]); self.assertLessEqual(mx[3],bx[3])
        # Source x=70..115 maps into the cropped content box, not original full-frame coordinates.
        self.assertNotEqual(mx,(70,25,116,76))
        session.process(); session.batch(0); self.assertEqual(session.encodes,0)
        session.close()

    def test_failed_enter_via_bridge_restores_every_mutation(self):
        model,host,p,owner,cfg,entries,_=self.fixtures(headshots=None,face=False)
        cfg['custom_mask']=None; p._khs_options=(cfg,owner)
        processing=NS(process_images_inner=lambda p:(_ for _ in ()).throw(AssertionError('sampling must not start')))
        runtime.install_bridge(processing,host)
        with patch('khs.runtime.registry',return_value=entries):
            with self.assertRaisesRegex(ValueError,'detected target face'):
                processing.process_images_inner(p)
        self.assertEqual((p.batch_size,p.n_iter,p.cfg_scale,p.denoising_strength),(2,1,6.5,0.22))
        self.assertIsNone(p.image_mask); self.assertIsNone(p._khs_session)
        self.assertEqual(model.ref_latents,['old-ref']); self.assertEqual(model.ini_latent,'old-ini')
        self.assertEqual(host.dynamic.ref_latents,['old-dynamic']); self.assertTrue(host.dynamic.is_referencing)
        self.assertTrue(host.dynamic.edit)

    def test_zit_zib_and_compact_names_are_zimage(self):
        for name in ('person-2026-ZIT','folder/ZIB CHAR LORA/person','ZImageTurbo/person','ZImageBase/person'):
            self.assertEqual(core.adapter_family(name)[0],'zimage',name)
        self.assertEqual(core.adapter_family('legitimate')[0],'unknown')

    def test_relative_subfolder_character_name_resolves_canonical_registry_key(self):
        from types import SimpleNamespace as NS
        cfg=core.normalize({'char_lora_name':'ZIT CHAR LORA/person-ZIT'})
        entry=NS(filename=r'G:\models\Lora\ZIT CHAR LORA\person-ZIT.safetensors',alias='person-ZIT',metadata={})
        with patch('khs.runtime.registry',return_value={'person-ZIT':entry}):
            self.assertEqual(runtime.resolve_adapters(cfg,None,'zimage'),('','person-ZIT'))

    def test_identity_audit_without_geometry_or_sharpness(self):
        from unittest.mock import Mock
        args=self.fixtures(headshots=[(Image.new('RGB',(64,64),'gray'),'identity')])
        model,host,p,owner,cfg,entries,custom=args
        cfg['identity_check']=True
        owner.auditor=Mock()
        session=self.session(*args)
        try:
            session.enter(); session.process(); session.batch(0)
            session.finish_image(Image.new('RGB',session.canvas_size,'gray'),0)
            owner.auditor.submit.assert_called_once()
        finally: session.close()

    def test_base_character_lora_preserves_native_guidance(self):
        args=self.fixtures()
        model,host,p,owner,cfg,entries,custom=args
        cfg['zimage_variant']='Base'
        name='person-ZIB'
        cfg['char_lora_name']=name
        entry=next(iter(entries.values()))
        entry.alias=name; entry.filename=name+'.safetensors'
        entry.metadata={'ss_base_model_version':'Tongyi-MAI/Z-Image'}
        entries.clear(); entries[name]=entry
        session=self.session(*args)
        try:
            session.enter(); session.process()
            self.assertEqual(p.cfg_scale,6.5)
            self.assertIn('<lora:person-ZIB:',p.all_prompts[0])
            self.assertEqual(session.encodes,0)
        finally: session.close()

    def test_nonpositive_character_strength_fails_before_generation(self):
        model,host,p,owner,cfg,entries,_=self.fixtures()
        for strength in (0,-0.5):
            cfg['char_lora_strength']=strength
            with patch('khs.runtime.registry',return_value=entries):
                with self.assertRaisesRegex(ValueError,'strength above 0'):
                    runtime.Session(p,cfg,owner,host)


if __name__=='__main__': unittest.main()
