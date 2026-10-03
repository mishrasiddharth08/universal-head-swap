import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from PIL import Image

from khs import core, runtime


class FakeMemory:
    def __init__(self,total_gib=24,reserved=2*1024**3,extra=0):
        self.total_vram=total_gib*1024
        self.SETTING_RESERVED_VRAM=reserved
        self.EXTRA_RESERVED_VRAM=extra
        self.calls=[]
    def extra_reserved_memory(self):
        return max(self.SETTING_RESERVED_VRAM,self.EXTRA_RESERVED_VRAM)
    def set_reserved_memory(self,allocated):
        self.calls.append(allocated)
        self.SETTING_RESERVED_VRAM=(1.0-allocated)*self.total_vram*1024*1024


class KreaCoreTests(unittest.TestCase):
    def test_adapter_detection_selection_and_flux_krea_exclusion(self):
        self.assertEqual(core.adapter_family('nested/bfs_head_swap_v1.1_krea2')[0], 'krea')
        self.assertEqual(core.adapter_family('FLUX.1-Krea-Dev-style')[0], 'other')
        entries={'old/bfs_head_swap_v1_krea2':{}, 'KREA2 ADDITIONAL LORA/bfs_head_swap_v1.1_krea2':{}}
        self.assertEqual(core.select_adapter(entries,None,'krea'),
                         'KREA2 ADDITIONAL LORA/bfs_head_swap_v1.1_krea2')

    def test_plan_uses_official_unweighted_prompt_and_retains_cfg(self):
        plan=core.build_plan('', '', {}, 1, core.choices({}), 'bfs_head_swap_v1.1_krea2',
                             host_cfg=1.0, weighted=False, family='krea')
        self.assertIn('head_swap: replace the head with the reference head.', plan.positive)
        self.assertTrue(plan.positive.startswith('head_swap: replace the head with the reference head.'))
        self.assertNotIn('(head_swap:', plan.positive)
        self.assertEqual(plan.cfg,1.0)
        self.assertNotIn('duplicate head',plan.negative)

    def test_raw_cfg_and_positive_only_semantics(self):
        dual=core.build_plan('', '', {'ban_channel':'Positive + Negative (uses at least CFG 1.1)'}, 1,
                             core.choices({}), 'bfs_head_swap_v1.1_krea2', host_cfg=4,
                             weighted=False, family='krea')
        fast=core.build_plan('', '', {'ban_channel':'Positive-only (fast, CFG 1.0)'}, 1,
                             core.choices({}), 'bfs_head_swap_v1.1_krea2', host_cfg=4,
                             weighted=False, family='krea')
        self.assertEqual(dual.cfg,4.0)
        self.assertEqual(fast.cfg,1.0)
        self.assertEqual(fast.negative,'')

    def test_auto_selection_refuses_body_swap(self):
        with self.assertRaisesRegex(ValueError,'Krea2 BFS face-swap'):
            core.select_adapter({'bfs_body_swap_v1_krea2':{}},None,'krea')

    def test_krea_rejects_known_incompatible_prompt_loras(self):
        for tag in ('bfs_head_v1_qwen_2.1','bfs_head_v1_flux-klein_9b','person-ZIT','FLUX.1-Krea-Dev-style'):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError,'does not match Krea2'):
                core.build_plan(f'<lora:{tag}:1>', '', {}, 1, core.choices({}),
                                'bfs_head_swap_v1.1_krea2', weighted=False, family='krea')
        plan=core.build_plan('<lora:generic_detail:0.5>', '', {}, 1, core.choices({}),
                             'bfs_head_swap_v1.1_krea2', weighted=False, family='krea')
        self.assertIn('<lora:generic_detail:0.5>',plan.positive)


    def test_explicit_trigger_is_kept_once(self):
        trigger='head_swap: replace the head with the reference head.'
        plan=core.build_plan(trigger+' Preserve the scene.', '', {}, 1, core.choices({}),
                             'bfs_head_swap_v1.1_krea2',host_cfg=1,weighted=False,family='krea')
        self.assertEqual(plan.positive.count(trigger),1)
        self.assertTrue(plan.positive.startswith(trigger))
        self.assertIn('Preserve the scene.',plan.positive)


class KreaRuntimeTests(unittest.TestCase):
    def reservation_fixture(self,family='krea',memory=None):
        config={'krea':'krea2','klein':'flux2','zimage':'zimage'}.get(family,'other')
        unet={'image_model':config}
        if family=='zimage': unet['z_image_modulation']=True
        model=NS(model_config=NS(unet_config=unet),ref_latents=[],ini_latent=None,
                 forge_objects=NS(vae=object()))
        if family=='qwen': model.text_processing_engine_qwen=object()
        opts=NS(krea2_do_reference=False,klein_do_reference=False)
        host=NS(shared=NS(opts=opts,state=NS(textinfo='')),
                dynamic=NS(edit=family=='qwen',klein=family=='klein',ref_latents=[],is_referencing=False),
                memory=memory or FakeMemory())
        p=NS(sd_model=model,init_images=[Image.new('RGB',(64,64))],batch_size=1,n_iter=1,
             scripts=None,width=512,height=512,_khs_session=None)
        owner=NS(last_report={})
        cfg=core.normalize({'enable':True,'headshots':[Image.new('RGB',(64,64))]})
        return p,cfg,owner,host

    def make_reservation_session(self,family='krea',memory=None):
        p,cfg,owner,host=self.reservation_fixture(family,memory)
        with patch('khs.runtime.resolve_adapters',return_value=('adapter','')), \
             patch('khs.runtime.registry',return_value={}):
            session=runtime.Session(p,cfg,owner,host)
        return session,host

    def test_krea_reserves_32_percent_and_restores_exact_previous_value(self):
        old=2*1024**3+123
        memory=FakeMemory(reserved=old)
        session,host=self.make_reservation_session('krea',memory)
        self.assertAlmostEqual(memory.extra_reserved_memory(),24*1024**3*0.32,delta=1)
        self.assertAlmostEqual(memory.calls[0],0.68,places=12)
        session.close()
        self.assertEqual(memory.SETTING_RESERVED_VRAM,old)

    def test_krea_preserves_larger_existing_reserve(self):
        memory=FakeMemory(reserved=10*1024**3)
        session,_=self.make_reservation_session('krea',memory)
        self.assertEqual(memory.calls,[])
        session.close()
        self.assertEqual(memory.SETTING_RESERVED_VRAM,10*1024**3)

    def test_krea_caps_reserve_at_12_gib_and_restores_auto_sentinel(self):
        memory=FakeMemory(total_gib=48,reserved=-1)
        session,_=self.make_reservation_session('krea',memory)
        self.assertEqual(memory.SETTING_RESERVED_VRAM,12*1024**3)
        self.assertAlmostEqual(memory.calls[0],0.75,places=12)
        session.close()
        self.assertEqual(memory.SETTING_RESERVED_VRAM,-1)

    def test_other_model_families_do_not_change_reservation(self):
        for family in ('klein','qwen','zimage'):
            memory=FakeMemory()
            with self.subTest(family=family):
                session,_=self.make_reservation_session(family,memory)
                session.close()
                self.assertEqual(memory.calls,[])
                self.assertEqual(memory.SETTING_RESERVED_VRAM,2*1024**3)

    def test_krea_constructor_failure_restores_exact_reserve(self):
        old=2*1024**3+321
        p,cfg,owner,host=self.reservation_fixture('krea',FakeMemory(reserved=old))
        with patch('khs.runtime.resolve_adapters',return_value=('adapter','')), \
             patch('khs.runtime.registry',return_value={}), \
             patch('khs.runtime.option_key',side_effect=RuntimeError('synthetic constructor failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic constructor failure'):
                runtime.Session(p,cfg,owner,host)
        self.assertEqual(host.memory.SETTING_RESERVED_VRAM,old)
        self.assertEqual(host.memory.calls,[])

    def test_nested_bfs_and_k2_character_resolve_as_a_pair(self):
        fs='KREA2 ADDITIONAL LORA/bfs_head_swap_v1.1_krea2'
        char='KREA2 CHAR LORA/rinmjq-woman-SuperLiteOriginalEdition-FE-2026-K2'
        metadata={'ss_base_model_version':'krea/Krea-2-Raw'}
        entries={
            fs:NS(alias=fs,filename=fs+'.safetensors',metadata=metadata),
            char:NS(alias=char,filename=char+'.safetensors',metadata=metadata),
        }
        cfg=core.normalize({'auto_model_adapter':False,'lora_dropdown':fs.split('/')[-1],
                            'char_lora_name':char.split('/')[-1]})
        with patch('khs.runtime.registry',return_value=entries):
            self.assertEqual(runtime.resolve_adapters(cfg,None,'krea'),(fs,char))

    def test_krea_character_metadata_names_and_wrong_family_rejection(self):
        krea_metadata={'ss_base_model_version':'krea/Krea-2-Raw'}
        names=(
            'rinmjq-woman-DollycastroEdition-2026-FE-KREA2',
            'rinmjq-woman-ItsraissabellinisEdition-2026-FE-KREA2',
            'rinmjq-woman-SuperLiteOriginalEdition-FE-2026-K2',
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(core.adapter_family(name,krea_metadata)[0],'krea')
        fs='nested/bfs_head_swap_v1.1_krea2'
        for char,metadata in (
            ('person-QWEN21',{'ss_base_model_version':'Qwen/Qwen-Image-Edit'}),
            ('person-ZIB',{'ss_base_model_version':'Tongyi-MAI/Z-Image'}),
            ('person-ZIT',{'ss_base_model_version':'Tongyi-MAI/Z-Image-Turbo'}),
            ('person-klein-9b',{'base_model':'Flux.2 Klein 9B'}),
        ):
            entries={
                fs:NS(alias=fs.split('/')[-1],filename=fs+'.safetensors',metadata=krea_metadata),
                char:NS(alias=char,filename='nested/'+char+'.safetensors',metadata=metadata),
            }
            cfg=core.normalize({'auto_model_adapter':False,'lora_dropdown':fs,'char_lora_name':char})
            with self.subTest(char=char), patch('khs.runtime.registry',return_value=entries):
                with self.assertRaisesRegex(ValueError,'does not match the active Krea2'):
                    runtime.resolve_adapters(cfg,None,'krea')

    def test_krea_pair_keeps_strengths_and_metadata_trigger_once(self):
        fs='KREA2 ADDITIONAL LORA/bfs_head_swap_v1.1_krea2'
        char='KREA2 CHAR LORA/rinmjq-woman-SuperLiteOriginalEdition-FE-2026-K2'
        trained='rinmjq-woman-SuperLiteOriginalEdition-FE-2026-K2'
        metadata={'ss_tag_frequency':{'subject':{trained:8}},'ss_output_name':trained}
        trigger=runtime.merge_character_trigger('studio portrait',metadata)
        plan=core.build_plan('', '', {'lora_strength':0.83,'char_lora_strength':0.47,
            'char_lora_trigger':trigger}, 1, core.choices({}), fs, char,
            host_cfg=3.5,weighted=False,family='krea')
        official='head_swap: replace the head with the reference head.'
        self.assertTrue(plan.positive.startswith(official))
        self.assertEqual(plan.positive.count(official),1)
        self.assertEqual(core.TOKEN.sub('',plan.positive).count(trained),1)
        self.assertIn('studio portrait',plan.positive)
        self.assertIn(f'<lora:{fs}:0.83>',plan.positive)
        self.assertIn(f'<lora:{char}:0.47>',plan.positive)
        self.assertEqual(plan.fs_strength,0.83)

    def test_native_krea_wins_over_shared_qwen_engine(self):
        Krea2=type('Krea2',(),{})
        model=Krea2(); model.model_config=NS(unet_config={}); model.text_processing_engine_qwen=object()
        self.assertEqual(runtime.model_family(model,NS(dynamic=NS(edit=True,krea2=True))),('krea',None))
        config_model=NS(model_config=NS(unet_config={'image_model':'krea2'}),text_processing_engine_qwen=object())
        self.assertEqual(runtime.model_family(config_model,NS(dynamic=NS(edit=True))),('krea',None))
        self.assertEqual(runtime.option_key(NS(krea2_do_reference=False),'krea'),('krea2_do_reference',True))

    def test_krea_pixel_reference_stays_on_cpu(self):
        class PixelTensor:
            def __init__(self,a): self.a=a
            def unsqueeze(self,i): return PixelTensor(np.expand_dims(self.a,i))
            def movedim(self,a,b): return PixelTensor(np.moveaxis(self.a,a,b))
            def contiguous(self): return self
            def cpu(self): return self
            def to(self,**kw): raise AssertionError('Krea pixels must stay on CPU')
            def numel(self): return self.a.size
            def element_size(self): return self.a.itemsize
        model=NS(ref_latents=[],ini_latent=None,forge_objects=NS(vae=object()))
        host=NS(torch=NS(from_numpy=PixelTensor),devices=NS(device='cuda'),dynamic=NS(is_referencing=False,edit=True))
        session=object.__new__(runtime.Session)
        session.model=model; session.host=host; session.family='krea'; session.cfg={'cache_encodes':True}
        session.cache=core.BoundedCache(); session.hits=0; session.encodes=0
        session._cancel_check=lambda:None
        tensor=session._encode(Image.new('RGB',(4,4),(0,128,255)))
        np.testing.assert_allclose(tensor.a[0,0,0],[0,128/255,1],rtol=1e-6)
        self.assertEqual(session.encodes,1)

    def test_batch_orders_source_then_single_identity_reference(self):
        source=Image.new('RGB',(8,8),'blue'); identity=Image.new('RGB',(8,8),'red')
        session=object.__new__(runtime.Session)
        session.family='krea'; session.error=None; session.target=source; session.refs=[identity]
        session.poses=[None]; session.target_pose=None; session.scored=[{'index':0,'score':1,'usable':True}]
        session.cfg={'no_ref_diag':False,'reference_framing':'Unmodified reference','reference_budget':1,
                     'latent_sharpness':0,'cache_encodes':True,'latent_kernel_size':'3x3'}
        session.model=NS(ref_latents=[],ini_latent=None); session.labels=['identity']; session.region=None
        session.original=source; session.analysis={}; session.hits=0; session.encodes=0
        session.plans=[NS(choices={},report=lambda:{})]; session.owner=NS(last_report=None)
        session.p=NS(clear_prompt_cache=lambda:None,extra_generation_params={})
        session.host=NS(memory=NS(get_total_memory=lambda d:8*1024**3,get_free_memory=lambda d:4*1024**3),
                        devices=NS(device='cuda'),dynamic=NS(ref_latents=[],edit=False),
                        shared=NS(state=NS(textinfo='')))
        session._cancel_check=lambda:None; session.requested_side=lambda:512
        session._encode=lambda im:'source' if im is source else 'identity'
        with patch.object(core,'select_reference',return_value=(0,'best')), \
             patch.object(core,'prepare_reference',return_value=identity), \
             patch.object(core,'prepare_pair',return_value=[source,identity]):
            session.batch(0)
        self.assertEqual(session.model.ini_latent,'source')
        self.assertEqual(session.model.ref_latents,['identity'])
        self.assertFalse(session.host.dynamic.edit)


if __name__=='__main__': unittest.main()
