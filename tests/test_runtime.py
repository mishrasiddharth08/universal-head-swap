"""Forge bridge contract tests with stand-ins. No GPU or Forge imports."""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core,runtime

class DummyTensor:
    def __init__(self,value='latent'): self.value=value
    def unsqueeze(self,*a): return self
    def to(self,*a,**kw): return self
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numel(self): return 8
    def element_size(self): return 4

class RuntimeTests(unittest.TestCase):
    def fixtures(self):
        model=NS(ref_latents=['previous'],ini_latent='initial',forge_objects=NS(vae=object()))
        shared=NS(opts=NS(klein_do_reference=False),state=NS(interrupted=False,stopping_generation=False,textinfo=''))
        host=NS(shared=shared,dynamic=NS(klein=True,ref_latents=['dynamic'],is_referencing=False),
            devices=NS(device='cpu',torch_gc=lambda:None),
            torch=NS(from_numpy=lambda a:DummyTensor(),inference_mode=contextlib.nullcontext,OutOfMemoryError=MemoryError))
        p=NS(sd_model=model,init_images=[Image.new('RGB',(256,256))],batch_size=2,n_iter=2,scripts=None,width=512,height=512,
             init=lambda *a:None,setup_conds=lambda:None,sample=lambda:None)
        owner=NS(analyzer=NS(faces=lambda im:[],mode=None,error='detector unavailable'),last_report={})
        cfg=core.normalize({'enable':True,'headshots':[Image.new('RGB',(256,256))]})
        return model,host,p,owner,cfg
    def session(self,p,cfg,owner,host):
        with patch('khs.runtime.resolve_adapters',return_value=('bfs_head_v1_flux-klein_9b','')),patch('khs.runtime.registry',return_value={}):
            return runtime.Session(p,cfg,owner,host)
    def test_reused_batch_inputs_do_not_accumulate_callback_negatives(self):
        model,host,p,owner,cfg=self.fixtures()
        p.prompt='original'; p.negative_prompt='negative'
        for _ in range(3):
            session=self.session(p,cfg,owner,host); session.enter()
            self.assertEqual(p.negative_prompt,'negative')
            p.prompt='expanded'; p.negative_prompt += ', expanded by AutoNeg'
            session.close()
        self.assertEqual((p.prompt,p.negative_prompt),('original','negative'))
    def test_mutated_prompt_lists_restore_original_objects(self):
        model,host,p,owner,cfg=self.fixtures()
        original=['original']; negative=['negative']
        p.prompt=original; p.negative_prompt=negative
        session=self.session(p,cfg,owner,host); session.enter()
        p.prompt.append('expanded'); p.negative_prompt[0]='changed'
        session.close()
        self.assertIs(p.prompt,original); self.assertIs(p.negative_prompt,negative)
        self.assertEqual(original,['original']); self.assertEqual(negative,['negative'])
    def test_option_polarities(self):
        self.assertEqual(runtime.option_key(NS(klein_do_reference=False)),('klein_do_reference',True))
        self.assertEqual(runtime.option_key(NS(klein_no_reference=True)),('klein_no_reference',False))
    def test_enter_cleanup_restores_all_owned_state(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host)
        session.enter(); self.assertTrue(host.shared.opts.klein_do_reference); self.assertEqual(p.batch_size,1); self.assertEqual(p.n_iter,4)
        session.close(); self.assertFalse(host.shared.opts.klein_do_reference)
        self.assertEqual(model.ref_latents,['previous']); self.assertEqual(model.ini_latent,'initial')
        self.assertEqual(host.dynamic.ref_latents,['dynamic']); self.assertEqual(p.batch_size,2); self.assertEqual(p.n_iter,2)
        session.close()
    def test_encoding_failure_restores_flag_and_temporary_reference_list(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host); session.enter()
        model.ref_latents=['intact']
        def encode(*a): model.ref_latents.append(DummyTensor()); raise RuntimeError('synthetic encode failure')
        host.encode=encode
        with self.assertRaisesRegex(RuntimeError,'synthetic'): session._encode(p.init_images[0])
        self.assertFalse(host.dynamic.is_referencing); self.assertEqual(model.ref_latents,['intact']); session.close()
    def test_cached_encoding_reused(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host); session.enter()
        host.encode=lambda *a:model.ref_latents.append(DummyTensor())
        first=session._encode(p.init_images[0]); second=session._encode(p.init_images[0])
        self.assertIs(first,second); self.assertEqual(session.encodes,1); self.assertEqual(session.hits,1); session.close()
    def test_cancel_stops_before_encoding(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host); session.enter()
        host.shared.state.interrupted=True
        with self.assertRaisesRegex(runtime.GenerationCancelled,'cancelled'): session._encode(p.init_images[0])
        session.close()
    def test_cancel_is_not_recorded_as_error(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host); session.enter()
        session.fail(runtime.GenerationCancelled('Stopped by request'))
        self.assertIsNone(session.error); self.assertEqual(owner.last_report['status'],'Cancelled')
        self.assertNotIn('error',owner.last_report); self.assertEqual(model.ref_latents,[])
        session.close(); self.assertEqual(model.ref_latents,['previous'])
    def test_original_failure_logged_once_even_when_guard_reraises(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host); session.enter()
        with patch('builtins.print') as log:
            session.fail(ValueError('broken reference')); message=session.error
            session.fail(RuntimeError(message))
        self.assertEqual(session.error,message); self.assertEqual(log.call_count,1); session.close()
    def test_cancel_guard_raises_distinct_signal(self):
        session=NS(p=object(),error=None,cancelled=None)
        def swallowed(p): session.cancelled='Stopped'
        proxy=runtime.ScriptGuard(NS(process_batch=swallowed),session)
        with self.assertRaises(runtime.GenerationCancelled): proxy.process_batch(session.p)
    def test_output_capture_happens_after_final_postprocessors(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host)
        p.prompts=p.all_prompts=['resolved']; p.negative_prompts=p.all_negative_prompts=['negative']
        p.seeds=p.all_seeds=[12]; p.subseeds=p.all_subseeds=[8]; p.batch_index=0
        session.processing=NS(create_infotext=lambda *a,**kw:'final metadata')
        original=Image.new('RGB',(16,16),'black'); final=Image.new('RGB',(16,16),'green')
        pp=NS(image=original,index=0)
        def finish(p,pp): pp.image=final
        proxy=runtime.ScriptGuard(NS(postprocess_image_after_composite=finish),session)
        proxy.postprocess_image_after_composite(p,pp)
        self.assertIs(session.completed[0]['image'],final)
        self.assertEqual(session.completed[0]['info'],'final metadata')
        self.assertEqual(session.completed[0]['seed'],12)
    def test_default_size_check_does_not_warp_finished_pixels(self):
        model,host,p,owner,cfg=self.fixtures(); cfg['match_sharpness']=False
        session=self.session(p,cfg,owner,host); session.target_pose={'box':(50,50,150,150),'head_h':100,'head_w':100}
        owner.analyzer.faces=lambda im:[{'box':(70,50,170,150),'head_h':100,'head_w':100}]
        p.extra_generation_params={}; image=Image.new('RGB',(256,256),'green')
        with patch('khs.core.correct_head_scale',side_effect=AssertionError('Must not warp')):
            result=session.finish_image(image,0)
        self.assertEqual(result.tobytes(),image.tobytes())
        self.assertFalse(owner.last_report['quality']['pixel_head_resizing_enabled'])
    def test_cancelled_bridge_returns_completed_images_and_restores_state(self):
        model,host,p,owner,cfg=self.fixtures(); p._khs_options=(cfg,owner)
        p.all_seeds=[12,13]; p.all_subseeds=[8,9]
        retained=Image.new('RGB',(256,256),'red')
        def original(p):
            p._khs_session.completed=[dict(image=retained,info='actual metadata',prompt='resolved',negative='negative',seed=12,subseed=8)]
            p._khs_session.fail(runtime.GenerationCancelled('Stopped'))
            raise runtime.GenerationCancelled('Stopped')
        def processed(p,images,**kw): return NS(images=images,comments='',**kw)
        processing=NS(process_images_inner=original,Processed=processed); runtime.install_bridge(processing,host)
        with patch('khs.runtime.resolve_adapters',return_value=('bfs_head_v1_flux-klein_9b','')),patch('khs.runtime.registry',return_value={}):
            result=processing.process_images_inner(p)
        self.assertEqual(result.images,[retained]); self.assertEqual(result.infotexts,['actual metadata'])
        self.assertEqual(result.all_seeds,[12]); self.assertEqual(p.batch_size,2)
        self.assertEqual(model.ref_latents,['previous']); self.assertIsNone(p._khs_session)
    def test_zero_output_cancel_does_not_invent_an_image(self):
        model,host,p,owner,cfg=self.fixtures(); session=self.session(p,cfg,owner,host); session.enter()
        p.all_seeds=[12]; p.all_subseeds=[8]
        session.processing=NS(Processed=lambda p,images,**kw:NS(images=images,comments='',**kw))
        session.fail(runtime.GenerationCancelled('Stopped'))
        result=session.cancelled_result(); self.assertEqual(result.images,[]); session.close()
    def test_strict_quality_rejects_undetected_output(self):
        model,host,p,owner,cfg=self.fixtures(); cfg['quality_strict']=True
        p.extra_generation_params={}
        session=self.session(p,cfg,owner,host); session.enter()
        with self.assertRaisesRegex(ValueError,'quality gate'): session.finish_image(p.init_images[0],0)
        self.assertFalse(owner.last_report['quality']['quality_gate_passed']); session.close()
    def test_full_canvas_returns_original_size_and_restores_sampling_size(self):
        model,host,p,owner,cfg=self.fixtures(); p.extra_generation_params={}
        p.init_images=[Image.new('RGB',(333,777))]
        session=self.session(p,cfg,owner,host); session.enter()
        output=session.finish_image(Image.new('RGB',(p.width,p.height)),0)
        self.assertEqual(output.size,(333,777)); session.close()
        self.assertEqual((p.width,p.height),(512,512)); self.assertEqual(p.init_images[0].size,(333,777))
    def test_script_guard_raises_after_swallowed_callback_error(self):
        session=NS(p=object(),error=None)
        def swallowed(p): session.error='failed second reference'
        proxy=runtime.ScriptGuard(NS(process_batch=swallowed,other=123),session)
        with self.assertRaisesRegex(RuntimeError,'second reference'): proxy.process_batch(session.p)
        self.assertEqual(proxy.other,123)
    def test_disabled_bridge_forwards_arguments_and_return(self):
        calls=[]; result=object()
        def original(*a,**kw): calls.append((a,kw)); return result
        processing=NS(process_images_inner=original); runtime.install_bridge(processing,NS())
        p=NS(_khs_options=None)
        self.assertIs(processing.process_images_inner(p,'future',test=True),result)
        self.assertEqual(calls,[((p,'future'),{'test':True})])
    def test_non_klein_rejected_before_session_mutation(self):
        model,host,p,owner,cfg=self.fixtures(); host.dynamic.klein=False; p._khs_options=(cfg,owner)
        processing=NS(process_images_inner=lambda *a:None); runtime.install_bridge(processing,host)
        with self.assertRaisesRegex(ValueError,'not Flux.2 Klein'): processing.process_images_inner(p)
        self.assertEqual(p.batch_size,2); self.assertFalse(host.shared.opts.klein_do_reference)
    def test_outer_exception_still_closes_session(self):
        model,host,p,owner,cfg=self.fixtures(); p._khs_options=(cfg,owner)
        def original(p): raise RuntimeError('sampling failure')
        processing=NS(process_images_inner=original); runtime.install_bridge(processing,host)
        with patch('khs.runtime.resolve_adapters',return_value=('bfs_head_v1_flux-klein_9b','')),patch('khs.runtime.registry',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'sampling failure'): processing.process_images_inner(p)
        self.assertEqual(p.batch_size,2); self.assertEqual(model.ref_latents,['previous'])
        self.assertFalse(host.shared.opts.klein_do_reference); self.assertIsNone(p._khs_session)
    def test_bridge_idempotent(self):
        processing=NS(process_images_inner=lambda *a:None); host=NS()
        runtime.install_bridge(processing,host); first=processing.process_images_inner
        runtime.install_bridge(processing,host); self.assertIs(first,processing.process_images_inner)

    def test_loaded_qwen_wins_over_stale_klein_flag(self):
        model=NS(text_processing_engine_qwen=object())
        host=NS(dynamic=NS(klein=True,edit=True))
        self.assertEqual(runtime.model_family(model,host),('qwen',None))
        host.dynamic.edit=False
        self.assertEqual(runtime.model_family(model,host),(None,None))

    def test_klein_architecture_without_preset_flag(self):
        for size in (4,9):
            model=NS(model_config=type(f'Flux2K{size}B',(),{})())
            self.assertEqual(runtime.model_family(model,NS(dynamic=NS(klein=False))),('klein',size))

    def test_bridge_returns_existing_wrapper(self):
        processing=NS(process_images_inner=lambda p:p)
        first=runtime.install_bridge(processing,NS())
        self.assertIs(runtime.install_bridge(processing,NS()),first)

    def test_qwen_reference_keeps_pixels_and_stays_on_cpu(self):
        import numpy as np
        class PixelTensor:
            def __init__(self,a): self.a=a
            def unsqueeze(self,i): return PixelTensor(np.expand_dims(self.a,i))
            def movedim(self,a,b): return PixelTensor(np.moveaxis(self.a,a,b))
            def contiguous(self): return self
            def cpu(self): return self
            def to(self,**kw): raise AssertionError('Qwen pixels must stay on CPU')
            def numel(self): return self.a.size
            def element_size(self): return self.a.itemsize
        model,host,p,owner,cfg=self.fixtures()
        session=self.session(p,cfg,owner,host)
        session.family='qwen'; host.dynamic.edit=True
        host.torch.from_numpy=PixelTensor
        original=model.ref_latents
        reference=Image.new('RGB',(8,8),(0,128,255))
        tensor=session._encode(reference)
        np.testing.assert_allclose(tensor.a[0,0,0],[0,128/255,1],rtol=1e-6)
        self.assertIs(model.ref_latents,original)
        self.assertIs(session._encode(reference),tensor)
        self.assertEqual(session.encodes,1)

if __name__=='__main__': unittest.main()
