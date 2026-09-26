"""Head Swap adapter for pipelines that bypass Forge's ScriptRunner callbacks."""
from contextlib import contextmanager
from types import SimpleNamespace
import json

from . import core, runtime


class ExternalSession(runtime.Session):
    def enter(self):
        full_size=(max(64,round(self.p.width/64)*64),max(64,round(self.p.height/64)*64))
        super().enter()
        self.external_canvas_size=None
        if self.region:
            # Qwen's character LoRAs can reframe tight head crops as portraits.
            # Condition on the full scene, then composite only the protected area.
            self.external_canvas_size=full_size
            self.target,self.external_canvas_box=core.fit_canvas(self.original,full_size)
            self.set_p('init_images',[self.target])
            self.set_p('width',full_size[0]); self.set_p('height',full_size[1])

    def finish_image(self,image,index):
        if self.external_canvas_size:
            from PIL import Image
            frame=image.convert('RGB').resize(self.external_canvas_size,Image.Resampling.LANCZOS)
            frame=frame.crop(self.external_canvas_box).resize(self.original.size,Image.Resampling.LANCZOS)
            image,_=core.fit_canvas(frame.crop(self.region.box),self.canvas_size)
        return super().finish_image(image,index)

    def prepare(self, index, prompt, negative, seed, cfg_scale, turbo=False):
        self._cancel_check()
        cfg = dict(self.cfg)
        if turbo:
            cfg['ban_channel'] = 'Positive-only (fast, CFG 1.0)'
        preset = self.owner.custom.get('negative_presets', {}).get(cfg['neg_preset_dropdown'], '')
        if preset:
            negative = core.merge_negatives(negative, [preset])
        plan = core.build_plan(prompt, negative, cfg, seed, self.owner.choices,
                               self.fs, self.char, cfg_scale,
                               (self.target_pose or {}).get('head_px'), None,
                               self.owned_aliases, self.target_location, weighted=False)
        # Dedicated Qwen 2.1 uses references in input order: target, identity.
        # The older native Qwen engine's zero-based remapping does not apply.
        plan.positive=plan.positive.replace('Picture 1','<image1>').replace('Picture 2','<image2>')
        plan.negative=plan.negative.replace('Picture 1','<image1>').replace('Picture 2','<image2>')
        plan.positive=plan.positive.replace('head_swap: use <image1> as the target body and scene; replace its head with the facial identity, eye color and nose structure of <image2>', 'Swap the head of the person in <image1> with the head of the person in <image2>')
        self.plans.append(plan)
        slot, reason = core.select_reference(self.scored, cfg, index, len(self.refs))
        self.selected_ref = slot
        reference = core.prepare_reference(self.refs[slot], self.poses[slot],
                                           self.target_pose, cfg['reference_framing'])
        _, total, free = runtime.device_policy(self.host)
        side, budget = core.memory_limits(total, free, self.requested_side(), cfg['reference_budget'])
        pair = core.prepare_pair(self.target, reference, side, budget)
        if cfg['no_ref_diag']:
            pair = []
        self.owner.last_report = {
            'version': core.VERSION, 'status': 'Generating', 'model_family': 'qwen21',
            'adapter': self.fs, 'character_adapter': self.char, 'image': index + 1,
            'selected_slot': slot + 1, 'reference_count': len(self.refs), 'reason': reason,
            'plan': plan.report(), 'analysis': self.analysis,
            'encoded_sizes': [im.size for im in pair],
        }
        if cfg['latent_sharpness']:
            self.owner.last_report['latent_sharpness'] = 'Klein-only; Qwen receives pixel references.'
        self.p.extra_generation_params['Universal Head Swap'] = f'{core.VERSION} | Qwen 2.1 | {self.fs}'
        self.p.extra_generation_params['Head Swap reference'] = f'{slot + 1}/{len(self.refs)}: {self.labels[slot]}'
        self.host.shared.state.textinfo = f'Head Swap: reference {slot + 1}/{len(self.refs)}; {self.fs}'
        print('[UniversalHeadSwap] ' + self.host.shared.state.textinfo)
        return plan, pair

    def metadata(self):
        return json.dumps(self.owner.last_report, ensure_ascii=False, separators=(',', ':'))


@contextmanager
def context(p, cfg, owner):
    if not cfg['enable'] or getattr(p, '_ad_inner', False) or getattr(p, 'is_hr_pass', False):
        yield None
        return
    if not getattr(p, 'init_images', None):
        raise ValueError('Head Swap needs a target image in img2img.')
    # Use a private reference state, never load or replace shared.sd_model.
    from modules import shared, devices
    from backend import memory_management
    import torch
    host = SimpleNamespace(shared=shared, devices=devices, memory=memory_management,
                           torch=torch, dynamic=SimpleNamespace(edit=True, klein=False,
                           ref_latents=[], is_referencing=False))
    model = SimpleNamespace(text_processing_engine_qwen=object(), model_config=None,
                            ref_latents=[], ini_latent=None)
    session = ExternalSession(p, cfg, owner, host, model=model)
    try:
        session.enter()
        yield session
    except Exception as exc:
        owner.last_report.update(status='Failed', error=str(exc))
        raise
    finally:
        session.close()
