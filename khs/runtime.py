"""Scoped Forge bridge: errors propagate outside swallowed script callbacks."""
from __future__ import annotations
from functools import wraps
import json
import copy
import math
import time

import numpy as np
from PIL import Image, ImageFilter

from . import core

class GenerationCancelled(Exception):
    """A requested stop, distinct from encoding and quality failures."""

def model_size(model):
    config=getattr(model,'model_config',None)
    typename=type(config).__name__.lower()
    if '4b' in typename: return 4
    if '9b' in typename: return 9
    hidden=getattr(config,'unet_config',{}).get('hidden_size') if config is not None else None
    if hidden==3072: return 4
    if hidden==4096: return 9
    return None

def model_family(model,host):
    """Detect the active model family from live engine state, not filenames alone.

    Returns ('klein', size) or ('qwen', None). Qwen is detected from
    dynamic_args.edit (set by Forge's loader for Qwen-Image-Edit checkpoints)
    plus the diffusion engine class; Klein from dynamic_args.klein.
    """
    # Loaded architecture wins over mutable flags left by UI presets.
    config_name=type(getattr(model,'model_config',None)).__name__.lower()
    engine_name=type(model).__name__.lower()
    if getattr(model,'text_processing_engine_qwen',None) is not None:
        if getattr(host.dynamic,'edit',False): return 'qwen',None
        return None,None
    if engine_name=='flux2' or 'flux2k' in config_name:
        return 'klein',model_size(model)
    if getattr(host.dynamic,'klein',False):
        return 'klein',model_size(model)
    return None,None

def device_policy(host):
    """Project-Invisible memory policy: probe the selected device, never assume GPU 0.

    Returns (backend_label, total_bytes, free_bytes). AMD (ROCm) and CUDA are
    both handled through torch's device interface; CPU gets conservative
    estimates so reference encodes stay bounded.
    """
    torch=host.torch; device=host.devices.device
    try:
        props=torch.cuda.get_device_properties(device)
        backend='rocm' if getattr(torch.version,'hip',None) else 'cuda'
        total=int(props.total_memory)
        free=int(host.memory.get_free_memory(device))
        return backend,total,free
    except Exception:
        try:
            total=int(host.memory.get_total_memory(device)); free=int(host.memory.get_free_memory(device))
            return 'cuda',total,free
        except Exception:
            return 'cpu',8*1024**3,2*1024**3

def registry():
    import networks
    # Reuse Forge's canonical registry to avoid duplicate basenames/aliases.
    if not networks.available_networks: networks.list_available_networks()
    return networks.available_networks

def resolve_adapters(cfg,model,family='klein'):
    entries=registry(); size=model_size(model) if family=='klein' else None
    wanted=cfg['lora_dropdown']
    if cfg.get('auto_model_adapter',True) or wanted in ('Auto (match model)','None',None,''):
        wanted=core.select_adapter({k:v.metadata for k,v in entries.items()},size,family)
    def lookup(name,strict):
        name=core.clean_name(name)
        matches=[(k,v) for k,v in entries.items() if name in (k,getattr(v,'alias',None)) or
                 core.clean_name(v.filename).lower()==name.lower()]
        if not matches:
            matches=[(k,v) for k,v in entries.items() if core.clean_name(v.filename).split('/')[-1].lower()==name.lower()]
        if len(matches)!=1: raise ValueError(f'Adapter {name!r} is missing or ambiguous. Refresh adapters and select its canonical name.')
        key,entry=matches[0]; core.compatible_adapter(key,size,entry.metadata,strict,family)
        return key
    fs=lookup(wanted,cfg['strict_adapter'])
    # Character LoRA is family-matched too: a Qwen character LoRA must never be
    # loaded into a Klein trunk (and vice versa) even with strict checking off.
    char=cfg['char_lora_name']
    char=lookup(char,False) if char and not str(char).startswith('None') else ''
    return fs,char

def token_counter(model):
    engine=getattr(model,'text_processing_engine_gemma',None)
    if engine is None: engine=getattr(model,'text_processing_engine_qwen',None)
    if engine is None: return None
    return lambda text:len(engine.tokenize([core.TOKEN.sub('',text)])[0])

def option_key(opts,family='klein'):
    if family=='qwen':
        # Qwen-Image-Edit needs no reference toggle: dynamic_args.edit already
        # selects its reference path, and there is no Klein-style option.
        return None,None
    if hasattr(opts,'klein_do_reference'): return 'klein_do_reference',True
    if hasattr(opts,'klein_no_reference'): return 'klein_no_reference',False
    raise RuntimeError('This Forge version exposes no supported Klein reference setting.')

class ScriptGuard:
    """Re-raise failures after Forge's ScriptRunner logs and swallows a callback error."""
    def __init__(self,runner,session):
        object.__setattr__(self,'_runner',runner); object.__setattr__(self,'_session',session)
    def __getattr__(self,name):
        value=getattr(self._runner,name)
        if name not in ('process','process_batch','postprocess_image_after_composite'): return value
        @wraps(value)
        def checked(*args,**kwargs):
            result=value(*args,**kwargs)
            if args and args[0] is self._session.p:
                if self._session.error: raise RuntimeError(self._session.error)
                if getattr(self._session,'cancelled',None): raise GenerationCancelled(self._session.cancelled)
                if name=='postprocess_image_after_composite':
                    self._session.record_output(args[1].image,int(getattr(args[1],'index',0)))
            return result
        return checked
    def __setattr__(self,name,value): setattr(self._runner,name,value)

class Session:
    def __init__(self,p,cfg,owner,host,model=None):
        self.p=p; self.cfg=cfg; self.owner=owner; self.host=host; self.model=p.sd_model if model is None else model
        self.family,self.family_size=model_family(self.model,host)
        if self.family is None:
            raise ValueError('Head Swap could not identify the loaded model as Flux.2 Klein or Qwen Image Edit. Load one of those checkpoints.')
        self.owner.last_report={'version':core.VERSION,'status':'Preparing current generation.','model_family':self.family,'model_size':self.family_size}
        self.error=None; self.cancelled=None; self.completed=[]; self.processing=None
        self.cache=core.BoundedCache(); self.hits=0; self.encodes=0
        self.snap={}; self.region=None; self.plans=[]; self.analysis={}; self.start=time.perf_counter()
        self.canvas_box=None; self.canvas_size=None; self.quality_history=[]
        self.selected_ref=None
        self.original=core.rgb_image(p.init_images[0]); self.refs,self.labels=core.gallery_images(cfg['headshots'])
        self.fs,self.char=resolve_adapters(cfg,self.model,self.family)
        entries=registry(); self.owned_aliases=[getattr(entries[n],'alias','') for n in (self.fs,self.char) if n in entries]
        self.key,self.edit_value=option_key(host.shared.opts,self.family)
        self.old_option=getattr(host.shared.opts,self.key) if self.key is not None else None
        self.old_refs=list(self.model.ref_latents); self.old_ini=self.model.ini_latent
        self.old_dynamic=list(host.dynamic.ref_latents); self.old_referencing=host.dynamic.is_referencing
        self.old_edit=getattr(host.dynamic,'edit',None)
        self.closed=False
    def set_p(self,key,value):
        if key not in self.snap: self.snap[key]=(hasattr(self.p,key),getattr(self.p,key,None))
        setattr(self.p,key,value)
    def enter(self):
        p=self.p; cfg=self.cfg; h=self.host
        # Other callbacks may expand or mutate these inputs. Folder batches reuse p.
        # Give callbacks a private list and restore the original input at close.
        for name in ('prompt', 'negative_prompt'):
            if hasattr(p, name): self.set_p(name, copy.deepcopy(getattr(p, name)))
        if self.key is not None:
            setattr(h.shared.opts,self.key,not self.edit_value if cfg['no_ref_diag'] else self.edit_value)
        if self.family=='qwen' and not cfg['no_ref_diag']:
            # Qwen-Image-Edit: keep dynamic_args.edit on so Forge routes
            # references through its vision-token path.
            h.dynamic.edit=True
        self.model.ref_latents=[]; self.model.ini_latent=None; h.dynamic.ref_latents=[]
        self.set_p('batch_size',1); self.set_p('n_iter',max(1,int(self.snap['batch_size'][1])*int(p.n_iter)))
        if p.scripts is not None: self.set_p('scripts',ScriptGuard(p.scripts,self))
        target_faces=self.owner.analyzer.faces(self.original)
        face_index=cfg['target_face']-1
        if target_faces and face_index>=len(target_faces): raise ValueError(f'Target face {face_index+1} not found; detected {len(target_faces)}.')
        self.target_pose=target_faces[face_index] if target_faces else None
        self.poses=[]
        for ref in self.refs:
            fs=self.owner.analyzer.faces(ref); self.poses.append(fs[0] if fs else None)
        self.scored=core.score_refs(self.refs,self.poses,self.original,self.target_pose,self.labels)
        self.analysis={'faces_detected':len(target_faces),'target_face':face_index+1,'head_px':(self.target_pose or {}).get('head_px'),
            'scores':self.scored,'detector':self.owner.analyzer.mode or self.owner.analyzer.error}
        core.select_reference(self.scored,cfg,0,len(self.refs))
        self.target=self.original
        self.target_location=None
        if len(target_faces)>1 and self.target_pose and cfg['edit_scope']=='Full image edit':
            x0,y0,x1,y1=self.target_pose['box']
            self.target_location=((x0+x1)/2/self.original.width,(y0+y1)/2/self.original.height)
            self.analysis['multiple_people_warning']='Full-image identity selection is prompt-guided. Protected head edit provides mask-based isolation.'
        if cfg['keep_original_canvas'] and getattr(p,'image_mask',None) is not None:
            raise ValueError('Keep original canvas needs the plain img2img tab. Turn it off to use Forge inpainting, or use the extension mask with Protected head edit.')
        if cfg['edit_scope']=='Protected head edit':
            if cfg['no_ref_diag']: raise ValueError('Protected head edit cannot be combined with no-reference diagnostics.')
            if getattr(p,'image_mask',None) is not None: raise ValueError('Use the plain img2img tab with Protected head edit; upload an optional mask inside the extension.')
            self.region=core.build_region(self.original,self.target_pose,cfg['crop_padding'],cfg['mask_feather'],cfg['custom_mask'])
            self.target=self.region.crop
            side=self.requested_side(); w,hh=self.target.size
            scale=side/max(w,hh); w=max(64,round(w*scale/64)*64); hh=max(64,round(hh*scale/64)*64)
            self.canvas_size=(w,hh)
            self.target,self.canvas_box=core.fit_canvas(self.target,self.canvas_size)
            self.set_p('init_images',[self.target]); self.set_p('width',w); self.set_p('height',hh)
            self.set_p('resize_mode',0); self.set_p('resize_by',1.0)
            self.set_p('color_corrections',[]); self.set_p('restore_faces',False)
        elif cfg['keep_original_canvas']:
            # Preserve the user's sampling-size budget while respecting target geometry.
            side=max(64,int(max(p.width,p.height))); w,hh=self.original.size
            scale=side/max(w,hh)
            self.canvas_size=(max(64,round(w*scale/64)*64),max(64,round(hh*scale/64)*64))
            self.target,self.canvas_box=core.fit_canvas(self.original,self.canvas_size)
            self.set_p('init_images',[self.target]); self.set_p('width',self.canvas_size[0]); self.set_p('height',self.canvas_size[1])
            self.set_p('resize_mode',0); self.set_p('resize_by',1.0)
        if cfg['geometry_match'] or cfg['match_sharpness']:
            # Host face restoration can replace identity before our quality comparison.
            self.set_p('restore_faces',False)
        if cfg['quality_strict']: self.set_p('color_corrections',[])
        # Any script-stage error is checked before VAE init and before sampling.
        for name in ('init','setup_conds','sample'):
            original=getattr(p,name,None)
            if not callable(original): continue
            def guard(*args,_original=original,**kwargs):
                if self.error: raise RuntimeError(self.error)
                if self.cancelled: raise GenerationCancelled(self.cancelled)
                return _original(*args,**kwargs)
            self.set_p(name,guard)
    def requested_side(self):
        import re
        match=re.search(r'\d+',str(self.cfg['resolution_dropdown']))
        side=int(match.group()) if match else 1024
        if self.cfg['auto_adapt'] and self.target_pose and self.target_pose['head_px']<220: side=max(side,1280)
        return min(2048,max(256,side))
    def fail(self,e):
        if isinstance(e,GenerationCancelled):
            if self.error or self.cancelled: return
            self.cancelled=str(e)
            self.model.ref_latents=[]; self.model.ini_latent=None; self.host.dynamic.ref_latents=[]
            self.owner.last_report.update(status='Cancelled',cancellation=self.cancelled,completed_outputs=len(self.completed))
            self.host.shared.state.textinfo='Klein: stopped; completed images retained.'
            print('[UniversalHeadSwap] Stopped during reference preparation; completed images retained.')
            return
        if self.error: return  # Preserve the first error instead of wrapping/logging it twice.
        self.error=f'Universal Head Swap: {type(e).__name__}: {e}'
        self.model.ref_latents=[]; self.model.ini_latent=None; self.host.dynamic.ref_latents=[]
        self.owner.last_report={**self.owner.last_report,'version':core.VERSION,'error':self.error,'analysis':self.analysis}
        self.host.shared.state.textinfo=self.error
        print(self.error)
    def process(self):
        p=self.p; cfg=dict(self.cfg)
        if any(core.has_speed_lora(text) for text in p.all_prompts):
            cfg['ban_channel']='Positive-only (fast, CFG 1.0)'
        self.input_prompts=list(p.all_prompts); self.input_negatives=list(p.all_negative_prompts)
        if cfg['seed_lock']:
            self.set_p('all_seeds',[p.all_seeds[0]]*len(p.all_seeds)); self.set_p('all_subseeds',[p.all_subseeds[0]]*len(p.all_subseeds))
        self.plans=[]
        qwen_delta=-1 if self.family=='qwen' else 0
        for i,text in enumerate(p.all_prompts):
            negative=p.all_negative_prompts[i]
            preset=self.owner.custom.get('negative_presets',{}).get(cfg['neg_preset_dropdown'],'')
            if preset: negative=core.merge_negatives(negative,[preset])
            plan=core.build_plan(text,negative,cfg,p.all_seeds[i],self.owner.choices,self.fs,self.char,
                                 p.cfg_scale,(self.target_pose or {}).get('head_px'),token_counter(self.model),self.owned_aliases,self.target_location)
            if qwen_delta:
                # Forge's Qwen engine numbers the target image as Picture 0 and
                # prepends its own image prompts; shift the plan's references.
                plan.positive=core.remap_picture_refs(plan.positive,qwen_delta)
                plan.negative=core.remap_picture_refs(plan.negative,qwen_delta)
            self.plans.append(plan)
        p.all_prompts=[x.positive for x in self.plans]; p.all_negative_prompts=[x.negative for x in self.plans]
        self.set_p('cfg_scale',self.plans[0].cfg)
        active=self.plans[0].cfg!=1.0
        p.extra_generation_params['Klein negative guidance']='active' if active else 'inactive (CFG 1.0)'
        print(f'[UniversalHeadSwap] CFG {self.plans[0].cfg:g}; negative guidance '+('active' if active else 'OFF. Select Positive + Negative to use removal negatives.'))
        p.extra_generation_params['Universal Head Swap']=f'{core.VERSION} | {cfg["edit_scope"]} | {self.fs}'
        p.extra_generation_params['Klein settings']=json.dumps({k:cfg[k] for k in core.SAVE_KEYS},ensure_ascii=False,separators=(',',':'))
        if self.region or self.canvas_box: p.extra_generation_params['Klein output size']=f'{self.original.width}x{self.original.height}'
    def _cancel_check(self):
        if self.host.shared.state.interrupted or self.host.shared.state.stopping_generation:
            raise GenerationCancelled('Generation cancelled during reference preparation.')

    def record_output(self,image,index):
        """Retain final callback outputs for a graceful stop between reference encodes."""
        p=self.p
        info=self.processing.create_infotext(p,p.prompts,p.seeds,p.subseeds,index=int(getattr(p,'batch_index',0)),
                                            all_negative_prompts=p.negative_prompts)
        self.completed.append({'image':image,'info':info,'prompt':p.all_prompts[index],
            'negative':p.all_negative_prompts[index],'seed':p.all_seeds[index],'subseed':p.all_subseeds[index]})

    def cancelled_result(self):
        p=self.p; done=self.completed
        result=self.processing.Processed(p,[x['image'] for x in done],
            seed=done[0]['seed'] if done else (p.all_seeds or [-1])[0],
            subseed=done[0]['subseed'] if done else (p.all_subseeds or [-1])[0],
            info=done[0]['info'] if done else 'Universal Head Swap stopped before an image was completed.',
            all_prompts=[x['prompt'] for x in done],all_negative_prompts=[x['negative'] for x in done],
            all_seeds=[x['seed'] for x in done],all_subseeds=[x['subseed'] for x in done],
            infotexts=[x['info'] for x in done],extra_images_list=getattr(p,'extra_result_images',[]))
        if self.region or self.canvas_box: result.width,result.height=self.original.size
        result.comments=(getattr(result,'comments','') or '')+'\nUniversal Head Swap: cancelled; completed outputs retained.'
        if p.scripts is not None: p.scripts.postprocess(p,result)
        return result
    def _encode(self,im):
        self._cancel_check(); h=self.host
        key=(id(self.model.forge_objects.vae),core.image_hash(im))
        cached=self.cache.get(key) if self.cfg['cache_encodes'] else None
        if cached is not None: self.hits+=1; return cached
        array=np.moveaxis(np.asarray(im,dtype=np.float32)/255,2,0).copy()
        tensor=h.torch.from_numpy(array).unsqueeze(0)
        if self.family!='qwen': tensor=tensor.to(device=h.devices.device)
        # Capture the reference from Forge itself (correct VAE scaling) in a temporary list.
        saved_refs=self.model.ref_latents; saved_ini=self.model.ini_latent
        previous=h.dynamic.is_referencing
        saved_edit=getattr(h.dynamic,'edit',None)
        self.model.ref_latents=[]
        try:
            if self.family=='qwen':
                # Qwen-Image-Edit consumes pixel-space start images appended to
                # model.ref_latents by its own encode_first_stage; no is_referencing
                # VAE encode happens here. Keep dynamic_args.edit enabled.
                if saved_edit is not None: h.dynamic.edit=True
                self.model.ref_latents=[tensor.movedim(1,-1).contiguous().cpu()]
                self.encodes+=1
                if self.cfg['cache_encodes']:
                    self.cache.put(key,self.model.ref_latents[0],
                                   self.model.ref_latents[0].numel()*self.model.ref_latents[0].element_size())
                return self.model.ref_latents[0]
            h.dynamic.is_referencing=True
            with h.torch.inference_mode(): h.encode(tensor,0,self.model)
            if len(self.model.ref_latents)!=1: raise RuntimeError('Forge did not produce exactly one reference latent. Check reference-mode compatibility.')
            latent=self.model.ref_latents[0].detach().cpu().contiguous()
        finally:
            self.model.ref_latents=saved_refs; self.model.ini_latent=saved_ini; h.dynamic.is_referencing=previous
            if saved_edit is not None: h.dynamic.edit=saved_edit
            del tensor
        self.encodes+=1
        if self.cfg['cache_encodes']: self.cache.put(key,latent,latent.numel()*latent.element_size())
        return latent
    def batch(self,index):
        self._cancel_check(); h=self.host; p=self.p; cfg=self.cfg
        if self.error: raise RuntimeError(self.error)
        if cfg['no_ref_diag']:
            self.model.ref_latents=[]; self.model.ini_latent=None; h.dynamic.ref_latents=[]; return
        selected,reason=core.select_reference(self.scored,cfg,index,len(self.refs))
        ref=core.prepare_reference(self.refs[selected],self.poses[selected],self.target_pose,cfg['reference_framing'])
        try:
            total=h.memory.get_total_memory(h.devices.device); free=h.memory.get_free_memory(h.devices.device)
        except Exception:
            total,free=8*1024**3,2*1024**3
        max_side,budget=core.memory_limits(total,free,self.requested_side(),cfg['reference_budget'])
        retries=0
        while True:
            pair=core.prepare_pair(self.target,ref,max_side,budget)
            try:
                latents=[self._encode(im) for im in pair]
                break
            except Exception as e:
                oom=isinstance(e,h.torch.OutOfMemoryError) or 'out of memory' in str(e).lower()
                if not oom or retries or max_side<=512: raise
                retries+=1; self.cache.clear(); h.devices.torch_gc(); max_side=max(512,int(max_side*0.75)//64*64); budget*=0.6
                print(f'[UniversalHeadSwap] Reference encoding ran out of memory; retrying once at maximum side {max_side}.')
        amount=cfg['latent_sharpness']
        if amount>0 and self.family=='klein':
            import torch.nn.functional as F
            latent=latents[1]; k=min(int(str(cfg['latent_kernel_size']).split('x')[0]),15)
            k=max(3,k if k%2 else k-1); k=min(k,min(latent.shape[-2:])*2-1)
            with h.torch.inference_mode():
                blurred=F.avg_pool2d(F.pad(latent.float(),(k//2,)*4,mode='reflect'),k,stride=1)
                enhanced=latent.float()+amount*(latent.float()-blurred)
                mean=latent.float().mean(dim=(-2,-1),keepdim=True); std=latent.float().std(dim=(-2,-1),keepdim=True).clamp_min(1e-6)
                latents[1]=enhanced.clamp(mean-4*std,mean+4*std).to(latent.dtype)
        self._cancel_check()
        if self.family=='qwen':
            # Qwen-Image-Edit: keep the pixel-space target in ini_latent (Forge
            # prepends it as Picture 0 itself) and hand over only the headshot.
            self.model.ref_latents=[latents[1]]; self.model.ini_latent=latents[0]
        else:
            self.model.ref_latents=latents; self.model.ini_latent=None
        h.dynamic.ref_latents=[]
        p.clear_prompt_cache()
        p.extra_generation_params['Klein reference']=f'slot {selected+1}: {self.labels[selected]} | {reason}'
        self.selected_ref=selected
        p.extra_generation_params['Klein reference dimensions']=' + '.join(f'{im.width}x{im.height}' for im in pair)
        p.extra_generation_params['Klein reference fingerprints']=','.join(core.image_hash(im)[:16] for im in (self.original,self.refs[selected]))
        p.extra_generation_params['Klein resolved choices']=json.dumps(self.plans[index].choices,ensure_ascii=False)
        report={'version':core.VERSION,'image':index+1,'selected_slot':selected+1,'reason':reason,'analysis':self.analysis,
            'plan':self.plans[index].report(),'encoded_sizes':[im.size for im in pair],'encode_cache_hits':self.hits,
            'vae_encodes':self.encodes,'memory_retry':bool(retries),'crop_box':self.region.box if self.region else None}
        self.owner.last_report=report
        h.shared.state.textinfo=f'Klein: reference {selected+1}/{len(self.refs)}; {reason}; {self.hits} cached encodes reused'
        print('[UniversalHeadSwap] '+h.shared.state.textinfo)
    def finish_image(self,image,index):
        cfg=self.cfg
        image=image.convert('RGB')
        grid_report=None
        if cfg['moire_enabled'] and cfg['moire_strength']>0:
            from .moire import auto_degrid
            image,grid_report=auto_degrid(image,cfg['moire_strength'])
        if self.region:
            x0,y0,x1,y1=self.region.box
            baseline=self.original.crop(self.region.box)
            image=image.resize(self.canvas_size,Image.Resampling.LANCZOS).crop(self.canvas_box).resize(baseline.size,Image.Resampling.LANCZOS)
            target_pose=core.scale_pose(self.target_pose,baseline.size,baseline.size,(x0,y0))
        else:
            baseline=self.original
            if self.canvas_box:
                image=image.resize(self.canvas_size,Image.Resampling.LANCZOS).crop(self.canvas_box).resize(self.original.size,Image.Resampling.LANCZOS)
            target_pose=core.scale_pose(self.target_pose,self.original.size,image.size)
        quality={'output_size':image.size,'geometry_enabled':cfg['geometry_match'],'sharpness_enabled':cfg['match_sharpness']}
        if grid_report is not None: quality['fine_grid_cleanup']=grid_report
        generated_pose=None
        if cfg['geometry_match'] or cfg['match_sharpness']:
            generated_pose=core.nearest_face(self.owner.analyzer.faces(image),target_pose,image.size)
            if not target_pose or not generated_pose:
                quality['warning']='Quality could not be measured: the target or corresponding generated face was not detected.'
            else:
                if cfg['geometry_match']:
                    quality['geometry_before']=core.geometry_report(target_pose,generated_pose)
                if cfg['geometry_match'] and cfg.get('geometry_correct',False):
                    try:
                        corrected,correction=core.correct_head_scale(image,target_pose,generated_pose)
                        verified=core.nearest_face(self.owner.analyzer.faces(corrected),target_pose,corrected.size)
                        if verified is None: raise ValueError('Face detection failed after head correction; kept the uncorrected image.')
                        measured=core.geometry_report(target_pose,verified)
                        before=quality['geometry_before']
                        height=max(8,target_pose['head_h'])
                        before_cost=before['head_height_error_percent']+100*before['head_center_error_px']/height
                        after_cost=measured['head_height_error_percent']+100*measured['head_center_error_px']/height
                        if after_cost>before_cost+1:
                            raise ValueError('Head correction did not improve measured scale; kept the uncorrected image.')
                        image=corrected; generated_pose=verified; quality['correction']=correction
                    except ValueError as e: quality['geometry_warning']=str(e)
                if cfg['match_sharpness']:
                    baseline_pose=core.scale_pose(target_pose,image.size,baseline.size)
                    image,quality['sharpness']=core.match_local_sharpness(image,baseline,baseline_pose,generated_pose)
        if cfg['dof_blur']>0:
            if self.region: pose=None
            else:
                faces=self.owner.analyzer.faces(image); pose=faces[min(cfg['target_face']-1,len(faces)-1)] if faces else None
            if pose:
                mask=core.build_region(image,pose,0.3,0.2).mask
                image=Image.composite(image,image.filter(ImageFilter.GaussianBlur(cfg['dof_blur'])),mask)
        if self.region: image=core.composite_region(image,self.region,cfg['color_match'])
        # Sharpening, tone mapping and grain run on the finished frame. Applied to the
        # protected crop instead, they stopped at the head mask and left a tonal and
        # texture step around the hairline and neck.
        if cfg['sharpness']>0: image=image.filter(ImageFilter.UnsharpMask(radius=1.5,percent=round(cfg['sharpness']),threshold=3))
        if cfg['hdr_enable']:
            x=np.asarray(image,dtype=np.float32)/255; linear=np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
            tone=np.clip((linear*(2.51*linear+0.03))/(linear*(2.43*linear+0.59)+0.14),0,1)
            y=(1-cfg['hdr_gain'])*linear+cfg['hdr_gain']*tone; x=np.where(y<=0.0031308,y*12.92,1.055*y**(1/2.4)-0.055)
            image=Image.fromarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
        if cfg['grain_amount']>0:
            seed=self.plans[min(index,len(self.plans)-1)].seed
            rng=np.random.default_rng(seed%(2**32)); arr=np.asarray(image,dtype=np.float32)
            image=Image.fromarray(np.clip(arr+rng.normal(0,cfg['grain_amount']*255,arr.shape[:2])[...,None],0,255).astype(np.uint8))
        # Measure the finished result again, including compositing and optional effects.
        if cfg['geometry_match'] or cfg['match_sharpness']:
            final_target=core.scale_pose(self.target_pose,self.original.size,image.size)
            final_face=core.nearest_face(self.owner.analyzer.faces(image),final_target,image.size)
            if final_target and final_face:
                quality['geometry_final']=core.geometry_report(final_target,final_face)
                original_score=core.metrics(core.face_crop(self.original,self.target_pose))['sharp']
                final_score=core.metrics(core.face_crop(image,final_face))['sharp']
                quality['detail_final']={'original_score':round(original_score,4),'output_score':round(final_score,4),
                                         'detail_target_met':bool(final_score>=0.95*original_score)}
        acceptable=(not cfg['geometry_match'] or quality.get('geometry_final',{}).get('geometry_target_met',False)) and (
                    not cfg['match_sharpness'] or quality.get('detail_final',{}).get('detail_target_met',False))
        quality['quality_gate_passed']=bool(acceptable)
        quality['pixel_head_resizing_enabled']=bool(cfg.get('geometry_correct',False))
        quality['removal_verification']='Prompt-guided only; no tattoo/piercing detector. Inspect all visible skin at 100%.'
        self.quality_history.append(quality)
        self.owner.last_report['quality']=quality
        self.owner.last_report['quality_history']=list(self.quality_history)
        self.p.extra_generation_params['Klein quality']=json.dumps(quality,ensure_ascii=False,separators=(',',':'))
        if cfg['quality_strict'] and not acceptable:
            raise ValueError('Output failed the head-size / detail quality gate and was not saved. Inspect Show last generation report; adjust reference, sampling size or mask context.')
        if not acceptable: self.host.shared.state.textinfo='Klein: output quality needs review; see Show last generation report.'
        auditor=getattr(self.owner,'auditor',None)
        if cfg['identity_check'] and auditor is not None and self.selected_ref is not None:
            from .identity import sample
            ticket=auditor.submit(sample(self.original,self.target_pose),sample(image,final_face),
                [sample(ref,pose) for ref,pose in zip(self.refs,self.poses)],self.selected_ref,cfg['identity_threshold'],
                {'image':index+1,'seed':self.plans[min(index,len(self.plans)-1)].seed},wait_seconds=0)
            self.owner.last_report['identity_ticket']=ticket
        return image
    def close(self):
        if self.closed: return
        self.closed=True
        try:
            if (self.error or self.cancelled) and getattr(self.p,'extra_network_data',None) and not getattr(self.p,'disable_extra_networks',False):
                try: self.host.extra_networks.deactivate(self.p,self.p.extra_network_data)
                except Exception as e: print(f'[UniversalHeadSwap] Adapter cleanup warning after failure: {e}')
        finally:
            if self.key is not None: setattr(self.host.shared.opts,self.key,self.old_option)
            self.model.ref_latents=self.old_refs; self.model.ini_latent=self.old_ini
            self.host.dynamic.ref_latents=self.old_dynamic; self.host.dynamic.is_referencing=self.old_referencing
            if self.old_edit is not None: self.host.dynamic.edit=self.old_edit
            self.cache.clear()
            for key,(exists,value) in self.snap.items():
                if exists: setattr(self.p,key,value)
                elif hasattr(self.p,key): delattr(self.p,key)
            if hasattr(self,'input_prompts'): self.p.all_prompts=self.input_prompts; self.p.all_negative_prompts=self.input_negatives
            self.p._khs_session=None

def install_bridge(processing,host):
    original=processing.process_images_inner
    if getattr(original,'_khs_bridge',False): return original
    @wraps(original)
    def wrapped(*args,**kwargs):
        p=args[0] if args else kwargs.get('p')
        options=getattr(p,'_khs_options',None)
        if not options or not options[0].get('enable') or getattr(p,'_ad_inner',False) or getattr(p,'is_hr_pass',False):
            return original(*args,**kwargs)
        existing=getattr(p,'_khs_session',None)
        if existing is not None and existing.p is p: return original(*args,**kwargs)
        cfg,owner=options
        # Guard before any mutation. Force mode cannot turn an unsupported model into a supported one.
        family,_=model_family(p.sd_model,host)
        if family is None:
            raise ValueError('Head Swap is enabled, but the loaded model is not Flux.2 Klein or Qwen Image Edit. Disable the extension or load a supported checkpoint.')
        if not getattr(p,'init_images',None): raise ValueError('Head Swap needs an img2img target image.')
        if any(im is not p.init_images[0] and core.image_hash(im)!=core.image_hash(p.init_images[0]) for im in p.init_images[1:]):
            raise ValueError('Process one target per request. Use Forge Batch for separate target files.')
        session=None
        try:
            session=Session(p,cfg,owner,host); session.processing=processing; p._khs_session=session; session.enter()
            result=original(*args,**kwargs)
            if session.error: raise RuntimeError(session.error)
            if session.cancelled: raise GenerationCancelled(session.cancelled)
            if session.region or session.canvas_box:
                result.width=session.original.width; result.height=session.original.height
            return result
        except GenerationCancelled as e:
            if session is None: raise
            session.fail(e)
            return session.cancelled_result()
        except Exception as e:
            if session is not None: session.fail(e)
            else: owner.last_report={'version':core.VERSION,'error':str(e)}
            raise
        finally:
            if session is not None: session.close()
    wrapped._khs_bridge=True; wrapped._khs_original=original
    processing.process_images_inner=wrapped
    return wrapped
