"""Universal Head Swap 7.0: dual-model UI and scoped lifecycle integration.

Serves both FLUX.2 Klein and Qwen Image Edit 2.1 checkpoints. The active mode
is detected from the loaded model at generation time; all features are shared.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import json
import gradio as gr
from PIL import Image, ImageDraw
import modules.scripts as scripts
from modules import script_callbacks
from modules.ui_components import InputAccordion

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from khs import core, runtime, saving, identity
from khs.vision import FaceAnalyzer

HOST=None; BRIDGE=None; BRIDGE_ERROR=''
try:
    import torch
    from modules import shared, devices, processing, extra_networks
    from modules.sd_samplers_common import images_tensor_to_samples
    from backend.args import dynamic_args
    from backend import memory_management
    HOST=SimpleNamespace(torch=torch,shared=shared,devices=devices,dynamic=dynamic_args,
                         memory=memory_management,encode=images_tensor_to_samples,extra_networks=extra_networks)
    BRIDGE=runtime.install_bridge(processing,HOST)
except Exception as e:
    HOST=None
    BRIDGE_ERROR=f'Forge bridge unavailable: {e}'
    print('[UniversalHeadSwap] '+BRIDGE_ERROR)
INSTANCES=[]
def unload():
    for instance in INSTANCES:
        instance.analyzer.close()
        instance.auditor.close()
    if BRIDGE is not None and processing.process_images_inner is BRIDGE:
        processing.process_images_inner=BRIDGE._khs_original
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_image_saved(saving.protect_jpeg_metadata)

class UniversalHeadSwap(scripts.Script):
    sorting_priority=900
    def __init__(self):
        self.analyzer=FaceAnalyzer(ROOT/'scripts/models/face_landmarker.task')
        self.auditor=identity.Auditor(ROOT/'scripts/models')
        self.custom={}; self.custom_error=''; self.components={}; self.capture_target=False
        self.last_report={'version':core.VERSION,'status':'No generation in this session yet.'}
        try: self.custom=core.load_custom(ROOT/'scripts/custom_data.json')
        except Exception as e: self.custom_error=str(e); print('[UniversalHeadSwap] '+self.custom_error)
        self.choices=core.choices(self.custom); INSTANCES.append(self)
    def title(self): return 'Universal Head Swap'
    def show(self,is_img2img): return scripts.AlwaysVisible if is_img2img else False
    def after_component(self,component,**kwargs):
        elem=kwargs.get('elem_id') or getattr(component,'elem_id',None)
        if elem=='forge_ui_preset': self.components['forge_preset']=component
        if elem=='setting_sd_model_checkpoint': self.components['checkpoint']=component
        if elem in ('img2img_prompt','img2img_neg_prompt','img2img_seed','img2img_cfg_scale'): self.components[elem]=component
        if elem=='img2img_image': self.capture_target=True
        classes=kwargs.get('elem_classes') or getattr(component,'elem_classes',[]) or []
        if self.capture_target and 'logical_image_background' in classes:
            self.components['target']=component; self.capture_target=False

    def ui(self,is_img2img):
        C={}; defaults=core.DEFAULTS
        def check(key,label,**kw):
            C[key]=gr.Checkbox(label=label,value=bool(defaults[key]),**kw); return C[key]
        def slide(key,label,low,high,step,**kw):
            C[key]=gr.Slider(low,high,value=defaults[key],step=step,label=label,**kw); return C[key]
        def drop(key,label,items,**kw):
            C[key]=gr.Dropdown(choices=items,value=defaults[key],label=label,**kw); return C[key]
        def text(key,label,**kw):
            C[key]=gr.Textbox(label=label,value=str(defaults[key] or ''),**kw); return C[key]
        try: names=list(runtime.registry()) if HOST else []
        except Exception as e: names=[]; print(f'[UniversalHeadSwap] Adapter list unavailable: {e}')
        with InputAccordion(False,label='Universal Head Swap') as C['enable']:
            gr.Markdown('Add your target in **img2img**, add headshots here, then press **Generate**. Model matching is automatic.')
            if BRIDGE_ERROR:
                gr.Markdown('**Unavailable:** '+BRIDGE_ERROR); C['enable'].interactive=False
            if self.custom_error: gr.Markdown('**Custom preset warning:** '+self.custom_error)
            with gr.Tabs() as tabs:
                with gr.Tab('Setup',id='setup'):
                    gr.Markdown('**LoRAs** · Auto selects BFS for your model. Choose your character LoRA below if needed.')
                    with gr.Row():
                        check('auto_model_adapter','Auto-select head-swap LoRA')
                        drop('lora_dropdown','Head-swap LoRA (BFS)',['Auto (match model)']+names)
                        slide('lora_strength','Head-swap strength',0.05,2,0.05)
                    with gr.Row():
                        drop('char_lora_name','Character LoRA (optional)',['None (skip)']+names)
                        slide('char_lora_strength','Character strength',-2,2,0.05)
                    text('char_lora_trigger','Character trigger words')
                    check('strict_adapter','Check face-swap adapter compatibility')
                    refresh=gr.Button('Refresh adapter list',size='sm')
                    gr.Markdown('Auto matches a registered BFS adapter to the loaded model: Klein 4B/9B or Qwen Image Edit.')
                    C['edit_scope']=gr.Radio(choices=[('Whole picture · swap + cleanup','Full image edit'),
                        ('Head only · protect the rest','Protected head edit')],value=defaults['edit_scope'],label='Editing mode')
                    C['headshots']=gr.Gallery(label='Reference headshots · up to 20',columns=5,height=165,type='pil',format='png',interactive=True,allow_preview=True)
                    summary=gr.Markdown('**Cleanup:** remove tattoos, piercings, forehead marks and jewelry. **Negative prompts active.**')
                    with gr.Row():
                        analyze=gr.Button('Check setup',variant='primary')
                        edit_mask=gr.Button('Edit head mask',visible=False)
                    C['ratio_status']=gr.Textbox(label='Setup status',value='Add headshots, then check the selected reference and edit area.',interactive=False,lines=1)
                    with gr.Accordion('Setup preview',open=False) as preview_section:
                        with gr.Row():
                            preview_image=gr.Image(label='Target and edit area',type='pil',format='png',height=250,interactive=False)
                            preview_ref=gr.Image(label='Selected reference',type='pil',format='png',height=250,interactive=False)
                    with gr.Accordion('Reference selection',open=False):
                        with gr.Row():
                            drop('pick_mode','Selection',['Best match (smart, no rotation)','Rotate good matches','Always rotation'])
                            drop('manual_slot','Specific headshot',['Auto (no override)']+[str(i) for i in range(1,21)])
                        with gr.Row():
                            slide('target_face','Target face · largest first',1,20,1)
                            drop('reference_framing','Reference framing',['Head crop (recommended)','Unmodified reference','Match target framing (experimental)'])
                        with gr.Row():
                            check('auto_pick_best','Score references automatically')
                            check('rotate_top_only','Rotate only good matches')
                        check('seed_lock','Reuse the first seed for this target')
                with gr.Tab('Appearance',id='look'):
                    gr.Markdown('**Cleanup preferences**')
                    labels={'bindi':'Forehead marks / sindoor','earrings':'Earrings','tattoos':'Tattoos / henna',
                            'piercings':'Body piercings','cross':'Cross symbols','jewelry':'Other jewelry'}
                    for row in (('tattoos','piercings','bindi'),('earrings','jewelry','cross')):
                        with gr.Row():
                            for key in row: drop('ban_'+key,labels[key],core.POLICIES)
                    check('removal_priority','Prioritize removal over conflicting prompts')
                    with gr.Row():
                        drop('ban_channel','Guidance mode',['Positive + Negative (uses at least CFG 1.1)','Positive-only (fast, CFG 1.0)'])
                        slide('blend_slider','Swap instruction strength',0,100,5)
                    guidance_note=gr.Markdown('Negative prompts active. Effective CFG is at least 1.1; additional guidance can take longer.')
                    use_negatives=gr.Button('Use negative prompts for cleanup',size='sm')
                    gr.Markdown('Remove jewelry or piercings also requires bare ears. Inspect visible skin: removal instructions are not a guarantee.')
                    with gr.Accordion('Hair, expression and style',open=False):
                        for category in core.CATEGORIES:
                            with gr.Row():
                                C[category]=gr.Dropdown(choices=self.choices[category],value=[],multiselect=True,
                                    allow_custom_value=True,label=category.replace('_',' ').capitalize(),scale=4)
                                if category!='earrings': check('rand_'+category,'Choose one per image',scale=1,min_width=145)
                        slide('expression_strength','Expression emphasis',0.1,2,0.05)
                    with gr.Accordion('Prompt controls',open=False):
                        drop('blend_order','Instruction order',['Extension instruction first','User prompt first'])
                        with gr.Row():
                            check('auto_prompt','Build swap wording when the main prompt is empty')
                            check('prevent_extra_head','Prevent a duplicate head')
                        drop('neg_preset_dropdown','Negative preset',['None']+list(self.custom.get('negative_presets',{})))
                        check('neg_prompt_enable','Append additional negatives')
                        text('neg_prompt_text','Additional negative prompt',lines=3)
                with gr.Tab('Quality & mask',id='detail'):
                    with gr.Row():
                        check('geometry_match','Check original head size and position')
                        check('match_sharpness','Match original face detail')
                    check('keep_original_canvas','Return original dimensions and aspect ratio')
                    with gr.Row():
                        check('ratio_lock','Also guide original head scale in the prompt')
                        drop('ratio_mode','Head-scale guidance',core.RATIO_MODES)
                    check('quality_strict','Reject outputs that fail enabled size or detail checks')
                    gr.Markdown('Checks measure face height, position and local contrast. Review hair, neck, shoulders and actual texture separately.')
                    with gr.Accordion('Fine-grid cleanup · optional',open=False):
                        check('moire_enabled','Automatically reduce fine 2-pixel grids')
                        slide('moire_strength','Cleanup strength',0,1,0.05)
                        gr.Markdown('Off by default. Skips images without a detected grid; limits changes to protect fine texture. Compare at 100% zoom. This is not tattoo or piercing removal.')
                    with gr.Accordion('Head mask and blending',open=False) as mask_section:
                        gr.Markdown('For Head only mode: **white edits, black preserves**. A custom mask must match the target dimensions.')
                        C['custom_mask']=gr.ImageEditor(label='Optional edit mask',type='pil',image_mode='RGB',height=300,
                            brush=gr.Brush(colors=['#ffffff','#000000'],color_mode='fixed'),format='png',
                            sources=['upload','clipboard'],transforms=[])
                        mask_button=gr.Button('Create mask from detected head')
                        with gr.Row():
                            slide('crop_padding','Context around head',0.2,1.5,0.05)
                            slide('mask_feather','Blend edge softness',0,0.25,0.01)
                        slide('color_match','Boundary color matching',0,0.5,0.05)
                    with gr.Accordion('Finishing effects',open=False):
                        check('hdr_enable','SDR HDR-look tone mapping')
                        with gr.Row():
                            slide('hdr_gain','Tone-map blend',0,1,0.05)
                            slide('sharpness','Manual output sharpening',0,200,5)
                        with gr.Row():
                            slide('grain_amount','Film grain',0,0.1,0.005)
                            slide('dof_blur','Background blur around head',0,20,0.5)
                        gr.Markdown('Tone mapping is SDR, not HDR10. Blur uses a head ellipse, not whole-body segmentation. Protected-mode effects stay inside its mask.')
                    with gr.Accordion('Experimental enhancements',open=False):
                        check('geometry_correct','Resize head pixels after generation · may affect hair/neck seams')
                        gr.Markdown('Off by default. Size checking and prompt guidance work without moving finished pixels. '
                            'Only small resizing adjustments are allowed; use a reviewed head mask for larger corrections.')
                        with gr.Row():
                            slide('latent_sharpness','Latent sharpening · 0 disables',0,2,0.1)
                            drop('latent_kernel_size','Latent sharpening kernel',['3x3','5x5','7x7','9x9','15x15'])
                        check('blend_lora_boost','Boost adapter above instruction strength 50')
                        check('tiny_head_boost','Small-head adapter boost')
                with gr.Tab('Speed & memory',id='memory'):
                    with gr.Row():
                        drop('resolution_dropdown','Maximum reference side',['512','768','1024','1280','1536','2048'])
                        slide('reference_budget','Combined reference budget · megapixels',0.5,8,0.25)
                    with gr.Row():
                        check('cache_encodes','Reuse unchanged encodings')
                        check('auto_adapt','Allow more reference detail for small heads')
                    gr.Markdown('Limits cover both references. Sampling memory also depends on the main Forge image size.')
                with gr.Tab('Saved setups',id='presets'):
                    try: presets=core.load_presets(ROOT/'scripts/headswap_presets.json')
                    except Exception as e: presets={}; gr.Markdown('**Preset error:** '+str(e))
                    with gr.Row():
                        C['preset_dropdown']=gr.Dropdown(choices=list(presets),label='Saved setup',value=None)
                        C['preset_save_name']=gr.Textbox(label='New preset name')
                    with gr.Row():
                        C['preset_save_btn']=gr.Button('Save current setup')
                        C['preset_delete_btn']=gr.Button('Delete selected preset')
                    preset_status=gr.Textbox(label='Preset status',interactive=False,lines=1)
                    gr.Markdown('Includes character adapter settings. Images and masks are excluded. Review cleanup choices after loading older presets.')
                with gr.Tab('Report',id='report'):
                    C['blend_preview']=gr.Textbox(label='Resolved next-image prompt',lines=9,interactive=False)
                    gr.Markdown('Reads plain img2img. Seed −1 shows a labeled example. For Batch/Inpaint, the actual generation report is authoritative.')
                    last_run=gr.Button('Show last generation report')
                    C['pos_status']=gr.Textbox(label='Last generation summary',lines=4,interactive=False)
                    analysis_json=gr.JSON(label='Detailed report')
                    check('no_ref_diag','Diagnostic generation without references')
            for key in core.ARG_KEYS:
                if key not in C: C[key]=gr.Checkbox(value=bool(defaults.get(key)),visible=False,label='Legacy '+key)
            all_inputs=[C[k] for k in core.ARG_KEYS]; save_inputs=[C[k] for k in core.SAVE_KEYS]
            policy_keys=['ban_'+k for k in core.APPEARANCE]
            def summarize(scope,channel,*policies):
                removed=[labels[key] for key,value in zip(core.APPEARANCE,policies) if value=='Remove']
                place='Inside head mask only' if scope=='Protected head edit' else 'Whole picture'
                cleanup='Remove: '+', '.join(removed) if removed else 'Automatic removal off; using your appearance choices'
                fast='Positive-only' in str(channel)
                note='**Negative prompts OFF at CFG 1.0.**' if fast else '**Negative prompts active.**'
                return f'**{place}.** {cleanup}. {note}',gr.update(visible=scope=='Protected head edit')
            summary_inputs=[C['edit_scope'],C['ban_channel']]+[C[k] for k in policy_keys]
            for component in summary_inputs: component.change(summarize,inputs=summary_inputs,outputs=[summary,edit_mask],queue=False)
            def guidance(channel):
                return ('**Your negative prompts are ignored at CFG 1.0.** Use Positive + Negative for removal negatives.'
                    if 'Positive-only' in str(channel) else 'Negative prompts active. Effective CFG is at least 1.1; additional guidance can take longer.')
            C['ban_channel'].change(guidance,inputs=C['ban_channel'],outputs=guidance_note,queue=False)
            use_negatives.click(lambda:gr.update(value='Positive + Negative (uses at least CFG 1.1)'),outputs=C['ban_channel'],queue=False)
            edit_mask.click(lambda:(gr.update(selected='detail'),gr.update(open=True)),outputs=[tabs,mask_section],queue=False)
            def sync_adapter(preset,checkpoint,automatic):
                if not automatic: return gr.update()
                key=str(preset).lower()
                if 'qwen' in key: family,size='qwen',None
                elif 'klein' in key:
                    checkpoint=str(checkpoint or '').lower()
                    family,size='klein',4 if '4b' in checkpoint or '4b' in key else 9
                else: return gr.update(value='Auto (match model)')
                try:
                    import networks
                    networks.list_available_networks()
                    entries=runtime.registry()
                    choice=core.select_adapter({name:item.metadata for name,item in entries.items()},size,family)
                    return gr.update(choices=['Auto (match model)']+list(entries),value=choice)
                except Exception: return gr.update(value='Auto (match model)')
            preset_component=self.components.get('forge_preset')
            checkpoint_component=self.components.get('checkpoint')
            try:
                from modules_forge import main_entry
                preset_component=getattr(main_entry,'ui_forge_preset',None) or preset_component
                checkpoint_component=getattr(main_entry,'ui_checkpoint',None) or checkpoint_component
            except ImportError:
                pass
            if preset_component is not None and checkpoint_component is not None:
                inputs=[preset_component,checkpoint_component,C['auto_model_adapter']]
                gr.context.Context.root_block.load(sync_adapter,inputs=inputs,outputs=C['lora_dropdown'],queue=False)
                for control in inputs:
                    control.change(sync_adapter,inputs=inputs,outputs=C['lora_dropdown'],queue=False)
            def refresh_adapters():
                import networks
                networks.list_available_networks(); ns=list(networks.available_networks)
                return gr.update(choices=['Auto (match model)']+ns),gr.update(choices=['None (skip)']+ns)
            refresh.click(refresh_adapters,outputs=[C['lora_dropdown'],C['char_lora_name']],queue=False)
            def save(name,*values):
                try:
                    ps=core.save_preset(ROOT/'scripts/headswap_presets.json',name,dict(zip(core.SAVE_KEYS,values)))
                    return gr.update(choices=list(ps),value=name),'Saved, including character adapter settings.'
                except Exception as e: raise gr.Error(str(e))
            def load(name):
                try:
                    cfg=core.load_presets(ROOT/'scripts/headswap_presets.json').get(name)
                    return [gr.update(value=cfg[k]) if cfg else gr.update() for k in core.SAVE_KEYS]
                except Exception as e: raise gr.Error(str(e))
            def delete(name):
                try:
                    ps=core.save_preset(ROOT/'scripts/headswap_presets.json',name,delete=True)
                    return gr.update(choices=list(ps),value=None),'Preset deleted.'
                except Exception as e: raise gr.Error(str(e))
            C['preset_save_btn'].click(save,inputs=[C['preset_save_name']]+save_inputs,outputs=[C['preset_dropdown'],preset_status],queue=False)
            C['preset_dropdown'].change(load,inputs=[C['preset_dropdown']],outputs=save_inputs,queue=False)
            C['preset_delete_btn'].click(delete,inputs=[C['preset_dropdown']],outputs=[C['preset_dropdown'],preset_status],queue=False)
            last_run.click(lambda:(json.dumps(self.generation_report(),indent=2,ensure_ascii=False),self.generation_report()),outputs=[C['pos_status'],analysis_json],queue=False)
            required=[self.components.get(k) for k in ('target','img2img_prompt','img2img_neg_prompt','img2img_seed','img2img_cfg_scale')]
            if all(x is not None for x in required):
                def preview_and_open(*values): return (*self.preview(*values),gr.update(open=True))
                analyze.click(preview_and_open,inputs=required+all_inputs,outputs=[C['blend_preview'],C['ratio_status'],preview_image,preview_ref,analysis_json,preview_section])
                def make_mask(image,face,padding):
                    try:
                        im=core.rgb_image(image); faces=self.analyzer.faces(im); idx=int(face)-1
                        if idx>=len(faces): raise ValueError('Selected face not detected. Upload a custom mask instead.')
                        mask=core.build_region(im,faces[idx],float(padding),0).mask.convert('RGB')
                        return {'background':mask,'layers':[],'composite':mask}
                    except Exception as e: raise gr.Error(str(e))
                mask_button.click(make_mask,inputs=[required[0],C['target_face'],C['crop_padding']],outputs=C['custom_mask'])
            else:
                analyze.interactive=False
                C['ratio_status'].value='Batch mode: see the Report tab after generation.'
        self.ui_controls=C
        return [C[k] for k in core.ARG_KEYS]


    def preview(self,image,prompt,negative,seed,guidance,*args):
        try:
            if not HOST: raise ValueError(BRIDGE_ERROR)
            cfg=core.normalize(args); im=core.rgb_image(image); refs,labels=core.gallery_images(cfg['headshots'])
            family,_=runtime.model_family(HOST.shared.sd_model,HOST)
            if family is None: raise ValueError('Load a supported checkpoint (Flux.2 Klein or Qwen Image Edit) to check setup.')
            fs,char=runtime.resolve_adapters(cfg,HOST.shared.sd_model,family)
            faces=self.analyzer.faces(im); index=cfg['target_face']-1
            if faces and index>=len(faces): raise ValueError('Selected target face not detected.')
            pose=faces[index] if faces else None; ref_poses=[]
            for ref in refs:
                found=self.analyzer.faces(ref); ref_poses.append(found[0] if found else None)
            scored=core.score_refs(refs,ref_poses,im,pose,labels); slot,reason=core.select_reference(scored,cfg,0,len(refs))
            preset=self.custom.get('negative_presets',{}).get(cfg['neg_preset_dropdown'],'')
            if preset: negative=core.merge_negatives(negative,[preset])
            example=int(seed or 0)<0; fixed=0 if example else int(seed or 0)
            entries=runtime.registry(); aliases=[getattr(entries[n],'alias','') for n in (fs,char) if n in entries]
            location=None
            if len(faces)>1 and pose and cfg['edit_scope']=='Full image edit':
                x0,y0,x1,y1=pose['box']; location=((x0+x1)/2/im.width,(y0+y1)/2/im.height)
            plan=core.build_plan(prompt,negative,cfg,fixed,self.choices,fs,char,float(guidance),(pose or {}).get('head_px'),runtime.token_counter(HOST.shared.sd_model),aliases,location)
            overlay=im.copy()
            if cfg['edit_scope']=='Protected head edit':
                region=core.build_region(im,pose,cfg['crop_padding'],cfg['mask_feather'],cfg['custom_mask'])
                overlay=Image.composite(Image.blend(im,Image.new('RGB',im.size,(50,200,120)),0.35),im,region.mask)
                ImageDraw.Draw(overlay).rectangle(region.box,outline=(50,255,150),width=max(2,im.width//400))
            elif pose: ImageDraw.Draw(overlay).rectangle(pose['box'],outline=(50,255,150),width=max(2,im.width//400))
            report={'version':core.VERSION,'preview_is_example':example,'plan':plan.report(),'scores':scored,'selected_slot':slot+1,'detector':self.analyzer.mode or self.analyzer.error}
            display=('SEED-0 EXAMPLE (main seed is random)\n' if example else f'Seed {fixed}\n')+f'CFG {plan.cfg:g}; LoRA {plan.fs_strength:.2f}\n\n'+plan.positive+('\n\nNEGATIVE (INACTIVE: CFG 1.0)\n' if plan.cfg==1.0 else '\n\nNEGATIVE (ACTIVE)\n')+plan.negative
            if plan.notes: display+='\n\nNOTES\n'+'\n'.join(plan.notes)
            guidance_status='negative prompts OFF' if plan.cfg==1 else 'negative prompts active'
            return display,f'{len(faces)} target faces; slot {slot+1}: {reason}; {guidance_status} (CFG {plan.cfg:g})',overlay,refs[slot],report
        except Exception as e: raise gr.Error(str(e))

    def generation_report(self):
        report=dict(self.last_report)
        report['identity_checks']=self.auditor.snapshot()
        return report

    def headswap_external_context(self,p):
        from khs.external import context
        values=list(getattr(p,'script_args',[]) or [])[self.args_from:self.args_to]
        return context(p,core.normalize(values),self)

    def before_process(self,p,*args):
        if getattr(p,'_ad_inner',False) or getattr(p,'is_hr_pass',False): return
        cfg=core.normalize(args)
        p._khs_options=(cfg,self) if cfg['enable'] else None
        if cfg['enable'] and HOST is None: raise RuntimeError(BRIDGE_ERROR)
    def process(self,p,*args):
        s=getattr(p,'_khs_session',None)
        if s is None or s.p is not p or getattr(p,'_ad_inner',False): return
        try: s.process()
        except Exception as e: s.fail(e)
    def process_batch(self,p,*args,**kwargs):
        s=getattr(p,'_khs_session',None)
        if s is None or s.p is not p or getattr(p,'_ad_inner',False): return
        try: s.batch(int(kwargs.get('batch_number',getattr(p,'iteration',0))))
        except Exception as e: s.fail(e)
    def postprocess_image_after_composite(self,p,pp,*args):
        s=getattr(p,'_khs_session',None)
        if s is None or s.p is not p or getattr(p,'_ad_inner',False): return
        try: pp.image=s.finish_image(pp.image,int(getattr(pp,'index',getattr(p,'iteration',0))))
        except Exception as e:
            s.fail(e); s.set_p('do_not_save_samples',True); s.set_p('do_not_save_grid',True)
            raise
    def postprocess(self,p,processed,*args):
        pass  # Scoped bridge cleans up in finally, including failures and cancellation.
