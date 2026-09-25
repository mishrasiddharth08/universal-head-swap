"""Deterministic planning, image geometry and versioned settings. No Forge imports."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from . import data
from .negative_catalog import CATALOG

VERSION = '7.0.0-rc1'
LEGACY_KEYS = [
    'enable','headshots','lora_dropdown','lora_strength','char_lora_name','char_lora_strength','char_lora_trigger',
    'hairstyle','hair_color','makeup','expression','expression_strength','lighting','age','ethnicity','camera',
    'neg_preset_dropdown','neg_prompt_enable','neg_prompt_text','resolution_dropdown','auto_prompt','force_klein',
    'prevent_extra_head','hdr_enable','hdr_gain','dof_blur','grain_amount','sharpness','latent_sharpness','latent_kernel_size',
    'pick_mode','no_ref_diag','rand_hairstyle','rand_hair_color','rand_makeup','rand_expression','rand_lighting','rand_age',
    'rand_ethnicity','rand_camera','preset_dropdown','preset_save_name','preset_save_btn','preset_delete_btn','seed_lock',
    'block_tattoos','block_crosses','block_sindoor_bindi','block_piercings','perfect_ratio','neg_cfg_boost','blend_slider',
    'blend_order','blend_lora_boost','blend_preview','auto_pick_best','rotate_top_only','manual_slot','auto_adapt',
    'ratio_lock','ratio_mode','ratio_status','dominant_pos','ban_bindi','ban_earrings','ban_tattoos','ban_piercings','ban_cross',
    'pos_status','ban_channel',
]
NEW_KEYS = ['edit_scope','target_face','crop_padding','mask_feather','custom_mask','earrings','reference_framing',
            'reference_budget','tiny_head_boost','strict_adapter','color_match','cache_encodes','ban_jewelry',
            'removal_priority','match_sharpness','geometry_match','quality_strict','keep_original_canvas',
            'identity_check','identity_threshold','geometry_correct','moire_enabled','moire_strength']
ARG_KEYS = LEGACY_KEYS + NEW_KEYS
APPEARANCE = ('bindi','earrings','tattoos','piercings','cross','jewelry')
POLICIES = ['Preserve', 'Remove', 'Use prompt / preset']
RATIO_MODES = ['Strict (match target head size exactly)', 'Strict + neck match', 'Balanced (allow tiny hair-volume change)']
CATEGORIES = ['hairstyle','hair_color','makeup','expression','lighting','age','ethnicity','camera','earrings']
DEFAULTS = {k:False for k in ARG_KEYS}
DEFAULTS.update(enable=False,headshots=None,lora_dropdown='Auto (match model)',lora_strength=1.0,
    char_lora_name='None (skip)',char_lora_strength=0.7,char_lora_trigger='',expression_strength=1.0,
    neg_preset_dropdown='None',neg_prompt_enable=False,neg_prompt_text='',resolution_dropdown='1024',auto_prompt=True,
    prevent_extra_head=True,hdr_gain=0.25,dof_blur=0.0,grain_amount=0.0,sharpness=0,latent_sharpness=0,
    latent_kernel_size='3x3',pick_mode='Best match (smart, no rotation)',manual_slot='Auto (no override)',
    auto_pick_best=True,rotate_top_only=True,auto_adapt=True,ratio_lock=True,ratio_mode=RATIO_MODES[1],
    dominant_pos=True,blend_slider=50,blend_order='Extension instruction first',blend_lora_boost=False,
    ban_channel='Positive + Negative (uses at least CFG 1.1)',edit_scope='Full image edit',target_face=1,crop_padding=0.55,
    mask_feather=0.08,custom_mask=None,reference_framing='Head crop (recommended)',reference_budget=2.5,
    tiny_head_boost=False,strict_adapter=True,color_match=0.0,cache_encodes=True,removal_priority=True,
    match_sharpness=True,geometry_match=True,quality_strict=False,keep_original_canvas=True,
    identity_check=True,identity_threshold=0.363,geometry_correct=False,moire_enabled=False,moire_strength=0.5)
DEFAULTS.update({k:[] for k in CATEGORIES})
DEFAULTS.update({'ban_'+k:'Remove' for k in APPEARANCE})
EXCLUDE_SAVE = {'enable','headshots','custom_mask','preset_dropdown','preset_save_name','preset_save_btn',
                'preset_delete_btn','blend_preview','ratio_status','pos_status'}
DEPRECATED = {'block_tattoos','block_crosses','block_sindoor_bindi','block_piercings','perfect_ratio',
              'neg_cfg_boost','force_klein','dominant_pos'}
SAVE_KEYS = [k for k in ARG_KEYS if k not in EXCLUDE_SAVE | DEPRECATED]
_PRESET_LOCK = threading.RLock()

def normalize(values=None):
    cfg = dict(DEFAULTS)
    if isinstance(values, dict):
        cfg.update({k:v for k,v in values.items() if k in cfg})
    elif values is not None:
        cfg.update(dict(zip(ARG_KEYS, values)))
    for key in APPEARANCE:
        v=cfg['ban_'+key]
        if isinstance(v,bool): v='Remove' if v else 'Preserve'
        cfg['ban_'+key]=v if v in POLICIES else 'Preserve'
    numeric = {'lora_strength':(0.05,2), 'char_lora_strength':(-2,2),'expression_strength':(0.1,2),
               'blend_slider':(0,100),'target_face':(1,20),'crop_padding':(0.2,1.5),'mask_feather':(0,0.25),
               'reference_budget':(0.5,8),'latent_sharpness':(0,2),'hdr_gain':(0,1),'dof_blur':(0,20),
               'grain_amount':(0,0.1),'sharpness':(0,200),'color_match':(0,0.5),'identity_threshold':(0,1),'moire_strength':(0,1)}
    for k,(low,high) in numeric.items():
        try: value=float(cfg[k])
        except (TypeError,ValueError): value=float(DEFAULTS[k])
        cfg[k]=max(low,min(high,value)) if math.isfinite(value) else float(DEFAULTS[k])
    cfg['target_face']=int(cfg['target_face'])
    for key,default in DEFAULTS.items():
        if isinstance(default,bool) and key not in {'ban_'+x for x in APPEARANCE}:
            value=cfg[key]
            cfg[key]=value.strip().lower() in ('true','1','yes','on') if isinstance(value,str) else bool(value)
    resolution=re.search(r'\d+',str(cfg['resolution_dropdown']))
    side=int(resolution.group()) if resolution else 1024
    cfg['resolution_dropdown']=str(min((512,768,1024,1280,1536,2048),key=lambda x:abs(x-side)))
    kernel=re.search(r'\d+',str(cfg['latent_kernel_size']))
    radius=int(kernel.group()) if kernel else 3
    radius=min((3,5,7,9,15),key=lambda x:abs(x-radius))
    cfg['latent_kernel_size']=f'{radius}x{radius}'
    cfg['ban_channel']='Positive-only (fast, CFG 1.0)' if 'Positive-only' in str(cfg['ban_channel']) else 'Positive + Negative (uses at least CFG 1.1)'
    if cfg['edit_scope'] not in ('Full image edit','Protected head edit'): cfg['edit_scope']='Full image edit'
    cfg['pick_mode']={'Always smart':'Best match (smart, no rotation)','Hybrid':'Rotate good matches'}.get(cfg['pick_mode'],cfg['pick_mode'])
    if cfg['pick_mode'] not in ('Best match (smart, no rotation)','Rotate good matches','Always rotation'):
        cfg['pick_mode']='Best match (smart, no rotation)'
    if cfg['ratio_mode'] not in RATIO_MODES: cfg['ratio_mode']=RATIO_MODES[1]
    if cfg['reference_framing'] not in ('Head crop (recommended)','Unmodified reference','Match target framing (experimental)'):
        cfg['reference_framing']='Head crop (recommended)'
    for k in CATEGORIES:
        cfg[k]=[str(x) for x in cfg[k]] if isinstance(cfg[k],(tuple,list)) else ([str(cfg[k])] if cfg[k] else [])
    return cfg

def load_custom(path):
    if not Path(path).exists(): return {}
    with open(path,encoding='utf-8-sig') as f: value=json.load(f)
    if not isinstance(value,dict): raise ValueError('custom_data.json must contain an object.')
    if 'negative_presets' in value:
        if not isinstance(value['negative_presets'],dict): raise ValueError('negative_presets must map preset names to text.')
        value['negative_presets']={str(k):str(v) for k,v in value['negative_presets'].items() if isinstance(v,str)}
    return value

def choices(custom):
    mapping={'hairstyle':('HAIRSTYLE','hairstyles'),'hair_color':('HAIR_COLOR','hair_color'),
        'makeup':('MAKEUP','makeup'),'expression':('EXPRESSION','expressions'),'lighting':('LIGHTING','lighting'),
        'age':('AGE','age'),'ethnicity':('ETHNICITY','ethnicity'),'camera':('CAMERA','camera')}
    result={}
    for k,(prefix,ck) in mapping.items():
        vals=list(getattr(data,prefix+'_SPECIAL',[]))+list(getattr(data,prefix+'_PRESETS',[]))
        vals += custom.get(ck,[]) if isinstance(custom.get(ck),list) else []
        result[k]=list(dict.fromkeys(str(x) for x in vals))
    earrings=custom.get('earrings',[])
    result['earrings']=list(dict.fromkeys(['None (no modifier)','Preserve target earrings']+(earrings if isinstance(earrings,list) else [])))
    return result

def load_presets(path):
    if not Path(path).exists(): return {}
    with _PRESET_LOCK, open(path,encoding='utf-8-sig') as f: payload=json.load(f)
    if not isinstance(payload,dict): raise ValueError('Preset file must contain an object.')
    source=payload.get('presets',{}) if 'schema_version' in payload else payload
    if not isinstance(source,dict): raise ValueError('Invalid preset collection.')
    result={}
    for name,raw in source.items():
        if not isinstance(raw,dict): continue
        raw=dict(raw)
        legacy={'bindi':'block_sindoor_bindi','tattoos':'block_tattoos','cross':'block_crosses','piercings':'block_piercings'}
        for k,old in legacy.items():
            if 'ban_'+k not in raw and old in raw: raw['ban_'+k]=raw[old]
        if 'schema_version' not in payload and raw.get('auto_pick_best') and raw.get('rotate_top_only'):
            raw['pick_mode']='Rotate good matches'
        cfg=normalize(raw)
        result[str(name)]={k:cfg[k] for k in SAVE_KEYS}
    return result

def save_preset(path,name,config=None,delete=False):
    name=str(name or '').strip()
    if not name: raise ValueError('Enter a preset name.')
    with _PRESET_LOCK:
        presets=load_presets(path)
        if delete: presets.pop(name,None)
        else:
            cfg=normalize(config)
            presets[name]={k:cfg[k] for k in SAVE_KEYS}
        atomic_json(path,{'schema_version':2,'extension_version':VERSION,'presets':presets})
        return presets

def atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
            temp=f.name; json.dump(payload,f,indent=2,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path); temp=None
    finally:
        if temp and os.path.exists(temp): os.unlink(temp)

TOKEN = re.compile(r'<lora:([^>]+)>',re.I)
BAN_PATTERNS={
 'bindi':r'\b(?:bindis?|sindoor|tikka|tika|tilaka?|kumkum|chandlo|bottu|pottu|vermilli?on|vermilion|maang tikka)\b',
 'earrings':r'\b(?:earrings?|ear rings?|jhumkas?|(?:hoop|stud|drop|dangling) earrings?|ear studs?)\b',
 'tattoos':r'\b(?:tattoos?|inked skin|body art|sleeve ink|henna|mehndi|body ink)\b',
 'piercings':r'\b(?:piercings?|belly rings?|navel rings?|nose rings?|nose pins?|septum rings?|belly button piercing)\b',
 'cross':r'\b(?:cross(?:es)?(?![- ](?:legged|body|armed|stitch))|crucifixes|crucifix)\b',
 'jewelry':r'\b(?:jewel(?:ry|lery)|earrings?|necklaces?|pendants?|chokers?|bracelets?|bangles?|anklets?|brooches?|nose pins?|wedding rings?|toe rings?)\b',
}
BAN_POS={'bindi':'remove every bindi, sindoor, tikka, tilak, kumkum and forehead ornament; bare forehead and unmarked hair parting',
         'earrings':'remove all earrings and ear ornaments; bare earlobes',
         'tattoos':'remove every tattoo, henna design, body ink and tattoo remnant anywhere on visible skin, including face, neck, torso, back, arms, hands, legs and feet; preserve natural pores and skin texture',
         'piercings':'remove all body piercings and piercing jewelry everywhere, including ears, nose, septum, lips, eyebrows, tongue, chest, navel and dermal studs; natural skin without rings, studs or barbells',
         'cross':'remove all cross symbols, crucifixes and cross ornaments anywhere in the image',
         'jewelry':'remove all jewelry, necklaces, pendants, bangles, bracelets, rings, anklets, brooches and nose pins'}
BAN_NEG={'bindi':['bindi','sindoor','forehead mark'],'earrings':['earrings','ear jewelry'],
         'tattoos':['tattoos','inked skin'],'piercings':['body piercing','navel piercing'],
         'cross':['cross symbol','crucifix']}
BAN_NEG.update({key:CATALOG[key] for key in APPEARANCE})

def affirmative_mention(text,key):
    for match in re.finditer(BAN_PATTERNS[key],str(text).lower()):
        prefix=re.split(r'[.!?;,]|\bbut\b',str(text).lower()[:match.start()])[-1]
        prefix=' '.join(prefix.split()[-6:])
        suffix=str(text).lower()[match.end():match.end()+8]
        if re.search(r'\b(no|not|without|avoid|remove|removing|exclude|free of|do not|don\W?t)\b',prefix) or suffix.startswith(('-free',' free')):
            continue
        return True
    return False

def clean_name(name):
    name=str(name or '').strip()
    m=TOKEN.fullmatch(name)
    if m: name=m.group(1).rsplit(':',1)[0]
    name=re.sub(r'\.(?:safetensors|pt|ckpt|bin)$','',name,flags=re.I)
    return name.replace('\\','/')

def adapter_family(name,metadata=None):
    text=' '.join([str(name)]+[str(v) for k,v in (metadata or {}).items() if k in
                  ('ss_base_model_version','modelspec.architecture','modelspec.title','ss_sd_model_name','base_model')]).lower()
    size=re.search(r'(?:klein[\W_]*|flux2k?[^\s]*?)([49])b\b',text) or re.search(r'\b([49])b\b',text.replace('_',' '))
    if 'klein' in text or re.search(r'flux[\W_]*2k(?:[\W_]*[49]b)?\b',text):
        return 'klein',int(size.group(1)) if size else None
    if 'qwen' in text.replace('_','-') or 'qwen' in text:
        qsize=re.search(r'qwen[\W_-]*(?:image[\W_-]*)?(?:edit[\W_-]*)?(\d+(?:\.\d+)?)',text)
        return 'qwen',float(qsize.group(1)) if qsize else None
    if any(x in text for x in ('sdxl','sd15','sd_1.5','sd1.5','sd 1.5','krea','flux1','flux.1','flux2_dev','flux2-dev')):
        return 'other',None
    return 'unknown',None

FAMILY_LABEL={'klein':'Klein','qwen':'Qwen Image'}

def compatible_adapter(name,size,metadata=None,strict=True,family='klein'):
    found_family,found=adapter_family(name,metadata)
    if found_family=='other' or (found_family!=family and found_family!='unknown'):
        raise ValueError(f'Adapter {name!r} does not match the active {FAMILY_LABEL.get(family,family)} model.')
    if family=='klein' and size and found and size!=found:
        raise ValueError(f'Adapter {name!r} does not match the active Klein {size}B model.')
    if strict and found_family=='unknown':
        raise ValueError(f'Cannot verify {name!r} as a {FAMILY_LABEL.get(family,family)} adapter. Choose a matching BFS adapter or disable strict adapter checking for a known compatible custom adapter.')

def select_adapter(entries,size,family='klein'):
    if family=='klein' and size not in (4,9):
        raise ValueError('Cannot identify the active Klein model size. Select a known compatible adapter manually.')
    eligible=[]
    for name,meta in entries.items():
        found_family,found=adapter_family(name,meta)
        if found_family!=family or not any(x in name.lower() for x in ('bfs','swap')): continue
        if family=='klein' and found!=size: continue
        eligible.append((40*bool(family=='klein' and size and found==size)+20*('bfs_head' in name.lower()),name))
    if not eligible:
        if family=='klein':
            raise ValueError('No verified matching face-swap adapter found. Load Klein and choose its 4B or 9B BFS LoRA.')
        raise ValueError('No verified Qwen Image BFS face-swap adapter found in the LoRA registry. Select one manually.')
    return sorted(eligible,key=lambda item:(-item[0],item[1].lower()))[0][1]

def remap_picture_refs(text,delta):
    """Shift 'Picture N' references by delta.

    Forge's Qwen-Image-Edit engine prepends the target image itself as
    'Picture 0' before the extension's instruction text, while the Klein path
    receives the target as Picture 1. Plans are written Klein-style
    (Picture 1 = target, Picture 2 = headshot); Qwen needs -1.
    """
    if not delta: return text
    return re.sub(r'Picture\s+(\d+)',lambda m:f'Picture {max(0,int(m.group(1))+delta)}',str(text))

def merge_negatives(original,terms):
    # Preserve the original text verbatim; normalize only the appended simple terms.
    existing={re.sub(r'\s+',' ',x.strip().lower()) for x in str(original).split(',')}
    extra=[]
    for term in terms:
        if term.lower() not in existing: extra.append(term); existing.add(term.lower())
    return str(original)+((', ' if original else '')+', '.join(extra) if extra else '')

def _resolve_category(category,values,cfg,all_choices,rng):
    if not values: return ''
    if len(values)>1 and not cfg.get('rand_'+category):
        return ', '.join(filter(None,(_resolve_category(category,[v],cfg,all_choices,rng) for v in values)))
    value=rng.choice(values) if len(values)>1 else values[0]
    if value.startswith(('🔀','🎲')):
        pool=[v for v in all_choices[category] if not v.startswith(('🔀','🎲','None','Retain','Preserve'))]
        value=rng.choice(pool) if pool else ''
    low=value.lower()
    if not value or low.startswith(('none',"don't do anything",'do nothing')): return ''
    if value=='📷 From Target Image' or value=='Preserve target earrings':
        return f'preserve the {category.replace("_"," ")} of Picture 1'
    if value=='📷 From Reference Image':
        return f'use the {category.replace("_"," ")} of Picture 2'
    if value=='🎨 From Character LoRA':
        return f'use the character adapter\'s {category.replace("_"," ")}'
    if category=='hair_color': return f'{value} hair'
    return value

@dataclass
class Plan:
    positive: str
    negative: str
    cfg: float
    seed: int
    fs_strength: float
    choices: dict
    appearance: dict
    notes: list
    token_count: int | None = None
    def report(self): return {**asdict(self),'negative_guidance_active':self.cfg!=1.0}

def build_plan(user,negative,cfg,seed,all_choices,fs_name,char_name='',host_cfg=1.0,head_px=None,token_counter=None,owned_aliases=(),target_location=None):
    cfg=normalize(cfg); rng=random.Random(int(seed)); notes=[]
    owned={clean_name(n).lower() for n in (fs_name,char_name,*owned_aliases) if n}
    extra_tags=[]
    def separate(m):
        body=m.group(1); name=body.split(':')[0]
        if clean_name(name).lower() not in owned: extra_tags.append(m.group(0))
        return ''
    user_text=TOKEN.sub(separate,str(user or '')).strip()
    resolved={}
    for category in CATEGORIES:
        kwname={'hairstyle':'HAIR','expression':'EXPRESSION'}.get(category,category.upper())+'_KEYWORDS'
        kws=getattr(data,kwname,())
        if category!='earrings' and any(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)',user_text,re.I) for w in kws): continue
        phrase=_resolve_category(category,cfg[category],cfg,all_choices,rng)
        if phrase and category=='expression' and cfg['expression_strength']!=1:
            phrase=f'({phrase}:{cfg["expression_strength"]:.2f})'
        if phrase: resolved[category]=phrase
    appearances={}
    for key in APPEARANCE:
        policy=cfg['ban_'+key]
        if policy=='Remove' and affirmative_mention(user_text,key) and not cfg['removal_priority']:
            policy='Use prompt / preset'; notes.append(f'{key}: explicit prompt overrides Remove')
        appearances[key]=policy
    if cfg['removal_priority'] and any(v=='Remove' for v in appearances.values()):
        notes.append('Removal priority is ON: Remove choices take precedence over conflicting text or presets.')
        conflicts=[key for key in APPEARANCE if appearances[key]=='Remove' and affirmative_mention(user_text,key)]
        if conflicts: notes.append('Conflicting user text retained for review; removal instruction takes priority: '+', '.join(conflicts))
    if appearances['jewelry']=='Remove' or appearances['piercings']=='Remove':
        if appearances['earrings']!='Remove': notes.append('Earrings removal is also required by Remove jewelry / piercings.')
        appearances['earrings']='Remove'
    if cfg['edit_scope']=='Protected head edit' and any(appearances[k]=='Remove' for k in ('tattoos','piercings','jewelry')):
        notes.append('Protected head edit only cleans inside the head mask. Use Full image edit for tattoos, piercings or jewelry elsewhere on the body.')
    extras=[]
    for cat,phrase in resolved.items():
        conflict=[k for k in APPEARANCE if appearances[k]=='Remove' and affirmative_mention(phrase,k)]
        if conflict: notes.append(f'Omitted {cat} preset conflicting with Remove: {", ".join(conflict)}')
        else: extras.append(phrase)
    swap=cfg['blend_slider']>0 and (bool(user_text) or cfg['auto_prompt'])
    clauses=[]
    if cfg['removal_priority'] and any(v=='Remove' for v in appearances.values()):
        clauses.append('prioritize all removal requirements below over conflicting appearance requests; reconstruct clean natural skin instead of hiding marks with blur')
    if swap:
        clauses.append('head_swap: use Picture 1 as the target body and scene; replace its head with the facial identity, eye color and nose structure of Picture 2; preserve the pose, expression, outfit, lighting and background of Picture 1')
        if target_location:
            clauses.append(f'apply the identity change only to the head centered {target_location[0]:.0%} from the left and {target_location[1]:.0%} from the top of Picture 1; preserve all other people')
        if not any('hairstyle' in v or 'hair' in v for v in extras): clauses.append('use the head and hair of Picture 2')
    if cfg['ratio_lock']:
        clauses.append('keep the original head scale relative to the shoulders and body of Picture 1')
        if 'neck' in cfg['ratio_mode'].lower(): clauses.append('match the original neck thickness and shoulder junction')
        if 'Balanced' in cfg['ratio_mode']: clauses.append('allow a small natural change in hair volume')
    if cfg['prevent_extra_head'] and swap: clauses.append('one replacement head; remove the original head completely')
    for key,policy in appearances.items():
        if policy=='Remove': clauses.append(BAN_POS[key])
        elif policy=='Preserve': clauses.append(f'preserve allowed {key} from Picture 1 except items explicitly required to be removed')
    if cfg['match_sharpness']:
        clauses.append('match the original focal sharpness and local contrast; preserve fine skin pores, age texture, individual hair strands and fabric detail; avoid waxy or airbrushed skin')
    weight=1.0+cfg['blend_slider']/200.0
    block='; '.join(clauses)
    if block: block=f'({block}:{weight:.2f})'
    strength=cfg['lora_strength']
    if cfg['blend_lora_boost'] and cfg['blend_slider']>50: strength*=1+(cfg['blend_slider']-50)/200
    if cfg['tiny_head_boost'] and head_px and head_px<220: strength*=1.08
    strength=min(2.0,strength)
    tags=list(dict.fromkeys(extra_tags+[f'<lora:{fs_name}:{strength:.2f}>']))
    if char_name: tags.append(f'<lora:{char_name}:{cfg["char_lora_strength"]:.2f}>')
    trigger=str(cfg['char_lora_trigger'] or '').strip() if char_name else ''
    core=[user_text,block] if cfg['blend_order']=='User prompt first' else [block,user_text]
    positive=', '.join(x for x in [trigger,*core,*extras] if x)+' '+ ' '.join(tags)
    negative=str(negative or '')
    if cfg['neg_prompt_enable'] and cfg['neg_prompt_text']: negative=merge_negatives(negative,[cfg['neg_prompt_text'].strip()])
    fast='Positive-only' in cfg['ban_channel']
    guidance=1.0 if fast else max(1.1,float(host_cfg))
    if fast and float(host_cfg)!=1: notes.append(f'Positive-only mode sets CFG {host_cfg:g} to 1.0')
    if fast: notes.append('Negative prompts are inactive at CFG 1.0. Choose Positive + Negative to use your removal negatives.')
    if not fast:
        terms=[t for k in APPEARANCE if appearances[k]=='Remove' for t in BAN_NEG[k]]
        if cfg['match_sharpness']: terms+=CATALOG['quality']
        if cfg['ratio_lock']: terms+=CATALOG['ratio']
        if cfg['prevent_extra_head']: terms+=['duplicate head','extra head']
        negative=merge_negatives(negative,terms)
    count=None
    if token_counter:
        try: count=int(token_counter(positive))
        except Exception as e: notes.append(f'Token count unavailable: {e}')
    if count and count>512: notes.append(f'{count} Qwen3 tokens; all user text retained. Long prompts cost more memory.')
    return Plan(positive.strip(),negative,guidance,int(seed),round(strength,2),resolved,appearances,notes,count)

def rgb_image(value):
    if isinstance(value,(tuple,list)): value=value[0] if value else None
    if isinstance(value,dict):
        value=next((value.get(k) for k in ('composite','image','background','name','path') if value.get(k) is not None),None)
    if hasattr(value,'name') and not isinstance(value,Image.Image): value=value.name
    if isinstance(value,str):
        if value.startswith('data:image/'):
            import base64,io
            with Image.open(io.BytesIO(base64.b64decode(value.split(',',1)[1]))) as im: return rgb_image(im)
        with Image.open(value) as im: return rgb_image(im)
    if isinstance(value,np.ndarray): value=Image.fromarray(value)
    if not isinstance(value,Image.Image): raise ValueError('Upload a valid image.')
    im=ImageOps.exif_transpose(value)
    if im.mode in ('RGBA','LA') or 'transparency' in im.info:
        rgba=im.convert('RGBA'); bg=Image.new('RGBA',im.size,'white'); bg.alpha_composite(rgba); return bg.convert('RGB')
    return im.convert('RGB').copy()

def gallery_images(items,limit=20):
    refs=[]; labels=[]
    if len(items or [])>limit: raise ValueError(f'Upload at most {limit} reference headshots; remove the extra uploads.')
    for i,item in enumerate((items or [])[:limit]):
        label=item[1] if isinstance(item,(tuple,list)) and len(item)>1 else ''
        source=item[0] if isinstance(item,(tuple,list)) else item
        if isinstance(source,dict): label=label or source.get('orig_name') or source.get('name') or ''
        elif isinstance(source,str): label=label or Path(source).name
        try: refs.append(rgb_image(source)); labels.append(str(label or f'Slot {i+1}'))
        except Exception as e: raise ValueError(f'Reference slot {i+1}: {e}') from e
    if not refs: raise ValueError('Upload at least one reference headshot.')
    return refs,labels

def image_hash(im):
    digest=hashlib.sha256(f'{im.mode}:{im.size}'.encode()); digest.update(im.tobytes()); return digest.hexdigest()

def face_crop(im,pose,padding=0.15):
    if not pose or not pose.get('box'): return im
    x0,y0,x1,y1=pose['box']; dx=(x1-x0)*padding; dy=(y1-y0)*padding
    box=(max(0,int(x0-dx)),max(0,int(y0-dy)),min(im.width,int(x1+dx)),min(im.height,int(y1+dy)))
    return im.crop(box) if box[2]>box[0] and box[3]>box[1] else im

def metrics(im):
    g=np.asarray(im.convert('L').resize((192,192)),dtype=np.float32)
    lap=-4*g[1:-1,1:-1]+g[:-2,1:-1]+g[2:,1:-1]+g[1:-1,:-2]+g[1:-1,2:]
    sharp=float(np.clip(np.log10(max(1,float(lap.var())))/3.5,0,1))
    return dict(sharp=sharp,bright=float(g.mean()/255),contrast=float(g.std()/255))

def score_refs(refs,poses,target,target_pose,labels=None,char_tokens=()):
    target_metrics=metrics(face_crop(target,target_pose)); scored=[]
    for i,ref in enumerate(refs):
        pose=poses[i]; stats=metrics(face_crop(ref,pose)); notes=[]
        if pose and target_pose:
            diff=lambda k: abs((pose[k]-target_pose[k]+180)%360-180)
            error=diff('yaw')+0.6*diff('pitch')+0.3*diff('roll'); pose_score=max(0,1-error/100)
        elif pose: pose_score=max(0,1-abs(pose['yaw'])/90)
        else: pose_score=0.25; notes.append('no face detected')
        usable=bool(pose and pose.get('head_px',0)>=48 and stats['sharp']>=0.12)
        framing=min(1,(pose or {}).get('head_px',0)/180)
        light=max(0,1-2*abs(stats['bright']-target_metrics['bright'])-abs(stats['contrast']-target_metrics['contrast']))
        label=(labels[i] if labels else '').lower()
        hint=min(0.02,0.01*sum(bool(re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)',label)) for t in char_tokens))
        score=0.45*pose_score+0.20*stats['sharp']+0.15*framing+0.20*light+hint
        scored.append(dict(index=i,score=round(score,4),usable=usable,pose=round(pose_score,3),sharp=round(stats['sharp'],3),
                           lighting=round(light,3),head_px=(pose or {}).get('head_px'),notes=notes))
    return sorted(scored,key=lambda d:(not d['usable'],-d['score'],d['index']))

def top_cluster(scored,within=0.15):
    if not scored: return []
    usable=[d for d in scored if d.get('usable',True)]
    if not usable: return scored[:1]
    best=usable[0]['score']; return [d for d in usable if d['score']>=best*(1-within)]

def select_reference(scored,cfg,index,ref_count):
    manual=str(cfg['manual_slot'])
    if manual.isdigit():
        slot=int(manual)-1
        if not 0<=slot<ref_count: raise ValueError(f'Reference slot {manual} is not uploaded.')
        return slot,'manual selection'
    mode=cfg['pick_mode']
    if mode=='Always rotation' or not cfg['auto_pick_best']: return index%ref_count,'sequential rotation'
    if mode=='Best match (smart, no rotation)': return scored[0]['index'],'best available match'
    pool=top_cluster(scored) if cfg['rotate_top_only'] else scored
    return pool[index%len(pool)]['index'],'good-match rotation'

def prepare_reference(im,pose,target_pose,mode):
    if mode=='Unmodified reference' or not pose: return im.copy()
    x0,y0,x1,y1=pose['box']; hh=max(1,y1-y0); hw=max(1,x1-x0)
    if mode=='Head crop (recommended)':
        box=(max(0,round(x0-hw*0.45)),max(0,round(y0-hh*0.65)),min(im.width,round(x1+hw*0.45)),min(im.height,round(y1+hh*0.45)))
        return im.crop(box)
    fill=max(0.12,min(0.65,(target_pose or {}).get('head_ratio',0.35)))
    ch=max(im.height,min(round(hh/fill),im.height*4)); cw=max(im.width,round(ch*0.8))
    arr=np.asarray(im); edges=np.concatenate([arr[:4].reshape(-1,3),arr[-4:].reshape(-1,3),arr[:,:4].reshape(-1,3),arr[:,-4:].reshape(-1,3)])
    canvas=Image.new('RGB',(cw,ch),tuple(int(v) for v in np.median(edges,axis=0)))
    px=max(0,min(round(cw/2-(x0+x1)/2),cw-im.width)); py=max(0,min(round(ch/2-(y0+y1)/2),ch-im.height))
    canvas.paste(im,(px,py)); return canvas

def fit_image(im,max_side,scale=1.0):
    factor=min(1.0,float(max_side)/max(im.size))*scale
    w=max(1,round(im.width*factor)); h=max(1,round(im.height*factor))
    # Fit inside aligned dimensions, padding instead of stretching face proportions.
    cw=max(64,min(int(max_side)//64*64,math.ceil(w/64)*64)); ch=max(64,min(int(max_side)//64*64,math.ceil(h/64)*64))
    thumb=ImageOps.contain(im,(cw,ch),Image.Resampling.LANCZOS)
    arr=np.asarray(im.resize((16,16))); color=tuple(int(x) for x in np.median(arr.reshape(-1,3),axis=0))
    result=Image.new('RGB',(cw,ch),color); result.paste(thumb,((cw-thumb.width)//2,(ch-thumb.height)//2)); return result

def fit_canvas(im,size):
    """Fit without stretching and return the exact content box for inverse mapping."""
    thumb=ImageOps.contain(im,size,Image.Resampling.LANCZOS)
    color=tuple(int(v) for v in np.median(np.asarray(im.resize((16,16))).reshape(-1,3),axis=0))
    canvas=Image.new('RGB',size,color); x=(size[0]-thumb.width)//2; y=(size[1]-thumb.height)//2
    canvas.paste(thumb,(x,y)); return canvas,(x,y,x+thumb.width,y+thumb.height)

def nearest_face(faces,target_pose,size):
    """Associate by spatial position, not largest-face ordering after generation."""
    if not faces or not target_pose: return None
    tx0,ty0,tx1,ty1=target_pose['box']; tx=(tx0+tx1)/2; ty=(ty0+ty1)/2
    def distance(p):
        x0,y0,x1,y1=p['box']
        return math.hypot((x0+x1)/2-tx,(y0+y1)/2-ty)
    candidate=min(faces,key=distance)
    if distance(candidate)>max(0.08*math.hypot(*size),0.7*max(tx1-tx0,ty1-ty0)): return None
    return candidate

def geometry_report(target_pose,generated_pose):
    tx0,ty0,tx1,ty1=target_pose['box']; gx0,gy0,gx1,gy1=generated_pose['box']
    if min(tx1-tx0,ty1-ty0,gx1-gx0,gy1-gy0)<=0: raise ValueError('Invalid face box for size measurement.')
    height_change=(gy1-gy0)/(ty1-ty0)-1
    width_change=(gx1-gx0)/(tx1-tx0)-1
    height_error=abs((gy1-gy0)/(ty1-ty0)-1)
    center_error=math.hypot((gx0+gx1-tx0-tx1)/2,(gy0+gy1-ty0-ty1)/2)
    return {'head_height_error_percent':round(height_error*100,2),'head_center_error_px':round(center_error,2),
            'head_height_change_percent':round(height_change*100,2),
            'head_width_change_percent':round(width_change*100,2),
            'head_area_change_percent':round(((1+height_change)*(1+width_change)-1)*100,2),
            'geometry_target_met':bool(height_error<=0.08 and center_error<=max(3,(ty1-ty0)*0.08)),
            'note':'Measured face height and center; excludes hair volume and shoulder/neck anatomy.'}

def memory_limits(total_bytes,free_bytes,requested=1024,megapixels=2.5):
    gib=1024**3; total=total_bytes/gib; free=free_bytes/gib
    cap=2048 if total>=28 else 1536 if total>=20 else 1280 if total>=14 else 1024
    # Conservative free-memory guard; Forge may offload models later, so this is advisory capacity, not an OOM guarantee.
    if free<2: cap=min(cap,768)
    elif free<4: cap=min(cap,1024)
    budget=min(float(megapixels),max(0.5,min(6.0,free*0.5)))
    return min(max(256,int(requested)),cap),budget

def prepare_pair(target,ref,max_side,megapixels):
    area_limit=megapixels*1_000_000
    pair=[fit_image(im,max_side) for im in (target,ref)]
    for _ in range(20):
        area=sum(im.width*im.height for im in pair)
        if area<=area_limit: return pair
        factor=min(0.94,math.sqrt(area_limit/area)*0.96)
        pair=[fit_image(im,max(64,int(max(im.size)*factor)//64*64)) for im in pair]
    raise ValueError('Reference pixel budget is too small for this pair.')

@dataclass
class EditRegion:
    original: Image.Image
    box: tuple
    mask: Image.Image
    crop: Image.Image

def build_region(original,pose,padding=0.55,feather=0.08,mask=None):
    if mask is not None:
        if isinstance(mask,dict): mask=mask.get('composite') if mask.get('composite') is not None else mask.get('background')
        if mask is None: raise ValueError('The custom mask is empty.')
        if not isinstance(mask,Image.Image): mask=rgb_image(mask)
        mask=ImageOps.exif_transpose(mask).convert('L')
        if mask.size!=original.size: raise ValueError('The custom mask must have the same dimensions as the target image.')
        bbox=mask.getbbox()
        if not bbox: raise ValueError('The custom mask has no white edit region.')
        x0,y0,x1,y1=bbox; hh=y1-y0; hw=x1-x0
    else:
        if not pose: raise ValueError('Protected head edit needs a detected target face or a custom white-on-black mask.')
        x0,y0,x1,y1=pose['box']; hh=y1-y0; hw=x1-x0
        # Include hair above the face, ears, and a short neck transition.
        x0-=hw*0.40; x1+=hw*0.40; y0-=hh*0.65; y1+=hh*0.40
        mask=Image.new('L',original.size,0)
        ImageDraw.Draw(mask).ellipse((x0,y0,x1,y1),fill=255)
    radius=max(0,round(min(hw,hh)*feather))
    if radius: mask=mask.filter(ImageFilter.GaussianBlur(radius))
    extent=mask.getbbox()
    if extent is None: raise ValueError('The edit region is empty.')
    x0,y0,x1,y1=extent; margin=max(hw,hh)*padding
    box=(max(0,int(x0-margin)),max(0,int(y0-margin)),min(original.width,math.ceil(x1+margin)),min(original.height,math.ceil(y1+margin)))
    return EditRegion(original.copy(),box,mask,original.crop(box))

def composite_region(generated,region,color_match=0):
    x0,y0,x1,y1=region.box; patch=generated.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS).convert('RGB')
    if color_match>0:
        base=np.asarray(region.original.crop(region.box),dtype=np.float32); arr=np.asarray(patch,dtype=np.float32)
        alpha=np.asarray(region.mask.crop(region.box),dtype=np.float32)/255
        ring=(alpha>0.05)&(alpha<0.8)
        if ring.sum()>20:
            shift=np.clip(base[ring].mean(0)-arr[ring].mean(0),-24,24)*color_match
            patch=Image.fromarray(np.clip(arr+shift,0,255).astype(np.uint8))
    layer=region.original.copy(); layer.paste(patch,(x0,y0))
    return Image.composite(layer,region.original,region.mask)

def scale_pose(pose,from_size,to_size,offset=(0,0)):
    if not pose: return None
    sx=to_size[0]/from_size[0]; sy=to_size[1]/from_size[1]
    x0,y0,x1,y1=pose['box']; ox,oy=offset
    result=dict(pose); result['box']=((x0-ox)*sx,(y0-oy)*sy,(x1-ox)*sx,(y1-oy)*sy)
    result['head_w']=(x1-x0)*sx; result['head_h']=(y1-y0)*sy
    result['head_px']=max(result['head_w'],result['head_h']); result['head_ratio']=result['head_h']/to_size[1]
    return result

def correct_head_scale(image,target_pose,generated_pose):
    """Correct measured face height/center isotropically, feathering the head region."""
    tx0,ty0,tx1,ty1=target_pose['box']; gx0,gy0,gx1,gy1=generated_pose['box']
    target_h=ty1-ty0; generated_h=gy1-gy0
    if min(target_h,generated_h)<8: raise ValueError('Head scale cannot be measured reliably below eight pixels.')
    factor=target_h/generated_h
    if not 0.85<=factor<=1.15: raise ValueError(f'Head resizing skipped to avoid hair/neck seams ({factor:.2f}x); use a protected head pass with a reviewed mask.')
    tx=(tx0+tx1)/2; ty=(ty0+ty1)/2; gx=(gx0+gx1)/2; gy=(gy0+gy1)/2
    if math.hypot(tx-gx,ty-gy)>max(3,target_h*0.08):
        raise ValueError('Head translation skipped to avoid dragging surrounding hair, neck or background.')
    dx=tx-factor*gx; dy=ty-factor*gy
    if abs(factor-1)<0.015 and math.hypot(tx-gx,ty-gy)<1:
        return image,{'scale_factor':1.0,'translation_px':[0,0],'skipped':'Already within a small resampling tolerance.'}
    # PIL affine maps output coordinates back into the input.
    moved=image.transform(image.size,Image.Transform.AFFINE,(1/factor,0,-dx/factor,0,1/factor,-dy/factor),
                          resample=Image.Resampling.BICUBIC)
    target_mask=build_region(image,target_pose,0.3,0.12).mask
    source_mask=build_region(image,generated_pose,0.3,0.12).mask
    union=Image.fromarray(np.maximum(np.asarray(target_mask),np.asarray(source_mask)))
    valid=Image.new('L',image.size,255).transform(image.size,Image.Transform.AFFINE,
            (1/factor,0,-dx/factor,0,1/factor,-dy/factor),resample=Image.Resampling.NEAREST)
    if np.any((np.asarray(union)>8)&(np.asarray(valid)==0)):
        raise ValueError('Head correction would sample outside the image. Increase crop context or use a better aligned reference.')
    corrected=Image.composite(moved,image,union)
    return corrected,{'scale_factor':round(factor,4),'translation_px':[round(dx,2),round(dy,2)],
                      'before_head_height_px':round(generated_h,2),'target_head_height_px':round(target_h,2)}

def match_local_sharpness(image,original,target_pose,generated_pose):
    """Choose a bounded unsharp mask using equally sampled face-region detail scores.

    The chosen amount is applied to the whole frame. Confining it to a head ellipse
    sharpened the hair, neck and background inside that ellipse but not just outside it,
    leaving a halo ring that reads as bad inpainting around the head. The surrounding
    pixels are either regenerated too (full image edit) or feathered back onto the
    original by composite_region (protected head edit), so a uniform pass keeps detail
    continuous in both modes.
    """
    reference=metrics(face_crop(original,target_pose))['sharp']
    before=metrics(face_crop(image,generated_pose))['sharp']
    best=image; after=before; percent=0
    goal=reference*0.95
    if before<goal:
        for amount in (40,70,100,130):
            candidate=image.filter(ImageFilter.UnsharpMask(radius=1.0,percent=amount,threshold=3))
            score=metrics(face_crop(candidate,generated_pose))['sharp']
            if score>after: best,after,percent=candidate,score,amount
            if score>=goal: break
    return best,{'original_detail_score':round(reference,4),'before_detail_score':round(before,4),
                 'after_detail_score':round(after,4),'automatic_unsharp_percent':percent,'detail_target_met':bool(after>=goal),
                 'applied_to':'whole frame (no head-shaped mask, so no sharpening ring)',
                 'note':'A local-contrast metric; not a guarantee of recovered identity detail or perceptual sharpness.'}

class BoundedCache:
    def __init__(self,max_bytes=256*1024**2): self.max_bytes=max_bytes; self.items=OrderedDict(); self.bytes=0
    def get(self,key):
        if key not in self.items: return None
        value,size=self.items.pop(key); self.items[key]=(value,size); return value
    def put(self,key,value,size):
        if key in self.items: self.bytes-=self.items.pop(key)[1]
        if size>self.max_bytes: return
        while self.items and self.bytes+size>self.max_bytes: self.bytes-=self.items.popitem(last=False)[1][1]
        self.items[key]=(value,size); self.bytes+=size
    def clear(self): self.items.clear(); self.bytes=0
