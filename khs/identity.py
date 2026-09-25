"""Local advisory face verification. One CPU worker, bounded inputs and history.

No downloads, Forge imports, GPU work, image writes, or persistent face embeddings.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, CancelledError
from dataclasses import dataclass
import copy
import hashlib
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from PIL import Image
from . import core

MODELS = {
    'face_detection_yunet_2023mar.onnx': '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    'face_recognition_sface_2021dec.onnx': '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
}

def normalized(vector):
    value=np.asarray(vector,dtype=np.float32).reshape(-1)
    norm=float(np.linalg.norm(value))
    if not value.size or not np.isfinite(value).all() or not math.isfinite(norm) or norm<1e-8:
        raise ValueError('Face descriptor is invalid.')
    return value/norm

def similarity(a,b):
    a,b=normalized(a),normalized(b)
    if a.shape!=b.shape: raise ValueError('Face descriptor dimensions differ.')
    return float(np.clip(np.dot(a,b),-1,1))

def compare(output, references, selected, threshold):
    """An outlier reference cannot turn a weak consensus into a passing audit."""
    scores=[None if r is None else similarity(output,r) for r in references]
    usable=[(i,s) for i,s in enumerate(scores) if s is not None]
    if not usable: raise ValueError('No usable reference face was found.')
    best=max(usable,key=lambda item:item[1]); median=float(np.median([s for _,s in usable]))
    selected_score=scores[selected] if 0<=selected<len(scores) else None
    inconsistent=[]
    for i,r in enumerate(references):
        others=[similarity(r,q) for j,q in enumerate(references) if q is not None and j!=i] if r is not None else []
        if others and float(np.median(others))<threshold: inconsistent.append(i+1)
    passed=selected_score is not None and selected_score>=threshold and median>=threshold
    return {'selected_similarity':None if selected_score is None else round(selected_score,4),
        'best_similarity':round(best[1],4),'best_reference_slot':best[0]+1,'median_similarity':round(median,4),
        'agreeing_references':sum(s>=threshold for _,s in usable),'usable_references':len(usable),
        'per_reference':[{'slot':i+1,'similarity':None if s is None else round(s,4)} for i,s in enumerate(scores)],
        'reference_disagreement_slots':inconsistent,'identity_above_threshold':bool(passed),
        'threshold':threshold,'score_type':'SFace cosine similarity; not an accuracy percentage'}

@dataclass
class Sample:
    image: Image.Image
    expected: dict | None
    scale: tuple
    offset: tuple
    fingerprint: str
    canvas_size: tuple

def sample(image,expected=None):
    """Limit retained pixels before handing data to a worker; keep coordinate mapping."""
    image=image.convert('RGB'); fingerprint=core.image_hash(image); canvas_size=image.size
    x=y=0
    if expected:
        a,b,c,d=expected['box']; w=c-a; h=d-b
        x=max(0,int(a-w)); y=max(0,int(b-h))
        image=image.crop((x,y,min(image.width,math.ceil(c+w)),min(image.height,math.ceil(d+h))))
    scale=min(1,768/max(image.size))
    size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    reduced=image.resize(size,Image.Resampling.LANCZOS)
    mapped=core.scale_pose(expected,image.size,size,(x,y)) if expected else None
    # Rounding may make x/y scaling slightly different; preserve them separately.
    return Sample(reduced,mapped,(size[0]/image.width,size[1]/image.height),(x,y),fingerprint,canvas_size)

class Engine:
    """Owned by a single worker; native inference explicitly uses the CPU backend."""
    def __init__(self,model_dir):
        self.model_dir=Path(model_dir); self.detector=None; self.recognizer=None; self.cv=None
        self.cache=OrderedDict()
    def ready(self):
        if self.detector is not None: return
        for name,digest in MODELS.items():
            path=self.model_dir/name
            if not path.is_file(): raise ValueError('Face-match models missing. See docs/IDENTITY.md for setup.')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Face-match model checksum mismatch: '+name)
        import cv2
        self.cv=cv2
        self.detector=cv2.FaceDetectorYN_create(str(self.model_dir/next(iter(MODELS))),'',(320,320),
            0.8,0.3,5000,cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)
        self.recognizer=cv2.FaceRecognizerSF_create(str(self.model_dir/list(MODELS)[1]),'',
            cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)
    def describe(self,item):
        expected=item.expected
        key=(item.fingerprint,tuple(expected['box']) if expected else None,item.offset,item.scale)
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        self.ready()
        bgr=np.ascontiguousarray(np.asarray(item.image)[:,:,::-1])
        self.detector.setInputSize(item.image.size)
        _,detected=self.detector.detect(bgr)
        if detected is None or not len(detected): raise ValueError('No clear face detected.')
        poses=[{'box':(float(r[0]),float(r[1]),float(r[0]+r[2]),float(r[1]+r[3])),
                'head_h':float(r[3]),'head_w':float(r[2])} for r in detected]
        if expected:
            found=core.nearest_face(poses,expected,item.image.size)
            if found is None: raise ValueError('The intended face could not be located.')
            chosen=poses.index(found)
            # Two near-identical spatial candidates are not a reliable association.
            cx=lambda p:((p['box'][0]+p['box'][2])/2,(p['box'][1]+p['box'][3])/2)
            center=cx(found)
            if any(i!=chosen and math.dist(cx(p),center)<min(found['head_h'],found['head_w'])*0.5 for i,p in enumerate(poses)):
                raise ValueError('Face association is ambiguous.')
        else:
            if len(poses)!=1: raise ValueError('Reference contains multiple faces; use one clear headshot.')
            chosen=0; found=poses[0]
        sx,sy=item.scale
        if min(found['head_w']/sx,found['head_h']/sy)<40 or min(found['head_w'],found['head_h'])<32:
            raise ValueError('Face too small for a reliable comparison.')
        aligned=self.recognizer.alignCrop(bgr,detected[chosen])
        vector=normalized(self.recognizer.feature(aligned))
        x0,y0,x1,y1=found['box']; ox,oy=item.offset
        pose={'box':(x0/sx+ox,y0/sy+oy,x1/sx+ox,y1/sy+oy),
              'head_h':found['head_h']/sy,'head_w':found['head_w']/sx}
        result=(vector,pose)
        self.cache[key]=result
        while len(self.cache)>128: self.cache.popitem(last=False)
        return result
    def audit(self,target,output,references,selected,threshold):
        start=time.perf_counter()
        try:
            vec,generated_pose=self.describe(output)
            vectors=[]; warnings=[]
            for i,ref in enumerate(references):
                try: vectors.append(self.describe(ref)[0])
                except Exception as e: vectors.append(None); warnings.append(f'Reference {i+1}: {e}')
            result=compare(vec,vectors,selected,threshold)
            try:
                _,target_pose=self.describe(target)
                target_pose=core.scale_pose(target_pose,target.canvas_size,output.canvas_size)
                result['geometry']=core.geometry_report(target_pose,generated_pose)
            except Exception as e: warnings.append('Size could not be verified: '+str(e))
            geometry=result.get('geometry',{})
            needs_review=bool(warnings) or not result['identity_above_threshold'] or not geometry.get('geometry_target_met',False)
            if not result['identity_above_threshold']: warnings.append('Selected-reference or median similarity is below the review threshold.')
            if geometry and not geometry.get('geometry_target_met',False): warnings.append('Face height or position differs from the original; inspect proportions.')
            # Width is advisory: head turn changes face-box width without changing anatomy.
            if abs(geometry.get('head_width_change_percent',0))>15: needs_review=True; warnings.append('Face width differs by more than 15%; inspect pose and head size.')
            if result['reference_disagreement_slots']: warnings.append('Headshots disagree; review the reference set before trusting a high best-match score.')
            result['status']='Needs review' if needs_review else 'Checked'
            result['warnings']=warnings
        except Exception as e: result={'status':'Could not verify','warnings':[str(e)]}
        result['elapsed_seconds']=round(time.perf_counter()-start,3)
        return result

class Auditor:
    """Thread-safe history; workers never touch Forge requests or images on disk."""
    def __init__(self,model_dir,engine=None,history_limit=300):
        self.model_dir=Path(model_dir); self.engine=engine or Engine(model_dir)
        self.executor=None; self.lock=threading.RLock(); self.slots=threading.BoundedSemaphore(2)
        self.records=OrderedDict(); self.futures={}; self.history_limit=history_limit; self.closed=False
    def availability(self):
        missing=[name for name in MODELS if not (self.model_dir/name).is_file()]
        return 'Ready for local CPU checks.' if not missing else 'Face-match models missing; see docs/IDENTITY.md.'
    def submit(self,target,output,references,selected,threshold,metadata,cancel=None,wait_seconds=30):
        ticket=uuid.uuid4().hex
        row=dict(metadata,id=ticket,status='Pending',selected_reference_slot=selected+1)
        with self.lock:
            if self.closed: return None
            self.records[ticket]=row
            while len(self.records)>self.history_limit: self.records.popitem(last=False)
        # At most two outputs retained by the worker. Backpressure is explicit and bounded.
        deadline=time.monotonic()+wait_seconds
        while not self.slots.acquire(timeout=0.1):
            if (cancel and cancel()) or time.monotonic()>=deadline:
                self._update(ticket,{'status':'Could not verify','warnings':['Audit queue busy or generation stopped; this output was not checked.']})
                return ticket
        with self.lock:
            if self.closed:
                self.slots.release(); self._update(ticket,{'status':'Cancelled'}); return ticket
            if self.executor is None: self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='KleinFaceAudit')
            try:
                future=self.executor.submit(self.engine.audit,target,output,references,selected,float(threshold))
            except Exception as e:
                self.slots.release(); self._update(ticket,{'status':'Could not verify','warnings':[str(e)]}); return ticket
            self.futures[ticket]=future
            future.add_done_callback(lambda done:self._complete(ticket,done))
        return ticket
    def _update(self,ticket,values):
        with self.lock:
            if ticket in self.records: self.records[ticket].update(values)
    def _complete(self,ticket,future):
        try:
            try: result=future.result()
            except CancelledError: result={'status':'Cancelled','warnings':['Extension unloaded before the check ran.']}
            except Exception as e: result={'status':'Could not verify','warnings':[str(e)]}
            self._update(ticket,result)
        finally:
            with self.lock: self.futures.pop(ticket,None)
            self.slots.release()
    def snapshot(self):
        with self.lock: return copy.deepcopy(list(self.records.values()))
    def close(self):
        with self.lock:
            self.closed=True; executor=self.executor
        if executor: executor.shutdown(wait=False,cancel_futures=True)

def table(records):
    rows=[]
    def score(value): return '—' if value is None else f'{value:.3f}'
    for r in reversed(records):
        geometry=r.get('geometry',{}); height=geometry.get('head_height_change_percent')
        rows.append([r.get('image',0),str(r.get('seed','')),r.get('target',''),r.get('selected_reference_slot',0),
            score(r.get('selected_similarity')),score(r.get('best_similarity')),score(r.get('median_similarity')),
            '—' if height is None else f'{height:+.1f}%',r['status'],'; '.join(r.get('warnings',[]))])
    return rows
