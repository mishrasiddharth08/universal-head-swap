"""Optional local MediaPipe analysis with bounded CPU caches."""
import math
from pathlib import Path
import threading
import numpy as np
from PIL import Image
from .core import BoundedCache,image_hash

class FaceAnalyzer:
    def __init__(self,model_path):
        self.model_path=Path(model_path); self.engine=None; self.mode=None; self.error=''
        self.lock=threading.RLock(); self.cache=BoundedCache(2*1024**2)
    def close(self):
        with self.lock:
            if self.engine is not None:
                try: self.engine.close()
                except Exception: pass
            self.engine=None; self.mode=None; self.error=''; self.cache.clear()
    def start(self):
        if self.engine is not None: return
        if self.error: return
        try:
            import mediapipe as mp
            if hasattr(mp,'tasks') and self.model_path.is_file():
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import FaceLandmarker,FaceLandmarkerOptions,RunningMode
                self.engine=FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.model_path)),running_mode=RunningMode.IMAGE,
                    num_faces=20,output_facial_transformation_matrixes=True,
                    min_face_detection_confidence=0.5,min_face_presence_confidence=0.5))
                self.mode='tasks'
            elif hasattr(mp,'solutions'):
                self.engine=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=20,
                    refine_landmarks=False,min_detection_confidence=0.5)
                self.mode='legacy'
            else: raise RuntimeError('Face landmarker model is missing. See README: optional face detector setup.')
        except Exception as e: self.error=str(e)
    @staticmethod
    def matrix_angles(matrix):
        rotation=np.asarray(matrix,dtype=float)[:3,:3]
        u,_,vh=np.linalg.svd(rotation); rotation=u@vh
        if np.linalg.det(rotation)<0: u[:,-1]*=-1; rotation=u@vh
        yaw=math.asin(float(np.clip(-rotation[2,0],-1,1)))
        pitch=math.atan2(rotation[2,1],rotation[2,2]); roll=math.atan2(rotation[1,0],rotation[0,0])
        return tuple(math.degrees(x) for x in (yaw,pitch,roll))
    def _detect(self,im):
        import mediapipe as mp
        pixels=np.ascontiguousarray(im.convert('RGB'),dtype=np.uint8)
        if self.mode=='tasks':
            result=self.engine.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=pixels))
            landmarks=result.face_landmarks
            matrices=result.facial_transformation_matrixes or []
        else:
            result=self.engine.process(pixels)
            landmarks=[x.landmark for x in (result.multi_face_landmarks or [])]; matrices=[]
        faces=[]; w,h=im.size
        for i,lm in enumerate(landmarks):
            xs=[p.x*w for p in lm]; ys=[p.y*h for p in lm]
            box=(max(0,min(xs)),max(0,min(ys)),min(w,max(xs)),min(h,max(ys)))
            fw=box[2]-box[0]; fh=box[3]-box[1]
            if fw<1 or fh<1: continue
            if i<len(matrices): yaw,pitch,roll=self.matrix_angles(matrices[i]); source='3D transformation'
            else:
                eye_l=lm[33]; eye_r=lm[263]; nose=lm[1]
                ex=(eye_r.x-eye_l.x)*w; ey=(eye_r.y-eye_l.y)*h; distance=max(1,math.hypot(ex,ey))
                yaw=math.degrees(math.atan2((nose.x-(eye_l.x+eye_r.x)/2)*w,distance))
                pitch=math.degrees(math.atan2((nose.y-(eye_l.y+eye_r.y)/2)*h,distance))-12
                roll=math.degrees(math.atan2(ey,ex)); source='2D estimate'
            faces.append(dict(box=box,head_px=max(fw,fh),head_ratio=fh/h,head_h=fh,head_w=fw,
                              yaw=yaw,pitch=pitch,roll=roll,pose_source=source))
        return faces
    def faces(self,im):
        key=image_hash(im)
        with self.lock:
            cached=self.cache.get(key)
            if cached is not None: return [dict(x) for x in cached]
            self.start()
            if self.engine is None: return []
            try:
                scale=min(1,1280/max(im.size)); thumb=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
                detected=self._detect(thumb)
                windows=[(detected,0,0,scale)]
                if not detected:
                    # Cover the complete upper frame, including off-center subjects.
                    for a,b in ((0,0.55),(0.45,1)):
                        x=int(im.width*a); y=0; crop=im.crop((x,y,int(im.width*b),max(1,int(im.height*0.65))))
                        factor=min(2,1280/max(crop.size)); zoom=crop.resize((round(crop.width*factor),round(crop.height*factor)))
                        windows.append((self._detect(zoom),x,y,factor))
                faces=[]
                for ds,ox,oy,factor in windows:
                    for item in ds:
                        p=dict(item); x0,y0,x1,y1=p['box']; p['box']=(x0/factor+ox,y0/factor+oy,x1/factor+ox,y1/factor+oy)
                        p['head_px']/=factor; p['head_h']/=factor; p['head_w']/=factor; p['head_ratio']=p['head_h']/im.height
                        cx=(p['box'][0]+p['box'][2])/2; cy=(p['box'][1]+p['box'][3])/2
                        if any(abs(cx-(q['box'][0]+q['box'][2])/2)<min(p['head_w'],q['head_w'])*0.5 and
                               abs(cy-(q['box'][1]+q['box'][3])/2)<min(p['head_h'],q['head_h'])*0.5 for q in faces): continue
                        faces.append(p)
                faces.sort(key=lambda p:(-p['head_w']*p['head_h'],p['box'][0]))
                self.cache.put(key,faces,max(512,len(faces)*512)); return [dict(x) for x in faces]
            except Exception as e:
                self.error=f'Face analysis failed: {e}'; return []
