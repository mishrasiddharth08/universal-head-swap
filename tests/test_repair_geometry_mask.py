import sys,unittest,tempfile,base64,io
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core,runtime
from khs.masking import editor_mask,resize_mask
from khs.geometry_repair import repair

def pose(box):
 x0,y0,x1,y1=box
 return dict(box=box,head_w=x1-x0,head_h=y1-y0,head_px=max(x1-x0,y1-y0),head_ratio=.2,yaw=0,pitch=0,roll=0)

class MaskInterpretationRepairs(unittest.TestCase):
 def test_photo_backed_editor_uses_only_painted_layers(self):
  background=Image.new('RGB',(24,24),(160,90,60));layer=Image.new('RGBA',(24,24),(255,255,255,0));ImageDraw.Draw(layer).rectangle((5,6,8,9),fill='white')
  composite=Image.alpha_composite(background.convert('RGBA'),layer).convert('RGB')
  mask=editor_mask(dict(background=background,layers=[layer],composite=composite),core.rgb_image)
  self.assertEqual(mask.getbbox(),(5,6,9,10));self.assertEqual(np.count_nonzero(np.asarray(mask)),16)
 def test_photo_without_paint_fails_clearly(self):
  photo=Image.new('RGB',(16,16),(100,70,40))
  with self.assertRaisesRegex(ValueError,'not an edit mask'):editor_mask(dict(background=photo,layers=[],composite=photo),core.rgb_image)
 def test_grayscale_portrait_requires_paint(self):
  photo=Image.linear_gradient('L').resize((32,32)).convert('RGB')
  with self.assertRaisesRegex(ValueError,'not an edit mask'):
   editor_mask(dict(background=photo,layers=[],composite=photo),core.rgb_image)
  layer=Image.new('RGBA',photo.size,(0,0,0,0));layer.putpixel((8,9),(255,255,255,255))
  self.assertEqual(editor_mask(dict(background=photo,layers=[layer],composite=photo),core.rgb_image).getbbox(),(8,9,9,10))
 def test_direct_rgba_alpha_is_honored(self):
  mask=Image.new('RGBA',(16,16),(255,255,255,0));mask.putpixel((7,8),(255,255,255,255))
  region=core.build_region(Image.new('RGB',(16,16)),None,padding=.2,feather=0,mask=mask)
  self.assertEqual(region.mask.getbbox(),(7,8,8,9))
 def test_file_and_data_uri_layers_keep_alpha_at_api_boundary(self):
  layer=Image.new('RGBA',(16,16),(255,255,255,0));layer.putpixel((7,8),(255,255,255,255))
  photo=Image.new('RGB',layer.size,(100,70,40));buffer=io.BytesIO();layer.save(buffer,format='PNG')
  uri='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'layer.png';path.write_bytes(buffer.getvalue())
   for value in [str(path),uri]:
    with self.subTest(value=value[:25]):
     self.assertEqual(editor_mask(dict(background=photo,layers=[value]),core.rgb_image).getbbox(),(7,8,8,9))
 def test_black_paint_preserves_in_existing_grayscale_mask(self):
  background=Image.new('RGB',(16,16),'white');composite=background.copy();ImageDraw.Draw(composite).rectangle((5,5,8,8),fill='black')
  mask=editor_mask(dict(background=background,composite=composite),core.rgb_image)
  self.assertEqual(mask.getpixel((6,6)),0);self.assertEqual(mask.getpixel((1,1)),255)
 def test_binary_resize_has_no_lanczos_halo(self):
  mask=Image.new('L',(16,16),0);ImageDraw.Draw(mask).rectangle((4,4,7,7),fill=255)
  region=core.build_region(Image.new('RGB',(32,32)),None,padding=.2,feather=0,mask=mask)
  self.assertEqual(set(np.asarray(region.mask).ravel()),{0,255});self.assertEqual(region.mask.getbbox(),(8,8,16,16))
 def test_soft_mask_resize_retains_soft_monotonic_edge(self):
  mask=Image.fromarray(np.tile(np.arange(16,dtype=np.uint8)*17,(8,1)),'L');scaled=np.asarray(resize_mask(mask,(32,16)))
  self.assertTrue(np.all(np.diff(scaled[8].astype(int))>=0));self.assertGreater(len(np.unique(scaled)),2)
 def test_custom_feather_never_expands_black_preserve_region(self):
  mask=Image.new('L',(64,64),0);ImageDraw.Draw(mask).rectangle((16,16,47,47),fill=255)
  region=core.build_region(Image.new('RGB',mask.size),None,feather=.2,mask=mask)
  alpha=np.asarray(region.mask);original=np.asarray(mask)
  self.assertTrue(np.all(alpha[original==0]==0));self.assertGreater(len(np.unique(alpha)),2)
  generated=core.composite_region(Image.new('RGB',region.crop.size,'white'),region)
  self.assertTrue(np.all(np.asarray(generated)[original==0]==0))
 def test_custom_soft_alpha_is_an_upper_bound_after_feather(self):
  mask=Image.new('L',(64,64),0);ImageDraw.Draw(mask).ellipse((16,16,47,47),fill=128)
  region=core.build_region(Image.new('RGB',mask.size),None,feather=.2,mask=mask)
  self.assertTrue(np.all(np.asarray(region.mask)<=np.asarray(mask)))

class FinalFrameMetricRepair(unittest.TestCase):
 def test_uniform_output_resize_does_not_report_body_ratio_failure(self):
  session=runtime.Session.__new__(runtime.Session)
  session.cfg=core.normalize(dict(geometry_match=False,match_sharpness=False,identity_check=True))
  session.original=Image.new('RGB',(1000,1000));session.target_pose=pose((100,100,300,300));session.region=None;session.canvas_box=None
  generated_pose=pose((50,50,150,150));session.owner=NS(analyzer=NS(faces=lambda image:[generated_pose]),last_report={})
  session.quality_history=[];session.family='klein';session.selected_ref=None;session.p=NS(extra_generation_params={});session.host=NS(shared=NS(state=NS(textinfo='')))
  session.finish_image(Image.new('RGB',(500,500)),0)
  report=session.owner.last_report['quality']['geometry_final'];self.assertEqual(report['body_to_head_ratio_change_percent'],0);self.assertTrue(report['geometry_target_met'])

class UniformProtectedRepair(unittest.TestCase):
 def test_uniform_correction_remeasured_after_final_mask_preserves_outside(self):
  image=Image.new('RGB',(200,200),'blue');ImageDraw.Draw(image).rectangle((76,75,123,144),fill='red')
  mask=Image.new('L',image.size,0);ImageDraw.Draw(mask).rectangle((40,40,160,170),fill=255)
  region=core.EditRegion(image.copy(),(40,40,161,171),mask,image.crop((40,40,161,171)))
  target=pose((80,80,120,140));generated=pose((76,75,124,145))
  def faces(im):
   arr=np.asarray(im);ys,xs=np.where((arr[:,:,0]>200)&(arr[:,:,1]<20)&(arr[:,:,2]<20))
   return [pose((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))] if len(xs) else []
  # Isolate geometric acceptance; a separate test exercises the detail gate.
  with patch.object(core,'metrics',return_value={'sharp':1.0}):
   result,report=repair(image,region,target,generated,NS(faces=faces),core)
  self.assertTrue(report['applied'],report);self.assertEqual(report['scale_x'],report['scale_y']);self.assertTrue(report['after']['geometry_target_met'])
  self.assertTrue(np.array_equal(np.asarray(result)[np.asarray(mask)==0],np.asarray(image)[np.asarray(mask)==0]))
 def test_detail_loss_rejects_otherwise_valid_uniform_correction(self):
  image=Image.new('RGB',(200,200),'blue');region=NS(mask=Image.new('L',image.size,255))
  target=pose((80,80,120,140));generated=pose((76,75,124,145))
  with patch.object(core,'metrics',side_effect=[{'sharp':1.0},{'sharp':.5}]):
   result,report=repair(image,region,target,generated,NS(faces=lambda im:[target]),core)
  self.assertIs(result,image);self.assertFalse(report['applied']);self.assertIn('detail',report['reason'])
 def test_demonstrated_anisotropic_failure_is_not_blindly_stretched(self):
  image=Image.new('RGB',(256,256),'blue');region=NS(mask=Image.new('L',image.size,255));analyzer=NS(faces=Mock())
  result,report=repair(image,region,pose((60,40,160,140)),pose((47.65,50.92,172.35,140)),analyzer,core)
  self.assertIs(result,image);self.assertFalse(report['applied']);self.assertTrue(report['requires_resampling']);analyzer.faces.assert_not_called()

if __name__=='__main__':unittest.main()
