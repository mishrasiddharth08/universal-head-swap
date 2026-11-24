"""Real Gradio preprocessing, independent of the Forge host and GPU."""
import tempfile,unittest,sys
from pathlib import Path
from PIL import Image,ImageDraw
import gradio as gr
from gradio.components.image_editor import EditorData
from gradio.data_classes import FileData
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from khs import core
from khs.masking import editor_mask

class GradioMaskBoundary(unittest.TestCase):
 def test_real_editor_preprocessing_retains_painted_layer_alpha(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory);photo=Image.new('RGB',(32,32),(120,70,40));layer=Image.new('RGBA',photo.size,(255,255,255,0));ImageDraw.Draw(layer).rectangle((8,8,15,15),fill='white')
   composite=Image.alpha_composite(photo.convert('RGBA'),layer)
   for name,im in [('background',photo),('layer',layer),('composite',composite)]:im.save(p/(name+'.png'))
   payload=EditorData(background=FileData(path=str(p/'background.png')),layers=[FileData(path=str(p/'layer.png'))],composite=FileData(path=str(p/'composite.png')))
   # Demonstrate why RGB is unsuitable at this actual UI boundary.
   old=gr.ImageEditor(type='pil',image_mode='RGB').preprocess(payload)
   fixed=gr.ImageEditor(type='pil',image_mode='RGBA').preprocess(payload)
   self.assertEqual(old['layers'][0].mode,'RGB')
   self.assertEqual(editor_mask(old,core.rgb_image).getbbox(),(0,0,32,32))
   self.assertEqual(editor_mask(fixed,core.rgb_image).getbbox(),(8,8,16,16))
