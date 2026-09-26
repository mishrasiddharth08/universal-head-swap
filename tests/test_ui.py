"""Build real Gradio controls with a stand-in host; no server or model loads."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from khs import core

@unittest.skipUnless(importlib.util.find_spec('gradio'),'Gradio not installed: loaded UI validation remains pending')
class UIConstructionTests(unittest.TestCase):
    def test_controls_construct_and_all_arguments_are_components(self):
        import gradio as gr
        modules=ModuleType('modules'); modules.__path__=[]
        scripts=ModuleType('modules.scripts'); scripts.Script=type('Script',(),{}); scripts.AlwaysVisible=True
        callbacks=ModuleType('modules.script_callbacks'); callbacks.on_script_unloaded=lambda fn:None; callbacks.on_before_image_saved=lambda fn:None
        components=ModuleType('modules.ui_components')
        @contextmanager
        def accordion(value=False,**kwargs):
            with gr.Accordion(label=kwargs.get('label',''),open=False):
                yield gr.Checkbox(value=value,visible=False)
        components.InputAccordion=accordion
        modules.scripts=scripts; modules.script_callbacks=callbacks
        replacements={'modules':modules,'modules.scripts':scripts,'modules.script_callbacks':callbacks,
                      'modules.ui_components':components,'torch':None}
        spec=importlib.util.spec_from_file_location('_khs_ui_contract',ROOT/'scripts/klein_face_reference.py')
        ui=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,replacements),patch.dict('os.environ',{'GRADIO_ANALYTICS_ENABLED':'False'}):
            spec.loader.exec_module(ui)
            self.assertIsNone(ui.HOST)
            with gr.Blocks(analytics_enabled=False):
                script=ui.UniversalHeadSwap()
                script.after_component(gr.Dropdown(choices=['klein','qwen_image21'],value='klein'),elem_id='forge_ui_preset')
                script.after_component(gr.Dropdown(choices=['klein-9b'],value='klein-9b'),elem_id='setting_sd_model_checkpoint')
                controls=script.ui(True)
                self.assertEqual(len(controls),len(core.ARG_KEYS))
                self.assertEqual(len({id(c) for c in controls}),len(controls))
                for c in controls: self.assertIsInstance(c,gr.components.Component)
                for key in ('geometry_match','match_sharpness','removal_priority','quality_strict','keep_original_canvas'):
                    self.assertTrue(controls[core.ARG_KEYS.index(key)].visible)
            ui.unload()

if __name__=='__main__': unittest.main()
