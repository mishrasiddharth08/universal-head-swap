import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class ReloadContractTests(unittest.TestCase):
    def test_entry_retires_stale_helpers_before_imports(self):
        text=(ROOT/'scripts/klein_face_reference.py').read_text(encoding="utf-8")
        call=text.index("_reload_extension_helpers()")
        helper_import=text.index("from khs import core")
        self.assertLess(call,helper_import)
        self.assertIn('khs.',text)
        self.assertIn("remove_current_script_callbacks",text)
        self.assertIn("importlib.reload(module)",text)
        self.assertNotIn("sys.modules.pop(name,None)",text)

if __name__=="__main__": unittest.main()
