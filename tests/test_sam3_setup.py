import tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from khs import sam3_setup

class SetupTests(unittest.TestCase):
    def test_existing_model_is_reused_without_network_or_install(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'sam3.pt';checkpoint.write_bytes(b'existing')
            with patch.object(sam3_setup,'model_directory',return_value=Path(folder)),patch.object(sam3_setup.importlib.util,'find_spec',return_value=object()),patch('huggingface_hub.hf_hub_download') as download,patch.object(sam3_setup.subprocess,'run') as run:
                for _ in range(2):
                    path,status=sam3_setup.download_once();self.assertEqual(path,str(checkpoint));self.assertIn('Reused',status)
                download.assert_not_called();run.assert_not_called()
    def test_access_failure_does_not_install_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sam3_setup,'model_directory',return_value=Path(folder)),patch('huggingface_hub.hf_hub_download',side_effect=RuntimeError('gated')),patch.object(sam3_setup.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'approved Hugging Face access'):sam3_setup.download_once()
                run.assert_not_called()
