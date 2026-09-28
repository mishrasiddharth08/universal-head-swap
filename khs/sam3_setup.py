"""Explicit one-time optional setup; never invoked by generation."""
from pathlib import Path
import importlib
import subprocess
import sys
import threading

SAM3_REVISION='2345a4ad109ac29c569da749c91d84f10dc08c40'
_SETUP_LOCK=threading.Lock()

def model_directory():
    return Path(__file__).resolve().parents[3]/'models'/'sam3'

def activate_runtime():
    runtime=model_directory()/'runtime'
    if (runtime/'sam3'/'__init__.py').is_file() and str(runtime) not in sys.path:
        sys.path.insert(0,str(runtime));importlib.invalidate_caches()

def download_once(token=None):
    with _SETUP_LOCK:
        folder=model_directory();folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'sam3.pt'
        reused=checkpoint.is_file() and checkpoint.stat().st_size>0
        if not reused:
            from huggingface_hub import hf_hub_download
            try:
                checkpoint=Path(hf_hub_download(repo_id='facebook/sam3',filename='sam3.pt',local_dir=str(folder),token=token or None))
            except Exception as exc:
                raise RuntimeError('SAM3 download failed. The official facebook/sam3 model requires approved Hugging Face access and a locally logged-in account. No license is accepted automatically. Existing downloads are retained for retry.') from exc
        activate_runtime()
        if importlib.util.find_spec('sam3') is None:
            runtime=folder/'runtime'
            result=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','--target',str(runtime),
                'git+https://github.com/facebookresearch/sam3.git@'+SAM3_REVISION],capture_output=True,text=True,timeout=600)
            if result.returncode:
                raise RuntimeError('SAM3 model saved at '+str(checkpoint)+'. Optional runtime setup failed; Forge packages were not upgraded. Check Git/pip availability and retry.')
            activate_runtime()
        return str(checkpoint),('Reused existing model. ' if reused else 'Downloaded model. ')+ 'Stored in '+str(folder)+'. Optional runtime is isolated; Forge dependencies were not upgraded.'
