"""Explicit one-time optional setup; never invoked by generation.

models/SAM 3 is the SAM3 home folder. Everything required is auto-detected
inside it and its subfolders: the official checkpoint (sam3.pt preferred),
the isolated runtime package, and the tokenizer vocabulary. models/sam3 and
models/SAM3 are also scanned for older installs.
"""
from pathlib import Path
import importlib
import subprocess
import sys
import threading

SAM3_REVISION='2345a4ad109ac29c569da749c91d84f10dc08c40'
_SETUP_LOCK=threading.Lock()
_HOME_NAMES=('SAM 3','sam3','SAM3')

def model_directory():
    return Path(__file__).resolve().parents[3]/'models'/'SAM 3'

def _home_folders():
    folder=model_directory().parent
    return [folder/name for name in _HOME_NAMES if (folder/name).is_dir()]

def _weights(folders):
    """Collect real weight files; prefer the official sam3.pt, then newest."""
    found=[]
    for folder in folders:
        found.extend(folder.glob('**/*.pt'))
        found.extend(folder.glob('**/*.safetensors'))
    found=[f for f in found if f.is_file() and f.stat().st_size>1024*1024
           and 'runtime' not in f.parts]
    if not found: return None
    official=[f for f in found if f.name.casefold()=='sam3.pt']
    if official:
        official.sort(key=lambda f:f.stat().st_mtime,reverse=True)
        return official[0]
    found.sort(key=lambda f:f.stat().st_mtime,reverse=True)
    return found[0]

def default_checkpoint():
    """Auto-detect the SAM3 checkpoint inside the SAM 3 folder(s) and subfolders."""
    return _weights(_home_folders())

def default_bpe():
    for folder in _home_folders():
        found=sorted(folder.glob('**/bpe_simple_vocab*.gz'))
        if found: return found[0]
    return None

def runtime_directory():
    for folder in _home_folders():
        runtime=folder/'runtime'
        if (runtime/'sam3'/'__init__.py').is_file(): return runtime
    return None

def activate_runtime():
    runtime=runtime_directory()
    if runtime is not None and str(runtime) not in sys.path:
        sys.path.insert(0,str(runtime));importlib.invalidate_caches()

def sam3_runtime_ready():
    try:
        activate_runtime()
        return importlib.util.find_spec('sam3') is not None
    except Exception:
        return False

def download_once(token=None):
    with _SETUP_LOCK:
        folder=model_directory();folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'sam3.pt'
        reused=checkpoint.is_file() and checkpoint.stat().st_size>0
        if not reused:
            existing=default_checkpoint()
            if existing is not None:
                checkpoint=existing; reused=True
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