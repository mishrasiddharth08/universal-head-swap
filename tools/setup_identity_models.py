"""Explicit, checksum-verified local setup. Run with Forge's Python while idle."""
import hashlib
import json
from pathlib import Path
import tempfile
import os
import urllib.request

def main():
    directory=Path(__file__).resolve().parents[1]/'scripts/models'
    manifest=json.loads((directory/'identity-models.json').read_text(encoding='utf-8'))
    for entry in manifest['models']:
        name=entry['name']
        if Path(name).name!=name or not entry['url'].startswith('https://media.githubusercontent.com/media/opencv/opencv_zoo/'):
            raise ValueError('Unexpected model path or source.')
        destination=directory/name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Existing model has a different checksum; preserve and inspect it before replacement: '+str(destination))
            print('Verified existing '+name); continue
        temporary=None
        try:
            request=urllib.request.Request(entry['url'],headers={'User-Agent':'KleinHeadSwap-explicit-setup'})
            digest=hashlib.sha256(); size=0
            with urllib.request.urlopen(request,timeout=60) as response, tempfile.NamedTemporaryFile(dir=directory,suffix='.tmp',delete=False) as handle:
                temporary=Path(handle.name)
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>entry['bytes']: raise ValueError('Download exceeds expected model size.')
                    digest.update(chunk); handle.write(chunk)
                handle.flush(); os.fsync(handle.fileno())
            if size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']: raise ValueError('Model download checksum mismatch.')
            # Never overwrite a model created by another setup while downloading.
            temporary.rename(destination); temporary=None
            print('Installed and verified '+name)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

if __name__=='__main__': main()
