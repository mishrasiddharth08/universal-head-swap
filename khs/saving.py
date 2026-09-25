"""Protect active Klein saves from JPEG APP1 metadata overflow."""
from pathlib import Path
from uuid import uuid4


def protect_jpeg_metadata(params):
    session = getattr(getattr(params, 'p', None), '_khs_session', None)
    if session is None or session.closed:
        return
    path = Path(params.filename)
    if path.suffix.lower() not in ('.jpg', '.jpeg'):
        return
    if not getattr(session.host.shared.opts, 'enable_pnginfo', True):
        return
    # Match Forge's actual Unicode UserComment serialization, including EXIF overhead.
    import piexif
    import piexif.helper
    info = params.pnginfo.get('parameters')
    if info is None:
        return
    payload = piexif.dump({'Exif': {
        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(info or '', encoding='unicode')
    }})
    if len(payload) + 2 <= 65535:
        return
    # JPEG filename allocation did not reserve a PNG name; use a unique suffix.
    params.filename = str(path.with_name(path.stem + '-metadata-' + uuid4().hex + '.png'))
    note = 'Oversized JPEG generation metadata: saved lossless PNG with full metadata instead.'
    session.owner.last_report.setdefault('save_warnings', []).append(note)
    print('[KleinHeadSwap] ' + note)
