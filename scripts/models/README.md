# Optional local detector

Put a compatible MediaPipe `face_landmarker.task` file here, following the [official setup guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).

Existing local detector files are preserved when this extension is updated. The source release does not redistribute detector weights or download them automatically. Model licenses remain upstream licenses. Restart Forge after setup, only when training and generation are idle.

## Optional face-match models

The two pinned ONNX models are listed in `identity-models.json`; their upstream license notices are included here. They are installed locally for this update but omitted from the clean source archive. Run `tools/setup_identity_models.py` explicitly with Forge's Python for another installation. Never overwrite a different existing model without inspecting it. See [Face-match setup and limitations](../../docs/IDENTITY.md).
