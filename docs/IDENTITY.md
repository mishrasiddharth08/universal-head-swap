# Face-match auditing and head-boundary safety — rc3

## Everyday use

Enable **Check face match in the background**, then generate normally. Click **Face-match results** and **Refresh results** to see completed checks. The panel is advisory: it does not reject, delete, replace, or retry output images. The most recent 300 records are retained across folder-batch targets in the current extension instance. Restarting Forge clears this in-memory history.

Each row identifies the image index, seed, target fingerprint and selected reference. It shows similarity to the selected headshot, the best score across usable headshots, the median score, signed face-height change, status and warnings. Expand details for every reference score, agreement count, face-width/area change and center displacement. **Export report** saves an explicitly requested JSON report under `outputs/identity-audits` inside the extension and offers it for download. Images and identity descriptors are not included.

**Checked** means the selected and median scores meet the configured threshold, every reference was measurable, and the advisory height/position/width checks did not request review. It is not proof of identity or a promise of photographic quality. **Needs review** explains low similarity, incomplete reference data or geometry concerns. **Could not verify** indicates missing models, small/absent/ambiguous faces, a checker error or a queue that could not accept the work. Pending results remain visibly pending. Reference-set disagreement is reported separately even when the output's selected and median scores pass; it is not proof that the references depict different people.

The threshold under **More options → Detail → Face-match settings** defaults to 0.363, an OpenCV LFW benchmark starting point. A score such as 0.70 is cosine similarity, not 70% identity accuracy. Pose, age, expression, lighting, occlusion, resolution and generated texture can change scores. Calibrate with representative authorized examples before treating any threshold as an acceptance rule. References should be clear single-person headshots of the intended person. Keep explicit rotation/manual choices if desired; **Best match (smart, no rotation)** is the stable selection starting point.

The selected-reference score alone cannot override a low median. Unusable references are explicitly listed; no fake descriptor or perfect score is substituted. Face detection is spatially associated with the expected target. When no expected location exists, only a single detected face can be used; multiple faces are refused. Faces under 40 original pixels on their smaller side, or under 32 pixels in the bounded detector input, are unverified.

## Why the head-boundary default changed

The rc2 implementation could transform and blend finished head pixels to match the original face position/height. That also moved nearby hair, neck and background pixels. Review of the owner's saved files found this operation recorded in 100 of 156 outputs, with some pre-correction center differences around 65–68 pixels. This provides a credible mechanism for seams, but an A/B regeneration is still needed to attribute every visible artifact.

In rc3 **Check original head size and position** measures without warping the finished picture. Original-scale prompt guidance remains available separately. The additional **Resize head pixels after generation** control is off by default, including for older presets that do not contain the new setting. It lives under **Detail → Experimental enhancements**. Even when explicitly enabled, resizing outside 0.85–1.15x or moving the face center by more than 8% of target face height (minimum allowance 3px) is refused; the blend feather is wider. These limits reduce risk and do not certify natural seams.

For a head-only pass, review the edit mask: it must include the intended hair, ears and neck transition without cutting across those features or painting over foreground objects. A feathered ellipse cannot segment hair or hands accurately. Protected mode keeps pixels outside the mask, including unwanted marks there; perform needed whole-body cleanup first. A mask example is not a guarantee of a good result for another pose.

The independent audit uses YuNet face boxes; the existing correction/detail path uses MediaPipe landmarks. Measurements from different detectors need not be numerically identical. Width warnings above 15% are advisory because head turning changes apparent width. Neither detector measures complete skull/hair volume or head-to-shoulder anatomy. The checker does not detect tattoos, jewelry or inpainting seams.

## Models and explicit setup

Local installation performed for this update includes the two models below. Clean source archives omit the binary weights. Pinned source revision, download URLs, sizes and SHA-256 digests are in `scripts/models/identity-models.json`. The YuNet MIT and SFace Apache 2.0 notices supplied by the official model directories are included alongside the manifest.

- `face_detection_yunet_2023mar.onnx`: 232,589 bytes.
- `face_recognition_sface_2021dec.onnx`: 38,696,353 bytes.

For another installation, explicitly run the setup tool from the extension root using that Forge installation's existing Python:

```powershell
& '<Forge>\venv\Scripts\python.exe' -B tools/setup_identity_models.py
```

The tool verifies downloaded hashes and preserves an existing mismatched file by failing rather than replacing it. No download or package installation occurs at extension import or generation. An OpenCV build exposing `FaceDetectorYN_create` and `FaceRecognizerSF_create` is required. The inspected local OpenCV 5.0.0 build ran the real-model CPU smoke checks; it emits a target-selection warning from its new graph engine. The chosen backend is OpenCV, CPU is requested, and this build exposes no CUDA DNN targets. Do not replace Torch, CUDA or Gradio for this feature.

## Worker and lifecycle

One worker belongs to each extension UI instance and starts only on the first submitted audit. Recognition models are lazily initialized there, not on the Forge generation thread. At most two audits are running/queued together (one executing). Each input is cropped around the expected face when available and capped at 768px; request references are reused. The descriptor cache is bounded to 128 entries. Reference/image hashes include image content; transformed crop location/scale are part of the cache key. No embeddings are persisted.

If the two slots are occupied, submission applies backpressure for at most 30 seconds, checking cancellation between waits. If it still cannot submit, that output receives an explicit unverified row rather than an unlimited queue or silent omission. Therefore overlap can reduce visible overhead but is not guaranteed to make checks free. Cancelling generation keeps checks for already completed outputs; unloading the extension cancels queued work and lets an already running native CPU call finish without touching Forge state.

The outer callback guard enqueues the final image after all `postprocess_image_after_composite` callbacks. Workers update only their own unique, synchronized audit records; they do not mutate requests, metadata, or another job's last-generation report. PNG infotext carries an audit ticket and an advisory marker, not a fabricated final score. Use the result table/export for later scores. Subsequent image-save callbacks, external editors and lossy JPEG compression may change the file after the audited image; the audit is of the final image-callback pixels, not a cryptographic certification of every saved file.

## Validation limits

Read `VALIDATION.md` for actual results. Standard sample portraits exercised identical-image, brightness-change, different-person, blank-image and tiny-face behavior with real models. Existing user outputs were audited against their actual reference folder after matching source fingerprints. These checks demonstrate inference/reporting and expose review candidates; they do not establish false-accept/false-reject rates or demonstrate improved newly generated images. No automatic rejection or retry is included in rc3.

## Primary sources

- [OpenCV face detection and recognition tutorial](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html).
- [Official SFace model directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface).
- [Official YuNet model directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet).
