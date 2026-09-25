# Current rc5 validation — 21 September 2026

**116 CPU tests passed**, zero failures/skips (2.668 seconds). Includes automatic grid detection, clean-image exact bypass, bounded correction, protected-mask preservation, operation before resizing/sharpening, preset/argument compatibility and real Gradio component construction with host stand-ins. No GPU or training work was performed for this change. UI tests require normal CPU detection; the restricted shell's Polars feature probe failed, and the unchanged dependency passed outside that shell.

The rc5 automatic filter is default-off. No fresh live Forge reload or real moire-photo comparison was performed for rc5; prior rc4 generation/cancellation evidence below is historical. Tattoo/piercing cleanup remains a separate unresolved acceptance issue. The new filter does not address it.

---

# Latest live acceptance — 20 September 2026

**Partial pass; cleanup failed.** An isolated Klein 9B request at 768x960 with AutoNeg completed and saved its JPEG grid. Cancellation at observed step 27/43 returned normally; a following two-iteration request completed in 35.1 seconds and saved its grid. Negative strings measured 3584 characters per recovery sample. This is not yet a native multi-target folder-batch test.

Fresh output passed the local single-reference advisory match threshold (selected/median 0.6064; uncalibrated threshold 0.363). Measured face-height error was 1.27%, center displacement 2.05 pixels; detail score changed from 0.3913 to 0.4403 without automatic sharpening. These metrics do not prove perfect anatomy, likeness or original-resolution sharpness.

Visual inspection found retained arm tattoos, navel piercing and earrings. **Whole-body cleanup is not accepted.** No clearly outlined head-shaped sharpening ring was apparent at this preview resolution; larger fixed-seed comparisons remain necessary.

Pending: native reused-request AutoNeg folder batch; live oversized JPEG-metadata fallback; populated background audit association/export; original-resolution visual acceptance; improved and verified whole-body cleanup. Prior 104 CPU tests remain valid and were not rerun without code changes. The initial test harness used an invalid adapter label and was corrected to the available Auto (match model); production code was unchanged. Test server processes were closed; no training or normal Forge settings were changed.

---

# Validation status — 6.0.0-rc4



12 September 2026. **104 CPU tests passed, zero failures/skips, 2.807 seconds.** This includes the preserved uniform-sharpening regression, seven new save/prompt regressions, and real Gradio component construction with host stand-ins. No GPU model was loaded and no active process was restarted.



An additional isolated execution of the installed Forge `save_image_with_geninfo` function saved and reopened an oversized-metadata fallback PNG with identical pixels and complete generation text. The regression reproduces the original `piexif.insert` JPEG failure before testing fallback. No Forge core file was modified.



Remaining: loaded Forge callback/lifecycle verification, a small reused-request folder batch with AutoNeg enabled and JPEG grids, cancellation/recovery, and fresh fixed-seed image comparisons for identity, proportions, hair/neck boundaries and whole-body cleanup. Prior recognition-model smoke cases and existing-output audits below are historical; they are not fresh rc4 image acceptance.



# Historical validation — 6.0.0-rc3



11 September 2026. **Offline and real recognition-model checks passed; newly generated rc3 visual acceptance remains pending.**



| Check | Actual result |

|---|---|

| Full offline suite after code changes | 96 passed, zero failures/skips; 2.474 seconds |

| Real Gradio 4.40 construction with host stand-ins | Passed within suite; 91 argument components |

| Queue bounds, snapshots, errors, unload, final-callback capture | Passed offline |

| Default geometry check does not warp finished pixels | Passed; original pixels compared |

| Real YuNet/SFace model inference | Five CPU smoke cases passed: identical image, brightness change, different person, no face, tiny face |

| Installed model integrity | Pinned official SHA-256 checks passed |

| Existing user outputs | 156 rc2 PNGs inspected and audited; all target/reference fingerprints mapped |

| Loaded rc3 Forge UI and cancellation | PENDING |

| Fresh fixed-seed rc3 identity, proportions and seams | PENDING |

| Whole-body tattoo/piercing/forehead cleanup | PENDING visual acceptance; no mark detector |

| User-specific identity threshold calibration | PENDING; scores are not accuracy percentages |



The latest full command was `python -B -m unittest discover -s tests -v` from the staged extension, using Forge's existing Python 3.13.12. Its full output is retained in the task workspace as `outputs/klein-head-swap-rc3-tests.log`. Model smoke results are in `outputs/klein-head-swap-identity-smoke.json`; existing-output review and scores are in `outputs/headswap-inspection`. These user-specific artifacts are not included in the public source package.



The real-model smoke used standard scikit-image/Matplotlib sample portraits. Identical-image similarity was 1.0, a brightness-adjusted version 0.9855, and the different-person case 0.1516. Blank and tiny-face cases returned Could not verify. First case including model load took 0.191 seconds; subsequent cases ranged 0.001–0.011 seconds. This tiny fixture set is not a general latency benchmark or accuracy calibration.



The local OpenCV 5.0.0 build emits a new-graph-engine target-selection warning. Inference succeeded using the requested OpenCV/CPU configuration; CUDA DNN targets were unavailable. No Torch, CUDA, Gradio or other dependency was upgraded, no diffusion model was loaded, and no Forge process was restarted or stopped.



Saved rc2 metadata recorded pixel-head correction in 100 of 156 inspected outputs; the largest reviewed pre-correction center discrepancy was 68.22px. Contact-sheet inspection and source review identified a credible seam mechanism from moving/blending adjacent pixels. Turning that operation off by default removes that implementation path; a new controlled generation is required to prove visual improvement and exclude remaining model/mask/postprocessor artifacts.



The prior rc2 results (70-test full run followed by 18 focused runtime tests, 71 distinct checks) and earlier rc1/v5.4 research are historical. Follow [the remaining procedure](docs/TESTING.md) and [identity-check limitations](docs/IDENTITY.md). Do not label existing rc2 image audits as new rc3 generation acceptance.

