# Changelog

All notable changes to CortexLab are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/) once tagged releases begin.

Until the first tagged release, "unreleased" tracks everything currently on
`master`.

## [Unreleased]

### Lesion-pipeline arc

A reviewer-grade modality-lesion pipeline for the BOLD Moments dataset
landed across PRs #40, #42, #43, #44, #45, #46, #49, #50, #54, #55. End-to-end:
features → ridge encoder → ablation → ROI summary → permutation p-values →
BH-FDR → cortical surface plots.

#### Added
- `cortexlab.features.TextFeatureExtractor` with `clip-text-vit-l-14` and
  `siglip2-text-vit-l` presets; mirrors the vision-side
  `FoundationFeatureExtractor` pattern (#40).
- `cortexlab.data.studies.lahner2024bold.load_subject` end-to-end loader
  consuming both fMRI betas and pre-extracted feature caches.
- `cortexlab.data.studies.lahner2024bold.load_captions` and
  `middle_frame_paths` helpers, aligned row-for-row with `list_stimulus_paths`.
- `cortexlab.data.studies.lahner2024bold.load_noise_ceiling` for per-subject
  BOLD Moments ceiling pickles; parses `n=10`, `n=1`, and `n=k` variants
  with multiple payload shapes (#49).
- `cortexlab.data.parcellations` module with `build_roi_indices`,
  `load_hcp_mmp_fsaverage`, and `load_hcp_mmp_from_freesurfer`. Default
  29-ROI subset covers V1-V4, LO/MT, FFC/PH, A1/A4/A5, STS subdivisions,
  Broca's area, IPL, ATL (#46).
- `cortexlab.analysis.lesion.run_modality_lesion` row-permutation test
  per modality with Phipson & Smyth `+1` smoothing; per-voxel one-sided
  p-values returned in `LesionResult.p_values` (#50).
- `cortexlab.analysis.stats` module with `bh_fdr` (numpy-only,
  NaN-aware, monotonicity-enforcing) and `fraction_significant` (#55).
- `experiments.build_feature_cache` CLI for vision + text feature
  extraction over BOLD Moments stimuli (#40).
- `scripts.postprocess_roi` CPU-only schema upgrade tool: applies
  parcellation + noise ceiling to existing whole-cortex lesion results
  without a GPU rerun (#51, plus follow-up `008d54f`).
- `experiments.causal_modality_ablation` orchestrator gained CLI flags
  for `--data-root`, `--feature-cache`, `--modalities` (#43);
  `--parcellation` + `--lh-annot` + `--rh-annot` (#46);
  `--noise-ceiling`, `--noise-ceiling-n`, `--noise-ceiling-split`,
  `--noise-ceiling-space` (#49, #54);
  `--permutations`, `--permutation-seed` (#50);
  `--fdr`, `--fdr-alpha` (#55).

#### Fixed
- `run_modality_lesion(device="cuda")` previously crashed in `_r2_score`
  with a device-mismatch RuntimeError because `Y_test` stayed on CPU
  while encoder predictions came back on CUDA (#42).
- `middle_frame_paths` resolves both bare `<stem>.jpg` symlink trees
  and the CSAIL `<stem>_<frame>_<total>.jpg` naming convention shipped
  in the real dataset (#44).
- `TextFeatureExtractor` survives transformers 5.x where
  `CLIPModel.get_text_features` returns a `BaseModelOutputWithPooling`
  object instead of a plain tensor (#45).
- `load_noise_ceiling` filename template matches the actual BOLD Moments
  release (`sub-XX_noiseceiling_space-{space}_task-{split}_hemi-{hemi}_n-{n}.pkl`),
  with new `space=` kwarg for fsaverage / MNI variants (#54).

### Visualisation arc

Three independent rendering engines for cortical surface maps, with
auto-fallback. PRs #56, #57, #58, plus follow-ups in `867035e`, `8a2250e`,
`ee7d50e`, `783acdf`, `ca48000`.

#### Added
- `scripts.plot_cortical_maps`: 4-panel fsaverage surface PNGs from a
  lesion output directory; auto-detects modalities, recomputes BH-FDR
  q-values at plot time, and emits q-masked variants (#56).
- `scripts.animate_cortical_maps`: rotating-brain GIF / MP4 with
  configurable frame count and engine.
- `cortexlab.viz.surface_renderer` abstraction with
  `MatplotlibRenderer`, `PlotlyRenderer`, `PyVistaRenderer` and a
  `make_renderer(engine='auto'|...)` factory. Auto-precedence is
  pyvista > plotly > matplotlib (#57, #58).
- `RenderConfig` frozen dataclass; `truncate_to_mesh` for
  zero-interpolation downsampling between fsaverage levels.
- `[viz]` extras (plotly + kaleido + Pillow + imageio + imageio-ffmpeg)
  for the lighter-weight WebGL rendering path. Heavier `[plotting]`
  extras already include pyvista (#57).

#### Fixed
- `MatplotlibRenderer` forces the `Agg` backend before importing
  pyplot, so it survives running after PyVista left matplotlib's
  interactive backend half-initialised (#58).
- Auto-fallback when kaleido v1+ Chrome binary is missing — the
  factory probes with a 1x1 figure up-front and falls back to
  matplotlib rather than failing mid-animation (`783acdf`).
- Pinned `kaleido<1.0` because v1.x spawns a fresh Chromium process
  per render call (no batching), making animations 30x slower than
  the 0.2.x series (`ca48000`).

### Operational
- Tightened the auto-assign bot: excludes OWNER/MEMBER/COLLABORATOR
  commenters and requires explicit opt-in regex like `/assign me`,
  preventing maintainer triage chatter from mis-assigning issues (#41).
- Added `Examples` section to `BrainAlignmentBenchmark.score_model`
  docstring (#52, closes #37).
- Bumped `x_transformers` from 1.27.20 to 1.43.0 to silence two
  `torch.cuda.amp.autocast` deprecation warnings (#53, closes #31).
- Renamed ambiguous `l` loop variable in `data/loader.py:get_topk_rois`
  (#59, closes #23).
- Ruff sweep across `src/`: 10 of 15 standing errors cleaned up,
  including dead `file()` reference past a `raise NotImplementedError`
  (#60).

### Test coverage
- 153 → 280 passing tests across the lesion-pipeline and
  visualisation arcs. CUDA-gated tests grew from 2 to 3.

## v0.2 — 2026-04 (PR #22)

### Added
- GPU voxelwise ridge encoder with `torch` and `Triton` backends
  (`cortexlab.gpu.ridge`).
- Foundation-model feature extractors for CLIP, DINOv2, SigLIP2,
  V-JEPA2, PaLiGemma2 (`cortexlab.features.extractors`).
- Causal modality lesion analysis (`cortexlab.analysis.lesion`).
- Noise-ceiling estimation (`cortexlab.analysis.noise_ceiling`).
- Brain-alignment benchmark (`cortexlab.analysis.brain_alignment`)
  with RSA, CKA, Procrustes; permutation tests; FDR correction;
  noise ceiling.
- Cognitive-load and temporal-dynamics analysis modules.
- ROI connectivity analysis (`cortexlab.analysis.connectivity`).
- SLURM submission templates (`scripts/slurm/`).
- Tutorial notebook (`notebooks/tutorial_analysis.ipynb`).

### Test coverage
- 89 → 143 passing tests.

## v0.1 — 2026-03

Initial open-source release. CortexLab forks and restructures Meta's
TRIBE v2 into a Python package layout (`src/cortexlab/...`), adds a
test suite, and ships an interactive Streamlit dashboard
(`cortexlab-dashboard` repo) with a 3D brain viewer and live-inference
mode.

[Unreleased]: https://github.com/siddhant-rajhans/cortexlab/compare/v0.2...HEAD
