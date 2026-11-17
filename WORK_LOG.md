# WORK_LOG

Chronological log of what was run, decided, and fixed. Times are local (KST), 2026-11-17.

## Environment (checked 09:3x)

| item | value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU | Intel Core i7-10510U, 4 cores / 8 threads |
| RAM | 32 GB |
| Disk free | ~269 GB on C: |
| Python | 3.12.7 (project venv `.venv`, global Python untouched) |
| GPU | NVIDIA GeForce GTX 1050 (3 GB) + Intel UHD |
| CUDA | **not usable**: `nvcuda.dll` reports driver CUDA version 10.1 (`cuDriverGetVersion = 10010`), `cuInit` returns 999, 0 devices. Current PyTorch CUDA wheels need CUDA >= 11.8, so a ~2.7 GB CUDA wheel was not downloaded. |
| MPS | not applicable (Windows) |
| Selected device | CPU (`torch 2.15.0+cpu`, 4 intra-op threads). `select_device()` still tries CUDA -> MPS -> CPU with an FFT + conv round trip and logs every check to `results/<tag>/environment.json`. |
| PDF tooling | no pandoc/LaTeX/wkhtmltopdf; Microsoft Edge present -> headless Edge `--print-to-pdf` used for `report.pdf` (no new system install). |

## Decisions

1. **Boundary**: circular convolution (FFT-diagonal, exact adjoint, closed-form Tikhonov and
   PnP data step). Observations are generated with the same circular model, so the only
   mismatch studied is the blur width. Limitation stated in the report.
2. **Kernel**: normalised Gaussian truncated at radius ceil(4 sigma), renormalised to sum 1.
3. **Noise**: y = A x + N(0, 0.02^2), *not* clipped (keeps the Gaussian model exact).
4. **Clipping**: final outputs of both methods are clipped to [0, 1] as post-processing;
   metrics and label-free scores use the clipped output. Tikhonov's unclipped PSNR is stored
   (`psnr_unclipped`) as a diagnostic.
5. **PnP**: PnP-ADMM, x0 = v0 = y, u0 = 0, fixed K iterations, output v^K. Hyper-parameters
   (rho, sigma_d, K) selected on validation. No convergence claim.
6. **Denoiser**: no pretrained model in the main experiment; a small noise-conditional
   residual CNN (8 conv layers, 32 channels, 56,385 parameters, circular padding) trained on
   400 synthetic training phantoms. Size chosen from a CPU benchmark (09:38):
   d8/w32 = 21 ms per 128x128 image per forward pass (batch 24) vs. 69 ms (d10/w48) and
   133 ms (d12/w64); d8/w32 keeps the full run (with 2 extra reconstructions per case for
   the sensitivity score) at roughly 30-40 min.
7. **Failure label** (fixed before test): PSNR < median(validation matched PSNR of the same
   method) - 3 dB (i.e. roughly MSE > 2x the typical matched MSE). Auxiliary label: worst
   20% MSE within each (method, ratio, noise seed).
8. **Validation noise**: one noise realisation per validation image for tuning (cost).
9. **Score direction** fixed in advance: higher = more suspicious for all three scores.
10. **Biomedical extension**: OrganCMNIST 128 (MedMNIST+ v3.0, CC BY 4.0, md5-verified,
    287.5 MB, opt-in). Chosen over BreastMNIST (11 MB) because its official split is by CT
    scan (115/16/70), whereas BreastMNIST is split by image with no patient IDs.
11. **Pretrained denoiser extension**: KAIR `ffdnet_gray.pth` (MIT, 1.95 MB) -- noise-level
    conditional like our CNN, light enough for CPU. DRUNet (130 MB, 32 M params) rejected as
    too slow on this CPU. Diffusion: not attempted (CPU only).

## Run log

- 09:36 venv + CPU torch installed; package installed editable.
- 09:37 first operator/solver tests: 8 failures -- test images (8x8, 12x12) smaller than the
  truncated kernel (13x13); the operator correctly raised. Tests enlarged -> 36/36 pass.
- 09:38 phantom visual check (12 test phantoms) and CNN CPU benchmark (see decision 6).
- 09:40 smoke denoiser stage OK (40 iterations -> ~identity denoiser, as expected).
- 09:41 **full denoiser training started** (4000 iterations, background;
  log: `results/full/logs/pipeline.log`).
- 09:44 smoke `reproduce` end-to-end OK (1 min 52 s) after which plotting fixes: overlapping
  direct labels, empty panels when a condition has no failures (now labelled "undefined").
- 09:50 compute-relevant run code moved to `experiments/reconstruct.py` so that cosmetic
  edits of `pipeline.py` do not invalidate the cached reconstructions.
- 09:51 tests: 67 passed, 1 skipped (KAIR weights not yet downloaded), 1 slow deselected.

- 10:18 denoiser done (2202 s, 4000 it). Held-out gain over noisy input: +6.72 / +8.78 /
  +10.98 / +12.16 dB at sigma 0.01 / 0.02 / 0.05 / 0.10; random-init model is worse than the
  input at every sigma. Checkpoint `den-2df14010572a`.
- 10:19-10:32 `reproduce --profile quick` (13 min). Selection on 12 val images:
  Tikhonov lambda = 0.0316; PnP rho = 0.1, sigma_d = 0.05, K = 40.
- 10:33 **K at grid edge**: validation-only check (quick val) showed PSNR flat between
  K = 40 and 60 (+0.006 dB) and slowly decreasing to K = 100 -> grid kept unchanged for full.
  Decided before looking at any test number. Later re-done on the full validation set and
  saved: `results/full/tuning/iteration_check.csv` (+0.0015 dB at K = 60, -0.066 dB at 100).
- 10:34-11:25 `reproduce --profile full` (51 min: tuning 11 min, PnP run 37.7 min). Same PnP
  selection as quick; Tikhonov lambda = 0.0316. 0 divergences, 0 non-finite outputs.
- 10:34 bug: `stage_data` passed the logger object instead of its `info` method to the
  MedMNIST loader (`TypeError: 'Logger' object is not callable`); fixed, re-run OK.
  OrganCMNIST download md5-verified (287,499,235 bytes).
- 11:26 plots: fig2 titles collided, fig1 legend overlapped the "matched reference" label; fixed.
- 11:30 added `ratio_vs_matched.csv` (paired bootstrap vs matched) because the full run showed
  Tikhonov slightly better at ratio 1.25 than at matched (+0.144 dB [0.098, 0.187]); this is
  reported as an observation, with only a hypothesised explanation.
- 11:26-11:46 OrganCMNIST extension, full scale. **Tuning landed on the grid boundary in all
  three PnP hyper-parameters** (rho = 0.05, sigma_d = 0.08, K = 5); validation surface is
  flat there (29.91 dB at K = 5 vs 29.85 at K = 40). Grid NOT extended after the fact; flagged
  in the report as a domain-shift symptom (synthetic-trained denoiser on CT).
- 11:46 FFDNet (KAIR) extension started on the quick profile (CPU cost: full scale would be
  ~1.5 h).
- Report self-check before rendering: one drafted sentence ("sensitivity is weakest under
  under-estimation") contradicted the CSV (weakest within-ratio Spearman is at the matched
  ratio) and was rewritten; hand-typed numbers replaced by CSV placeholders.

- 11:46-12:04 FFDNet extension (quick): selection rho = 0.05 (grid edge), sigma_d = 0.05,
  K = 30. Pure denoising better than our CNN, but PnP worse at matched and much worse at
  ratio 0.5 (no divergence). Reported as an anecdote (12 images); cause not investigated.
- 12:05 report build failed: `TypeError` because an aggregate column named `div` shadowed
  `DataFrame.div`; renamed. First PDF was 8 pages -> runtime and per-ratio capture tables folded
  into text/CSV references, extension table made wide, print CSS tightened -> 5 pages.
- 12:09 wording fix: negative differences were printed as "-2.12 dB below Tikhonov" (double
  negative). Added an `abs:` placeholder format; all such sentences rewritten.
- 12:10-12:23 **reproducibility check**: deleted the cached quick reconstructions and the frozen
  selection, re-ran `reproduce --profile quick` (12 min 49 s). Every non-timing value of
  `per_image.csv` and the whole `selection.json` were bit-for-bit identical to the first run.
- 12:24 trained checkpoint committed under `results/checkpoints/` (sha256 dade27cb...); the
  pipeline restores it when the cache key matches (tested by hiding the cached copy).
- 12:26 logs redacted (absolute paths containing the local user name -> `<repo>`), logging now
  redacts automatically.
- 12:27 final test suite: **71 passed** (incl. slow smoke end-to-end), 0 skipped.

## Final status

| item | status |
|---|---|
| synthetic data, blur + noise model, Tikhonov, PnP-ADMM, mismatch experiment | done, verified by tests |
| denoiser trained and checked (beats input at every held-out sigma; random init does not) | done |
| validation-only tuning, frozen thresholds | done (`results/*/tuning/`) |
| smoke / quick / full runs | done (`results/smoke`, `results/quick`, `results/full`) |
| label-free scores, failure capture (within / pooled / condition-only), cluster bootstrap | done |
| figures, tables, report (md/html/pdf 5 pages), README | done, numbers generated from CSVs |
| public biomedical extension (OrganCMNIST) | done at full scale; PnP tuning on grid boundary (flagged) |
| pretrained denoiser extension (KAIR FFDNet) | done at quick scale only (CPU time) |
| diffusion-based reconstruction | **not attempted** (CPU only; not required) |
| GPU execution | **not possible** on this machine (driver supports CUDA 10.1 only) |
| fresh-from-scratch denoiser retraining via `reproduce --profile quick` | not re-timed separately; the same training ran once via `train-denoiser --profile full` (identical cache key) |
| git | result files were produced before the first commit, so their `environment.json` records the code SHA-256 (not a commit hash). |
