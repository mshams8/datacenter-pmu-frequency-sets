# Data-Center Synchrophasor Frequency-Set Benchmark

Portable data and Python replay code accompanying **Estimator-Specific Closed-Catalogue Frequency Sets from Reporting-Rate-Aliased Synchrophasor Measurements: A Data-Center Case Study** (manuscript under review). Version 1.0.0 preserves the revised study's evidence; it is not a claim of acceptance or field validation.

## Contents

The data archive contains 104 source-bus three-phase voltage records exported without quantization from saved nonlinear ParaEMT simulations, 384 source-bus PMU-derived streams, eight high-rate terminal-voltage records, AC-solved case definitions, forcing metadata, and the retained numerical results. The code archive contains estimator implementations, inference and replay scripts, tests, and this documentation. Numerical arrays are NPZ files readable with `allow_pickle=False`; simulator pickle files are not distributed.

* **64 original trials:** two AC-solved operating conditions (bus 4, 0.95 loading, intact network; bus 39, 1.05 loading, physical 6–11 line outage), four frequencies (14.7, 29, 31, 45.3 Hz), two modulation amplitudes (0.005 and 0.02 pu), and four onset/phase alignments. There are eight dependency groups. Each trial has one- and two-cycle filtered views at 30, 60, and 120 frames/s.
* **40 additional physical trials:** two conditions, four nominal frequencies, and five settings: nominal, frequency offsets of −0.25/+0.25 Hz, and frequency slopes of −0.25/+0.25 Hz/s. Eight nominal cases additionally retain 20-kHz terminal histories.
* **Measurement studies:** native waveform replay through SDFT-pre, TF-p2 and SDFT-reported research estimators; reference/filtered processing; controlled waveform response tests; 76,800 dependent channel-noise trials; threshold sensitivity and noise-only checks.

## Obtain and run

Download `benchmark-code-v1.0.0.zip` and `benchmark-data-v1.0.0.zip` from the versioned archive linked below. Extract the code ZIP, enter `datacenter-pmu-frequency-sets`, then extract the data ZIP there so that `data/native_manifest.csv` exists. Alternatively clone this repository and extract the data ZIP in its root.

```console
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/verify_dataset.py
python -m pytest -q
python scripts/67_tim_revision_studies.py
python scripts/68_tim_waveform_study.py
python scripts/69_tim_native_replay.py
python scripts/compare_reference.py
python scripts/70_tim_revision_figures.py
```

Use Python 3.11 or newer. The pinned environment records the versions used to check this release. Runs use CPU numerical routines and do not require a GPU. All generated files go to `reproduced/`; retained results in `data/reference/` are never overwritten. The full noise and waveform studies may take several minutes. `--pilot` provides a smoke run but does not reproduce the full study. Read [DATA_DICTIONARY.md](DATA_DICTIONARY.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before interpreting results.

## Interpretation

This is a closed-catalogue frequency-set procedure applied after an event detector supports the catalogue hypothesis. A result may be a singleton, unresolved set, outside the calibrated response, or insufficient channel support. Calibration is empirical and estimator-specific; it does not establish a distribution-free guarantee for the dependent physical cases. An incorrect fold can exclude the true frequency. Off-catalogue forcing and drift can produce misleading catalogue singletons. These negative results are included.

The 1-kHz saved waveform is resampled to 4.8 kHz only for estimator compatibility; this does not recover bandwidth above the original 500-Hz Nyquist limit. Research estimators and simplified PMU-derived streams are not commercial PMU measurements or conformance tests. There are no confidential data-center measurements or third-party field recordings in this release.

## Provenance and license

Own code: BSD-3-Clause (`LICENSE`). Own simulation data and documentation: CC BY 4.0 (`DATA_LICENSE.md`). Upstream ParaEMT acknowledgment and license: `licenses/ParaEMT-BSD-3-Clause.txt`. The simulation tool is [ParaEMT_public](https://github.com/NatLabRockies/ParaEMT_public), upstream reference commit `d79d735a4a587d56c5b88187d1a499195b6b2b84`. That simulator is not bundled. The released forcing definitions and solved cases document the study; this package reproduces the measurement analysis from saved waveforms, not a standalone rerun of every EMT solver state.

File hashes, source lineage, and validation results accompany the archive. Please cite the versioned Zenodo dataset using `CITATION.cff`. Repository: https://github.com/mshams8/datacenter-pmu-frequency-sets.

## Versioned archive

Dataset and code: https://doi.org/10.5281/zenodo.23808827 (version 1.0.0).
