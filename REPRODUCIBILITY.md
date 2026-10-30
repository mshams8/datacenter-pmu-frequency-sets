# Reproducibility and evidence boundaries

The release changes file paths and simulation-object loading, not feature formation, estimator formulas, fitted splits, thresholds, or decision logic. The source-bus NPZ arrays are extracted directly from trusted local simulation outputs and contain no object pickles. Other buses and large solver objects are omitted. The original source hashes and the numeric export checks provide lineage; parent hashes cannot be recomputed from a source-bus subset.

`67_tim_revision_studies.py` uses the 384 original filtered views. It reproduces nominal outcomes, the 76,800 channel-noise trials, group bootstrap intervals, threshold studies, and noise-only checks. Original within-condition alignment 0/1 is design, 2/3 calibration, and the other condition is test; the direction is then reversed. Derived views and perturbations are not independent physical events.

`68_tim_waveform_study.py` synthesizes controlled three-phase waveforms to test the research estimator chain under modulation, noise, offsets, drift and channel delays. These are controlled measurement experiments, distinct from nonlinear EMT simulations. It refits deterministic response models and does not distribute pickled fitted objects.

`69_tim_native_replay.py` processes 104 exported native waveform records. The original 64 cases train reciprocal-condition models; these models remain fixed for the 40 additional physical trials. The saved 1-kHz abc samples are resampled by 24/5 to 4.8 kHz, with exactly the same terminal-sample omission as the original study. Eight additional nominal cases also compare with a 20-kHz terminal-history route. The exported simulation phasor and PLL frequency support the separate simplified filter study; they are not measured commercial-PMU channels.

`compare_reference.py` checks per-event and aggregate numerical results against retained tables with declared tolerances. Protocol script hashes change after packaging, and `simulation_validation.csv` in the replay hashes NPZ files rather than original simulator pickles; those two provenance differences are intentionally not compared as scientific outputs. The original simulation-validation table is retained in `data/reference/`.

Expected native nominal outcomes across 64 cases: SDFT-pre accepts 63 reference and 62 filtered correct singletons; TF-p2 accepts 0 reference and 40 filtered correct singletons. The nearest-centre comparison resolves 64/64 in each of those four settings. These are within-benchmark results, not expected field performance. SDFT-reported retains unresolved sets. See all retained tables for rejection, incorrect-fold, off-catalogue and drift outcomes.

EMT reruns require the original facility integration and solver environment beyond this portable replay package. The released solved cases and forcing definitions are supplied for inspection and adaptation; they are not represented as a complete substitute for that environment. No commercial PMU, hardware-in-the-loop, or field-data-center validation is claimed.

## Release verification

All 25 included tests passed, and all 18 scientific result tables were reproduced. The maximum observed calibration-radius difference was 3.64e-8 after source-bus column extraction; the comparison tolerance is 5e-8 for that table and 1e-9 absolute / 1e-8 relative for other numerical tables. All event decisions are unchanged. The unchanged older response helpers remain for API completeness and regression tests; the paper uses the empirical decision logic in script 67.
