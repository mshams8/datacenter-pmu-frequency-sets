# Data dictionary

All frequencies are in Hz; time is in seconds; source-bus voltage is in the simulation per-unit convention. Loading factors and forcing amplitude are dimensionless. File names preserve case identifiers across measurements and result tables.

| Path | Meaning |
|---|---|
| `data/native_manifest.csv` | 104 cases; source bus, original/additional designation, NPZ path, parent simulation SHA-256, optional high-rate path |
| `data/native/*.npz` | `time_s` (N), `voltage_abc_pu` (N,3; instantaneous a,b,c), `voltage_phasor_pu` (N; complex simulation bus-state voltage), `pll_frequency_hz` (N; simulation bus-state PLL), `source_bus` (scalar) |
| `data/terminal_20khz/*.npz` | Eight nominal additional cases; terminal `time_s` and `voltage_abc_pu`; replay interpolates to the declared 20-kHz grid before 4.8-kHz conversion |
| `data/original/case_manifest.csv` | Original 64 cases, dependency group, source bus, condition, frequency, amplitude, alignment, onset, phase, ramp, expected load modulation, parent hash |
| `data/original/pmu/open_*cycle/*fps/*.npz` | 384 source-bus views: `time_s`, `bus_ids` (one element), `voltage_magnitude_pu`, `voltage_angle_deg`, `frequency_hz`, `rocof_hz_per_s` (N,1). Original numeric precision is retained |
| `data/pmu_parent_hashes.csv` | Hashes of the original all-bus view files from which the source-bus columns were exported |
| `data/protocols/` | Original frozen measurement protocols and explicit additional-case parameters; script hashes refer to the original study scripts |
| `data/solved_cases/` | The two separately AC-solved IEEE 39-bus condition definitions |
| `data/reference/` | Retained per-event results, calibration quantities, summaries, and original simulation validation |
| `provenance/source_hashes.json` | SHA-256 hashes of original scientific sources before portability changes |
| `data/SHA256SUMS.txt` | SHA-256 of every distributed data file except this manifest |

`truth` is the stationary physical forcing frequency; drift cases intentionally have no stationary truth. In decision tables, `a` denotes the selected reported fold where present, `outcome` identifies the returned result, `correct`/`incorrect` identify singleton correctness, and `fold_correct` checks compatibility with the true catalogue fold. `outside` means no calibrated candidate remains; it is not a universal out-of-distribution detector. `group` identifies dependencies, not an additional independent sample. Empty group entries for additional trials do not imply independent repetitions. The authoritative equations and physical scope are in the associated manuscript; the scripts expose each numerical operation.
