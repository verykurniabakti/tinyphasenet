# Supplementary data

`per_seed_results.csv` holds the measurements on the ESP32-S3 for each training seed (42, 43, 44), the mean and the sample standard deviation over the three seeds, for the six measured configurations and for the single-channel Float32 model with the arena in external PSRAM.

| Column | Meaning |
|---|---|
| `configuration` | model and numeric format |
| `seed` | training seed, or `mean` / `sd` over the three seeds |
| `latency_ms` | latency of the `Invoke()` call per 30.72 s window |
| `arena_kb` | size of the tensor arena |
| `energy_mj` | mean board power during `Invoke()` multiplied by the invocation time (board level, includes idle consumption) |
| `f1_device_percent` | F1-score of the first pass over the 100 test windows, with the threshold chosen on the STEAD validation set |
| `differing_decisions` | decisions that differ from the interpreter on a computer, out of `decisions_compared` |
| `decisions_compared` | 100 evaluations (Float32 and three-channel models) or 200 (single-channel 16x8 and INT8, two passes over the windows) |

The file was generated from `results/final/hw_tiga_seed.json` and replaces the per-seed table that was an appendix of the manuscript.
