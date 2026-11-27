# Data

No seismic waveforms are redistributed in this release. All waveforms come from public datasets that must be obtained from their original sources:

| Dataset | Used for | Reference |
|---|---|---|
| STEAD (Stanford Earthquake Dataset) | training, validation and test of all models; the 100 windows run on the device | Mousavi et al., IEEE Access 2019, doi:10.1109/ACCESS.2019.2947848; license CC BY 4.0 (official repository, checked 10 Nov 2026) |
| INSTANCE | cross-dataset accuracy figures taken from the companion study (not recomputed here) | Michelini et al., Earth Syst. Sci. Data 13, 2021, doi:10.5194/essd-13-5509-2021; dataset doi:10.13127/instance; license CC BY 4.0 (DataCite record and INGV page, checked 10 Nov 2026) |
| TXED | same as above | Chen et al., Seismol. Res. Lett. 95, 2024, doi:10.1785/0220230327. The repository README states an MIT license for the TXED package; **no license for the data itself was found** (checked 10 Nov 2026), so no TXED-derived data is redistributed here |
| ETHZ (SED, Swiss Seismological Service) | same as above | per-network DOIs (CH 10.12686/sed/networks/ch, C4, 8D, S, XT 10.12686/alparray/xt_2014). **No license statement was found** in the SeisBench documentation, in the DataCite records of these network DOIs, or on the CH network page (checked 10 Nov 2026), so no ETHZ-derived data is redistributed here |

## The 100 test windows run on the device

The windows are the same for every model and seed. They are drawn by `select()` in `scripts/pipeline/10_make_hw_eval_inputs.py`: windows of the STEAD test split with a flat (zero-variance) trace excluded, then 50 noise and 50 earthquake windows sampled with `random_state = config.SEED`, then shuffled with the same seed. The firmware headers that embed the window samples (`hw_eval*.h`, `test_windows.h`) are **not** included because they contain STEAD data; `scripts/pipeline/88_make_fw_seed_inputs.py`, `163_make_fw_int8_final.py`, `165_make_fw_3ch_int8.py` and `170_make_fw_float32.py` regenerate them from STEAD.

The 100 selected windows are listed in `test_windows_100.csv` (firmware order, STEAD `trace_name`, label 1 = earthquake / 0 = noise, and the start sample of the 30.72 s window within the trace). The list was produced with the original `select()` and its label sequence is identical to the label sequence stored in the firmware input of seed 42 (`results/final/fw_input_seed42.json`).

The per-window scores and labels of the reference computation (Python) are in `results/final/fw_input_*.json`.

## Paths

`scripts/core/config.py` reads the dataset locations from environment variables, with the defaults `data_external/<file name>`: `STEAD_CSV`, `STEAD_HDF5`, `INSTANCE_EQ_CSV`, `INSTANCE_EQ_HDF5`, `INSTANCE_NOISE_CSV`, `INSTANCE_NOISE_HDF5` and `ETHZ_DIR`. Set them to your own copies before running anything that reads waveforms. `scripts/core/phasenet_keras.py` reads the checkpoint of the original PhaseNet from `PHASENET_CKPT` (default `phasenet/model/190703-214543/model_95.ckpt`), used only by the companion scripts that compare with the published weights.
