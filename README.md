# Preprocessing for AIND ephys pipeline
## aind-ephys-preprocessing


### Description

This capsule is designed to preprocess data for the AIND pipeline.

This capsule preprocessed the ephys stream with the following steps:

- Phase shift correction (see [phase_shift](https://spikeinterface.readthedocs.io/en/latest/modules/preprocessing.html#phase-shift))
- High-pass filtering
- Bad channel removal (using the method [developed by IBL](https://spikeinterface.readthedocs.io/en/latest/modules/preprocessing.html#detect-bad-channels-interpolate-bad-channels))
- Denoising: using one of the following methods
    - Common Median Referenc (CMR)
    - High-pass spatial filtering (see [highpass_spatial_filter()](https://spikeinterface.readthedocs.io/en/latest/modules/preprocessing.html#highpass-spatial-filter))
- (optional) Drift correction (estimation and interpolation)


### Inputs

The `data/` folder must include a single recorded session (e.g., "ecephys_664438_2023-04-12_14-59-51") with the `ecephys` (uncompressed Open Ephys output) or the `ecepys_compressed` and `ecephys_clipped` folders (processed with [aind-data-transfer](https://github.com/AllenNeuralDynamics/aind-data-transfer)). 
In addition, at least one JSON file generated by the [aind-ephys-job-dispatch](https://github.com/AllenNeuralDynamics/aind-ephys-job-dispatch) capsule is required (1 is recommended).

### Parameters

The `code/run` script takes 4 arguments:

- `preprocessing_strategy`: `cmr` (default) | `destripe`. The preprocessing strategy to use. `cmr` is the common median reference, `destripe` is the high-pass spatial filtering.
- `debug`: `false` (default) | `true`. If `true`, the capsule will run in debug mode, processing only a small subset of the data.
- `debug_duration_s`: `60` (default). The duration of the debug subset, in seconds.
- `drift`: `estimate` (default) | `apply` | `skip`. The drift correction strategy to use. `estimate` will estimate the drift and save it to the output folder. `apply` will estimate the drift and interpolate the traces using the estimated motion. `skip` will skip the drift correction.

A full list of parameters can be found at the top of the `code/run_capsule.py` script and is reported here:

```python
preprocessing_params = dict(
        preprocessing_strategy="cmr", # 'destripe' or 'cmr'
        highpass_filter=dict(freq_min=300.0,
                             margin_ms=5.0),
        phase_shift=dict(margin_ms=100.),
        detect_bad_channels=dict(method="coherence+psd",
                                 dead_channel_threshold=-0.5,
                                 noisy_channel_threshold=1.,
                                 outside_channel_threshold=-0.3,
                                 n_neighbors=11,
                                 seed=0),
        remove_out_channels=True,
        remove_bad_channels=True,
        max_bad_channel_fraction_to_remove=0.5,
        common_reference=dict(reference='global',
                              operator='median'),
        highpass_spatial_filter=dict(n_channel_pad=60,
                                     n_channel_taper=None,
                                     direction="y",
                                     apply_agc=True,
                                     agc_window_length_s=0.01,
                                     highpass_butter_order=3,
                                     highpass_butter_wn=0.01),
        motion_correction=dict(compute=True,
                               apply=False,
                               preset="nonrigid_accurate",)
    )
```

### Output

The output of this capsule is the following:

- `results/preprocessed_{recording_name}` folder, containing the preprocessed data saved to binary
- `results/motion_{recording_name}.json` file, containing the motion estimation data (if `drift` is `estimate` or `apply`)
- `results/preprocessed_{recording_name}.json` file, containing the JSON file to reload the processing chain
- `results/preprocessedviz_{recording_name}.json` file, all the information required to make a visualization of the preprocessing downstream
- `results/data_process_preprocessing_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

