import warnings
warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
import shutil
import json
import sys
import time
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-capsule-ephys-preprocessing"
VERSION = "0.1.0"

import wavpack_numcodecs

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
                                     highpass_butter_wn=0.01)
    )

job_kwargs = {
    'n_jobs': -1,
    'chunk_duration': '1s',
    'progress_bar': True
}

data_folder = Path("../data/")
results_folder = Path("../results/")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        PREPROCESSING_STRATEGY = sys.argv[1]
        if sys.argv[2] == "true":
            DEBUG = True
            DURATION_S = float(sys.argv[3])
        else:
            DEBUG = False
            DURATION_S = None
    else:
        PREPROCESSING_STRATEGY = "cmr"
        DEBUG = False
        DURATION_S = None

    print(f"Wavpack-numcodecs version: {wavpack_numcodecs.__version__}")

    data_processes_folder = results_folder / "data_processes" / "preprocessing"
    data_processes_folder.mkdir(exist_ok=True, parents=True)

    if DEBUG:
        print(f"DEBUG ENABLED - Only running with {DURATION_S} seconds")
    
    si.set_global_job_kwargs(**job_kwargs)

    assert PREPROCESSING_STRATEGY in ["cmr", "destripe"], f"Preprocessing strategy can be 'cmr' or 'destripe'. {PREPROCESSING_STRATEGY} not supported."
    preprocessing_params["preprocessing_strategy"] = PREPROCESSING_STRATEGY

    # load job json files
    job_config_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    print(f"Found {len(job_config_json_files)} json configurations")

    if len(job_config_json_files) > 0:
        ####### PREPROCESSING #######
        print("\n\nPREPROCESSING")
        datetime_start_preproc = datetime.now()
        t_preprocessing_start = time.perf_counter()
        preprocessing_notes = ""
        preprocessing_vizualization_data = {}
        print(f"Preprocessing strategy: {PREPROCESSING_STRATEGY}")

        preprocessed_output_folder = results_folder / "preprocessed"
        preprocessed_viz_folder = results_folder / "visualization_preprocessed"
        preprocessed_viz_folder.mkdir(exist_ok=True)

        for job_config_file in job_config_json_files:
            with open(job_config_file, "r") as f:
                job_config = json.load(f)
            session_name = job_config["session"]
            session = data_folder / session_name
            assert session.is_dir(), f"Could not find {session_name} in data folder"

            ecephys_full_folder = session / "ecephys"
            ecephys_compressed_folder = session / "ecephys_compressed"
            compressed = False
            if ecephys_compressed_folder.is_dir():
                compressed = True
                ecephys_folder = session / "ecephys_clipped"
            else:
                ecephys_folder = ecephys_full_folder

            experiment_name = job_config["experiment_name"]
            stream_name = job_config["stream_name"]
            block_index = job_config["block_index"]
            segment_index = job_config["segment_index"]
            recording_name = job_config["recording_name"]

            preprocessing_vizualization_data[recording_name] = {}
            preprocessing_output_process_json = data_processes_folder / f"preprocessing_{recording_name}.json"

            exp_stream_name = f"{experiment_name}_{stream_name}"
            if not compressed:
                recording = se.read_openephys(ecephys_folder, stream_name=stream_name, block_index=block_index)
            else:
                recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")

            if DEBUG:
                recording_list = []
                for segment_index in range(recording.get_num_segments()):
                    recording_one = si.split_recording(recording)[segment_index]
                    recording_one = recording_one.frame_slice(start_frame=0, end_frame=int(DURATION_S*recording.sampling_frequency))
                    recording_list.append(recording_one)
                recording = si.append_recordings(recording_list)

            if segment_index is not None:
                recording = si.split_recording(recording)[segment_index]

            print(f"Preprocessing recording: {recording_name}")
            print(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

            recording_ps_full = spre.phase_shift(recording, **preprocessing_params["phase_shift"])

            recording_hp_full = spre.highpass_filter(recording_ps_full, **preprocessing_params["highpass_filter"])
            preprocessing_vizualization_data[recording_name]["timeseries"] = {}
            preprocessing_vizualization_data[recording_name]["timeseries"]["full"] = dict(
                                                            raw=recording.to_dict(),
                                                            phase_shift=recording_ps_full.to_dict(),
                                                            highpass=recording_hp_full.to_dict()
                                                        )

            # IBL bad channel detection
            _, channel_labels = spre.detect_bad_channels(recording_hp_full, **preprocessing_params["detect_bad_channels"])
            dead_channel_mask = channel_labels == "dead"
            noise_channel_mask = channel_labels == "noise"
            out_channel_mask = channel_labels == "out"
            print(f"\tBad channel detection:")
            print(f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}")
            dead_channel_ids = recording_hp_full.channel_ids[dead_channel_mask]
            noise_channel_ids = recording_hp_full.channel_ids[noise_channel_mask]
            out_channel_ids = recording_hp_full.channel_ids[out_channel_mask]

            all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

            skip_processing = False
            max_bad_channel_fraction_to_remove = preprocessing_params["max_bad_channel_fraction_to_remove"]
            if len(all_bad_channel_ids) >= int(max_bad_channel_fraction_to_remove * recording.get_num_channels()):
                print(f"\tMore than {max_bad_channel_fraction_to_remove * 100}% bad channels ({len(all_bad_channel_ids)}). "
                      f"Skipping further processing for this recording.")            
                preprocessing_notes += f"\n- Found {len(all_bad_channel_ids)} bad channels. Skipping further processing\n"
                skip_processing = True
                # in this case, processed timeseries will not be visualized
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
                recording_drift = recording_hp_full
            else:
                if preprocessing_params["remove_out_channels"]:
                    print(f"\tRemoving {len(out_channel_ids)} out channels")
                    recording_rm_out = recording_hp_full.remove_channels(out_channel_ids)
                    preprocessing_notes += f"{recording_name}:\n- Removed {len(out_channel_ids)} outside of the brain."
                else:
                    recording_rm_out = recording_hp_full

                recording_processed_cmr = spre.common_reference(recording_rm_out, **preprocessing_params["common_reference"])

                bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))
                recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
                recording_hp_spatial = spre.highpass_spatial_filter(recording_interp, **preprocessing_params["highpass_spatial_filter"])
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = dict(
                                                                highpass=recording_rm_out.to_dict(),
                                                                cmr=recording_processed_cmr.to_dict(),
                                                                highpass_spatial=recording_hp_spatial.to_dict()
                                                            )

                preproc_strategy = preprocessing_params["preprocessing_strategy"]
                if preproc_strategy == "cmr":
                    recording_processed = recording_processed_cmr
                else:
                    recording_processed = recording_hp_spatial

                if preprocessing_params["remove_bad_channels"]:
                    print(f"\tRemoving {len(bad_channel_ids)} channels after {preproc_strategy} preprocessing")
                    recording_processed = recording_processed.remove_channels(bad_channel_ids)
                    preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"
                recording_saved = recording_processed.save(folder=preprocessed_output_folder / recording_name)
                recording_drift = recording_saved

                # store recording for drift visualization
                preprocessing_vizualization_data[recording_name]["drift"] = dict(
                                                        recording=recording_drift.to_dict()
                                                    )
                with open(preprocessed_viz_folder / f"{recording_name}.json", "w") as f:
                    json.dump(check_json(preprocessing_vizualization_data), f, indent=4)
                

        t_preprocessing_end = time.perf_counter()
        elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)

        # save params in output
        preprocessing_params["recording_name"] = recording_name
        preprocessing_process = DataProcess(
                name="Ephys preprocessing",
                version=VERSION, # either release or git commit
                start_date_time=datetime_start_preproc,
                end_date_time=datetime_start_preproc + timedelta(seconds=np.floor(elapsed_time_preprocessing)),
                input_location=str(data_folder),
                output_location=str(results_folder),
                code_url=URL,
                parameters=preprocessing_params,
                notes=preprocessing_notes
            )
        with open(preprocessing_output_process_json, "w") as f:
            f.write(preprocessing_process.json(indent=3))

        print(f"PREPROCESSING time: {elapsed_time_preprocessing}s")


