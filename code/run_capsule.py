import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads 
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import numpy as np
from pathlib import Path
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
from aind_data_schema.core.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing"
VERSION = "0.1.0"

preprocessing_params = dict(
    denoising_strategy="cmr",  # 'destripe' or 'cmr'
    min_preprocessing_duration=120,  # if less than this duration, processing is skipped (probably a test recording)
    highpass_filter=dict(freq_min=300.0, margin_ms=5.0),
    phase_shift=dict(margin_ms=100.0),
    detect_bad_channels=dict(
        method="coherence+psd",
        dead_channel_threshold=-0.5,
        noisy_channel_threshold=1.0,
        outside_channel_threshold=-0.3,
        n_neighbors=11,
        seed=0,
    ),
    remove_out_channels=True,
    remove_bad_channels=True,
    max_bad_channel_fraction=0.5,  # above this fraction, processing is skipped
    common_reference=dict(reference="global", operator="median"),
    highpass_spatial_filter=dict(
        n_channel_pad=60,
        n_channel_taper=None,
        direction="y",
        apply_agc=True,
        agc_window_length_s=0.01,
        highpass_butter_order=3,
        highpass_butter_wn=0.01,
    ),
    motion_correction=dict(
        compute=True,
        apply=False,
        preset="nonrigid_accurate",
    ),
)

n_jobs_co = os.getenv("CO_CPUS")
n_jobs = int(n_jobs_co) if n_jobs_co is not None else -1

job_kwargs = {"n_jobs": n_jobs, "chunk_duration": "1s", "progress_bar": False}

data_folder = Path("../data/")
results_folder = Path("../results/")


# filter and resample LFP
lfp_filter_kwargs = dict(freq_min=0.1, freq_max=500)
lfp_sampling_rate = 2500

# default event line from open ephys
data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

si.set_global_job_kwargs(**job_kwargs)


parser = argparse.ArgumentParser(description="Preprocess AIND Neurpixels data")

debug_group = parser.add_mutually_exclusive_group()
debug_help = "Whether to run in DEBUG mode"
debug_group.add_argument("--debug", action="store_true", help=debug_help)
debug_group.add_argument("static_debug", nargs="?", default="false", help=debug_help)


# positional arguments
denoising_group = parser.add_mutually_exclusive_group()
denoising_help = "Which denoising strategy to use. Can be 'cmr' or 'destripe'"
denoising_group.add_argument("--denoising", choices=["cmr", "destripe"], help=denoising_help)
denoising_group.add_argument("static_denoising", nargs="?", default="cmr", help=denoising_help)

remove_out_channels_group = parser.add_mutually_exclusive_group()
remove_out_channels_help = "Whether to remove out channels"
remove_out_channels_group.add_argument("--no-remove-out-channels", action="store_true", help=remove_out_channels_help)
remove_out_channels_group.add_argument(
    "static_remove_out_channels", nargs="?", default="true", help=remove_out_channels_help
)

remove_bad_channels_group = parser.add_mutually_exclusive_group()
remove_bad_channels_help = "Whether to remove bad channels"
remove_bad_channels_group.add_argument("--no-remove-bad-channels", action="store_true", help=remove_bad_channels_help)
remove_bad_channels_group.add_argument(
    "static_remove_bad_channels", nargs="?", default="true", help=remove_bad_channels_help
)

max_bad_channel_fraction_group = parser.add_mutually_exclusive_group()
max_bad_channel_fraction_help = (
    "Maximum fraction of bad channels to remove. If more than this fraction, processing is skipped"
)
max_bad_channel_fraction_group.add_argument(
    "--max-bad-channel-fraction", default=0.5, help=max_bad_channel_fraction_help
)
max_bad_channel_fraction_group.add_argument(
    "static_max_bad_channel_fraction", nargs="?", default="0.5", help=max_bad_channel_fraction_help
)

motion_correction_group = parser.add_mutually_exclusive_group()
motion_correction_help = "How to deal with motion correction. Can be 'skip', 'compute', or 'apply'"
motion_correction_group.add_argument("--motion", choices=["skip", "compute", "apply"], help=motion_correction_help)
motion_correction_group.add_argument("static_motion", nargs="?", default="skip", help=motion_correction_help)

debug_duration_group = parser.add_mutually_exclusive_group()
debug_duration_help = (
    "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
)
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default="30", help=debug_duration_help)


if __name__ == "__main__":
    args = parser.parse_args()

    DEBUG = args.debug or args.static_debug == "true"
    DENOISING_STRATEGY = args.denoising or args.static_denoising
    REMOVE_OUT_CHANNELS = False if args.no_remove_out_channels else args.static_remove_out_channels == "true"
    REMOVE_BAD_CHANNELS = False if args.no_remove_bad_channels else args.static_remove_bad_channels == "true"
    MAX_BAD_CHANNEL_FRACTION = float(args.max_bad_channel_fraction or args.static_max_bad_channel_fraction)
    motion_arg = args.motion or args.static_motion
    COMPUTE_MOTION = True if motion_arg != "skip" else False
    APPLY_MOTION = True if motion_arg == "apply" else False
    DEBUG_DURATION = float(args.debug_duration or args.static_debug_duration)

    data_process_prefix = "data_process_preprocessing"

    print(f"Running preprocessing with the following parameters:")
    print(f"\tDENOISING_STRATEGY: {DENOISING_STRATEGY}")
    print(f"\tREMOVE_OUT_CHANNELS: {REMOVE_OUT_CHANNELS}")
    print(f"\tREMOVE_BAD_CHANNELS: {REMOVE_BAD_CHANNELS}")
    print(f"\tMAX BAD CHANNEL FRACTION: {MAX_BAD_CHANNEL_FRACTION}")
    print(f"\tCOMPUTE_MOTION: {COMPUTE_MOTION}")
    print(f"\tAPPLY_MOTION: {APPLY_MOTION}")

    if DEBUG:
        print(f"\nDEBUG ENABLED - Only running with {DEBUG_DURATION} seconds\n")

    si.set_global_job_kwargs(**job_kwargs)

    preprocessing_params["denoising_strategy"] = DENOISING_STRATEGY
    preprocessing_params["remove_out_channels"] = REMOVE_OUT_CHANNELS
    preprocessing_params["remove_bad_channels"] = REMOVE_BAD_CHANNELS
    preprocessing_params["max_bad_channel_fraction"] = MAX_BAD_CHANNEL_FRACTION
    preprocessing_params["motion_correction"]["compute"] = COMPUTE_MOTION
    preprocessing_params["motion_correction"]["apply"] = APPLY_MOTION

    # load job json files
    job_config_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    print(f"Found {len(job_config_json_files)} json configurations")

    if len(job_config_json_files) > 0:
        ####### PREPROCESSING #######
        print("\n\nPREPROCESSING")
        t_preprocessing_start_all = time.perf_counter()
        preprocessing_vizualization_data = {}

        for job_config_file in job_config_json_files:
            datetime_start_preproc = datetime.now()
            t_preprocessing_start = time.perf_counter()
            preprocessing_notes = ""

            with open(job_config_file, "r") as f:
                job_config = json.load(f)
            session_name = job_config["session_name"]
            session_folder_path = job_config["session_folder_path"]

            session = data_folder / session_folder_path
            assert session.is_dir(), (
                f"Could not find {session_name} in {str((data_folder / session_folder_path).resolve())}. "
                f"Make sure mapping is correct!"
            )

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

            skip_processing = False
            preprocessing_vizualization_data[recording_name] = {}
            preprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
            preprocessing_output_folder = results_folder / f"preprocessed_{recording_name}"
            preprocessingviz_output_file = results_folder / f"preprocessedviz_{recording_name}.json"
            preprocessing_output_json = results_folder / f"preprocessed_{recording_name}.json"

            exp_stream_name = f"{experiment_name}_{stream_name}"
            if not compressed:
                recording = se.read_openephys(ecephys_folder, stream_name=stream_name, block_index=block_index)
            else:
                recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")

            if DEBUG:
                recording_list = []
                for segment_index in range(recording.get_num_segments()):
                    recording_one = si.split_recording(recording)[segment_index]
                    recording_one = recording_one.frame_slice(
                        start_frame=0, end_frame=int(DEBUG_DURATION * recording.sampling_frequency)
                    )
                    recording_list.append(recording_one)
                recording = si.append_recordings(recording_list)

            if segment_index is not None:
                recording = si.split_recording(recording)[segment_index]

            print(f"Preprocessing recording: {recording_name}")
            print(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

            preprocessing_vizualization_data[recording_name]["timeseries"] = dict()
            preprocessing_vizualization_data[recording_name]["timeseries"]["full"] = dict(
                raw=recording.to_dict(relative_to=data_folder, recursive=True)
            )
            # maybe a recording is from a different source and it doesn't need to be phase shifted
            if "inter_sample_shift" in recording.get_property_keys():
                recording_ps_full = spre.phase_shift(recording, **preprocessing_params["phase_shift"])
                preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                    dict(phase_shift=recording_ps_full.to_dict(relative_to=data_folder, recursive=True))
                )
            else:
                recording_ps_full = recording

            recording_hp_full = spre.highpass_filter(recording_ps_full, **preprocessing_params["highpass_filter"])
            preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                dict(highpass=recording_hp_full.to_dict(relative_to=data_folder, recursive=True))
            )

            if recording.get_total_duration() < preprocessing_params["min_preprocessing_duration"] and not DEBUG:
                print(f"\tRecording is too short ({recording.get_total_duration()}s). Skipping further processing")
                preprocessing_notes += (
                    f"\n- Recording is too short ({recording.get_total_duration()}s). Skipping further processing\n"
                )
                skip_processing = True
            else:
                # IBL bad channel detection
                _, channel_labels = spre.detect_bad_channels(
                    recording_hp_full, **preprocessing_params["detect_bad_channels"]
                )
                dead_channel_mask = channel_labels == "dead"
                noise_channel_mask = channel_labels == "noise"
                out_channel_mask = channel_labels == "out"
                print(f"\tBad channel detection:")
                print(
                    f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}"
                )
                dead_channel_ids = recording_hp_full.channel_ids[dead_channel_mask]
                noise_channel_ids = recording_hp_full.channel_ids[noise_channel_mask]
                out_channel_ids = recording_hp_full.channel_ids[out_channel_mask]

                all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

                skip_processing = False
                max_bad_channel_fraction = preprocessing_params["max_bad_channel_fraction"]
                if len(all_bad_channel_ids) >= int(max_bad_channel_fraction * recording.get_num_channels()):
                    print(
                        f"\tMore than {max_bad_channel_fraction * 100}% bad channels ({len(all_bad_channel_ids)}). "
                        f"Skipping further processing for this recording."
                    )
                    preprocessing_notes += (
                        f"\n- Found {len(all_bad_channel_ids)} bad channels. Skipping further processing\n"
                    )
                    skip_processing = True
                else:
                    if preprocessing_params["remove_out_channels"]:
                        print(f"\tRemoving {len(out_channel_ids)} out channels")
                        recording_rm_out = recording_hp_full.remove_channels(out_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(out_channel_ids)} outside of the brain."
                    else:
                        recording_rm_out = recording_hp_full

                    recording_processed_cmr = spre.common_reference(
                        recording_rm_out, **preprocessing_params["common_reference"]
                    )

                    bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))
                    recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
                    recording_hp_spatial = spre.highpass_spatial_filter(
                        recording_interp, **preprocessing_params["highpass_spatial_filter"]
                    )
                    preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = dict(
                        highpass=recording_rm_out.to_dict(relative_to=data_folder, recursive=True),
                        cmr=recording_processed_cmr.to_dict(relative_to=data_folder, recursive=True),
                        highpass_spatial=recording_hp_spatial.to_dict(relative_to=data_folder, recursive=True),
                    )

                    denoising_strategy = preprocessing_params["denoising_strategy"]
                    if denoising_strategy == "cmr":
                        recording_processed = recording_processed_cmr
                    else:
                        recording_processed = recording_hp_spatial

                    if preprocessing_params["remove_bad_channels"]:
                        print(f"\tRemoving {len(bad_channel_ids)} channels after {denoising_strategy} preprocessing")
                        recording_processed = recording_processed.remove_channels(bad_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"

                    # motion correction
                    if preprocessing_params["motion_correction"]["compute"]:
                        preset = preprocessing_params["motion_correction"]["preset"]
                        print(f"\tComputing motion correction with preset: {preset}")
                        motion_folder = results_folder / f"motion_{recording_name}"
                        recording_corrected = spre.correct_motion(
                            recording_processed, preset=preset, folder=motion_folder
                        )
                        if preprocessing_params["motion_correction"]["apply"]:
                            print(f"\tApplying motion correction")
                            recording_processed = recording_corrected

                    recording_saved = recording_processed.save(folder=preprocessing_output_folder)
                    recording_processed.dump_to_json(preprocessing_output_json, relative_to=data_folder)
                    recording_drift = recording_saved

            if skip_processing:
                # in this case, processed timeseries will not be visualized
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
                recording_drift = recording_hp_full

            # store recording for drift visualization
            preprocessing_vizualization_data[recording_name]["drift"] = dict(
                recording=recording_drift.to_dict(relative_to=data_folder, recursive=True)
            )
            with open(preprocessingviz_output_file, "w") as f:
                json.dump(check_json(preprocessing_vizualization_data), f, indent=4)

            t_preprocessing_end = time.perf_counter()
            elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)

            # save params in output
            preprocessing_params["recording_name"] = recording_name
            preprocessing_outputs = dict(
                channel_labels=channel_labels.tolist(),
            )
            preprocessing_process = DataProcess(
                name="Ephys preprocessing",
                software_version=VERSION,  # either release or git commit
                start_date_time=datetime_start_preproc,
                end_date_time=datetime_start_preproc + timedelta(seconds=np.floor(elapsed_time_preprocessing)),
                input_location=str(data_folder),
                output_location=str(results_folder),
                code_url=URL,
                parameters=preprocessing_params,
                outputs=preprocessing_outputs,
                notes=preprocessing_notes,
            )
            with open(preprocessing_output_process_json, "w") as f:
                f.write(preprocessing_process.model_dump_json(indent=3))

        t_preprocessing_end_all = time.perf_counter()
        elapsed_time_preprocessing_all = np.round(t_preprocessing_end_all - t_preprocessing_start_all, 2)

        print(f"PREPROCESSING time: {elapsed_time_preprocessing_all}s")
