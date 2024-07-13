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
VERSION = "1.0"


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")


# define argument parser
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
    "static_max_bad_channel_fraction", nargs="?", default=None, help=max_bad_channel_fraction_help
)

motion_correction_group = parser.add_mutually_exclusive_group()
motion_correction_help = "How to deal with motion correction. Can be 'skip', 'compute', or 'apply'"
motion_correction_group.add_argument("--motion", choices=["skip", "compute", "apply"], help=motion_correction_help)
motion_correction_group.add_argument("static_motion", nargs="?", default="compute", help=motion_correction_help)

motion_preset_group = parser.add_mutually_exclusive_group()
motion_preset_help = (
    "What motion preset to use. Can be 'nonrigid_accurate', 'kilosort_like', or 'nonrigid_fast_and_accurate'"
)
motion_preset_group.add_argument(
    "--motion-preset",
    choices=["nonrigid_accurate", "kilosort_like", "nonrigid_fast_and_accurate"],
    help=motion_preset_help,
)
motion_preset_group.add_argument("static_motion_preset", nargs="?", default=None, help=motion_preset_help)

t_start_group = parser.add_mutually_exclusive_group()
t_start_help = (
    "Start time of the recording in seconds (assumes recording starts at 0). "
    "This parameter is ignored in case of multi-segment or multi-block recordings."
    "Default is None (start of recording)"
)
t_start_group.add_argument("static_t_start", nargs="?", default=None, help=t_start_help)
t_start_group.add_argument("--t-start", default=None, help=t_start_help)

t_stop_group = parser.add_mutually_exclusive_group()
t_stop_help = (
    "Stop time of the recording in seconds (assumes recording starts at 0). "
    "This parameter is ignored in case of multi-segment or multi-block recordings."
    "Default is None (end of recording)"
)
t_stop_group.add_argument("static_t_stop", nargs="?", default=None, help=t_stop_help)
t_stop_group.add_argument("--t-stop", default=None, help=t_stop_help)

debug_duration_group = parser.add_mutually_exclusive_group()
debug_duration_help = (
    "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
)
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default=None, help=debug_duration_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default=None, help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")


if __name__ == "__main__":
    args = parser.parse_args()

    DEBUG = args.debug or args.static_debug == "true"
    DENOISING_STRATEGY = args.denoising or args.static_denoising
    REMOVE_OUT_CHANNELS = False if args.no_remove_out_channels else args.static_remove_out_channels == "true"
    REMOVE_BAD_CHANNELS = False if args.no_remove_bad_channels else args.static_remove_bad_channels == "true"
    MAX_BAD_CHANNEL_FRACTION = float(args.static_max_bad_channel_fraction or args.max_bad_channel_fraction)
    motion_arg = args.motion or args.static_motion
    MOTION_PRESET = args.static_motion_preset or args.motion_preset
    COMPUTE_MOTION = True if motion_arg != "skip" else False
    APPLY_MOTION = True if motion_arg == "apply" else False
    T_START = args.static_t_start or args.t_start
    if isinstance(T_START, str) and T_START == "":
        T_START = None
    T_STOP = args.static_t_stop or args.t_stop
    if isinstance(T_STOP, str) and T_STOP == "":
        T_STOP = None
    DEBUG_DURATION = float(args.static_debug_duration or args.debug_duration)

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    N_JOBS = int(N_JOBS_CO) if N_JOBS_CO is not None else N_JOBS

    print(f"Running preprocessing with the following parameters:")
    print(f"\tDENOISING_STRATEGY: {DENOISING_STRATEGY}")
    print(f"\tREMOVE_OUT_CHANNELS: {REMOVE_OUT_CHANNELS}")
    print(f"\tREMOVE_BAD_CHANNELS: {REMOVE_BAD_CHANNELS}")
    print(f"\tMAX BAD CHANNEL FRACTION: {MAX_BAD_CHANNEL_FRACTION}")
    print(f"\tCOMPUTE_MOTION: {COMPUTE_MOTION}")
    print(f"\tAPPLY_MOTION: {APPLY_MOTION}")
    print(f"\tMOTION PRESET: {MOTION_PRESET}")
    print(f"\tT_START: {T_START}")
    print(f"\tT_STOP: {T_STOP}")
    print(f"\tN_JOBS: {N_JOBS}")

    if DEBUG:
        print(f"\nDEBUG ENABLED - Only running with {DEBUG_DURATION} seconds\n")

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_preprocessing"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    preprocessing_params = processing_params["preprocessing"]
    preprocessing_params["denoising_strategy"] = DENOISING_STRATEGY
    preprocessing_params["remove_out_channels"] = REMOVE_OUT_CHANNELS
    preprocessing_params["remove_bad_channels"] = REMOVE_BAD_CHANNELS
    preprocessing_params["max_bad_channel_fraction"] = MAX_BAD_CHANNEL_FRACTION
    motion_params = processing_params["motion_correction"]
    motion_params["compute"] = COMPUTE_MOTION
    motion_params["apply"] = APPLY_MOTION
    if MOTION_PRESET is not None:
        motion_params["preset"] = MOTION_PRESET

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
            recording_name = job_config["recording_name"]
            recording_dict = job_config["recording_dict"]

            try:
                recording = si.load_extractor(recording_dict, base_folder=data_folder)
            except:
                raise RuntimeError(
                    f"Could not find load recording {recording_name} from dict. "
                    f"Make sure mapping is correct!"
                )

            skip_processing = False
            preprocessing_vizualization_data[recording_name] = {}
            preprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
            preprocessing_output_folder = results_folder / f"preprocessed_{recording_name}"
            preprocessingviz_output_file = results_folder / f"preprocessedviz_{recording_name}.json"
            preprocessing_output_json = results_folder / f"preprocessed_{recording_name}.json"
            motioncorrected_output_json = results_folder / f"motioncorrected_{recording_name}.json"
            binary_output_json = results_folder / f"binary_{recording_name}.json"


            if DEBUG:
                recording_list = []
                for segment_index in range(recording.get_num_segments()):
                    recording_one = si.split_recording(recording)[segment_index]
                    recording_one = recording_one.frame_slice(
                        start_frame=0, end_frame=int(DEBUG_DURATION * recording.sampling_frequency)
                    )
                    recording_list.append(recording_one)
                recording = si.append_recordings(recording_list)

            print(f"Preprocessing recording: {session_name} - {recording_name}")

            if (T_START is not None or T_STOP is not None) and not DEBUG:
                if recording.get_num_segments() > 1:
                    print(f"\tRecording has multiple segments. Ignoring T_START and T_STOP")
                else:
                    if T_START is None:
                        T_START = 0
                    if T_STOP is None:
                        T_STOP = recording.get_duration()
                    T_START = float(T_START)
                    T_STOP = float(T_STOP)
                    T_STOP = min(T_STOP, recording.get_duration())
                    print(f"\tOriginal recording duration: {recording.get_duration()} -- Clipping to {T_START}-{T_STOP} s")
                    start_frame = int(T_START * recording.get_sampling_frequency())
                    end_frame = int(T_STOP * recording.get_sampling_frequency() + 1)
                    recording = recording.frame_slice(start_frame=start_frame, end_frame=end_frame)

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
                channel_labels = None
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
                    print(f"\tMore than {max_bad_channel_fraction * 100}% bad channels ({len(all_bad_channel_ids)}). ")
                    preprocessing_notes += f"\n- Found {len(all_bad_channel_ids)} bad channels."
                    if preprocessing_params["remove_bad_channels"]:
                        skip_processing = True
                        print("\tSkipping further processing for this recording.")
                        preprocessing_notes += f" Skipping further processing for this recording.\n"
                    else:
                        preprocessing_notes += "\n"

                if not skip_processing:
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
                    # protection against short probes
                    try:
                        recording_hp_spatial = spre.highpass_spatial_filter(
                            recording_interp, **preprocessing_params["highpass_spatial_filter"]
                        )
                    except:
                        recording_hp_spatial = None
                    preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = dict(
                        highpass=recording_rm_out.to_dict(relative_to=data_folder, recursive=True),
                        cmr=recording_processed_cmr.to_dict(relative_to=data_folder, recursive=True),
                    )
                    if recording_hp_spatial is not None:
                        preprocessing_vizualization_data[recording_name]["timeseries"]["proc"].update(
                            dict(highpass_spatial=recording_hp_spatial.to_dict(relative_to=data_folder, recursive=True))
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

                    # save to binary to speed up downstream processing
                    recording_bin = recording_processed.save(folder=preprocessing_output_folder)

                    # motion correction
                    recording_corrected = None
                    recording_bin_corrected = None
                    if motion_params["compute"]:
                        from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion

                        preset = motion_params["preset"]
                        print(f"\tComputing motion correction with preset: {preset}")

                        detect_kwargs = motion_params.get("detect_kwargs", {})
                        select_kwargs = motion_params.get("select_kwargs", {})
                        localize_peaks_kwargs = motion_params.get("localize_peaks_kwargs", {})
                        estimate_motion_kwargs = motion_params.get("estimate_motion_kwargs", {})
                        interpolate_motion_kwargs = motion_params.get("interpolate_motion_kwargs", {})

                        motion_folder = results_folder / f"motion_{recording_name}"

                        concat_motion = False
                        if recording_processed.ger_num_segments() > 1:
                            recording_bin_c = si.concatenate_recordings(recording_bin)
                            recording_processed_c = si.concatenate_recordings(recording_processed)
                            concat_motion = True
                        else:
                            recording_bin_c = recording_bin
                            recording_processed_c = recording_processed

                        recording_bin_corrected, motion_info = spre.correct_motion(
                            recording_bin_c,
                            preset=preset,
                            folder=motion_folder,
                            output_motion_info=True,
                            detect_kwargs=detect_kwargs,
                            select_kwargs=select_kwargs,
                            localize_peaks_kwargs=localize_peaks_kwargs,
                            estimate_motion_kwargs=estimate_motion_kwargs,
                            interpolate_motion_kwargs=interpolate_motion_kwargs
                        )
                        recording_corrected = interpolate_motion(
                            recording_processed_c,
                            motion=motion_info["motion"],
                            temporal_bins=motion_info["temporal_bins"],
                            spatial_bins=motion_info["spatial_bins"],
                            **interpolate_motion_kwargs
                        )

                        # split segments back
                        if concat_motion:
                            rec_corrected_list = []
                            rec_corrected_bin_list = []
                            for seg in recording_bin_corrected.get_num_segments():
                                num_samples = recording_bin.get_num_samples(seg)
                                if seg == 0:
                                    start_frame = 0
                                else:
                                    start_frame = recording_bin.get_num_samples(seg - 1)
                                end_frame = start_frame + num_samples
                                rec_split_corrected = recording_corrected.frame_slice(
                                    start_frame=start_frame,
                                    end_frame=end_frame
                                )
                                rec_corrected_list.append(rec_split_corrected)
                                rec_split_bin = recording_bin_corrected.frame_slice(
                                    start_frame=start_frame,
                                    end_frame=end_frame
                                )
                                rec_corrected_bin_list.append(rec_split_bin)
                            # append all segments
                            recording_corrected = si.append_recordings(rec_corrected_list)
                            recording_bin_corrected = si.append_recordings(rec_corrected_bin_list)

                        if motion_params["apply"]:
                            print(f"\tApplying motion correction")
                            recording_processed = recording_corrected
                            recording_bin = recording_bin_corrected

                    # this is used to reload the binary traces downstream
                    recording_bin.dump_to_json(binary_output_json, relative_to=results_folder)

                    # this is to reload the recordings lazily
                    recording_processed.dump_to_json(preprocessing_output_json, relative_to=data_folder)
                    if recording_corrected is not None:
                        recording_corrected.dump_to_json(motioncorrected_output_json, relative_to=data_folder)
                    recording_drift = recording_bin
                    drift_relative_folder = results_folder

            if skip_processing:
                # in this case, processed timeseries will not be visualized
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
                recording_drift = recording_hp_full
                drift_relative_folder = data_folder
                # make a dummy file if too many bad channels to skip downstream processing
                preprocessing_output_folder.mkdir()
                error_file = preprocessing_output_folder / "error.txt"
                error_file.write_text("Too many bad channels")

            # store recording for drift visualization
            preprocessing_vizualization_data[recording_name]["drift"] = dict(
                recording=recording_drift.to_dict(relative_to=drift_relative_folder, recursive=True)
            )
            with open(preprocessingviz_output_file, "w") as f:
                json.dump(check_json(preprocessing_vizualization_data), f, indent=4)

            t_preprocessing_end = time.perf_counter()
            elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)

            # save params in output
            preprocessing_params["recording_name"] = recording_name
            if channel_labels is not None:
                preprocessing_outputs = dict(
                    channel_labels=channel_labels.tolist(),
                )
            else:
                preprocessing_outputs = dict()
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
