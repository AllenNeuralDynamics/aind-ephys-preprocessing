import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
import shutil
import numpy as np
from pathlib import Path
import json
import pickle
import time
import logging
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.preprocessing as spre

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema.core.processing import DataProcess

try:
    from aind_log_utils import log

    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing"
VERSION = "1.0"


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

motion_presets = spre.get_motion_presets()

# define argument parser
parser = argparse.ArgumentParser(description="Preprocess AIND Neurpixels data")

# positional arguments
denoising_group = parser.add_mutually_exclusive_group()
denoising_help = "Which denoising strategy to use. Can be 'cmr' or 'destripe'"
denoising_group.add_argument("--denoising", choices=["cmr", "destripe"], help=denoising_help)
denoising_group.add_argument("static_denoising", nargs="?", default="cmr", help=denoising_help)

filter_group = parser.add_mutually_exclusive_group()
filter_help = "Which filter to use. Can be 'highpass' or 'bandpass'"
filter_group.add_argument("--filter-type", choices=["highpass", "bandpass"], help=filter_help)
filter_group.add_argument("static_filter_type", nargs="?", default="highpass", help=filter_help)

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
    f"What motion preset to use. Supported presets are: {', '.join(motion_presets)}."
)
motion_preset_group.add_argument(
    "--motion-preset",
    choices=motion_presets,
    help=motion_preset_help,
)
motion_preset_group.add_argument("static_motion_preset", nargs="?", default=None, help=motion_preset_help)

motion_temporal_bin_s_group = parser.add_mutually_exclusive_group()
motion_temporal_bin_s_help = (
    ""
)
motion_temporal_bin_s_group.add_argument(
    "--motion-temporal-bin-s", default=1, help=motion_temporal_bin_s_help
)
motion_temporal_bin_s_group.add_argument(
    "static_motion_temporal_bin_s", nargs="?", default=None, help=motion_temporal_bin_s_help
)

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

min_duration_group = parser.add_mutually_exclusive_group()
min_duration_help = (
    "Minimum duration of a recording to be preprocessed."
)
min_duration_group.add_argument("static_min_duration_for_preprocessing", nargs="?", default=None, help=min_duration_help)
min_duration_group.add_argument("--min-duration-for-preprocessing", default=None, help=min_duration_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default=None, help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")



def dump_to_json_or_pickle(recording, results_folder, base_name, relative_to):
    if recording.check_serializability("json"):
        recording.dump_to_json(results_folder / f"{base_name}.json", relative_to=relative_to)
    else:
        recording.dump_to_pickle(results_folder / f"{base_name}.pkl", relative_to=relative_to)


if __name__ == "__main__":
    args = parser.parse_args()

    PARAMS = args.params
    if PARAMS is not None:
        try:
            # try to parse the JSON string first to avoid file name too long error
            preprocessing_params = json.loads(PARAMS)
        except json.JSONDecodeError:
            if Path(PARAMS).is_file():
                with open(PARAMS, "r") as f:
                    preprocessing_params = json.load(f)
            else:
                raise ValueError(f"Invalid parameters: {PARAMS} is not a valid JSON string or file path")

        DENOISING_STRATEGY = preprocessing_params.pop("denoising_strategy", "cmr")
        FILTER_TYPE = preprocessing_params.pop("filter_type", "highpass")
        REMOVE_OUT_CHANNELS = preprocessing_params.pop("remove_out_channels", False)
        REMOVE_BAD_CHANNELS = preprocessing_params.pop("remove_bad_channels", False)
        MAX_BAD_CHANNEL_FRACTION = preprocessing_params.pop("max_bad_channel_fraction", 0.5)
        MIN_DURATION_FOR_PREPROCESSING = preprocessing_params.pop("min_preprocessing_duration", 120)
        motion_params = preprocessing_params.get("motion_correction", None)
        MOTION_PRESET = motion_params.pop("preset", None)
        MOTION_TEMPORAL_BIN_S = motion_params.pop("temporal_bin_s", 1)
        COMPUTE_MOTION = motion_params.pop("compute", True)
        APPLY_MOTION = motion_params.pop("apply", False)
    else:
        with open("params.json", "r") as f:
            preprocessing_params = json.load(f)
        DENOISING_STRATEGY = args.denoising or args.static_denoising
        FILTER_TYPE = args.filter_type or args.static_filter_type
        REMOVE_OUT_CHANNELS = False if args.no_remove_out_channels else args.static_remove_out_channels == "true"
        REMOVE_BAD_CHANNELS = False if args.no_remove_bad_channels else args.static_remove_bad_channels == "true"
        MAX_BAD_CHANNEL_FRACTION = float(args.static_max_bad_channel_fraction or args.max_bad_channel_fraction)
        motion_arg = args.motion or args.static_motion
        MOTION_PRESET = args.static_motion_preset or args.motion_preset
        MOTION_TEMPORAL_BIN_S = float(args.static_motion_temporal_bin_s or args.motion_temporal_bin_s)
        COMPUTE_MOTION = True if motion_arg != "skip" else False
        APPLY_MOTION = True if motion_arg == "apply" else False
        MIN_DURATION_FOR_PREPROCESSING = args.static_min_duration_for_preprocessing or args.min_duration_for_preprocessing

    T_START = args.static_t_start or args.t_start
    if isinstance(T_START, str) and T_START == "":
        T_START = None
    T_STOP = args.static_t_stop or args.t_stop
    if isinstance(T_STOP, str) and T_STOP == "":
        T_STOP = None

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)

    # Use CO_CPUS/SLURM_CPUS_ON_NODE env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("SLURM_CPUS_ON_NODE")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else N_JOBS

    # setup AIND logging before any other logging call
    ecephys_session_folders = [
        p for p in data_folder.iterdir() if "ecephys" in p.name.lower() or "behavior" in p.name.lower()
    ]
    ecephys_session_folder = None
    aind_log_setup = False
    if len(ecephys_session_folders) == 1:
        ecephys_session_folder = ecephys_session_folders[0]
        if HAVE_AIND_LOG_UTILS:
            # look for subject.json and data_description.json files
            subject_json = ecephys_session_folder / "subject.json"
            subject_id = "undefined"
            if subject_json.is_file():
                subject_data = json.load(open(subject_json, "r"))
                subject_id = subject_data["subject_id"]

            data_description_json = ecephys_session_folder / "data_description.json"
            session_name = "undefined"
            if data_description_json.is_file():
                data_description = json.load(open(data_description_json, "r"))
                session_name = data_description["name"]

            log.setup_logging(
                "Preprocess Ecephys",
                subject_id=subject_id,
                asset_name=session_name,
            )
            aind_log_setup = True

    if not aind_log_setup:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info(f"Running preprocessing with the following parameters:")
    logging.info(f"\tDENOISING_STRATEGY: {DENOISING_STRATEGY}")
    logging.info(f"\tFILTER TYPE: {FILTER_TYPE}")
    logging.info(f"\tREMOVE_OUT_CHANNELS: {REMOVE_OUT_CHANNELS}")
    logging.info(f"\tREMOVE_BAD_CHANNELS: {REMOVE_BAD_CHANNELS}")
    logging.info(f"\tMAX BAD CHANNEL FRACTION: {MAX_BAD_CHANNEL_FRACTION}")
    logging.info(f"\tCOMPUTE_MOTION: {COMPUTE_MOTION}")
    logging.info(f"\tAPPLY_MOTION: {APPLY_MOTION}")
    logging.info(f"\tMOTION PRESET: {MOTION_PRESET}")
    logging.info(f"\tMOTION TEMPORAL BIN S: {MOTION_TEMPORAL_BIN_S}")
    logging.info(f"\tT_START: {T_START}")
    logging.info(f"\tT_STOP: {T_STOP}")
    logging.info(f"\tMIN_DURATION FOR PREPROCESSING: {MIN_DURATION_FOR_PREPROCESSING}")
    logging.info(f"\tN_JOBS: {N_JOBS}")

    data_process_prefix = "data_process_preprocessing"

    job_kwargs = preprocessing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    preprocessing_params["denoising_strategy"] = DENOISING_STRATEGY
    preprocessing_params["remove_out_channels"] = REMOVE_OUT_CHANNELS
    preprocessing_params["remove_bad_channels"] = REMOVE_BAD_CHANNELS
    preprocessing_params["max_bad_channel_fraction"] = MAX_BAD_CHANNEL_FRACTION
    motion_params = preprocessing_params["motion_correction"]
    motion_params["compute"] = COMPUTE_MOTION
    motion_params["apply"] = APPLY_MOTION
    if MOTION_PRESET is not None:
        motion_params["preset"] = MOTION_PRESET
    if MIN_DURATION_FOR_PREPROCESSING is None:
        MIN_DURATION_FOR_PREPROCESSING = preprocessing_params["min_preprocessing_duration"]
    MIN_DURATION_FOR_PREPROCESSING = float(MIN_DURATION_FOR_PREPROCESSING)

    # load job files
    job_config_files = [p for p in data_folder.iterdir() if (p.suffix == ".json" or p.suffix == ".pickle" or p.suffix == ".pkl") and "job" in p.name]
    logging.info(f"Found {len(job_config_files)} configurations")

    if len(job_config_files) > 0:
        ####### PREPROCESSING #######
        logging.info("\n\nPREPROCESSING")
        t_preprocessing_start_all = time.perf_counter()
        preprocessing_visualization_data = {}

        for job_config_file in job_config_files:
            datetime_start_preproc = datetime.now()
            t_preprocessing_start = time.perf_counter()
            preprocessing_notes = ""

            if job_config_file.suffix == ".json":
                with open(job_config_file, "r") as f:
                    job_config = json.load(f)
            else:
                with open(job_config_file, "rb") as f:
                    job_config = pickle.load(f)

            session_name = job_config["session_name"]
            recording_name = job_config["recording_name"]
            recording_dict = job_config["recording_dict"]
            skip_times = job_config.get("skip_times", False)
            debug = job_config.get("debug", False)

            try:
                recording = si.load(recording_dict, base_folder=data_folder)
            except:
                raise RuntimeError(
                    f"Could not find load recording {recording_name} from dict. "
                    f"Make sure mapping is correct!"
                )
            if skip_times:
                logging.info("Resetting recording timestamps")
                recording.reset_times()

            skip_processing = False
            visualization_file_is_json_serializable = True

            preprocessing_visualization_data[recording_name] = {}
            preprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
            preprocessing_output_folder = results_folder / f"preprocessed_{recording_name}"
            preprocessingviz_output_filename = f"preprocessedviz_{recording_name}"
            preprocessing_output_filename = f"preprocessed_{recording_name}"
            motioncorrected_output_filename = f"motioncorrected_{recording_name}"
            binary_output_filename = f"binary_{recording_name}"

            logging.info(f"Preprocessing recording: {session_name} - {recording_name}")

            if (T_START is not None or T_STOP is not None):
                if recording.get_num_segments() > 1:
                    logging.info(f"\tRecording has multiple segments. Ignoring T_START and T_STOP")
                else:
                    if T_START is None:
                        T_START = 0
                    if T_STOP is None:
                        T_STOP = recording.get_duration()
                    T_START = float(T_START)
                    T_STOP = float(T_STOP)
                    T_STOP = min(T_STOP, recording.get_duration())
                    logging.info(f"\tOriginal recording duration: {recording.get_duration()} -- Clipping to {T_START}-{T_STOP} s")
                    start_frame = int(T_START * recording.get_sampling_frequency())
                    end_frame = int(T_STOP * recording.get_sampling_frequency() + 1)
                    recording = recording.frame_slice(start_frame=start_frame, end_frame=end_frame)

            logging.info(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

            preprocessing_visualization_data[recording_name]["timeseries"] = dict()
            preprocessing_visualization_data[recording_name]["timeseries"]["full"] = dict(
                raw=recording.to_dict(relative_to=data_folder, recursive=True)
            )
            if not recording.check_serializability("json"):
                visualization_file_is_json_serializable = False
            # maybe a recording is from a different source and it doesn't need to be phase shifted
            if "inter_sample_shift" in recording.get_property_keys():
                recording_ps_full = spre.phase_shift(recording, **preprocessing_params["phase_shift"])
                preprocessing_visualization_data[recording_name]["timeseries"]["full"].update(
                    dict(phase_shift=recording_ps_full.to_dict(relative_to=data_folder, recursive=True))
                )
            else:
                recording_ps_full = recording

            if FILTER_TYPE == "highpass":
                recording_filt_full = spre.highpass_filter(recording_ps_full, **preprocessing_params["highpass_filter"])
                preprocessing_visualization_data[recording_name]["timeseries"]["full"].update(
                    dict(highpass=recording_filt_full.to_dict(relative_to=data_folder, recursive=True))
                )
                preprocessing_params["filter_type"] = "highpass"
            elif FILTER_TYPE == "bandpass":
                recording_filt_full = spre.bandpass_filter(recording_ps_full, **preprocessing_params["bandpass_filter"])
                preprocessing_visualization_data[recording_name]["timeseries"]["full"].update(
                    dict(bandpass=recording_filt_full.to_dict(relative_to=data_folder, recursive=True))
                )
                preprocessing_params["filter_type"] = "bandpass"
            else:
                raise ValueError(f"Filter type {FILTER_TYPE} not recognized")

            if recording.get_total_duration() < MIN_DURATION_FOR_PREPROCESSING and not debug:
                logging.info(f"\tRecording is too short ({recording.get_total_duration()}s). Skipping further processing")
                preprocessing_notes += (
                    f"\n- Recording is too short ({recording.get_total_duration()}s). Skipping further processing\n"
                )
                channel_labels = None
                skip_processing = True
            else:
                # IBL bad channel detection
                _, channel_labels = spre.detect_bad_channels(
                    recording_filt_full, **preprocessing_params["detect_bad_channels"]
                )
                dead_channel_mask = channel_labels == "dead"
                noise_channel_mask = channel_labels == "noise"
                out_channel_mask = channel_labels == "out"
                logging.info(f"\tBad channel detection:")
                logging.info(
                    f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}"
                )
                dead_channel_ids = recording_filt_full.channel_ids[dead_channel_mask]
                noise_channel_ids = recording_filt_full.channel_ids[noise_channel_mask]
                out_channel_ids = recording_filt_full.channel_ids[out_channel_mask]

                all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

                skip_processing = False
                max_bad_channel_fraction = preprocessing_params["max_bad_channel_fraction"]
                if len(all_bad_channel_ids) >= int(max_bad_channel_fraction * recording.get_num_channels()):
                    logging.info(f"\tMore than {max_bad_channel_fraction * 100}% bad channels ({len(all_bad_channel_ids)}). ")
                    preprocessing_notes += f"\n- Found {len(all_bad_channel_ids)} bad channels."
                    if preprocessing_params["remove_bad_channels"]:
                        skip_processing = True
                        logging.info("\tSkipping further processing for this recording.")
                        preprocessing_notes += f" Skipping further processing for this recording.\n"
                    else:
                        preprocessing_notes += "\n"

                if not skip_processing:
                    if preprocessing_params["remove_out_channels"]:
                        logging.info(f"\tRemoving {len(out_channel_ids)} out channels")
                        recording_rm_out = recording_filt_full.remove_channels(out_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(out_channel_ids)} outside of the brain."
                    else:
                        recording_rm_out = recording_filt_full

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
                    preprocessing_visualization_data[recording_name]["timeseries"]["proc"] = dict(
                        highpass=recording_rm_out.to_dict(relative_to=data_folder, recursive=True),
                        cmr=recording_processed_cmr.to_dict(relative_to=data_folder, recursive=True),
                    )
                    if recording_hp_spatial is not None:
                        preprocessing_visualization_data[recording_name]["timeseries"]["proc"].update(
                            dict(highpass_spatial=recording_hp_spatial.to_dict(relative_to=data_folder, recursive=True))
                        )

                    denoising_strategy = preprocessing_params["denoising_strategy"]
                    if denoising_strategy == "cmr":
                        recording_processed = recording_processed_cmr
                    else:
                        recording_processed = recording_hp_spatial

                    if preprocessing_params["remove_bad_channels"]:
                        logging.info(f"\tRemoving {len(bad_channel_ids)} channels after {denoising_strategy} preprocessing")
                        recording_processed = recording_processed.remove_channels(bad_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"

                    # save to binary to speed up downstream processing
                    recording_bin = recording_processed.save(folder=preprocessing_output_folder)

                    # motion correction
                    recording_corrected = None
                    recording_bin_corrected = None
                    if motion_params["compute"]:
                        from spikeinterface.sortingcomponents.motion import interpolate_motion

                        preset = motion_params["preset"]
                        logging.info(f"\tComputing motion correction with preset: {preset}")

                        detect_kwargs = motion_params.get("detect_kwargs", {})
                        select_kwargs = motion_params.get("select_kwargs", {})
                        localize_peaks_kwargs = motion_params.get("localize_peaks_kwargs", {})
                        estimate_motion_kwargs = motion_params.get("estimate_motion_kwargs", {})

                        estimate_motion_kwargs["bin_s"] = MOTION_TEMPORAL_BIN_S
                        logging.info(f"\t\tUsing bin_s: {MOTION_TEMPORAL_BIN_S}")

                        # the win_step_norm/win_scale_norm define the win_step_um/win_scale_um based on the probe_span
                        probe_span = np.ptp(recording.get_channel_locations()[:, 1])
                        if "win_step_norm" in estimate_motion_kwargs:
                            win_step_norm = estimate_motion_kwargs.pop("win_step_norm")
                        else:
                            win_step_norm = None
                        if "win_scale_norm" in estimate_motion_kwargs:
                            win_scale_norm = estimate_motion_kwargs.pop("win_scale_norm")
                        else:
                            win_scale_norm = None
                        if win_step_norm is not None:
                            win_step_um = win_step_norm * probe_span
                            estimate_motion_kwargs["win_step_um"] = win_step_um
                            logging.info(f"\t\tUsing win_step_um: {win_step_um}")
                        if win_scale_norm is not None:
                            win_scale_um = win_scale_norm * probe_span
                            estimate_motion_kwargs["win_scale_um"] = win_scale_um
                            logging.info(f"\t\tUsing win_scale_um: {win_scale_um}")

                        motion_folder = results_folder / f"motion_{recording_name}"
                        interpolate_motion_kwargs = motion_params.get("interpolate_motion_kwargs", {})

                        concat_motion = False
                        recording_corrected = None
                        if recording_processed.get_num_segments() > 1:
                            recording_bin_c = si.concatenate_recordings([recording_bin])
                            recording_processed_c = si.concatenate_recordings([recording_processed])
                            concat_motion = True
                        else:
                            recording_bin_c = recording_bin
                            recording_processed_c = recording_processed

                        # use compute motion
                        motion = spre.compute_motion(
                            recording_bin_c,
                            preset=preset,
                            folder=motion_folder,
                            detect_kwargs=detect_kwargs,
                            select_kwargs=select_kwargs,
                            localize_peaks_kwargs=localize_peaks_kwargs,
                            estimate_motion_kwargs=estimate_motion_kwargs,
                            raise_error=False
                        )
                        if motion is not None:
                            logging.info(f"\tMotion computed successfully!")
                            if motion_params["apply"]:
                                logging.info(f"\tApplying motion correction")
                                recording_bin_corrected = interpolate_motion(
                                    recording_bin_c.astype("float32"),
                                    motion=motion,
                                    **interpolate_motion_kwargs
                                )
                                recording_corrected = interpolate_motion(
                                    recording_processed_c.astype("float32"),
                                    motion=motion,
                                    **interpolate_motion_kwargs
                                )

                                # split segments back
                                if concat_motion:
                                    rec_corrected_list = []
                                    rec_corrected_bin_list = []
                                    for segment_index in range(recording_bin.get_num_segments()):
                                        num_samples = recording_bin.get_num_samples(segment_index)
                                        if segment_index == 0:
                                            start_frame = 0
                                        else:
                                            start_frame = recording_bin.get_num_samples(segment_index - 1)
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
                                logging.info(f"\tApplying motion correction")
                                recording_processed = recording_corrected
                                recording_bin = recording_bin_corrected
                        else:
                            logging.info(f"\tMotion computation failed. Skipping motion correction")
                            preprocessing_notes += "\n- Motion computation failed. Skipping motion correction.\n"

                    # this is used to reload the binary traces downstream
                    dump_to_json_or_pickle(
                        recording_bin,
                        results_folder,
                        binary_output_filename,
                        relative_to=results_folder
                    )

                    # this is to reload the recordings lazily            
                    dump_to_json_or_pickle(
                        recording_processed,
                        results_folder,
                        preprocessing_output_filename,
                        relative_to=results_folder
                    )

                    # this is to reload the motion-corrected recording lazily
                    if recording_corrected is not None:     
                        dump_to_json_or_pickle(
                            recording_corrected,
                            results_folder,
                            motioncorrected_output_filename,
                            relative_to=results_folder
                        )

                    recording_drift = recording_bin
                    drift_relative_folder = results_folder

            if skip_processing:
                # in this case, processed timeseries will not be visualized
                preprocessing_visualization_data[recording_name]["timeseries"]["proc"] = None
                recording_drift = recording_filt_full
                drift_relative_folder = data_folder
                # make a dummy file if too many bad channels to skip downstream processing
                preprocessing_output_folder.mkdir()
                error_file = preprocessing_output_folder / "error.txt"
                error_file.write_text("Too many bad channels")

            # store recording for drift visualization
            preprocessing_visualization_data[recording_name]["drift"] = dict(
                recording=recording_drift.to_dict(relative_to=drift_relative_folder, recursive=True)
            )

            if visualization_file_is_json_serializable:            
                with open(results_folder / f"{preprocessingviz_output_filename}.json", "w") as f:
                    json.dump(check_json(preprocessing_visualization_data), f, indent=4)
            else:
                # then dump to pickle
                with open(results_folder / f"{preprocessingviz_output_filename}.pkl", "wb") as f:
                    pickle.dump(preprocessing_visualization_data, f)

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

            # copy data_description and subject json
            if ecephys_session_folder is not None:
                metadata_json_files = [p for p in ecephys_session_folder.iterdir() if p.suffix == ".json"]
                for metadata_file in metadata_json_files:
                    if "data_description" in metadata_file.name or "subject" in metadata_file.name:
                        shutil.copy(metadata_file, results_folder / f"preprocessing_{recording_name}_{metadata_file.name}")

        t_preprocessing_end_all = time.perf_counter()
        elapsed_time_preprocessing_all = np.round(t_preprocessing_end_all - t_preprocessing_start_all, 2)

        logging.info(f"PREPROCESSING time: {elapsed_time_preprocessing_all}s")
