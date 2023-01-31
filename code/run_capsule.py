import warnings
warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
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
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
from wavpack_numcodecs import WavPack


# AIND
from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess


URL = "https://github.com/AllenNeuralDynamics/aind-capsule-ephys-preprocessing"
VERSION = "1.0.0.dev"


preprocessing_params = dict(
        highpass_filter=dict(freq_min=300.0,
                             margin_ms=5.0),
        phase_shift=dict(margin_ms=500.),
        common_reference=dict(reference='global',
                              operator='median'),
    )

job_kwargs = {
    'n_jobs': os.cpu_count(),
    'chunk_duration': '1s',
    'progress_bar': True
}

data_folder = Path("../data/")
results_folder = Path("../results/")


def run(*args):
    """ basic run function """
    DEBUG = False
    DURATION_S = None
    CONCAT = False

    if len(args) == 3:
        if args[0] == "true":
            DEBUG = True
            DURATION_S = float(args[1])
        CONCAT = args[2]

    if DEBUG:
        print(f"DEBUG ENABLED - Only running with {DURATION_S} seconds")

    datetime_start_preproc = datetime.now()
    t_preprocessing_start = time.perf_counter()
    preprocessed_output_folder = results_folder

    # load data assets: 2 options
    # 1. A data asset is loaded "manually" in the capsule capsule
    #    In this case, we expect a tree as follows "ecephys_{session}/ecephys"
    # 2. The data folder is mapped in a pipeline.
    #    In this case we have the "ecephys" folder already in the data folder

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session = ecephys_sessions[0]
    session_name = session.name

    print(f"Preprocessing session: {session_name}")
    ecephys_full_folder = session / "ecephys"
    ecephys_compressed_folder = session / "ecephys_compressed"
    compressed = False
    if ecephys_compressed_folder.is_dir():
        compressed = True
        ecephys_folder = session / "ecephys_clipped"
    else:
        ecephys_folder = ecephys_full_folder

    # if (data_folder / "ecephys").is_dir():
    #     oe_folder = data_folder / "ecephys"
    #     session_name = "From pipeline"
    # else:
    #     ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    #     assert len(ecephys_sessions) == 1, "Attach one session at a time"
    #     session = ecephys_sessions[0]
    #     session_name = session.name

    # get blocks/experiments and streams info
    num_blocks = se.get_neo_num_blocks("openephys", ecephys_folder)
    stream_names, stream_ids = se.get_neo_streams("openephys", ecephys_folder)

    # load first stream to map block_indices to experiment_names
    rec_test = se.read_openephys(ecephys_folder, block_index=0, stream_name=stream_names[0])
    record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
    experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
    exp_ids = list(experiments.keys())
    experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

    print(f"Session: {session_name} - Num. Blocks {num_blocks} - Num. streams: {len(stream_names)}")

    recording_names = []
    for block_index in range(num_blocks):
        for stream_name in stream_names:
            # skip NIDAQ and NP1-LFP streams
            if "NI-DAQ" not in stream_name and "LFP" not in stream_name:
                experiment_name = experiment_names[block_index]
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

                if CONCAT:
                    recordings = [recording]
                else:
                    recordings = si.split_recording(recording)

                print(recordings)

                for i_r, recording in enumerate(recordings):
                    if CONCAT:
                        recording_name = f"{exp_stream_name}_recording"
                    else:
                        recording_name = f"{exp_stream_name}_recording{i_r + 1}"

                    recording_names.append(recording_name)
                    print(f"Preprocessing recording: {recording_name}")

                    recording_ps = spre.phase_shift(recording, **preprocessing_params["phase_shift"])

                    recording_hp = spre.highpass_filter(recording_ps, **preprocessing_params["highpass_filter"])

                    recording_cmr = spre.common_reference(recording_hp, **preprocessing_params["common_reference"])

                    # cast to int16
                    recording_cmr = spre.scale(recording_cmr, dtype="int16")

                    recording_saved = recording_cmr.save(folder=preprocessed_output_folder / recording_name)


    t_preprocessing_end = time.perf_counter()
    elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)
    print(f"Preprocessing+recording vizualization took {elapsed_time_preprocessing}s")

    # save params in output
    preprocessing_process = DataProcess(
            name="Ephys preprocessing",
            version=VERSION, # either release or git commit
            start_date_time=datetime_start_preproc,
            end_date_time=datetime_start_preproc + timedelta(seconds=np.floor(elapsed_time_preprocessing)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=preprocessing_params,
        )

    # save params in output
    with (results_folder / "preprocessing_params.json").open("w") as f:
        json.dump(preprocessing_params, f, indent=4)


if __name__ == "__main__": 
    run(*sys.argv[1:])


