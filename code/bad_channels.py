"""Bad-channel detection with the PSD noise threshold scaled to each probe."""

import numpy as np
import spikeinterface.preprocessing as spre
from scipy.signal import welch
from spikeinterface.core.time_series_tools import get_random_sample_slices


def probe_psd_hf(recording, num_random_chunks=100, chunk_duration_s=0.3, nyquist_threshold=0.8,
                 welch_window_ms=10.0, seed=None):
    """Per-channel median over random chunks of the high-frequency PSD that `detect_bad_channels` thresholds.

    Uses the same chunks, window and band as SpikeInterface's coherence+psd method, in uV^2/Hz.
    """
    fs = recording.sampling_frequency
    slices = get_random_sample_slices(
        recording, num_chunks_per_segment=num_random_chunks, chunk_size=int(chunk_duration_s * fs), seed=seed
    )
    psd_hf = []
    for segment_index, start_frame, end_frame in slices:
        traces = recording.get_traces(
            segment_index=segment_index, start_frame=start_frame, end_frame=end_frame, return_in_uV=True
        ).astype("float64")
        traces -= traces.mean(axis=0, keepdims=True)
        freqs, psd = welch(traces, fs=fs, axis=0, window="hann", nperseg=int(welch_window_ms * fs / 1000))
        psd_hf.append(psd[freqs > fs / 2 * nyquist_threshold].mean(axis=0))
    return np.median(np.stack(psd_hf), axis=0)


def detect_bad_channels_relative(recording, highpass_freq_min, psd_hf_relative_threshold=None, **detect_kwargs):
    """Label channels on a highpass view of `recording`, optionally with a PSD threshold relative to the probe.

    Detection always sees the highpass view: a lowpass empties the band above 0.8 x Nyquist that the PSD
    test measures. NP2.0 probes differ by ~2x in that band's broadband floor, and a typical floor sits at
    SpikeInterface's absolute 0.02 uV^2/Hz, so `psd_hf_relative_threshold` replaces it with that multiple
    of the probe's median channel.

    Returns
    -------
    bad_channel_ids, channel_labels, psd_hf_threshold
        `psd_hf_threshold` is the absolute threshold applied, in uV^2/Hz, or None for SpikeInterface's default.
    """
    recording_hp = spre.highpass_filter(recording, freq_min=highpass_freq_min)
    if psd_hf_relative_threshold is not None:
        psd_hf = probe_psd_hf(
            recording_hp,
            **{
                k: detect_kwargs[k]
                for k in ("num_random_chunks", "chunk_duration_s", "nyquist_threshold", "welch_window_ms", "seed")
                if k in detect_kwargs
            },
        )
        detect_kwargs["psd_hf_threshold"] = float(psd_hf_relative_threshold * np.median(psd_hf))
    bad_channel_ids, channel_labels = spre.detect_bad_channels(recording_hp, **detect_kwargs)
    return bad_channel_ids, channel_labels, detect_kwargs.get("psd_hf_threshold")
