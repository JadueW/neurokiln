from __future__ import annotations

import numpy as np

from .model import Epochs


def make_synthetic_epochs(
    n_trials_per_class: int = 40,
    n_classes: int = 2,
    n_channels: int = 8,
    n_times: int = 64,
    sfreq: float = 128.0,
    subjects: tuple[str, ...] = ("S01",),
    sessions: tuple[str, ...] = ("0",),
    seed: int = 0,
    noise: float = 1.0,
    class_sep: float = 0.6,
    class_noise_step: float = 0.0,
) -> Epochs:
    """Class-dependent channel means plus noise; enough to exercise protocols.

    class_noise_step makes trial noise scale class-dependent too, so variance
    features and mean features carry different amounts of signal.
    """
    rng = np.random.default_rng(seed)
    n_trials = n_trials_per_class * n_classes
    labels = np.repeat(np.arange(n_classes), n_trials_per_class)
    means = rng.normal(0, class_sep, size=(n_classes, n_channels))
    data = np.empty((n_trials, n_channels, n_times), dtype=np.float64)
    for i, label in enumerate(labels):
        scale = noise * (1.0 + class_noise_step * float(label))
        data[i] = means[label][:, None] + rng.normal(0, scale, size=(n_channels, n_times))
    subject_ids = np.array([subjects[i % len(subjects)] for i in range(n_trials)], dtype=str)
    session_ids = np.array([sessions[(i // n_trials_per_class) % len(sessions)] for i in range(n_trials)], dtype=str)
    order = rng.permutation(n_trials)
    return Epochs(
        data=data[order],
        labels=labels[order],
        sfreq=sfreq,
        channel_names=tuple(f"C{i + 1}" for i in range(n_channels)),
        tmax=n_times / sfreq,
        subject_ids=subject_ids[order],
        session_ids=session_ids[order],
        dataset="synthetic",
        dataset_version="1",
    )


def _trial_metadata(
    n_trials: int,
    n_trials_per_class: int,
    subjects: tuple[str, ...],
    sessions: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_ids = np.array([subjects[i % len(subjects)] for i in range(n_trials)], dtype=str)
    session_ids = np.array(
        [sessions[(i // max(1, n_trials_per_class)) % len(sessions)] for i in range(n_trials)],
        dtype=str,
    )
    order = rng.permutation(n_trials)
    return subject_ids[order], session_ids[order], order


def make_synthetic_p300_epochs(
    n_trials_per_class: int = 40,
    n_channels: int = 8,
    sfreq: float = 128.0,
    n_times: int = 128,
    subjects: tuple[str, ...] = ("S01",),
    sessions: tuple[str, ...] = ("0",),
    seed: int = 0,
    noise: float = 0.8,
) -> Epochs:
    """Target trials carry a positive 300 ms deflection on a channel subset."""
    rng = np.random.default_rng(seed)
    n_trials = n_trials_per_class * 2
    labels = np.repeat(np.array([0, 1], dtype=int), n_trials_per_class)
    times = np.arange(n_times, dtype=float) / sfreq
    erp_channels = np.arange(0, n_channels, 2)
    data = rng.normal(0.0, noise, size=(n_trials, n_channels, n_times))
    for index, label in enumerate(labels):
        if label != 1:
            continue
        latency = 0.30 + rng.normal(0.0, 0.02)
        width = 0.07 + rng.normal(0.0, 0.008)
        amplitude = 2.2 + rng.normal(0.0, 0.25)
        wave = amplitude * np.exp(-0.5 * ((times - latency) / width) ** 2)
        data[index, erp_channels, :] += wave[None, :]
    subject_ids, session_ids, order = _trial_metadata(
        n_trials, n_trials_per_class, subjects, sessions, rng
    )
    return Epochs(
        data=data[order],
        labels=labels[order],
        sfreq=sfreq,
        channel_names=tuple(f"C{i + 1}" for i in range(n_channels)),
        tmin=0.0,
        tmax=n_times / sfreq,
        subject_ids=subject_ids,
        session_ids=session_ids,
        paradigm="P300",
        paradigm_meta={"task": "p300", "positive_label": "Target", "window_seconds": n_times / sfreq},
        dataset="synthetic_p300",
        dataset_version="1",
        label_names=("NonTarget", "Target"),
    )


def make_synthetic_ssvep_epochs(
    n_trials_per_class: int = 12,
    n_classes: int = 8,
    n_channels: int = 8,
    sfreq: float = 256.0,
    n_times: int = 256,
    subjects: tuple[str, ...] = ("S01",),
    sessions: tuple[str, ...] = ("0",),
    seed: int = 0,
    noise: float = 0.55,
) -> Epochs:
    """Sinusoidal steady-state responses with a distinct frequency per class."""
    rng = np.random.default_rng(seed)
    n_trials = n_trials_per_class * n_classes
    labels = np.repeat(np.arange(n_classes, dtype=int), n_trials_per_class)
    frequencies = np.linspace(8.0, 15.0, n_classes)
    times = np.arange(n_times, dtype=float) / sfreq
    channel_gain = rng.uniform(0.6, 1.2, size=(n_classes, n_channels))
    data = rng.normal(0.0, noise, size=(n_trials, n_channels, n_times))
    for index, label in enumerate(labels):
        frequency = float(frequencies[label])
        phase = rng.uniform(0.0, 2.0 * np.pi)
        carrier = np.sin(2.0 * np.pi * frequency * times + phase)
        harmonic = 0.35 * np.sin(2.0 * np.pi * 2.0 * frequency * times + phase / 2.0)
        data[index] += channel_gain[label][:, None] * (carrier + harmonic)[None, :]
    subject_ids, session_ids, order = _trial_metadata(
        n_trials, n_trials_per_class, subjects, sessions, rng
    )
    label_names = tuple(f"{frequency:.2f}" for frequency in frequencies)
    return Epochs(
        data=data[order],
        labels=labels[order],
        sfreq=sfreq,
        channel_names=tuple(f"C{i + 1}" for i in range(n_channels)),
        tmin=0.0,
        tmax=n_times / sfreq,
        subject_ids=subject_ids,
        session_ids=session_ids,
        paradigm="SSVEP",
        paradigm_meta={
            "task": "ssvep",
            "stimulus_frequencies": [float(value) for value in frequencies],
            "window_seconds": n_times / sfreq,
        },
        dataset="synthetic_ssvep",
        dataset_version="1",
        label_names=label_names,
    )
