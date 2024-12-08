import random
import numpy as np
import librosa


def crop_fixed_length(
    array, sample_rate, sample_length, fit_audio_to_sample_length=True
):
    """
    Generate a fixed-length audio sample. Optionally repeat the audio if total length < sample length.

    Args:
        array (np.ndarray): The input audio array.
        sample_rate (int): Sample rate of the audio in Hz.
        sample_length (float): Desired sample length in seconds (can be a decimal).
        fit_audio_to_sample_length (bool): If True, repeat audio to fit the sample length if it's shorter.

    Returns:
        np.ndarray: A fixed-length audio sample.

    Raises:
        ValueError: If fit_audio_to_sample_length is False and total audio length is less than sample length.
    """
    # Total audio length in seconds
    total_length = len(array) / sample_rate

    if fit_audio_to_sample_length and total_length < sample_length:
        # Repeat the audio to match the required length
        num_repeats = int(np.ceil(sample_length / total_length))
        repeated_array = np.tile(array, num_repeats)

        # Trim to exact sample length
        return repeated_array[: int(sample_length * sample_rate)]

    elif not fit_audio_to_sample_length and total_length < sample_length:
        raise ValueError(
            f"Audio length ({total_length:.2f} seconds) is shorter than the required sample length "
            f"({sample_length} seconds), and fit_audio_to_sample_length is set to False."
        )

    else:
        # Calculate starting index in seconds (ensuring valid range)
        start_idx = random.uniform(0, total_length - sample_length)

        # Convert start and end indices to array indices (ensure integers)
        start_idx_array = int(start_idx * sample_rate)
        end_idx_array = int((start_idx + sample_length) * sample_rate)

        # Return the sample slice
        return array[start_idx_array:end_idx_array]


def random_crop(array, sample_rate, sample_max_length):
    # total audio length in seconds
    total_length = len(array) / sample_rate

    # ensure max sample length does not exceed total length
    sample_max_length = min(sample_max_length, total_length)

    # sample length in seconds (convert to integer for indices)
    sample_length = random.randint(3, int(sample_max_length))

    return crop_fixed_length(
        array, sample_rate, sample_length, fit_audio_to_sample_length=True
    )


def add_gaussian_noise(audio_sample, noise_level=0.01):
    """Adds Gaussian noise to the audio sample."""
    noise = np.random.normal(0, noise_level, audio_sample.shape)
    return audio_sample + noise


def adjust_volume(audio_sample, volume_range=(0.1, 0.5)):
    """Randomly adjusts the volume of the audio sample."""
    volume_adjust = np.random.uniform(volume_range[0], volume_range[1])
    return audio_sample * volume_adjust


def pitch_shift(audio_sample, sample_rate, pitch_shift_steps=2):
    """Shifts the pitch of the audio sample."""
    return librosa.effects.pitch_shift(
        audio_sample.astype(float), sr=sample_rate, n_steps=pitch_shift_steps
    )


def time_stretch(audio_sample, time_stretch_rate=1.1):
    """Stretches or compresses the time of the audio sample."""
    return librosa.effects.time_stretch(
        audio_sample.astype(float), rate=time_stretch_rate
    )
