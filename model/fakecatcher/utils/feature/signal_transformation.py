import numpy as np
from typing import Tuple
from scipy.signal import csd
from scipy.stats import entropy
from scipy.fft import fft, fftfreq

## ====================================================================
## ====================== Signal Transformations ======================
## ====================================================================

def log(signal: np.ndarray) -> np.ndarray:
    return np.log(signal+1e9)


def spectral_line(r_xxs: np.ndarray, threshold_factor=1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the number of spectral lines and maximum spectral line magnitude for each row.
    """
    magnitudes = np.abs(r_xxs)
    thresholds = threshold_factor * np.mean(magnitudes, axis=1, keepdims=True)
    spectral_lines = magnitudes > thresholds
    num_spectral_lines = np.sum(spectral_lines, axis=1)
    max_spectral_lines = np.max(magnitudes, axis=1)
    return num_spectral_lines, max_spectral_lines

def spectral_auto_correlation(signals: np.ndarray) -> np.ndarray:
    """
    Calculate the spectral autocorrelation for each row of a 2D signal array.
    params:
    signals: np.ndarray (num_signal, signal_length)
    return:
    np.ndarray (num_signal, len(delta_f_list))
    """
    _, signal_length = signals.shape
    delta_f_list = np.arange(-(signal_length - 1) // 2, (signal_length + 1) // 2)
    X_f = fft(signals, axis=1)  # FFT along rows
    r_xxs = []

    for delta_f in delta_f_list:
        X_f_shifted = np.roll(X_f, shift=delta_f, axis=1)
        r_xx = np.mean(X_f * np.conjugate(X_f_shifted), axis=1)
        r_xxs.append(r_xx)

    return np.array(r_xxs).T  # Transpose to have (num_rows, len(delta_f_list))

def autocorrelation(signals: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation for each signal in a 2D array.

    Parameters:
    signals (np.ndarray): Input 2D signal array of shape (num_signals, signal_length).

    Returns:
    np.ndarray: Autocorrelations for each signal, shape (num_signals, signal_length).
    """
    num_signals, signal_length = signals.shape
    autocorrelations = np.zeros((num_signals, signal_length))  # Initialize result array

    for i in range(num_signals):
        signal = signals[i]
        max_lag = signal_length

        # Normalize signal
        normalized_signal = signal - np.mean(signal)

        # Compute full autocorrelation
        full_autocorrelation = (
            np.correlate(normalized_signal, normalized_signal, mode='full')
            / (np.var(signal) * len(signal))
        )

        # Extract positive lags
        autocorrelations[i] = full_autocorrelation[len(signal) - 1 : len(signal) - 1 + max_lag]

    return autocorrelations

def shannon_entropy(signal: np.ndarray, num_windows: int):
    """
    Shannon entropy
    param:
    signal: (signal_length,)
    """
    counts, _ = np.histogram(signal, bins=num_windows, range=(0, np.max(signal)))
    probabilities = counts / np.sum(counts)

    sh_entropy = entropy(probabilities, base=2)
    return sh_entropy

def narrow_pulse(r_xxs: np.ndarray, threshold_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the number and average magnitude of narrow pulses for each row.
    """
    magnitudes = np.abs(r_xxs)
    thresholds = threshold_factor * np.mean(magnitudes, axis=1, keepdims=True)
    pulses = magnitudes > thresholds
    num_pulses = np.sum(pulses, axis=1)
    
    avg_pulses = np.where(num_pulses > 0, np.mean(magnitudes * pulses, axis=1), 0)

    return num_pulses, avg_pulses

def pairwise_cross_spectral_density(signal1:np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """
    Calculate F1 features: Mean and max cross spectral density differences.

    Parameters:
    signal (np.ndarray): (num_signal, each_signal_length)  original ppg transformed signals
    Returns:
    np.ndarray:
    """
    freqs, csd_values = csd(signal1, signal2)
    # Return as a feature vector
    return freqs, np.abs(csd_values)

if __name__=="__main__":
    signal2 = np.random.randn(1000)

    print(signal2.shape)
    print(autocorrelation(signal2))