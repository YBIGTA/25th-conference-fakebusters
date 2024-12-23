import numpy as np
from scipy.fft import fft, ifft

def frequency_resample(signal: np.ndarray, time_interval: int, original_fps: int, target_fps: int) -> np.ndarray:
    """
    주파수 기반 보간을 사용하여 1차원 신호를 target_fps로 변환.

    Args:
        signal (np.ndarray): 입력 신호 (1차원 배열).
        original_fps (int): 입력 신호의 샘플링 주파수(FPS).
        target_fps (int): 목표 샘플링 주파수(FPS).
    
    Returns:
        np.ndarray: 보간된 신호 (1차원 배열).
    """
    if original_fps == target_fps:
        # FPS가 동일하면 보간 없이 원본 신호 반환
        return signal
    
    # 1. FFT를 통해 주파수 도메인으로 변환
    fft_signal = fft(signal)

    # 2. 목표 길이 계산
    target_length = int(time_interval * target_fps)

    # 3. Zero Padding 또는 Truncation
    interpolated_fft = np.zeros(target_length, dtype=complex)
    min_length = min(len(fft_signal), target_length)
    interpolated_fft[:min_length // 2] = fft_signal[:min_length // 2]
    interpolated_fft[-min_length // 2:] = fft_signal[-min_length // 2:]

    # 4. IFFT를 통해 시간 도메인으로 변환
    interpolated_signal = ifft(interpolated_fft).real

    return interpolated_signal
