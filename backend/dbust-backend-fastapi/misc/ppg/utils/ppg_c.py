import numpy as np
from scipy.signal import butter, filtfilt, windows
import math


class PPG_C:
    """
    G. de Haan and V. Jeanne, “Robust pulse rate from chrominance-based
    rPPG,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 10,
    pp. 2878–2886, Oct 2013.
    """
    def __init__(self, RGB_mean_dict: dict, fps: int):
        """
        ROI 경로로 비디오 정보를 읽어 인스턴스 초기화.
        """
        self.RGB_mean_dict = RGB_mean_dict
        self.RGB = self.extract_mean_rgb()
        self.fps = fps

    @classmethod
    def from_RGB(cls, RGB: np.ndarray, fps: float):
        """
        생성자 오버로딩 느낌
        """
        obj = cls.__new__(cls)
        obj.RGB_mean_dict = None
        obj.RGB = RGB
        obj.fps = fps
        return obj
    ## ====================================================================
    ## ========================== Core Methods ============================
    ## ====================================================================

    def extract_mean_rgb(self) -> np.ndarray:
        """
        각 프레임에서 RGB 평균값을 계산하여 반환.

        Returns:
            RGB: (N, 3) 형태의 평균 RGB 배열.
        """
        RGB = np.vstack((self.RGB_mean_dict["R"], self.RGB_mean_dict["G"], self.RGB_mean_dict["B"]))
        RGB = RGB.transpose()
        return RGB

    def bandpass_filter(self, lpf: float, hpf: float) -> tuple:
        """
        대역 통과 필터 설계.

        Args:
            lpf: 저주파 차단 주파수.
            hpf: 고주파 차단 주파수.
        
        Returns:
            B, A: 필터 계수.
        """
        nyquist_freq = 0.5 * self.fps
        B, A = butter(3, [lpf / nyquist_freq, hpf / nyquist_freq], btype='bandpass')
        return B, A

    def compute_signal(self, lpf=0.7, hpf=2.5, win_sec=1.6) -> np.ndarray:
        """
        CHROM 알고리즘을 사용해 PPG 신호를 계산.

        Args:
            lpf: 저주파 차단 주파수.
            hpf: 고주파 차단 주파수.
            win_sec: 슬라이딩 윈도우 길이 (초).
        
        Returns:
            BVP: PPG 신호.
        """
        # 1. RGB 평균값 추출
        num_frames = self.RGB.shape[0]

        # 2. Band-pass filtering
        B, A = self.bandpass_filter(lpf, hpf)
        
        # 3. 슬라이딩 윈도우 설정
        win_length = math.ceil(win_sec * self.fps)
        if win_length % 2:
            win_length += 1  # 짝수로 맞춤

        # 4. PPG 신호 계산
        COMPUTED_PPG_SIGNAL = np.zeros(num_frames)  # 결과 배열을 num_frames 크기로 초기화
        win_start = 0

        while win_start + win_length <= num_frames:
            win_end = win_start + win_length

            # 현재 윈도우에 대한 평균 및 정규화
            rgb_base = np.mean(self.RGB[win_start:win_end], axis=0)
            rgb_base += 1e-6
            rgb_norm = self.RGB[win_start:win_end] / rgb_base

            Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
            Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]

            # 필터 적용
            Xf = filtfilt(B, A, Xs, axis=0)
            Yf = filtfilt(B, A, Ys)

            std_Yf = np.std(Yf)
            if std_Yf > 1e-6:
                alpha = np.std(Xf) / std_Yf
            else:
                alpha = 0.0

            S_window = Xf - alpha * Yf
            S_window *= windows.hann(win_length)

            # 결과 반영
            COMPUTED_PPG_SIGNAL[win_start:win_end] += S_window

            # 다음 윈도우로 이동
            win_start += win_length // 2  # 윈도우가 절반만큼 겹침

        return COMPUTED_PPG_SIGNAL