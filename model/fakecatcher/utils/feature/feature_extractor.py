import numpy as np
from typing import Tuple, Dict, List
from scipy.signal import csd
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
from .signal_transformation import *

class FeatureExtractor:
    """
    """
    def __init__(self, fps, *args):
        """
        params:
        signals: np.ndarray, order: [G_L, G_M, G_R, C_L, C_M, C_R]
        """
        self.fps = fps
        self.G_L_o, self.G_M_o, self.G_R_o, self.C_L_o, self.C_M_o, self.C_R_o = args

        self.S = self.prepare_S(self.G_L_o, self.G_R_o, self.G_M_o, self.C_L_o, self.C_R_o, self.C_M_o)
        self.S_C = self.prepare_S_C(self.C_L_o, self.C_R_o, self.C_M_o)
        self.D = self.prepare_D(self.C_L_o, self.G_L_o, self.C_R_o, self.G_R_o, self.C_M_o, self.G_M_o)
        self.D_C = self.prepare_D_C(self.C_L_o, self.C_M_o, self.C_R_o)

    def prepare_S(self, G_L, G_R, G_M, C_L, C_R, C_M):
        return np.vstack([G_L, G_R, G_M, C_L, C_R, C_M])

    def prepare_S_C(self, C_L, C_R, C_M):
        return np.vstack([C_L, C_R, C_M]) 

    def prepare_D(self, C_L, G_L, C_R, G_R, C_M, G_M):
        return np.vstack([
                np.abs(C_L - G_L),
                np.abs(C_R - G_R),
                np.abs(C_M - G_M)
            ])

    def prepare_D_C(self, C_L, C_M, C_R):
        return np.vstack([
                np.abs(C_L - C_M),
                np.abs(C_L - C_R),
                np.abs(C_R - C_M)
            ])
    
    def feature_union(self):
        """
        f = F1(log(D_C ) ∪ F3(F3(log(S) ∪ Ap(DC))) ∪ F4(log(S) ∪ Ap(DC )) ∪ µAˆ(S) ∪ max(Aˆ(S))
        """
        log_D_C = log(self.D_C)
        S_union_D_C = np.vstack((log(self.S), autocorrelation(self.D_C)))
        S_sac = spectral_auto_correlation(self.S)
        f1 = self.F1(log_D_C) # (2, )
        f3 = self.F3(S_union_D_C)
        f4 = self.F4(S_union_D_C, self.fps)
        mean_S = np.mean(S_sac, axis=1).real
        max_S = np.max(S_sac, axis=1).real
        f1_flat = f1.flatten()
        f3_flat = f3.flatten()
        f4_flat = f4.flatten()
        mean_S_flat = mean_S.flatten()
        max_S_flat = max_S.flatten()

        features = np.concatenate([f1_flat, f3_flat, f4_flat, mean_S_flat, max_S_flat])

        return features
    ## ====================================================================
    ## ========================= Feature Creation =========================
    ## ====================================================================

    def F1(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate F1 features: Mean and max cross spectral density differences.
        Parameters:
        signal (np.ndarray): (num_signal, each_signal_length)  original ppg transformed signals
        Returns:
        np.ndarray: Array of shape (2, num_signal) for the 2 features in F1.
        [[ max1 max2]
        [ mean1 mean2]]
        """
        num_signals, _ = signal.shape
        csd_features = []
        if num_signals > 1: 
            for i in range(num_signals):
                for j in range(i+1, num_signals):  # Avoid duplicate pairs (i, j) where i == j
                    _, csd_values = pairwise_cross_spectral_density(signal[i], signal[j])
                    mean_csd = np.mean(csd_values)
                    max_csd = np.max(csd_values)

                    # Append the features for this pair
                    csd_features.append([mean_csd, max_csd])
        else: 
            _, csd_values = pairwise_cross_spectral_density(signal[0], signal[0])
            mean_csd = np.mean(csd_values)
            max_csd = np.max(csd_values)
            csd_features.append([mean_csd, max_csd])

        return np.array(csd_features).T

    def F3(self, signals: np.ndarray) -> np.ndarray:
        """
        Extract F3 features for each row of a 2D signal array using numpy vectorization.
        params:
        signals (np.ndarray): (num_signals, signal_length)
        returns:
        np.ndarray: Array of shape (4, num_signals) containing the 4 features for each row.
        """
        _, signal_length = signals.shape
        r_xxs = spectral_auto_correlation(signals)
        num_pulses, avg_pulses = narrow_pulse(r_xxs)
        num_spectral_lines, max_spectral_lines = spectral_line(r_xxs)

        features = np.vstack((num_pulses, num_spectral_lines, avg_pulses, max_spectral_lines))
        return features


    def F4(self, signals: np.ndarray, window_size: int): 
        """
        Calculate 7 features for each row of a 2D signal array using numpy.
        param:
        signal (np.ndarray): Input signal of shape (num_signals, signal_length).
        window_size (int): Size of the window for analysis.(fps)

        returns:
        np.ndarray: Array of shape (7, num_signals) containing the 7 features for each row.
        """
        num_signals, signal_length = signals.shape

        num_windows = signal_length//window_size  
        assert num_windows > 1, "segment length는 fps의 2배여야합니다다"  
        reshaped_signals = signals[:, :num_windows * window_size].reshape(num_signals, num_windows, window_size)

        # 1. std
        std = np.std(signals, axis=1)

        # 2. sdann
        # Reshape signal to (num_windows, window_size) and truncate extra values
        window_means = reshaped_signals.mean(axis=2)
        sdann = np.std(window_means, axis=1)

        # 3.rmssd(1sec difference)
        # 1초 간격의 signal끼리 차분
        
        successive_difference =  np.diff(reshaped_signals[:,:,0], axis=1)
        
        # 제곱의 평균의 루트 계산
        rmssd = np.sqrt(np.mean(successive_difference**2,axis=1))
        # 4. sdnni
        window_std = reshaped_signals.std(axis=2)
        sdnni = np.mean(window_std, axis=1)

        # 5. sdsd(standard deviation of differences)
        successive_differences = np.diff(signals, axis=1)
        sdsd = np.std(successive_differences, axis=1)

        # 6. mean of autocorrelation
        mean_autocorrelations = np.mean(autocorrelation(signals), axis=1)

        # 7. Shannon entropy
        sh_entropy = np.array([shannon_entropy(signal, num_windows) for signal in signals])
        
        features = np.vstack([std, sdann, rmssd, sdnni, sdsd, mean_autocorrelations, sh_entropy])

        assert not np.any(np.isinf(features)), "nan이 존재함함"
        return features    

import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVR

def split_segments(data, segment_length):
    """
    Split data into segments of fixed length.
    """
    valid_length = len(data) - (len(data) % segment_length)
    return [np.array(data[i:i + segment_length]) for i in range(0, valid_length, segment_length)]

def combine_segments(*segments):
    """
    Combine multiple segments based on indices.
    """
    return list(zip(*segments))

def majority_voting(probabilities):
    """
    Perform majority voting on the predicted probabilities.
    """
    mean_prob = np.mean(probabilities)
    majority_vote = np.round(mean_prob)
    return majority_vote

if __name__ == "__main__":
    from utils.roi import ROIProcessor

    video_path = "/root/audio-visual-forensics/test.mp4"
    model_path = "misc/face_landmarker.task"
    landmarker = ROIProcessor(video_path, model_path)
    R_means_dict, L_means_dict, M_means_dict, fps = landmarker.detect_with_calculate()



    R_ROI_G_PPG = PPG_G(R_means_dict, fps).compute_signal()
    print("R_ROI_G_PPG: ", len(R_ROI_G_PPG))

    R_ROI_C_PPG = PPG_C(R_means_dict, fps).compute_signal()
    print("R_ROI_C_PPG: ", len(R_ROI_C_PPG))

    L_ROI_G_PPG = PPG_G(L_means_dict, fps).compute_signal()
    print("L_ROI_G_PPG: ", len(L_ROI_G_PPG))

    L_ROI_C_PPG = PPG_C(L_means_dict, fps).compute_signal()
    print("L_ROI_C_PPG: ", len(L_ROI_C_PPG))

    M_ROI_G_PPG = PPG_G(M_means_dict, fps).compute_signal()
    print("M_ROI_G_PPG: ", len(M_ROI_G_PPG))

    M_ROI_C_PPG = PPG_C(M_means_dict, fps).compute_signal()
    print("M_ROI_C_PPG: ", len(M_ROI_C_PPG))

    R_ROI_G_segments = split_segments(R_ROI_G_PPG, fps)
    R_ROI_C_segments = split_segments(R_ROI_C_PPG, fps)
    L_ROI_G_segments = split_segments(L_ROI_G_PPG, fps)
    L_ROI_C_segments = split_segments(L_ROI_C_PPG, fps)
    M_ROI_G_segments = split_segments(M_ROI_G_PPG, fps)
    M_ROI_C_segments = split_segments(M_ROI_C_PPG, fps)

    combined_segments = combine_segments(
    R_ROI_G_segments, 
    R_ROI_C_segments, 
    L_ROI_G_segments, 
    L_ROI_C_segments, 
    M_ROI_G_segments, 
    M_ROI_C_segments
    )

    features = []
    for ppg in combined_segments:
        fe = FeatureExtractor(fps, *ppg)
        feature = fe.feature_union()
        features.append(feature)
    features = np.array(features)
