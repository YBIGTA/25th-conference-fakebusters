import numpy as np
from .ppg_c import PPG_C

class PPG_MAP:
    """
    """
    def __init__(self, transformed_frames: list[np.ndarray], fps: int):
        """
        ROI 경로로 비디오 정보를 읽어 인스턴스 초기화.
        """
        self.transformed_frames = transformed_frames
        self.fps = fps

    def compute_map(self):
        """
        """
        ## (1200, 600, 1320, 3)
        num_frames = self.transformed_frames.shape[0]
        region_height = int(self.transformed_frames.shape[1]/4)
        region_width = int(self.transformed_frames.shape[2]/8)

        ## (1200, 4, 150, 8, 165, 3)
        regions_reshaped = self.transformed_frames.reshape(
            self.transformed_frames.shape[0],
            self.transformed_frames.shape[1] // region_height, region_height,
            self.transformed_frames.shape[2] // region_width, region_width,
            self.transformed_frames.shape[3]
        )

        ## (1200, 4, 8, 3)
        region_means = regions_reshaped.mean(axis=(2, 4))
        num_cols = region_means.shape[1]
        num_rows = region_means.shape[2]
        grid_rows = num_cols * num_rows * 2
        grid = np.empty(shape=(grid_rows, num_frames))

        iter = 0
        for row in range(num_rows):
            for col in range(num_cols):
                PPG = PPG_C.from_RGB(RGB=region_means[:, col, row, :], fps=self.fps)
                signal = PPG.compute_signal()
                fft_values = np.fft.fft(signal)
                psd_values = np.abs(fft_values)**2
                normalized_signal = ((signal - signal.min()) / (signal.max() - signal.min()) * 255).astype(np.uint8)
                normalized_density = ((psd_values - psd_values.min()) / (psd_values.max() - psd_values.min()) * 255).astype(np.uint8)
                grid[iter,:] = normalized_signal
                iter+=1
                grid[iter,:] = normalized_density
                iter+=1
        return grid