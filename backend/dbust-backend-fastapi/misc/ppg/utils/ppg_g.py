import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

class PPG_G:
    """
    C. Zhao, C.-L. Lin, W. Chen, and Z. Li, “A novel framework for remote
    photoplethysmography pulse extraction on compressed videos,” in The
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
    Workshops, June 2018.
    """
    def __init__(self, RGB_mean_dict: dict, fps: int):
        """
        ROI 경로로 비디오 정보를 읽어 인스턴스 초기화.
        """
        self.RGB_mean_dict = RGB_mean_dict
        self.fps = fps
        self.trace = None
        self.extract_G_trace()

    @classmethod
    def from_RGB(cls, trace: np.ndarray, fps: float):
        """
        생성자 오버로딩 느낌
        """
        obj = cls.__new__(cls)
        obj.RGB_mean_dict = None
        obj.fps = fps
        obj.trace = trace[:,1]
        return obj
    ## ====================================================================
    ## ============================== Utils ===============================
    ## ====================================================================
    
    def diagonal_average(self, matrix: np.ndarray) -> np.ndarray:
        """
        행렬 diagonal average 적용.

        a11 a12 a13   (a11)             / 1
        a21 a22 a23   (a21 + a12)       / 2
        a31 a32 a33   (a31 + a22 + a13) / 3
                      (a32 + a23)       / 2
                      (a33)             / 1

        Args:
            matrix: diagonal averaging 적용할 행렬.

        Returns:
            averages: "/" 형태의 대각선 상의 값들에 대한 평균값 배열.
        """
        ## 행렬 행과 열 개수 확인.
        rows, cols = matrix.shape
        ## 가능한 대각선 경우의 수 계산.
        num_diags = rows + cols - 1
        ## 평균값 저장할 배열 초기화.
        averages = np.zeros(num_diags)

        ## 각 대각선 별로 평균값 계산.
        for diag in range(num_diags):
            elements = []
            for row_idx in range(rows):
                col_idx = diag - row_idx
                if 0 <= col_idx < cols:
                    elements.append(matrix[row_idx, col_idx])
            if elements:
                averages[diag] = np.mean(elements)
        return averages
    
    def dominant_frequency(self, array: np.ndarray) -> float:
        """
        Fast fourier transform을 통해서 시계열 값을 주파수 성분으로 분해했을 때,
        그 진폭이 가장 강한 성분의 주파수 값을 반환.

        Args: 
            array: 고속 푸리에 변환을 적용할 1D 배열.
        
        Returns:
            dominant_frequency: 진폭이 가장 강한 성분의 주파수 값.
        """
        ## 주어진 시퀀스를 주파수 성분별로 분해.
        ## 주파수 성분의 진폭과 위상을 포함하는 복소수 배열
        fourier = fft(array)

        ## 각 주파수 성분에 해당하는 주파수 값을 나타내는 배열
        frequency = fftfreq(n=len(array), d=1/self.fps)

        ## 진폭 계산.
        amplitude = np.abs(fourier)

        ## 진폭이 가장 큰 주파수 성분의 인덱스를 이용해서 dominant frequency 추출.
        dominant_index = np.argmax(amplitude)
        dominant_frequency = frequency[dominant_index]
        return dominant_frequency

    ## ====================================================================
    ## ================ Single-channel Band Pass Filtering ================
    ## ====================================================================

    def extract_G_trace(self) -> np.ndarray:
        """
        ROI 영상 G채널의 프레임 별 평균값 배열 반환.

        Returns:
            raw_G_trace: ROI 영상 G채널의 프레임 별 평균값을 저장한 넘파이 배열. [F]
        """
        raw_G_trace = self.RGB_mean_dict["G"]
        self.raw_G_trace = raw_G_trace

    def filter_G_trace(self, raw_G_trace: np.ndarray) -> np.ndarray:
        """
        Single-channel Band Pass Filtering.

        Args:
            raw_G_trace: ROI 영상 G채널의 프레임 별 평균값을 저장한 넘파이 배열. [F]

        Returns:
            filtered_G_trace: single-channel band-pass filtering 적용한 배열. [F]
        """
        ## Hz 기준으로 컷오프 설정.
        low = 0.8
        high = 5.0

        ## FPS 절반을 기준으로 컷오프 정규화.
        low = low / (0.5 * self.fps)
        high = high / (0.5 * self.fps)

        # Butterworth band-pass filter 적용.
        b, a = butter(N=4, Wn=[low, high], btype='band')
        
        # 양방향 필터링 적용하여 필터링에 의한 위상 왜곡 제거.
        filtered_G_trace = filtfilt(b, a, raw_G_trace)
        return filtered_G_trace

    ## ====================================================================
    ## ======================= SSA and RC Selection =======================
    ## ====================================================================

    def SSA(self, filtered_G_trace: np.ndarray, window_size: int) -> np.ndarray:
        """
        Singular Spectrum Analysis Decomposition.

        Args:
            filtered_G_trace: single-channel band-pass filtering 적용한 배열. [F]
            window_size: 생성되는 한켈 행렬의 행 개수. W

        Returns:
            rc_array: SSA decomposition을 거쳐 얻는 Reconstructed Components.
                      매칭되는 고유값 기준으로 내림차순 정렬된 다차원 배열. [W, F]
        """
        ## Filtered G trace 배열을 한켈 행렬 Y로 변환.
        N = len(filtered_G_trace)
        K = N - window_size + 1
        Y = np.array([filtered_G_trace[i:i + window_size] for i in range(K)]).T

        ## Y에 SVD를 적용해서 [U][Sigma][V]ᵗ 형태로 계산.
        U, Sigma, Vt = np.linalg.svd(Y, full_matrices=False)

        ## 각 [Sigmaᵢ][Uᵢ][Vᵢ]ᵗ에 대해 diagonal averaging을 적용하여 reconstructed component 뽑기.
        rc_dict = {}
        for idx, sigma in enumerate(Sigma):
            rc_component = sigma * np.outer(U[:,idx], Vt[idx,:])
            rc_component = self.diagonal_average(rc_component)
            rc_dict[sigma] = rc_component

        ## 고유값 기준으로 내림차순 정렬.
        rc_dict = dict(sorted(rc_dict.items(), key=lambda item: item[0], reverse=True))
        rc_array = np.array(list(rc_dict.values()))

        ## 상위 10개 이내의 컴포넌트만 반환.
        rc_array = rc_array[:10] if len(rc_array)>10 else rc_array
        return rc_array

    def RC_selection(self, rc_array: np.ndarray, tolerance:float) -> np.ndarray:
        """
        RC Selection.

        Args:
            rc_array: SSA decomposition을 거쳐 얻는 Reconstructed Components.
                      매칭되는 고유값 기준으로 내림차순 정렬된 다차원 배열. [W, F]
            tolerance: dominant frequency 비교에 사용되는 absolute tolerance.

        Returns:
            rc_trace: RC selection을 거쳐 얻은 Reconstructed Components의 element-wise 합. [F]
        """
        ## twice-relationship 대조를 위한 후보 리스트.
        candidates=[]

        ## 후보 리스트에 각 Reconstructed Component의 dominant frequency 순차적으로 저장.
        for rc_component in rc_array:
            candidates.append(self.dominant_frequency(rc_component))

        ## dominant frequency만 비교하여 twice relationship을 만족하는 페어의 인덱스 추출.
        satisfies_twice = set()
        for source_idx, source_freq in enumerate(candidates):
            for target_idx, target_freq in enumerate(candidates):
                if source_idx < target_idx:
                    s_t = np.isclose(source_freq, 2 * target_freq, atol=tolerance)
                    t_s = np.isclose(target_freq, 2 * source_freq, atol=tolerance)
                    if s_t or t_s:
                        satisfies_twice.add(source_idx)
                        satisfies_twice.add(target_idx)

        ## 인덱스를 기준으로 기존 rc_array에서 최종 컴포넌트 추출.
        rc_array = rc_array[list(satisfies_twice)]

        ## element-wise 합으로 rc_trace 계산.
        rc_trace = np.sum(rc_array, axis=0)
        return rc_trace
    
    ## ====================================================================
    ## ========================== Overlap Adding ==========================
    ## ====================================================================

    def overlap_add(self, array: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """
        Overlap Adding.

        Args:
            array: overlap adding을 적용할 1차원 배열. [F]
            window_size: 윈도우 크기.
            step_size: 스텝 사이즈.
        
        Returns:
            overlap_sum: 각 윈도우에 Hanning window를 element-wize 곱처리하고,
                         윈도우 분할에 쓰인 스텝사이즈를 재활용하여 원본 배열의 크기에 맞춰 더한 배열. [F]
        """
        ## 입력 배열과 동일한 크기의 배열 0으로 초기화.
        overlap_sum = np.zeros_like(array)
        ## Hanning window 초기화.
        hann_window = np.hanning(window_size)
        
        for start in range(0, len(array) - window_size + 1, step_size):
            ## 마지막 인덱스 설정.
            end = start + window_size
            ## 인덱싱으로 원본 배열에서 윈도우 추출, hanning window와 곱처리.
            windowed_segment = array[start:end] * hann_window
            ## 결과값의 같은 인덱스 범위에 값 추가. 
            overlap_sum[start:end] += windowed_segment
        return overlap_sum

    ## ====================================================================
    ## ===================== SSA and Spectral Masking =====================
    ## ====================================================================

    def instantaeous_HR(self, preliminary: np.ndarray, window_size: int, step_size: int) -> float:
        """
        SSA decompositon, RC selection과 overlap adding으로 얻은 preliminary를 레퍼런스로 재활용.

        Args:
            preliminary: SSA decompositon, RC selection과 overlap adding으로 얻은 배열.

        Returns:
            f_r: 마스킹 기준으로 사용되는 dominant frequency 값.
        """
        ## 윈도우 단위로 Fast Fourrier Transform 적용.
        ## 각각의 윈도우에 대한 dominant frequency 계산 후 평균값 도출.
        freqs = []
        for start in range(0, len(preliminary) - window_size + 1, step_size):
            end = start + window_size
            windowed_segment = preliminary[start:end]
            dominant_frequency = self.dominant_frequency(windowed_segment)
            freqs.append(dominant_frequency)
        f_r = sum(freqs) / len(freqs)
        return f_r

    def spectral_mask(self, rc_array: np.ndarray, f_r: float,
                      window_size: int, step_size: int) -> np.ndarray:
        """
        SSA decomposition으로 얻은 reconstructed component 후보군 배열을
        instantaeous_HR에서 구한 마스킹 기준, f_r을 사용해서 필터링하고
        마지막으로 overlap adding을 적용해서 PPG 시그널 도출.

        Args:
            rc_array: SSA decomposition으로 얻은 reconstructed component 후보군 배열.
            f_r: 마스킹 기준으로 사용되는 dominant frequency 값.
            window_size: 마스킹 윈도우 크기.
            step_size: 마스킹 스텝 크기.
        
        Returns:
            pulse_signal: PPG 시그널.
        """
        ## Hanning window 초기화.
        hann_window = np.hanning(window_size)

        ## 마스킹을 통과하고 overlap adding이 적용된 최종 결과물 저장하기 위한 배열.
        pulse_signal = np.zeros_like(rc_array[0])

        ## 윈도우 단위로 마스킹 적용.
        for start in range(0, len(rc_array[0]) - window_size + 1, step_size):
            ## 마지막 인덱스 설정.
            end = start + window_size
            ## 마스킹 통과한 RC component에 대한 element-wise addition 값 저장할 배열.
            sum = np.zeros(window_size)

            ## rc_array 안에는 최대 10개의 rc_component가 담겨있다.
            for rc_component in rc_array:
                ## 주어진 시작, 끝 인덱스를 사용해 rc_component 슬라이싱.
                windowed_segment = rc_component[start:end]
                ## dominant frequency 계산.
                f_i = self.dominant_frequency(rc_component)
                ## 마스킹 조건을 만족하면 sum에 더해주기.
                if f_r - (window_size / 2) <= f_i <= f_r + (window_size / 2):
                    sum += windowed_segment

            ## Hanning window 적용하기.
            sum = sum * hann_window
            ## 최종 결과 배열에 더해주기.
            pulse_signal[start:end] += sum
        return pulse_signal

    ## ====================================================================
    ## ============================ Execution =============================
    ## ====================================================================

    def compute_signal(self) -> np.ndarray:
        """
        엑조디아.

        Returns:
            pulse_signal: PPG-G 시그널.
        """
        window_size = int(self.fps * 1.6)
        step_size = int(self.fps * 0.8)
        filtered_G_trace = self.filter_G_trace(self.trace)
        rc_array = self.SSA(filtered_G_trace, window_size)
        rc_trace = self.RC_selection(rc_array, tolerance=0.2)
        preliminary = self.overlap_add(rc_trace, window_size, step_size)
        f_r = self.instantaeous_HR(preliminary, window_size, step_size)
        pulse_signal = self.spectral_mask(rc_array, f_r, window_size, step_size)
        return pulse_signal
        