import os
import cv2
import dlib
import yaml
import skvideo
import argparse
import skvideo.io
import subprocess
import numpy as np
from tqdm import tqdm
from collections import deque
from skimage import transform as tf

class VideoROIExtractor:
    
    def __init__(self,
                 input_video_path: str,
                 detector: dlib.get_frontal_face_detector,
                 predictor: dlib.shape_predictor,
                 mean_face: object,
                 config: dict):
        """
        If you intend to import this class for extracting ROIs,
        you only need to provide a dictionary of config settings for the following:
        roi_target, skip_frames, resized_frame_width, resized_frame_height,
        stablePntsIDs, std_size, crop_width, crop_height, and window_margin.
        """
        ## Arguments
        self.input_video_path = input_video_path
        self.detector = detector
        self.predictor = predictor
        self.mean_face = mean_face
        self.config = config

        ## Directory Management
        self.roi_directory = config['roi_directory']
        self.audio_directory = config['audio_directory']

        ## ROI Settings
        self.roi_target = config['roi_target']
        self.custom_target = config['custom_target']
        self.roi_indices = None
        self.get_roi_indices()

        ## Video Landmark Options
        self.skip_frames = config['skip_frames']
        self.resized_frame_width = config['resized_frame_width']
        self.resized_frame_height = config['resized_frame_height']

        ## Video Crop Options
        self.stablePntsIDs = config['stablePntsIDs']
        self.std_size = config['std_size']
        self.crop_height = config['crop_height']
        self.crop_width = config['crop_width']
        self.window_margin = config['window_margin']
        
        ## Video Metadata
        self.frame_total_count = 0
        self.frame_per_second = 0
        self.current_frames = 1
        self.get_metadata()

        ## Video Data
        self.frames = None
        self.load_video()

    ## =====================================================================
    ## ============================= UTILITIES =============================
    ## =====================================================================

    def load_video(self):
        """
        비디오 파일 읽어 프레임 이미지 배열로 반환.
        """
        videogen = skvideo.io.vread(self.input_video_path)
        frames = np.array([frame for frame in videogen])
        self.frames = frames

    def yield_frame(self):
        """
        주어진 프레임 이미지 배열로부터 이미지 한장씩 반환.
        """
        for frame in self.frames:
            yield frame

    def get_metadata(self):
        """
        비디오 파일을 읽어 총 프레임 수와 fps 정보를 저장.
        """
        ## 비디오 메타데이터 열람.
        meta_data = skvideo.io.ffprobe(self.input_video_path)
        fps = meta_data['video']['@avg_frame_rate']
        ftc = meta_data['video']['@nb_frames']
        ## 문자열 정보 정수로 변환.
        numer = int(fps.split('/')[0])
        denom = int(fps.split('/')[1])
        self.frame_per_second = int(numer/denom)
        self.frame_total_count = int(ftc)

    def get_roi_indices(self):
        """
        사용자 설정에서 읽어온 roi 타겟을 랜드마크 인덱스 리스트로 변환.
        """
        roi_dict = {
            "mouth": [48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67],
            "nose": [27,28,29,30,31,32,33,34,35],
            "right eye": [36,37,38,39,40,41],
            "right cheek": [0,1,2,3,4,5,48,31,39,40,41,36],
            "right eyebrow": [17,18,19,20,21],
            "left eye": [42,43,44,45,46,47],
            "left cheek": [16,15,14,13,12,11,35,42,47,46,45],
            "left eyebrow": [22,23,24,25,26],
            "face": list(range(0, 68))
        }
        if self.custom_target:
            self.roi_indices = self.custom_target
        else:
            self.roi_indices = roi_dict[self.roi_target]

    def save(self, rois):
        """
        추출한 ROI는 mp4 파일로, 오디오는 wav 파일로 저장.
        """
        ## 경로 할당.
        video_name = os.path.basename(self.input_video_path[:-4])
        roi_name = "custom" if self.custom_target else self.roi_target
        roi_path = os.path.join(self.roi_directory, video_name +'_' + roi_name + '_roi.mp4')
        audio_fn = os.path.join(self.audio_directory, video_name + '.wav')

        # 비디오 저장.
        skvideo.io.vwrite(roi_path, rois, inputdict={'-r': str(self.frame_per_second)})
        
        # 오디오 추출.
        cmd = f"/root/miniconda3/envs/video/bin/ffmpeg -i {self.input_video_path} -f wav -vn -y {audio_fn} -loglevel quiet"
        subprocess.call(cmd, shell=True)

    ## ======================================================================
    ## ============================== LANDMARK ==============================
    ## ======================================================================

    def detect_landmark(self, frame: np.ndarray) -> np.ndarray:
        """
        이미지로부터 얼굴 랜드마크 추출.
        
        Args: 
            frame: 비디오 프레임

        Returns:
            landmark: 랜드마크 좌표값
        """
        ## 프레임 길이, 너비 저장.
        original_frame_height = frame.shape[0]
        original_frame_width = frame.shape[1]

        ## 빠른 연산을 위한 프레임 리사이징 후 그레이스케일 변환.
        resize_dim = (self.resized_frame_width, self.resized_frame_height, )
        resized_frame = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
        del(frame)

        ## 얼굴이 포함된 바운딩 박스들을 모두 받아온다.
        rect, score, _ = self.detector.run(gray, upsample_num_times=1, adjust_threshold=0.5)

        ## 검출된 바운딩 박스가 없으면 None 리턴.
        if len(rect)==0:
            return None
        
        ## 가장 점수가 높은 바운딩 박스 선택
        rect, _ = max(zip(rect, score), key=lambda x: x[1])

        ## 바운딩 박스 내부에서 얼굴 랜드마크 추출.
        shape = self.predictor(gray, rect)
        
        ## 랜드마크 좌표 값을 넘파이 배열로 저장하기 위해 0으로 채운 배열 초기화.
        landmark = np.zeros((68, 2), dtype=np.int32)

        ## 추출한 랜드마크 좌표 값들 원본 프레임 크기로 리사이징 후, 배열 안에 저장.
        for i in range(68):
            x_resized = shape.part(i).x
            y_resized = shape.part(i).y
            x_original = int(x_resized * (original_frame_width / self.resized_frame_width))
            y_original = int(y_resized * (original_frame_height / self.resized_frame_height))
            landmark[i] = (x_original, y_original)
        return landmark

    def linear_interpolate(self, landmarks: list,
                           start_idx: int,
                           stop_idx: int) -> list:
        """
        얼굴 랜드마크 추출을 실패한 프레임에 대한 선형 보간 작업 실행.
        
        Args: 
            landmarks: 비디오 하나에 대해 프레임 단위로 추출한 얼굴 랜드마크 리스트.
            start_idx: 결측치 발생 직전의 랜드마크
            stop_idx: 결측치 발생 이후 최초의 랜드마크

        Returns:
            landmarks: 선형 보간 작업을 거친 얼굴 랜드마크 리스트.
        """
        ## 결측치 발생 직전의 랜드마크와 결측치 발생 이후 최초의 랜드마크 불러오고, 값 차이 계산.
        start_landmarks = landmarks[start_idx]
        stop_landmarks = landmarks[stop_idx]
        delta = stop_landmarks - start_landmarks

        ## 선형 보간 작업.
        for idx in range(1, stop_idx-start_idx):
            landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
        return landmarks

    def landmarks_interpolate(self, landmarks: list) -> list:
        """
        얼굴 랜드마크 추출을 실패한 프레임 구간 탐지 후 선형 보간.
        
        Args: 
            landmarks: 주어진 비디오에 대해 프레임 단위로 추출한 얼굴 랜드마크 리스트.

        Returns:
            landmarks: 선형 보간 작업을 거친 얼굴 랜드마크 리스트.
        """
        ## 랜드마크가 제대로 추출된 프레임들의 인덱스만 저장하는 리스트.
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        
        ## 추출된 랜드마크가 하나도 없으면 None 반환.
        ## for-loop를 돌면서 연속적으로 잘 추출된 프레임들은 무시한다.
        ## 인덱스 간 차이가 1보다 크면 바로 linear_interpolate 메소드에 전달.
        if not valid_frames_idx:
            return None
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                continue
            else:
                landmarks = self.linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])

        ## 마지막으로 비디오 시작과 끝 부분에 대한 랜드마크 결측 발생 시
        ## 가장 첫 랜드마크와 가장 마지막 랜드마크를 단순히 복사+붙여넣기 하는 방식으로 보간.
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        
        ## 최종 결측치 체크.
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
        return np.array(landmarks)

    ## ======================================================================
    ## ==============================   CROP   ==============================
    ## ======================================================================

    def warp_img(self, src: np.ndarray,
                 dst: np.ndarray,
                 img: np.ndarray
                 ) -> tuple[np.ndarray, tf.SimilarityTransform]:
        """
        주어진 좌표들을 기반으로 이미지를 변환 후 변환 행렬과 함께 지정된 크기로 출력.

        Args:
            src   : 원본 이미지에서의 좌표
            dst   : 목표 이미지에서의 좌표
            img   : 변형할 이미지

        Returns:
            warped: 변형된 이미지
            tform : 유사 변환 행렬
        """
        # 유사 변환 행렬 계산
        tform = tf.estimate_transform('similarity', src, dst)

        # 이미지 변형
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=self.std_size)

        # 0~1 범위에서 0~255 범위로 스케일링
        warped = warped * 255

        # uint8 형식으로 변환
        warped = warped.astype('uint8')
        return warped, tform

    def apply_transform(self, transform: tf.SimilarityTransform,
                        img: np.ndarray) -> np.ndarray:
        """
        주어진 변환 행렬을 사용하여 이미지를 변환하고, 변환된 이미지를 반환.

        Args:
            transform: 유사 변환 행렬
            img      : 변형할 이미지

        Returns:
            warped   : 변형된 이미지
        """
        # 변환 적용
        warped = tf.warp(img, inverse_map=transform.inverse, output_shape=self.std_size)

        # 0~1 범위에서 0~255 범위로 스케일링
        warped = warped * 255

        # uint8 형식으로 변환
        warped = warped.astype('uint8')
        return warped

    def cut_patch(self, img: np.ndarray,
                  landmarks: np.ndarray,
                  height: int,
                  width: int,
                  threshold: int = 5) -> np.ndarray:
        """
        주어진 이미지에서 랜드마크를 기준으로 패치를 잘라서 반환.

        Args:
            img      : 입력 이미지
            landmarks: 랜드마크 좌표
            height   : 잘라낼 패치의 높이
            width    : 잘라낼 패치의 너비
            threshold: 잘라낼 영역이 너무 크거나 작은 경우를 방지하는 임계값

        Raises:
            Exception: 이미지의 크기와 잘라낼 패치 크기 차이가 너무 클 경우 예외를 발생시킴

        Returns:
            잘라낸 이미지 패치
        """
        # 랜드마크 중심 계산
        center_x, center_y = np.mean(landmarks, axis=0)

        # 이미지 크기를 벗어나지 않도록 중심 좌표 조정
        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')

        # 잘라낼 영역이 이미지 크기를 넘지 않도록 조정
        if center_y + height > img.shape[0]:
            center_y = img.shape[0] - height
        if center_y + height > img.shape[0] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[1]:
            center_x = img.shape[1] - width
        if center_x + width > img.shape[1] + threshold:
            raise Exception('too much bias in width')

        # 이미지에서 패치 추출
        cut_img = np.copy(
            img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                int(round(center_x) - round(width)): int(round(center_x) + round(width))]
        )
        return cut_img

    def crop_patch1(self, landmarks: list[np.ndarray]) -> np.ndarray:
        """
        주어진 비디오 프레임에서 랜드마크를 기준으로 ROI를 추출하여 패치 시퀀스를 반환.
        패치는 프레임 간의 일관성을 유지하기 위해 랜드마크를 스무딩 처리하고,
        기하학적 변환을 적용하여 정렬된 형태로 생성.

        Args:
            landmarks: 각 프레임의 보간된 랜드마크 목록. 

        Returns:
            잘라낸 ROI 패치 시퀀스.

        Raises:
            Exception: 랜드마크를 처리할 수 없는 경우 예외 처리.
        """
        ## 비디오 불러오고, 프레임 인덱스 초기화.
        frame_gen = self.yield_frame()
        frame_idx = 0

        ## 비디오 총 프레임 수와 윈도우 크기와 비교, 더 작은 값을 롤링 버퍼의 최대 용량으로 지정한다.
        margin = min(self.frame_total_count, self.window_margin)

        ## 프로세스 트래킹을 위한 프로그레스 바 설정.
        with tqdm(total=len(landmarks)-int(margin), desc="관심영역 추출", leave=False) as pbar:
            
            while True:
                ## 비디오 프레임을 순차적으로 읽어오기.
                try:
                    frame = next(frame_gen)
                except StopIteration:
                    break
                
                ## 프레임, 랜드마크 저장용 롤링 버퍼 초기화.
                ## 크롭한 시퀀스 저장할 빈 리스트 초기화.
                if frame_idx == 0:
                    q_frame = deque(maxlen=margin)
                    q_landmarks = deque(maxlen=margin)
                    sequence = []

                ## 현재 프레임, 랜드마크 롤링 버퍼에 추가.
                q_landmarks.append(landmarks[frame_idx])
                q_frame.append(frame)

                ## 주어진 최대 용량을 채우는 시점부터 스무딩 및 얼굴 정렬 시작.
                if len(q_frame) == margin:

                    ## 롤링 윈도우에서 랜드마크를 평균값을 구하고,
                    ## 버퍼 내부에서 가장 오래된 프레임과 랜드마크를 가져온다.
                    smoothed_landmarks = np.mean(q_landmarks, axis=0)
                    cur_landmarks = q_landmarks.popleft()
                    cur_frame = q_frame.popleft()

                    ## 얼굴 랜드마크에서 가장 안정적인 인덱스들을 기준으로
                    ## 현재 버퍼 내부의 랜드마크 평균값과 레퍼런스 랜드마크 사이의 변환 행렬을 구한다.
                    ## 앞서 추출한 프레임과 랜드마크에 변환 행렬 계산을 적용한다.
                    trans_frame, trans = self.warp_img(
                        smoothed_landmarks[self.stablePntsIDs, :],
                        self.mean_face[self.stablePntsIDs, :],
                        cur_frame
                    )
                    trans_landmarks = trans(cur_landmarks)
                    
                    ## 랜드마크의 인덱스를 활용하여 지정한 ROI 영역에 한정된 패치로 자른다.
                    ## 자른 패치를 최종 반환될 시퀀스에 저장한다.
                    sequence.append(
                        self.cut_patch(
                            trans_frame,
                            trans_landmarks[self.roi_indices],
                            self.crop_height // 2,
                            self.crop_width // 2
                        )
                    )

                ## 프레임 인덱스가 마지막 값에 도달할 때, 버퍼 안에는 margin-1개의 데이터가 남아있다.
                ## 따라서 남은 값들에 대해서도 동일한 연산을 수행하고, 얻은 패치들을 최종 시퀀스에 저장한다.
                if frame_idx == len(landmarks) - 1:
                    while q_frame:
                        cur_frame = q_frame.popleft()
                        trans_frame = self.apply_transform(trans, cur_frame)
                        trans_landmarks = trans(q_landmarks.popleft())
                        sequence.append(
                            self.cut_patch(
                                trans_frame,
                                trans_landmarks[self.roi_indices],
                                self.crop_height // 2,
                                self.crop_width // 2
                            )
                        )
                    return np.array(sequence)
                
                ## 프레임 인덱스 업데이트.
                frame_idx += 1
                pbar.update(1)

        ## 처리할 프레임과 랜드마크가 없으면 None 반환.
        return None

    def crop_patch2(self, landmarks: np.ndarray) -> np.ndarray:
        # 메모리에 존재하는 프레임 배열
        frames = self.frames
        
        # 시퀀스를 저장할 리스트 초기화
        sequence = []

        pbar = tqdm(total=len(landmarks), desc="관심영역 추출", leave=False)
        
        for frame_idx in range(len(landmarks)):
            frame = frames[frame_idx]  # 이미 메모리에 존재하는 프레임 배열에서 직접 접근
            cur_landmarks = landmarks[frame_idx][self.roi_indices]  # 현재 랜드마크 사용

            # 랜드마크에 맞춰서 ROI 자르기
            patch = self.cut_patch(
                frame,
                cur_landmarks,
                self.crop_height // 2,
                self.crop_width // 2
            )

            # 자른 패치를 시퀀스에 추가
            sequence.append(patch)
            pbar.update(1)

        pbar.close()
        return np.array(sequence)

    ## ======================================================================
    ## ==============================  PROCESS  =============================
    ## ======================================================================

    def preprocess_video(self) -> np.ndarray:
        """
        비디오 파일에서 얼굴을 감지하고, 해당 얼굴의 랜드마크를 기준으로 얼굴 영역을 자르고, 
        얼굴을 정렬하여 비디오를 전처리하는 함수.

        주어진 입력 비디오 파일에서 얼굴을 감지하고, 랜드마크를 추출하여 얼굴 영역을 자른 후 
        표준 크기로 얼굴을 정렬하여 새로운 비디오를 생성. 이때 음성도 추출하여 저장.
        """
        ## 랜드마크 추출
        landmarks = []
        ## 프로세스 트래킹을 위한 프로그레스 바 설정.
        with tqdm(total=len(self.frames), desc="랜드마크 추출", leave=False) as pbar:
            for frame in self.frames:
                if self.current_frames % self.skip_frames != 0:
                    self.current_frames+=1
                    landmarks.append(None)
                else:
                    self.current_frames+=1
                    landmark = self.detect_landmark(frame)
                    landmarks.append(landmark)
                pbar.update(1)

        ## 랜드마크 보간
        preprocessed_landmarks = self.landmarks_interpolate(landmarks)

        ## ROI 패치 자르기
        rois = self.crop_patch1(landmarks=preprocessed_landmarks)
        return rois
        
    ## ======================================================================
    ## ==============================   MAIN   ==============================
    ## ======================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    video_directory = config['video_directory']
    face_predictor_path = config['face_predictor_path']
    mean_face_path = config['mean_face_path']

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mean_face = np.load(mean_face_path)
    videos = os.listdir(video_directory)

    for video in tqdm(videos, desc="전체 진행률", total=len(videos)):
        video_path = os.path.join(video_directory, video)

        if video_path.endswith('.mp4') and not video_path.endswith('_roi.mp4'):
            processor = VideoROIExtractor(video_path, detector, predictor, mean_face, config)
            rois = processor.preprocess_video()
            processor.save(rois)