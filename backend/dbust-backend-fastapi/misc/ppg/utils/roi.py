import cv2
import yaml
import skvideo.io
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

class ROIProcessor:
    """
    EXAMPLE USAGE

    Initialization:
        roi = ROIProcessor(video_path, model_path)

    Visualization:
        roi.detect_with_draw("OUTPUT_PATH")

    RGB Averaging:
        R_means_dict, L_means_dict, M_means_dict = roi.detect_with_calculate()

    Affine Transformation:
        transformed_frames = roi.detect_with_map()
    """

    ## ROI 랜드마크 인덱스.
    POLYGONS = {
        "R": [139, 100, 203, 135],
        "L": [368, 329, 423, 364],
        "M": [226, 25 , 110, 24 , 23 , 22 , 26 , 112, 243, 244,
              245, 122, 6  , 351, 465, 464, 463, 341, 256, 252,
              253, 254, 339, 255, 446, 265, 372, 345, 352, 280,
              411, 416, 434, 432, 436, 426, 423, 358, 279, 360,
              363, 281, 5  , 51 , 134, 131, 49 , 129, 203, 206,
              216, 212, 214, 192, 187, 50 , 123, 116, 143, 35]
    }

    ## ROI 색.
    COLORS = {
        "R": (0, 0, 255),
        "L": (255, 0, 0), 
        "M": (0, 255, 0)
    }

    ## ROI "M" 삼각형 인덱스.
    TRIANGLE_INDICES = np.array([
        [214, 212, 216], [214, 216, 207], [192, 214, 207], [187, 192, 207], [187, 207, 205],
        [50 , 187, 205], [50 , 205, 101], [50 , 101, 118], [50 , 118, 117], [123, 50 , 117],
        [116, 123, 117], [116, 117, 111], [143, 116, 111], [143, 111, 35 ], [35 , 111, 31 ],
        [226, 35 , 31 ], [226, 31 , 25 ], [25 , 31 , 228], [31 , 111, 117], [31 , 117, 228],
        [228, 118, 229], [229, 118, 119], [118, 101, 119], [205, 36 , 101], [207, 216, 206],
        [207, 206, 205], [205, 206, 36 ], [206, 203, 36 ], [36 , 203, 129], [36 , 129, 142],
        [36 , 142, 100], [101, 36 , 100], [101, 100, 119], [100, 120, 119], [119, 120, 230],
        [229, 119, 230], [25 , 228, 110], [110, 228, 229], [110, 229, 24 ], [24 , 229, 230],
        [24 , 230, 23 ], [230, 120, 231], [23 , 230, 231], [23 , 231, 22 ], [22 , 232, 26 ],
        [22 , 231, 232], [231, 121, 232], [231, 120, 121], [120, 47 , 121], [120, 100, 47 ],
        [100, 126, 47 ], [100, 142, 126], [142, 209, 126], [142, 129, 209], [129, 49 , 209],
        [49 , 131, 198], [49 , 198, 209], [209, 198, 126], [126, 198, 217], [126, 217, 47 ],
        [47 , 217, 114], [228, 117, 118], [47 , 114, 121], [114, 128, 121], [121, 128, 232],
        [232, 128, 233], [232, 233, 112], [26 , 232, 112], [112, 233, 243], [243, 233, 244],
        [244, 233, 128], [244, 128, 245], [245, 128, 114], [245, 114, 188], [188, 114, 174],
        [114, 217, 174], [217, 198, 174], [198, 236, 174], [198, 131, 134], [198, 134, 236],
        [134, 51 , 236], [236, 51 , 3  ], [174, 236, 196], [236, 3  , 196], [188, 174, 196],
        [188, 196, 122], [245, 188, 122], [122, 196, 6  ], [196, 197, 6  ], [196, 3  , 197],
        [197, 3  , 195], [3  , 51 , 195], [51 , 5  , 195], [6  , 197, 419], [6  , 351, 419],
        [197, 248, 419], [197, 195, 248], [195, 248, 281], [195, 5  , 281], [248, 281, 456],
        [281, 363, 456], [248, 456, 419], [419, 456, 399], [419, 399, 412], [419, 412, 351],
        [351, 412, 465], [465, 412, 343], [465, 343, 357], [465, 357, 464], [464, 357, 453],
        [464, 453, 463], [463, 453, 341], [412, 399, 343], [399, 437, 343], [399, 420, 437],
        [399, 456, 420], [456, 363, 420], [363, 360, 420], [360, 279, 420], [420, 279, 429],
        [437, 420, 355], [420, 429, 355], [437, 355, 277], [343, 437, 277], [343, 277, 350],
        [357, 343, 350], [357, 350, 452], [453, 357, 452], [341, 453, 452], [341, 452, 256],
        [256, 452, 252], [452, 451, 252], [452, 350, 451], [350, 349, 451], [350, 277, 349],
        [277, 329, 349], [277, 355, 329], [355, 371, 329], [355, 429, 371], [429, 358, 371],
        [429, 279, 358], [371, 358, 266], [358, 423, 266], [371, 266, 329], [329, 266, 330],
        [329, 330, 348], [349, 329, 348], [349, 348, 450], [451, 349, 450], [451, 450, 253],
        [252, 451, 253], [266, 423, 426], [266, 426, 425], [266, 425, 330], [425, 426, 427],
        [426, 436, 427], [436, 432, 434], [427, 436, 434], [427, 434, 416], [427, 416, 411],
        [425, 427, 411], [425, 411, 280], [330, 425, 280], [347, 330, 280], [348, 330, 347],
        [346, 347, 280], [346, 280, 352], [346, 352, 345], [450, 348, 449], [449, 348, 347],
        [449, 347, 448], [448, 347, 346], [448, 346, 261], [261, 346, 340], [261, 340, 265],
        [265, 340, 372], [340, 346, 345], [340, 345, 372], [253, 450, 254], [254, 450, 449],
        [254, 449, 339], [339, 449, 448], [339, 448, 255], [255, 448, 261], [255, 261, 446],
        [446, 261, 265]], dtype=np.int32)
    
    ## ROI "M" 랜드마크 인덱스 직사각형 메쉬 전환용 좌표.
    INDEX_COORDS = {
        226 : [0  ,0  ], 35  : [0  ,12 ], 143 : [0  ,18 ], 116 : [0  ,24 ],
        123 : [0  ,30 ], 50  : [0  ,36 ], 187 : [0  ,42 ], 192 : [0  ,48 ],
        214 : [0  ,54 ], 212 : [0  ,60 ], 216 : [12 ,60 ], 206 : [18 ,60 ],
        203 : [24 ,60 ], 129 : [30 ,60 ], 49  : [36 ,60 ], 131 : [42 ,60 ],
        134 : [54 ,60 ], 51  : [60 ,60 ], 5   : [66 ,60 ], 281 : [72 ,60 ],
        363 : [78 ,60 ], 360 : [90 ,60 ], 279 : [96 ,60 ], 358 : [102,60 ],
        423 : [108,60 ], 426 : [114,60 ], 436 : [120,60 ], 446 : [132,0  ],
        265 : [132,12 ], 372 : [132,18 ], 345 : [132,24 ], 352 : [132,30 ],
        280 : [132,36 ], 411 : [132,42 ], 416 : [132,48 ], 434 : [132,54 ],
        432 : [132,60 ], 25  : [6  ,0  ], 110 : [12 ,0  ], 24  : [18 ,0  ],
        23  : [24 ,0  ], 22  : [30 ,0  ], 26  : [36 ,0  ], 112 : [42 ,0  ],
        243 : [46 ,0  ], 244 : [50 ,0  ], 245 : [54 ,0  ], 122 : [60 ,0  ],
        6   : [66 ,0  ], 351 : [72 ,0  ], 465 : [78 ,0  ], 464 : [82 ,0  ],
        463 : [86 ,0  ], 341 : [92 ,0  ], 256 : [96 ,0  ], 252 : [102,0  ],
        253 : [108,0  ], 254 : [114,0  ], 339 : [120,0  ], 255 : [126,0  ],
        31  : [6  ,12 ], 111 : [6  ,18 ], 228 : [12 ,12 ], 117 : [12 ,24 ],
        207 : [12 ,44 ], 229 : [18 ,12 ], 118 : [18 ,30 ], 205 : [18 ,40 ],
        230 : [24 ,12 ], 119 : [24 ,24 ], 101 : [24 ,36 ], 36  : [24 ,48 ],
        231 : [30 ,12 ], 120 : [30 ,24 ], 100 : [30 ,36 ], 142 : [30 ,48 ],
        232 : [36 ,10 ], 121 : [36 ,20 ], 47  : [36 ,30 ], 126 : [36 ,40 ],
        209 : [36 ,50 ], 233 : [42 ,10 ], 128 : [42 ,20 ], 114 : [42 ,30 ],
        217 : [42 ,40 ], 198 : [42 ,50 ], 188 : [54 ,15 ], 174 : [54 ,30 ],
        236 : [54 ,45 ], 196 : [60 ,20 ], 3   : [60 ,40 ], 197 : [66 ,20 ],
        195 : [66 ,40 ], 419 : [72 ,20 ], 248 : [72 ,40 ], 412 : [78 ,15 ],
        399 : [78 ,30 ], 456 : [78 ,45 ], 453 : [90 ,10 ], 357 : [90 ,20 ],
        343 : [90 ,30 ], 437 : [90 ,40 ], 420 : [90 ,50 ], 452 : [96 ,10 ],
        350 : [96 ,20 ], 277 : [96 ,30 ], 355 : [96 ,40 ], 429 : [96 ,50 ],
        451 : [102,12 ], 349 : [102,24 ], 329 : [102,36 ], 371 : [102,48 ],
        450 : [108,12 ], 348 : [108,24 ], 330 : [108,36 ], 266 : [108,48 ],
        261 : [126,12 ], 340 : [126,18 ], 448 : [120,12 ], 346 : [120,24 ],
        427 : [120,44 ], 449 : [114,12 ], 347 : [114,30 ], 425 : [114,40 ]}

    def __init__(self, video_path: str, config: dict):
        self.video_path = video_path
        self.model_path = config["model_path"]
        self.fps_standard = int(config["fps_standard"])
        self.seg_time_interval = int(config["seg_time_interval"])
        self.frame_list = []
        self.detection_result_list = []

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO)
        
        with FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(self.video_path)
            self.fps_local = int(cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timestep_ms= int((frame_idx / self.fps_local) * 1000)
                detection_result = landmarker.detect_for_video(mp_image, timestep_ms)
                self.frame_list.append(frame)
                self.detection_result_list.append(detection_result)
                frame_idx += 1
            cap.release()
            cv2.destroyAllWindows()

    def map(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> np.ndarray:
        """
        단일 프레임 아핀 변환.

        Args:
            frame: 비디오 프레임 이미지.
            detection_result: frame에 대한 모델 추론 결과.

        Returns:
            destin_frame: ROI "M" 내부의 삼각 랜드마크를 직사각형으로 펼쳐놓은 이미지.
        """
        ## 주어진 프레임에 얼굴이 잡히지 않은 경우 비어있는 이미지 반환.
        try:
            face_landmark = detection_result.face_landmarks[0]
        except Exception:
            return np.zeros((600, 1320, 3), dtype=np.float32)

        ## 프레임 복사본 생성하고 길이, 너비 확인.
        origin_frame = np.copy(frame)
        height, width, _ = frame.shape

        ## 랜드마크 내부의 정규화되어 있는 좌표들을 프레임 길이, 너비를 사용해 원본으로 복구.
        landmark_points = [[int(landmark.x * width), int(landmark.y * height)] for landmark in face_landmark]

        ## 랜드마크 옮겨줄 타겟 이미지 생성.
        destin_frame = np.zeros((600, 1320, 3), dtype=np.float32)

        ## 원본 이미지 내부 삼각 랜드마크를 직사각형 내부에 배정된 칸으로 아핀 변환을 적용해 끼워맞춘다.
        for triangle_index in ROIProcessor.TRIANGLE_INDICES:
            ## 원본 이미지의 삼각 랜드마크 좌표.
            origin_coords = np.array([landmark_points[i] for i in triangle_index], dtype=np.float32)
            ## 원본 이미지의 삼각 랜드마크 좌표에 대응하는 타겟 이미지의 삼각 랜드마크 좌표.
            destin_coords = np.array([ROIProcessor.INDEX_COORDS[i] for i in triangle_index], dtype=np.float32)
            ## 현재 최대 길이와 최대 너비가 60,132로 제한되어 있는 타겟 이미지 10배 스케일링.
            destin_coords *= 10

            ## 원본, 타겟 이미지의 삼각 랜드마크를 담을 바운딩 박스 좌표 계산. (x,y,w,h)
            origin_rect = cv2.boundingRect(origin_coords)
            destin_rect = cv2.boundingRect(destin_coords)

            ## 바운딩 박스의 왼쪽 상단 코너 좌표값을 기준으로 삼각형 좌표 재정렬. 
            origin_trig_cropped = []
            destin_trig_cropped = []
            for i in range(0, 3):
                origin_trig_cropped.append(((origin_coords[i][0] - origin_rect[0]), (origin_coords[i][1] - origin_rect[1])))
                destin_trig_cropped.append(((destin_coords[i][0] - destin_rect[0]), (destin_coords[i][1] - destin_rect[1])))
            
            ## 원본 이미지를 바운딩 박스 크기로 자르기.
            ox1 = origin_rect[0]
            ox2 = origin_rect[0] + origin_rect[2]
            oy1 = origin_rect[1]
            oy2 = origin_rect[1] + origin_rect[3]
            cropped_frame = origin_frame[oy1:oy2, ox1:ox2]
            
            ## 원본 이미지의 삼각 랜드마크로부터 타겟 이미지의 삼각 랜드마크로의 아핀 변환 행렬 계산.
            warpMat = cv2.getAffineTransform(np.float32(origin_trig_cropped), np.float32(destin_trig_cropped))
            
            ## 원본 이미지로부터 잘라낸 바운딩 박스를 변환 행렬을 사용해 워프.
            try:
                warped_frame = cv2.warpAffine(cropped_frame, warpMat, (destin_rect[2], destin_rect[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            except Exception:
                return np.zeros((600, 1320, 3), dtype=np.float32)
            
            ## 타겟 이미지 바운딩 박스 크기의 마스크 생성.
            mask = np.zeros((destin_rect[3], destin_rect[2], 3), dtype = np.float32)
            ## 타겟 삼각 랜드마크 내부의 픽셀값을 1로 조정.
            cv2.fillConvexPoly(mask, np.int32(destin_trig_cropped), (1,1,1), 16, 0)
            ## 변환한 바운딩 박스에 마스크를 적용.
            warped_frame = warped_frame * mask

            ## 직사각형의 바운딩 박스 영역 가져와서 픽셀값 더해주기.
            dx1 = destin_rect[0]
            dy1 = destin_rect[1]
            dx2 = destin_rect[0] + destin_rect[2]
            dy2 = destin_rect[1] + destin_rect[3]
            ## 아핀 변환을 적용하는 과정에서 결과값 차원이 미세하게 틀어질 수 있기 때문에 리사이징으로 다시 맞춰주기.
            warped_frame = cv2.resize(warped_frame, (destin_frame[dy1:dy2, dx1:dx2].shape[1], destin_frame[dy1:dy2, dx1:dx2].shape[0]))
            destin_frame[dy1:dy2, dx1:dx2] = destin_frame[dy1:dy2, dx1:dx2] + warped_frame
        return destin_frame

    def draw(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> np.ndarray:
        """
        단일 프레임 시각화.

        Args:
            frame: 비디오 프레임 이미지.
            detection_result: frame에 대한 모델 추론 결과.

        Returns:
            annotated_frame: 얼굴 우측, 중심, 좌측 ROI 폴리곤 시각화한 프레임.
        """
        ## 주어진 프레임에 얼굴이 잡히지 않은 경우 원본 프레임 그대로 반환.
        try:
            face_landmark = detection_result.face_landmarks[0]
        except Exception:
            return frame
        
        ## 그림 그릴 프레임 복사본 생성하고 길이, 너비 확인.
        annotated_frame = np.copy(frame)
        height, width, _ = frame.shape
        
        ## 랜드마크 내부의 정규화되어 있는 좌표들을 프레임 길이, 너비를 사용해 원본으로 복구.
        landmark_points = [[int(landmark.x * width), int(landmark.y * height)] for landmark in face_landmark]

        ## 프레임 이미지 위에 ROI "R", "L", "M" 경계선 그리기.
        for region, indices in ROIProcessor.POLYGONS.items():
            polygon_points = np.array([landmark_points[i] for i in indices], dtype=np.int32)
            cv2.polylines(annotated_frame, [polygon_points], isClosed=True, color=ROIProcessor.COLORS[region], thickness=2)

        ## 프레임 이미지 위에 ROI "M" 내부 삼각 랜드마크 그리기.
        for indices in ROIProcessor.TRIANGLE_INDICES:
            polygon_points = np.array([landmark_points[i] for i in indices], dtype=np.int32)
            cv2.polylines(annotated_frame, [polygon_points], isClosed=True, color=[220,220,220], thickness=1)
        return annotated_frame

    def calculate(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> list[np.ndarray]:
        """
        단일 프레임 RGB 평균 계산.

        Args:
            frame: 비디오 프레임 이미지.
            detection_result: frame에 대한 모델 추론 결과.

        Returns:
            R_mean: 오른쪽 뺨 ROI의 RGB 평균값.
            L_mean: 왼쪽 뺨 ROI의 RGB 평균값.
            M_mean: 얼굴 중간 ROI의 RGB 평균값.
        """
        ## 폴리곤 형태에 맞춰서 마스킹 씌우고 마스킹 내부 영역 픽셀에 대한 평균값 계산하는 내부함수 설정.
        def get_mean_color(landmark_indices):
            points = np.array([landmark_points[i] for i in landmark_indices], dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            pixels = frame[mask == 255]
            return np.mean(pixels, axis=0) if pixels.size > 0 else [0, 0, 0]

        ## 주어진 프레임에 얼굴이 잡히지 않은 경우 np.nan 반환.
        try:
            face_landmark = detection_result.face_landmarks[0]
        except Exception:
            return [np.array([np.nan, np.nan, np.nan]),
                    np.array([np.nan, np.nan, np.nan]),
                    np.array([np.nan, np.nan, np.nan])]
        
        ## 프레임 길이, 너비 확인.
        height, width, _ = frame.shape
        ## 랜드마크 내부의 정규화되어 있는 좌표들을 프레임 길이, 너비를 사용해 원본으로 복구.
        landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmark]
        
        ## 각 ROI 영역에 대한 RGB 평균 계산. 각각 (1,3) 차원의 넘파이 배열.
        R_mean = get_mean_color(ROIProcessor.POLYGONS["R"])
        L_mean = get_mean_color(ROIProcessor.POLYGONS["L"])
        M_mean = get_mean_color(ROIProcessor.POLYGONS["M"])
        return R_mean, L_mean, M_mean
    
    def detect_with_map(self, crop: bool) -> list[np.ndarray, int]:
        """
        Mediapipe의 FaceLandmark 모델로 검출한 얼굴 중심부 ROI를 삼각형 단위로 나눈다.
        삼각형 영역별로 아핀 변환을 적용하고, 변환된 이미지를 타겟 영역에 배치하여 직사각형 형태로 재구성한다.

        Returns:
            transformed_frames: 직사각형 이미지 리스트.
        """        
        ## 아핀 변환 적용한 프레임 이미지 저장할 리스트.
        transformed_frames = []

        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for frame, detection_result in zip(self.frame_list, self.detection_result_list):
                transformed_frames.append(self.map(frame, detection_result))
                pbar.update(1)

        transformed_frames = np.array(transformed_frames)
        if crop==False:
            return transformed_frames, self.fps_local
        
        segment_size = int(self.fps_local * self.seg_time_interval)
        padding_size = (segment_size - (self.frame_count % segment_size)) % segment_size
        segment_num = int((self.frame_count + padding_size) / segment_size)

        if padding_size > 0:
            transformed_frames = np.pad(transformed_frames,((0, padding_size),(0, 0),(0, 0),(0, 0)), mode='constant', constant_values=0)

        transformed_frames = transformed_frames.reshape(segment_num, segment_size, 600, 1320, 3)
        transformed_frames = transformed_frames[:-1, :, :, :, :]
        return transformed_frames, self.fps_local

    def detect_with_draw(self) -> np.ndarray:
        """
        Args:
            output_path: 결과 시각화 비디오 저장 경로.

        Mediapipe의 FaceLandmark 모델로 검출한 얼굴 중심부, 왼쪽 뺨, 오른쪽 뺨 ROI의 경계선을 시각화한다.
        모델이 도출하는 랜드마크는 사람의 얼굴에 들로네 삼각변환을 적용한 형태이고, 이중 얼굴 중심부에 포함되는 삼각형들도 시각화한다.
        최종 결과물을 동영상으로 합쳐서 저장한다.
        """

        ## 랜드마크 경계 및 삼각형 표시된 프레임 담을 리스트.
        annotated_frames = []

        ## tqdm 진행도 설정.
        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for frame, detection_result in zip(self.frame_list, self.detection_result_list):
                annotated_frames.append(self.draw(frame, detection_result))                
                pbar.update(1)
        annotated_frames = np.array(annotated_frames)
        return annotated_frames
    
    def detect_with_calculate(self, crop: bool) -> list[list[np.ndarray], int]:
        """
        Mediapipe의 FaceLandmark 모델로 검출한 얼굴 중심부, 왼쪽 뺨, 오른쪽 뺨 ROI의 경계 바깥을 마스킹 처리하여
        주어진 ROI 내부의 픽셀만 남기고, RGB 각 채널에 대한 픽셀 평균값을 계산한다.
        마지막으로 결측치에 대한 선형 보간 작업을 수행하여 딕셔너리 형태로 반환한다.

        Returns:
            R_means_dict: "R", "G", "B" 키에 대한 np.array(shape=(#frames,)) 평균값 밸류 보유.
            L_means_dict: "R", "G", "B" 키에 대한 np.array(shape=(#frames,)) 평균값 밸류 보유.
            M_means_dict: "R", "G", "B" 키에 대한 np.array(shape=(#frames,)) 평균값 밸류 보유.
        """
        def interpolate(array: np.ndarray) -> np.ndarray:
            for row in range(len(array)):
                # 현재 행 데이터 가져오기
                current_row = array[row]
                # 결측치의 인덱스와 비결측치 인덱스 구분
                mask = np.isnan(current_row)
                # 결측치가 존재하면 보간 수행
                if np.any(mask):
                    current_row[mask] = np.interp(
                        np.flatnonzero(mask),
                        np.flatnonzero(~mask),
                        current_row[~mask])
            return array

        ## ROI별 RGB 평균값 담을 리스트.
        R_means_list = []
        L_means_list = []
        M_means_list = []
        
        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for frame, detection_result in zip(self.frame_list, self.detection_result_list):
                R_mean, L_mean, M_mean = self.calculate(frame, detection_result)
                R_means_list.append(R_mean)
                L_means_list.append(L_mean)
                M_means_list.append(M_mean)
                pbar.update(1)

        ## RGB 리스트 결측치 선형 보간.
        R_means_array = interpolate(np.array(R_means_list).T)
        L_means_array = interpolate(np.array(L_means_list).T)
        M_means_array = interpolate(np.array(M_means_list).T)

        if crop==False:
            return R_means_array, L_means_array, M_means_array, self.fps_local

        segment_size = int(self.fps_local * self.seg_time_interval)
        padding_size = (segment_size - (self.frame_count % segment_size)) % segment_size
        segment_num = int((self.frame_count + padding_size) / segment_size)
        
        if padding_size > 0:
            R_means_array = np.pad(R_means_array, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
            L_means_array = np.pad(L_means_array, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
            M_means_array = np.pad(M_means_array, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)

        R_means_array = R_means_array.reshape(3, segment_num, segment_size).transpose(1, 2, 0)
        L_means_array = L_means_array.reshape(3, segment_num, segment_size).transpose(1, 2, 0)
        M_means_array = M_means_array.reshape(3, segment_num, segment_size).transpose(1, 2, 0)

        R_means_array = R_means_array[:-1, :, :]
        L_means_array = L_means_array[:-1, :, :]
        M_means_array = M_means_array[:-1, :, :]
        return R_means_array, L_means_array, M_means_array, self.fps_local