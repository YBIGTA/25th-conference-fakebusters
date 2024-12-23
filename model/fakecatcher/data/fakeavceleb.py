import pandas as pd
from typing import List, Tuple
import os

def load_data(root_dir, csv_path: str) -> Tuple[List[str]]:
    """
    CSV 파일에서 source/id 형식의 비디오 경로를 생성하여 반환합니다.

    Args:
        csv_path (str): CSV 파일 경로.

    Returns:
        list: 비디오 경로 리스트.
    """
    # CSV 파일 읽기
    data = pd.read_csv(csv_path)

    # 비디오 경로 생성
    video_paths = data.apply(lambda row: os.path.join(root_dir, row['vid'], row['path']), axis=1).tolist()
    labels = data['type'].apply(lambda x: 1 if x.split('-')[0] == 'RealVideo' else 0).tolist()

    return video_paths, labels

if __name__ == "__main__":
    # CSV 파일 경로
    csv_path = "/root/audio-visual-forensics/data/FakeAVCeleb/meta_data.csv"

    # 비디오 경로 로드
    video_paths = load_data(csv_path)
    
    # 출력 예시
    for path in video_paths:
        print(path)