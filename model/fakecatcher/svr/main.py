from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
from pathlib import Path
from inference import inference
import uvicorn
import argparse

app = FastAPI()

# 비디오 저장 디렉토리 설정
VIDEO_UPLOAD_DIR = Path("uploaded_videos")
VIDEO_UPLOAD_DIR.mkdir(exist_ok=True)  # 디렉터리가 없으면 생성

# 명령줄 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
args = parser.parse_args()

@app.get("/")
def read_root():
    """
    Root 엔드포인트: 간단한 상태 메시지 반환
    """
    return {"message": "PPG map-based Fake Video Detection API"}

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    비디오 파일을 업로드받아 서버에 저장하는 API
    """
    # 비디오 파일 MIME 타입 확인
    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a valid video file.")

    # 저장 경로 설정
    file_path = VIDEO_UPLOAD_DIR / file.filename

    # 비디오 파일 저장
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    
    # 비디오 파일 분석
    try:
        accuracy = inference(str(file_path), args.config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    return {"message": "Video uploaded successfully!", "file_name": file.filename, "file_path": str(file_path), "score": str(accuracy)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
