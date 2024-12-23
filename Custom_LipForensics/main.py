from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
from pathlib import Path

app = FastAPI()

# 비디오 저장 디렉토리 설정
VIDEO_UPLOAD_DIR = Path("uploaded_videos")
VIDEO_UPLOAD_DIR.mkdir(exist_ok=True)  # 디렉터리가 없으면 생성

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


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
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    

    return {"message": "Video uploaded successfully!", "file_name": file.filename, "file_path": str(file_path)}
