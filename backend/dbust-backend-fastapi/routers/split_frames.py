import os
import subprocess
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/api/split-frames",
    tags=["split-frames"],
    responses={404: {"description": "Not found"}},
)

@router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    video_path = "original.mp4"
    frames_directory = "data/frames/"
    os.makedirs(frames_directory, exist_ok=True)

    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract frames using ffmpeg
    subprocess.run(["ffmpeg", "-i", video_path, os.path.join(frames_directory, "frame%04d.jpg")])

    def iter_images():
        image_files = sorted([f for f in os.listdir(frames_directory) if f.startswith("frame") and f.endswith(".jpg")])
        for image_file in image_files:
            with open(os.path.join(frames_directory, image_file), "rb") as img:
                yield img.read()
            # os.remove(os.path.join(frames_directory, image_file))  # Clean up the image file after sending

    return StreamingResponse(iter_images(), media_type="multipart/x-mixed-replace; boundary=frame")