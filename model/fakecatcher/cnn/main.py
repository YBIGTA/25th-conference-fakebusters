from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
from pathlib import Path
from inference import predict
import uvicorn
import argparse

app = FastAPI()

# Set the directory for saving videos
VIDEO_UPLOAD_DIR = Path("uploaded_videos")
VIDEO_UPLOAD_DIR.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

# Command line argument handling
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
args = parser.parse_args()

@app.get("/")
def read_root():
    """
    Root endpoint: Returns a simple status message
    """
    return {"message": "PPG map-based Fake Video Detection API"}

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    API to receive and save the uploaded video file on the server
    """
    # Check the MIME type of the video file
    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a valid video file.")

    # Set the save path
    file_path = VIDEO_UPLOAD_DIR / file.filename

    # Save the video file
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    
    # Analyze the video file
    try:
        accuracy = predict(str(file_path), args.config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    return {"message": "Video uploaded successfully!", "file_name": file.filename, "file_path": str(file_path), "score": str(accuracy)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8282, reload=True)
