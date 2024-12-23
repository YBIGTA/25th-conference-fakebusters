from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import random 
import httpx
import os, shutil

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)

VIDEO_DIR = "data/video"

@router.post("/lipforensic")
async def roi_model(file: UploadFile = File(...)):
    '''
    Simulate the lipforensic model server
    Input: filekey: str
    Output: response: StreamingResponse which contains the video file and score in the header
    '''
    
    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")
    score = random.randint(0, 100)

    response = StreamingResponse(video_file, media_type="video/mp4")
    response.headers['Score'] = f"{score}"
    return response

@router.post("/mmnet")
async def model(file: UploadFile = File(...)):
    '''
    get response from mmnet model server
    Input: filekey: str
    Output: response: dict {message (text), status (text), reults (float)}
    '''

    model_sever_url = "http://165.132.46.87:30310/process_video/"
    
    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")
    score = random.randint(0, 100)
    
    response = StreamingResponse(video_file, media_type="video/mp4")
    response.headers['Score'] = f"{score}"
    return response

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(model_sever_url, files={"file": video_file})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@router.post("/faceroi")
async def face_roi_model(file: UploadFile = File(...)):
    '''
    Simulate the Face ROI server
    Input: filekey: str
    Output: response: StreamingResponse which contains the video file and score in the header
    '''
    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")
    score = random.randint(0, 100)
    
    response = StreamingResponse(video_file, media_type="video/mp4")