from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import random 
import httpx
import os, shutil
import numpy as np
import json

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)

VIDEO_DIR = "data/video"


hard_path = "data/video/chimmark.mp4"


@router.post("/test")
async def main(file: UploadFile = File(...)):
    # Define the directory to save the file
    save_dir = "data/video"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a unique filename
    file_path = os.path.join(save_dir, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    def iterfile():
        with open(file_path, mode="rb") as file:
            yield from file
    
    score = random.randint(0, 100)
    
    headers = {
        "File-Path": file_path,
        "Score": f"{score}",
        "Access-Control-Expose-Headers": "File-Path, Score"
    }

    return StreamingResponse(iterfile(), media_type="video/mp4", headers=headers)



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
  
    model_server_url = "http://165.132.46.83:32274/upload-video/"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(model_server_url, files={"file": video_file})
            response.raise_for_status()
            
            score = response.headers["score"]
            return StreamingResponse(content=response.iter_bytes(), headers={"score": score})
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")

@router.post("/mmnet")
async def model(file: UploadFile = File(...)):
    '''
    get response from mmnet model server
    Input: filekey: str
    Output: response: dict {message (text), status (text), reults (float)}
    '''

    model_sever_url = "http://165.132.46.87:32116/process_video/"
    
    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")
    headers = {"Accept-Charset": "utf-8"}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(model_sever_url, files={"file": video_file}, headers=headers)
            response.raise_for_status()
            
            video_data = response.content
            score = response.headers["X-Inference-Result"]
            
            def video_iterator(data):
                yield data
                
            return StreamingResponse(video_iterator(video_data), headers={"score": score})
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
  
    
@router.post("/ppg")
async def face_roi_model(file: UploadFile = File(...)):
    '''
    Simulate the ppg server
    Input: filekey: str
    Output: response: StreamingResponse which contains the video file and score in the header
    '''
    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")
    
    model_sever_url = "http://165.132.46.85:32697/upload-video/"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(model_sever_url, files={"file": video_file})
            response.raise_for_status()
            
            return response
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")


@router.post("/visual-ppg")
async def get_visual(file: UploadFile = File(...)):
    
    model_sever_url = "http://165.132.46.83:30409/upload-video"
    file_path = os.path.join(VIDEO_DIR, file.filename)
    
    if os.path.exists(file_path):
        video_file = open(file_path, "rb")
    else:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        video_file = open(file_path, "rb")
        
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(model_sever_url, files={"file": video_file})
            response.raise_for_status()
            return {"message": "Success", "status": "200", "results": response.headers}
            
            results_path = os.path.join("data/npz", file.filename + ".npz")
            with open(results_path, "wb") as buffer:
                buffer.write(response.content)
            
            results = np.load(results_path)
            
            total_frames = results["total_frames"]
            masked_frames = results["masked_frames"]
            transformed_frames = results["transformed_frames"]
            r_means = results["R_means"]
            l_means = results["L_means"]
            m_means = results["M_means"]
            
            return {"masked_frames": masked_frames.tolist()}
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")
