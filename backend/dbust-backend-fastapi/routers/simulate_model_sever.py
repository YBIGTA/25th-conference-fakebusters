from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import random 
import httpx
import os, shutil
import numpy as np
import json
import subprocess

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
  
    model_server_url = "https://358e-165-132-46-83.ngrok-free.app/upload-video/"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(model_server_url, files={"file": video_file})
            response.raise_for_status()
            
            headers = {
                "File-Path": file_path,
                "Score":str(response.headers["score"]),
                "Access-Control-Expose-Headers": "File-Path, Score"
            }
            return StreamingResponse(content=response.iter_bytes(), headers=headers)
        
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
    model_server_url = "https://9d4f-165-132-46-93.ngrok-free.app/process_video/"

    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")
    headers = {"Accept-Charset": "utf-8"}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(model_server_url, files={"file": video_file}, headers=headers)
            response.raise_for_status()
            
            video_data = response.content
            
            headers = {
                "File-Path": file_path,
                "Score": str(response.headers["X-Inference-Result"]),
                "Access-Control-Expose-Headers": "File-Path, Score"
            }
            
            def video_iterator(data):
                yield data
                
            return StreamingResponse(video_iterator(video_data), headers=headers)
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

@router.post("/visual-ppg")
async def get_visual(file: UploadFile = File(...)):
    
    PPG_DIR = "misc/ppg"
    file_path = os.path.join(PPG_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        subprocess.run(["python", PPG_DIR +"/main.py", "-v", file_path, "-c", PPG_DIR+"/config.yaml"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error in processing video {e}")
    
    ppg_graph_name = file.filename.split(".")[0] + "_graph.mp4"
    ppg_mask_name = file.filename.split(".")[0] + "_mask.mp4"
    ppg_transformed_name = file.filename.split(".")[0] + "_transformed.mp4"
    
    return JSONResponse(content={
        "videos": {
            "ppg_graph": ppg_graph_name,
            "ppg_mask": ppg_mask_name,
            "ppg_transformed": ppg_transformed_name
        }
    })
    
    
@router.get("/video/{file_name}")
async def get_video(file_name: str):
    '''
    Get the video file
    Input: file_path: str
    Output: response: StreamingResponse which contains the video file
    '''
    file_path = os.path.join("misc/ppg/output", file_name)
    video_file = open(file_path, "rb")
    
    def video_iterator(file):
        yield from file
    
    return StreamingResponse(video_iterator(video_file), media_type="video/mp4")


@router.post("/fakecatcher-cnn")
async def get_result(file: UploadFile = File(...)):
    model_server_url = "https://534e-165-132-46-85.ngrok-free.app/upload-video/"
    file_path = os.path.join(VIDEO_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(model_server_url, files={"file": video_file})
            response.raise_for_status()
            
            headers = {
                "File-Path": file_path,
                "Access-Control-Expose-Headers": "File-Path"
            }
            score = response.json()["score"]
            return JSONResponse(content={"Score": score}, headers=headers)
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")



@router.post("/fakecatcher-feature")
async def get_result(file: UploadFile = File(...)):
    model_server_url = "https://be4e-165-132-46-92.ngrok-free.app/upload-video"
    file_path = os.path.join(VIDEO_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video_file = open(file_path, "rb")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(model_server_url, files={"file": video_file})
            response.raise_for_status()
            
            headers = {
                "File-Path": file_path,
                "Access-Control-Expose-Headers": "File-Path"
            }
            score = response.json()["score"]
            return JSONResponse(content={"Score": score}, headers=headers)
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")

