from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import random 
import httpx

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)

@router.get("/lip/{filekey}")
async def roi_model(filekey: str):
    '''
    Simulate the lipforensic model server
    Input: filekey: str
    Output: response: StreamingResponse which contains the video file and score in the header
    '''
    # todo: add logic to get the video file from lipforensic model server
    video_file_path = "lip_sample.mp4"

    video_file = open(video_file_path, "rb")
    score = random.randint(0, 100)

    response = StreamingResponse(video_file, media_type="video/mp4")
    response.headers['Score'] = f"{score}"
    response.headers['FileKey'] = filekey
    return response

@router.get("/mmnet/{filekey}")
async def model(filekey: str):
    '''
    get response from mmnet model server
    Input: filekey: str
    Output: response: dict {message (text), status (text), reults (float)}
    '''

    model_sever_url = "http://165.132.46.87:30310/process_video/"
    video_file_path = "lip_sample.mp4"
    # headers = {
    #     "Content-Type": "multipart/form-data"
    # }

    try: 
        with open(video_file_path, "rb") as video_file:
            async with httpx.AsyncClient() as client:
                response = await client.post(model_sever_url, files={"file": video_file})
                response.raise_for_status()
                return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from model server: {e.response.text}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@router.get("/faceroi/{filekey}")
async def face_roi_model(filekey: str):
    '''
    Simulate the Face ROI server
    Input: filekey: str
    Output: response: StreamingResponse which contains the video file and score in the header
    '''
