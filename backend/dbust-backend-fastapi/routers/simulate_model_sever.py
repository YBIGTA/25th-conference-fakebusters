from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import random 

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)

@router.get("/lip/{filekey}")
async def roi_model(filekey: str):
    '''
    Simulate the ROI model server
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
