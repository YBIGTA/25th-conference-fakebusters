from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import time
from csv_service import log_upload_metrics

router = APIRouter(
    prefix="/api/files",
    tags=["files"],
    responses={404: {"description": "Not found"}},
)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    print(f"Upload started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}")

    file_key = upload_file_to_s3(file)

    end_time = time.time()
    print(f"Upload finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}")
    print(f"Upload duration: {int((end_time - start_time) * 1000)} ms")

    # Log metrics to CSV
    file_name = file.filename
    file_type = file.content_type
    file_size = file.size
    upload_time = int((end_time - start_time) * 1000)

    log_upload_metrics(file_name, file_type, file_size, upload_time)

    return JSONResponse(content={"fileKey": file_key})