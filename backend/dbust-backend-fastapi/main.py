from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from routers import file_upload, split_frames
import asyncio
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(file_upload.router)
app.include_router(split_frames.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.websocket("/ws/images")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    image_directory = "data/video/chim/frames"
    image_files = sorted(os.listdir(image_directory))
    for image_file in image_files:
        with open(os.path.join(image_directory, image_file), "rb") as f:
            image_data = f.read()
            await websocket.send_bytes(image_data)
        await asyncio.sleep(1)  # Simulate delay between images
    await websocket.close()

# run with uvicorn main:app --reload