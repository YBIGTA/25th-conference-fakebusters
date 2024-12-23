from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import file_upload, simulate_model_sever
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
app.include_router(simulate_model_sever.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


# run with uvicorn main:app --reload