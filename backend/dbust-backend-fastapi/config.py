import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = "fakebuster"
REGION = "ap-northeast-2"
CSV_FILE_PATH = "upload_metrics.csv"

lipforensic_server_url = "localhost:8000/upload-video/"
mmnet_server_url = "localhost:8001/process_video/"
fakecatcher_cnn_server_url = "localhost:8002/upload-video/"
fakecatcher_feature_server_url = "localhost:8003/upload-video/"
