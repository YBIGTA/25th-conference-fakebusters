import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = "fakebuster"
REGION = "ap-northeast-2"
CSV_FILE_PATH = "upload_metrics.csv"