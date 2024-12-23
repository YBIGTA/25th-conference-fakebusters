import boto3
import uuid
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME, REGION

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

def upload_file_to_s3(file):
    file_key = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
    s3_client.upload_fileobj(file.file, BUCKET_NAME, file_key)
    return file_key