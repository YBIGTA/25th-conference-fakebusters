import os
import csv
from config import CSV_FILE_PATH

# Ensure CSV file exists and has the correct header
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "file_type", "file_size(B)", "upload_time(ms)"])

def log_upload_metrics(file_name, file_type, file_size, upload_time):
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file_name, file_type, file_size, upload_time])