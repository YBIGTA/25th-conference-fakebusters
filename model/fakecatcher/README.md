# FakeCatcher Inference Code

This code was written by adapting the contents of paper [FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals] https://arxiv.org/abs/1901.02212 

## How to Start
These commands should be run after you execute `setup.sh`.

```bash
conda activate fakecatcher
export PYTHONPATH=/25th-conference-fakebusters/model/fakecatcher # path to fakecather dir
```
## How to Use (Dataset)

https://github.com/ondyari/FaceForensics

Generate a CSV file that includes the data's location and labels.

```bash
cd /25th-conference-fakebusters/model/fakecatcher/data
python fakeforensics.py -b path/to/dataset
```
Then `train_video_list.csv` and `test_video_list.csv` files are generated. 

## How to Use (CNN based Model)

### 1. Train step

#### 1-1. Preprocess PPG-map
```bash
cd /25th-conference-fakebusters/model/fakecatcher
python preprocess_map.py -c model/fakecatcher/utils/config.yaml -l model/fakecatcher/data/ppg_map.log -o model/fakecatcher/data
```

#### 1-2. Train cnn model
```bash
python cnn/train.py -c model/fakecatcher/utils/config.yaml -l model/fakecatcher/data/ppg_cnn.log -i model/fakecatcher/data/ppg_maps.json -o model/fakecatcher/model_state.pth
```

### 2. Inference step
```bash
python cnn/main.py -c model/fakecatcher/utils/config.yaml
```

Now your model is running on the uvicorn!


## How to Use (SVR based Model)

### 1. Train step

#### 1-1. Extract PPG and Features
```bash
cd /25th-conference-fakebusters/model
python fakecatcher/svr/preprocess_feature.py -c fakecatcher/utils/config.yaml -d fakecatcher/data/train_video_list.csv
```

#### 1-2. Train svr model
```bash
python fakecatcher/svr/train.py -f fakecatcher/misc/features.pkl
```
### 2. Inference step
```bash
python fakecatcher/svr/main.py -c fakecatcher/utils/config.yaml
```
Now your model is running on the uvicorn!

## Inference request API
- Request
    - URL: POST `https://{your server url}/upload-video/`
    - Content-Type: multipart/form-data
    - Form-Data field: 
        - `file`: The `.mp4` video file to be uploaded.

    - request example
        ```bash
        curl -X POST https://{your server url}/upload-video/ \
        -H "Content-Type: multipart/form-data" \
        -F "file=@{file path}"
        ```
- Response
    - format: application/json
    - Success response example:
        ```json
        {
        "message": "Video uploaded successfully!",
        "file_name": "KakaoTalk_20241219_172238270.mp4",
        "file_path": "uploaded_videos/KakaoTalk_20241219_172238270.mp4",
        "score": "0.8333333333333334"
        }
        ```
        - `message`: Upload result message
        - `file_name`: video file name
        - `file_path`: saved video path in server
        - `score`:  real video score(The `score` ranges from 0 to 1, where values closer to **1** indicate a real video, and values closer to **0** indicate a fake video.)
    

    - Fail response example:
        - Returned when the server fails to save the uploaded file.
        - status code: 500
        - Returned when an error occurs during saving video or inference.
      ```json
        {
          "detail": "File save failed: [error message]"
        }
      ```
      
      ```json
        {
          "detail": "Prediction failed: [error message]"
        }
      ```

        
