# MM-Det Inference Code

This code is originally from [SparkleXFantasy/MM-Det](https://github.com/SparkleXFantasy/MM-Det).  
The test code has been revised to support inference, and additional inference functionality has been added.

## How to Use

### 1. Download the weight from
```bash
https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy
```
<img width="409" alt="스크린샷 2025-01-19 오후 2 52 55" src="https://github.com/user-attachments/assets/cdf97cf0-e007-4866-aebc-aa4f7e52bdea" />

### 2. Inference
To perform inference, following the command. 

```bash
python main.py -v video_path
```

if inference went succesfully, you can see
<img width="240" alt="스크린샷 2025-01-19 오후 2 52 03" src="https://github.com/user-attachments/assets/3281e75f-77bf-43f7-aebc-5ed1eb03ab9e" />



### 3. Minimum GPU Memory
GPU must have > 48Gb memory or 24GB * 2
