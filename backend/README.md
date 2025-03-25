# 백엔드
fakebusters에서 제공하는 4가지 모델의 시연을 위한 데모페이지의 백엔드 코드

fastapi, uvicorn 사용

### 최소 스펙
v2 CPU, 16GB RAM (동영상 처리를 위함)

### 백엔드 실행
1. config.py 설정
- 각 모델 서버의 url 설정 필요
- 각 모델 서버가 실행되고 있지 않으면, 프론트에 404 응답이 반한됨

2. 실행 환경 설정
프로젝트 전체 최상단 폴더에 위치한 setup.sh 로 backend conda 가상환경 실행이 필요함
```bash
conda activate backend
```

3. 백엔드 코드 실행
```bash
cd dbust-backend-fastapi
uvicorn main:app --reload                       #localhost, 8000으로 실행
uvicorn main:app --host 0.0.0.0 --port 8000     #호스트와 포트 지정해 실행
```

4. 참고: visual-ppg 실행시
- 해당 api는 ppg 방식의 동작 과정을 보여주기 위해 backend 서버에서 따로 동영상을 처리하므로, 이를 호출할 시 backend 서버의 추가 환경 세팅이 필요함.
```bash
sudo yum install -y mesa-libGL mesa-libGL-devel
conda install -c conda-forge ffmpeg
```
