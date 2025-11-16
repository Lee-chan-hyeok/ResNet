FROM python:3.10-slim

# 시스템 업데이트 + 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# Python 패키지 버전 고정
COPY requirements.txt .

# pip 최신화 및 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

CMD ["python3"]