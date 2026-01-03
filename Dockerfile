FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# 🔧 [추가] apt-get 설치 중 사용자 입력(타임존 질문) 방지
ENV DEBIAN_FRONTEND=noninteractive

# 🔧 [추가] tzdata가 물어보지 않도록 타임존 명시
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    tzdata \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu111

COPY . .

CMD ["bash"]