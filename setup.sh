#!/bin/bash
set -e

echo "NLP 테스트 환경 구축을 시작합니다 (All in /data/ephemeral)..."

# ==========================================
# 0. 핵심: 모든 저장소를 /data/ephemeral로 강제 지정
# ==========================================
echo "[0/6] 작업 공간 및 캐시 경로 설정..."

# 메인 작업 디렉토리 생성
WORK_DIR="/data/ephemeral/nlp_workspace"
mkdir -p "$WORK_DIR"
chmod 777 "$WORK_DIR"

# 임시 디렉토리도 ephemeral로!
export TMPDIR="/data/ephemeral/tmp"
mkdir -p "$TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# 캐시 디렉토리 설정 (Root 용량 부족 방지)
export XDG_CACHE_HOME="/data/ephemeral/.cache"
export PIP_CACHE_DIR="/data/ephemeral/.cache/pip"
export UV_CACHE_DIR="/data/ephemeral/.cache/uv"
export HF_HOME="/data/ephemeral/.cache/huggingface"

mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$HF_HOME"

# 작업 공간으로 이동 (이제부터 모든 파일은 여기에 생김)
cd "$WORK_DIR"
echo "현재 작업 위치: $(pwd)"

# 타임존 설정
echo "타임존을 Asia/Seoul로 설정합니다..."
export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Seoul
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
echo $TZ | tee /etc/timezone

# ==========================================
# 1. 시스템 패키지 설치
# ==========================================
echo "📦 [1/6] 시스템 패키지 업데이트 및 설치..."
apt-get update
apt-get install -y tzdata vim wget build-essential cmake

# 설치 후 정리
apt-get clean
rm -rf /var/cache/apt/archives/*
sync
sleep 2

# ==========================================
# 2. CUDA 설치 (설치 파일 및 경로 모두 ephemeral)
# ==========================================
if [ ! -d "/data/ephemeral/cuda-12.2" ]; then
    echo " [2/6] CUDA 12.2 설치 중..."
    
    # 다운로드 (ephemeral에 다운됨)
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
    chmod +x cuda_12.2.0_535.54.03_linux.run
    
    # 설치 (Toolkit 경로를 ephemeral로 지정)
    sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit --toolkitpath=/data/ephemeral/cuda-12.2
    
    # 심볼릭 링크
    ln -sf /data/ephemeral/cuda-12.2 /usr/local/cuda
    
    # 설치 파일 삭제 및 정리
    rm -f cuda_12.2.0_535.54.03_linux.run
    sync
    sleep 3
    echo "CUDA 설치 완료 및 정리됨"
else
    echo "CUDA가 이미 설치되어 있습니다. 스킵합니다."
fi

# ==========================================
# 3. 환경변수 설정
# ==========================================
echo "[3/6] 환경변수 설정 중..."

# 캐시 경로 영구 등록
if ! grep -q "XDG_CACHE_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'CACHE_EOF'
# Ephemeral 캐시 경로
export TMPDIR="/data/ephemeral/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XDG_CACHE_HOME="/data/ephemeral/.cache"
export PIP_CACHE_DIR="/data/ephemeral/.cache/pip"
export UV_CACHE_DIR="/data/ephemeral/.cache/uv"
export HF_HOME="/data/ephemeral/.cache/huggingface"
CACHE_EOF
fi

# CUDA 경로 영구 등록
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'CUDA_EOF'
# CUDA 환경변수
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
CUDA_EOF
fi

# 현재 세션 적용
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

# ==========================================
# 4. uv 설치 및 프로젝트 초기화
# ==========================================
echo "[4/6] Python 3.12 & uv 초기화..."
pip install uv

# 현재 폴더(/data/ephemeral/nlp_workspace)에 초기화
uv init --python 3.12.1 . 
uv sync

sync
sleep 2

# ==========================================
# 5. 라이브러리 설치
# ==========================================
echo "[5/6] 라이브러리 설치 (GPU 가속 포함)..."
source .venv/bin/activate

# requirements.txt가 없으면 생성
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt 파일을 생성합니다..."
    cat <<REQ_EOF > requirements.txt
llama-index
langchain
langchain-community
adalflow[ollama]
mlflow
llama-cpp-python[server]
unsloth
unsloth_zoo
transformers
datasets
sentence-transformers
jedi>=0.16
autogen
autogen-agentchat
autogen-ext[openai]
llama-index-llms-llama-cpp
llama-index-llms-openai
llama-index-llms-upstage
llama-index-embeddings-huggingface
llama-index-embeddings-upstage
llama-index-retrievers-bm25
llama-index-readers-wikipedia
llama-index-readers-file
llama-index-graph-stores-neo4j
llama-index-vector-stores-neo4jvector
ollama
neo4j
SPARQLWrapper
wikipedia
wikipedia-api
REQ_EOF
fi

CMAKE_ARGS="-DGGML_CUDA=on" uv pip install -r requirements.txt

# 중간 정리
sync
sleep 2

uv pip install unsloth unsloth_zoo --upgrade

# 최종 정리
sync
sleep 2

# ==========================================
# 6. 데이터 다운로드 및 압축 해제
# ==========================================
echo " [6/6] 데이터 다운로드 및 압축 해제..."

# data.tar.gz가 없으면 다운로드
if [ ! -f "data.tar.gz" ]; then
    echo "데이터 파일 다운로드 중..."
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000270/data/data.tar.gz
else
    echo " data.tar.gz 파일이 이미 존재합니다."
fi

# data 폴더가 없으면 압축 해제
if [ ! -d "data" ]; then
    echo "압축 해제 중..."
    tar -zxvf data.tar.gz
    echo " 압축 해제 완료"
else
    echo " data 폴더가 이미 존재합니다. 압축 해제를 스킵합니다."
fi

# 정리
sync

echo ""
echo " 모든 설정이 완료되었습니다!"
echo " 설치 위치: /data/ephemeral/nlp_workspace"
echo " 데이터 위치: /data/ephemeral/nlp_workspace/data"
echo " 디스크 사용량:"
df -h | grep -E "Filesystem|/data|/$"
echo ""
echo " 다음 명령어로 이동 및 활성화하세요:"
echo "   cd /data/ephemeral/nlp_workspace"
echo "   source .venv/bin/activate"