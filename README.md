# 1. env는 python/numpy/matplotlib만 conda로 생성
conda create -n stereo-compare python=3.10 numpy matplotlib -y

# 2. env 활성화
conda activate stereo-compare

# 3. open3d는 pip로 설치
pip install open3d

# 4. 실행
python main.py --repeat 10 --warmup 2