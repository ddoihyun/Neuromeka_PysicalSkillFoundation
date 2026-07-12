# FoundationStereo Demo

FoundationStereo를 이용하여 **Stereo Image → Depth Map → Point Cloud**를 생성하는 예제입니다.

본 프로젝트는 다음 두 가지 방법을 지원합니다.

- **Demo Mode** : 제공된 이미지로 실행
- **RealSense Mode** : Intel RealSense D435 IR Stereo Camera로 직접 촬영

---

# 1. Download Pretrained Model

Pretrained model을 다운로드합니다.

https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf

다운로드 후 아래와 같이 배치합니다.

```text
weights/
└── 23-51-11/
    ├── cfg.yaml
    └── model_best_bp2.pth
```

---

# 2. Dataset Format

FoundationStereo는 아래 형식의 Stereo Dataset을 입력으로 사용합니다.

```text
dataset/
├── left.png
├── right.png
└── K.txt
```

### left.png

Left stereo image

### right.png

Right stereo image

### K.txt

Camera intrinsic matrix와 stereo baseline을 저장합니다.

Format

```text
fx 0 cx
0 fy cy
0 0 1

baseline(m)
```

---

# 3. Demo

기본 제공되는 예제를 실행합니다.

## Activate Environment

```bash
conda activate foundation_stereo
```

## Run

```bash
python scripts/run_demo.py \
    --left_file assets/left.png \
    --right_file assets/right.png \
    --intrinsic_file assets/K.txt \
    --ckpt_dir weights/23-51-11/model_best_bp2.pth \
    --out_dir test_outputs
```

---

# 4. Output

실행이 완료되면 다음 결과가 생성됩니다.

```text
test_outputs/
├── vis.png
├── depth_meter.npy
├── cloud.ply
└── cloud_denoise.ply
```

| File | Description |
|------|-------------|
| vis.png | Depth visualization |
| depth_meter.npy | Depth map (meter) |
| cloud.ply | Raw point cloud |
| cloud_denoise.ply | Denoised point cloud |

Open3D Viewer가 자동으로 실행됩니다.

ESC를 누르면 종료됩니다.

---

# 5. RealSense D435 Capture

Stereo IR 영상을 직접 촬영하는 방법입니다.

## Install SDK

```bash
pip install pyrealsense2
```

---

## Check Camera

```bash
rs-enumerate-devices
```

Example

```text
Intel RealSense D435
USB Type : 3.x
```

---

## Capture Stereo Images

기본 실행

```bash
python capture_d435.py
```

기본적으로

```text
dataset/
├── left.png
├── right.png
└── K.txt
```

가 생성됩니다.

---

## Optional Arguments

### High Resolution

```bash
python capture_d435.py \
    --width 1280 \
    --height 720
```

---

### Change FPS

```bash
python capture_d435.py \
    --fps 15
```

---

### Enable IR Projector

```bash
python capture_d435.py \
    --emitter 1
```

---

### Change Output Directory

```bash
python capture_d435.py \
    --out_dir scene01
```

---

# 6. Run FoundationStereo

촬영한 Stereo Pair를 이용하여 Depth를 생성합니다.

```bash
python scripts/run_demo.py \
    --left_file dataset/left.png \
    --right_file dataset/right.png \
    --intrinsic_file dataset/K.txt \
    --ckpt_dir weights/23-51-11/model_best_bp2.pth \
    --out_dir test_outputs
```

---

# 7. Workflow

```text
Intel RealSense D435
        │
        ▼
capture_d435.py
        │
        ├─────────────┐
        ▼             ▼

 left.png      right.png

        │
        ▼

      K.txt
(Intrinsic + Baseline)

        │
        ▼

 FoundationStereo

        │
        ▼

 scripts/run_demo.py

        │
        ├───────────────┬────────────────┐
        ▼               ▼                ▼

   Disparity        Depth Map      Point Cloud
                                         │
                                         ▼
                                      cloud.ply
```

---

# Camera Model

FoundationStereo computes depth from stereo disparity using

```text
depth = fx × baseline / disparity
```

where

- **fx** : focal length
- **baseline** : distance between left/right IR cameras
- **disparity** : pixel disparity

---

# Repository Structure

```text
FoundationStereo/
├── core/
├── scripts/
│   └── run_demo.py
├── weights/
│   └── 23-51-11/
│       └── model_best_bp2.pth
├── capture_d435.py
├── README.md
└── ...
```