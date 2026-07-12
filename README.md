# FoundationStereo vs Fast-FoundationStereo 정량 비교

`FoundationStereo`(정밀/고정밀 모델)와 `Fast-FoundationStereo`(경량/고속 모델), 두 스테레오 매칭 모델을
**같은 입력 이미지 쌍**으로 돌려서 속도·자원 사용량·출력 품질을 자동으로 비교하는 도구입니다.

---

## 1. 폴더 구조

```
Stereo/
├── main.py                          # 비교 실행 orchestrator (이 프로젝트의 핵심)
├── dataset/
│   ├── left.png / right.png         # 스테레오 입력 이미지
│   └── K.txt                        # 카메라 intrinsic + baseline
├── FoundationStereo/
│   └── scripts/run_demo.py          # ★ patch된 버전으로 교체 필요
├── Fast-FoundationStereo/
│   └── scripts/run_demo.py          # ★ patch된 버전으로 교체 필요
└── outputs/
    ├── FoundationStereo/            # fs 실행 결과 (main.py가 자동 생성/삭제)
    ├── Fast-FoundationStereo/       # ffs 실행 결과 (main.py가 자동 생성/삭제)
    └── comparison/                  # 최종 비교 결과 (그래프 + 요약 json)
```

---

## 2. 사전 준비 (conda 환경 3개)

| env 이름 | 용도 | 필요 패키지 |
|---|---|---|
| `foundation_stereo` | FoundationStereo 모델 실행 | 기존 요구사항 + `psutil`, `pynvml` |
| `ffs` | Fast-FoundationStereo 모델 실행 | 기존 요구사항 + `psutil`, `pynvml` |
| `stereo-compare` (또는 위 둘 중 하나 재사용) | `main.py` 실행 전용 | `numpy`, `matplotlib`, `open3d`(pip) |

```bash
# 모델 실행 env 2개에 모니터링용 패키지 추가
conda activate foundation_stereo && pip install psutil pynvml
conda activate ffs             && pip install psutil pynvml

# main.py 실행용 가벼운 env (신규 생성 시)
conda create -n stereo-compare python=3.10 numpy matplotlib -y
conda activate stereo-compare
pip install open3d   # conda defaults 채널엔 없으므로 반드시 pip로 설치
```

> **주의**: `open3d`는 conda `defaults` 채널에 없어서 `conda create ... open3d`로 하면 실패합니다.
> 반드시 `python`/`numpy`만 conda로 만들고 `open3d`는 `pip install open3d`로 따로 설치하세요.

---

## 3. patch된 `run_demo.py` 적용

원본 `run_demo.py`는 결과를 보려면 `cv2.waitKey(0)` / Open3D 창에서 사람이 직접 키를 눌러야
종료되는 구조라 자동화가 불가능합니다. 그래서 아래 기능을 추가한 patch 버전을 사용합니다.

**패치 내용:**
- `--headless 1` : 시각화 창을 띄우지 않고 바로 종료 (자동화용)
- `--warmup N` : 측정 전 N회 미리 실행 (첫 실행은 CUDA 컴파일 때문에 느리므로 측정에서 제외)
- `--repeat N` : forward를 N회 반복 측정 → 평균/표준편차 계산
- 추론 시간, GPU/CPU/RAM 사용량을 측정해 `metrics.json`으로 저장

**적용:**
```bash
cp run_demo_fs.py  FoundationStereo/scripts/run_demo.py
cp run_demo_ffs.py Fast-FoundationStereo/scripts/run_demo.py
```
> ⚠️ 파일명이 헷갈리기 쉬우니 꼭 `_fs` → `FoundationStereo`, `_ffs` → `Fast-FoundationStereo` 로
> 정확히 매칭해서 복사하세요. (반대로 넣으면 `ImportError` / `unrecognized arguments` 에러가 납니다)

---

## 4. `main.py` 실행

`main.py`는 **base(또는 stereo-compare) 환경에서 한 번만** 실행하면 됩니다.
내부적으로 `conda run -n <env>`를 이용해 각 모델을 해당 conda 환경에서 자동으로 호출하기 때문에,
사용자가 직접 conda activate를 두 번 할 필요가 없습니다.

```bash
conda activate stereo-compare
python main.py --repeat 10 --warmup 2
```

### 주요 옵션
| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--repeat` | 5 | 각 모델 forward를 몇 번 반복 측정할지 (많을수록 평균값이 안정적) |
| `--warmup` | 2 | 측정 전 워밍업 횟수 |
| `--parallel` | (off) | 두 모델을 동시에 실행 (기본은 순차 실행 — GPU 자원 경쟁을 피해 더 정확한 벤치마크가 됨) |

### 실행 시 하는 일
1. `outputs/FoundationStereo`, `outputs/Fast-FoundationStereo` 폴더를 **비우고** 새로 생성
   (이전 실행의 옛날 결과가 남아 잘못 비교되는 것을 방지)
2. `conda run -n foundation_stereo python FoundationStereo/scripts/run_demo.py ...` 실행
3. `conda run -n ffs python Fast-FoundationStereo/scripts/run_demo.py ...` 실행
4. 각 모델이 생성한 `metrics.json`, `depth_meter.npy`, `cloud_denoise.ply`를 읽어서 비교
5. 비교 결과를 콘솔에 표로 출력하고, `outputs/comparison/`에 그래프(png)와 요약(json) 저장

---

## 5. 결과 해석 가이드

### (1) `outputs/{FoundationStereo,Fast-FoundationStereo}/metrics.json`
모델 1개당 자동 생성되는 개별 성능 기록.

| 필드 | 의미 |
|---|---|
| `inference_time_sec_mean/std` | forward pass 평균 시간 / 표준편차 (초) |
| `gpu_peak_mem_mb`, `gpu_mem_percent_of_total` | GPU 메모리 사용량 (절대값 / 전체 VRAM 대비 %) |
| `gpu_util_percent_mean/max` | GPU 코어 사용률 (`nvidia-smi`의 Volatile GPU-Util과 동일 개념) |
| `cpu_percent_mean_of_total` | CPU 사용률 — **시스템 전체 코어** 대비 % |
| `ram_percent_of_total` | RAM 사용량 — 시스템 전체 RAM 대비 % |
| `num_points_raw` / `num_points_denoised` | 포인트클라우드 원본/노이즈 제거 후 포인트 개수 |
| `outlier_removed_ratio` | 노이즈로 판정되어 제거된 포인트 비율 |

### (2) `outputs/comparison/summary.json`
두 모델의 `metrics.json` + 아래 비교 지표를 합친 최종 결과 파일.

**`depth_comparison`** (GT 없이, 두 모델 depth map 간 픽셀 단위 상호 일치도)
| 필드 | 의미 |
|---|---|
| `valid_pixel_ratio` | 두 모델 모두 유효 depth를 낸 픽셀 비율 |
| `depth_MAE_m` / `depth_RMSE_m` | 두 모델 depth 차이의 평균절대오차 / RMSE (미터) |
| `bad_pixel_ratio_2cm` / `_5cm` | 차이가 2cm/5cm 넘는 픽셀 비율 |

**`pointcloud_comparison`** (ICP로 두 포인트클라우드를 정합)
| 필드 | 의미 |
|---|---|
| `icp_fitness` | 정합 후 겹치는 포인트 비율 (1에 가까울수록 두 모델의 3D 형상이 일치) |
| `icp_inlier_rmse_m` | 겹치는 부분끼리의 평균 거리 오차 (미터) |

> **주의**: GT(정답) depth가 없으므로 이 비교는 "누가 더 정확한가"가 아니라
> **"두 모델이 서로 얼마나 다른 결과를 내는가"**를 보는 지표입니다.

### (3) `outputs/comparison/comparison_bars.png`
아래 6개 항목을 막대그래프로 시각화:
1. 추론 시간 (초)
2. GPU 메모리 사용률 (전체 VRAM 대비 %)
3. GPU 코어 사용률 (%)
4. CPU 사용률 (전체 코어 대비 %)
5. RAM 사용률 (전체 시스템 RAM 대비 %)
6. 포인트클라우드 개수 (노이즈 제거 후)

---

## 6. 예시 결과 (참고용 — 실제 값은 하드웨어/데이터셋에 따라 다름)

```
                      metric |             FoundationStereo |        Fast-FoundationStereo
------------------------------------------------------------------------------------------
      inference_time_sec_mean |                       0.9928 |                       0.0886
               gpu_peak_mem_mb |                    5021.6870 |                     869.8760
                   valid_iters |                           32 |                            8
                  cpu_count 등 |         (시스템 스펙에 따라 다름)

[Depth 비교]
  depth_MAE_m: 0.0248        (평균 2.5cm 차이)
  bad_pixel_ratio_2cm: 0.140 (14%는 2cm 이상 차이)

[Point Cloud 정합(ICP)]
  icp_fitness: 0.967         (96.7%가 5cm 이내로 겹침 → 전체 형상은 거의 동일)
```

**이 예시의 해석:** Fast-FoundationStereo는 FoundationStereo보다 **약 11배 빠르고 GPU 메모리를
약 1/6만 사용**하지만, depth 디테일에서 약 14%의 픽셀은 2cm 이상 차이가 납니다. 다만 전체적인
3D 형상(포인트클라우드)은 96.7% 일치하므로, "큰 구조는 거의 같지만 세부 디테일에서 정확도를
일부 희생한 경량화 모델"이라고 해석할 수 있습니다. (GT가 없어 "어느 쪽이 실제로 더 정확한지"는
별도의 정답 depth 데이터가 있어야 확정할 수 있습니다.)

---

## 7. 자주 발생하는 문제 (Troubleshooting)

| 증상 | 원인 | 해결 |
|---|---|---|
| `Command 'python' not found` | `conda deactivate` 후 시스템에 `python`(3.x) 심볼릭 링크가 없음 (`python3`만 있음) | conda 환경을 activate 하고 실행 (`main.py`는 별도 가벼운 env에서 실행 권장) |
| `PackagesNotFoundInChannelsError: open3d` | conda `defaults` 채널에 `open3d` 패키지가 없음 | `conda create`에서 `open3d` 빼고, 생성 후 `pip install open3d`로 별도 설치 |

---

## 8. 설정 변경이 필요한 곳

`main.py` 상단의 `CONFIG` 딕셔너리에서 아래 값들을 **본인 프로젝트 경로에 맞게** 수정하세요:
- `env_name` : conda 환경 이름
- `ckpt_dir` / `model_dir` : 각 모델의 가중치 파일 경로
- `out_dir` : 결과 저장 위치 (기본값 그대로 써도 무방)