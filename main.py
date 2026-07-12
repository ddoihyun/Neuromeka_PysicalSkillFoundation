"""
main.py
--------------------------------------------------------------------------
FoundationStereo (fs) vs Fast-FoundationStereo (ffs) 정량 비교 스크립트

사용법:
    (base) $ python main.py                 # 순차 실행 (기본, 벤치마크 정확도 권장)
    (base) $ python main.py --parallel      # 두 conda env를 동시에 실행 (2터미널처럼)
    (base) $ python main.py --repeat 10     # 각 모델 forward를 10회씩 측정해 평균/표준편차 산출

주의:
  - 이 main.py 자체는 어떤 conda 환경에서 실행해도 됩니다 (subprocess로 다른 env를 호출하기 때문).
    단, `conda`가 PATH에 잡혀 있어야 하고, `conda run` 이 동작해야 합니다.
  - CONFIG 아래의 env 이름 / 경로들은 본인 프로젝트 구조에 맞게 반드시 수정하세요.
--------------------------------------------------------------------------
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ============================================================
# 프로젝트 환경에 맞게 수정하세요
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

CONFIG = {
    "fs": {
        "env_name": "foundation_stereo",  # conda env 이름 (오타 주의: foundaton_stereo 아님)
        "script": os.path.join(PROJECT_ROOT, "FoundationStereo/scripts/run_demo.py"),
        "out_dir": os.path.join(PROJECT_ROOT, "outputs/FoundationStereo"),
        "extra_args": [
            "--left_file", os.path.join(PROJECT_ROOT, "dataset/left.png"),
            "--right_file", os.path.join(PROJECT_ROOT, "dataset/right.png"),
            "--intrinsic_file", os.path.join(PROJECT_ROOT, "dataset/K.txt"),
            "--ckpt_dir", os.path.join(PROJECT_ROOT, "FoundationStereo/weights/23-51-11/model_best_bp2.pth"),
        ],
    },
    "ffs": {
        "env_name": "ffs",
        "script": os.path.join(PROJECT_ROOT, "Fast-FoundationStereo/scripts/run_demo.py"),
        "out_dir": os.path.join(PROJECT_ROOT, "outputs/Fast-FoundationStereo"),
        "extra_args": [
            "--left_file", os.path.join(PROJECT_ROOT, "dataset/left.png"),
            "--right_file", os.path.join(PROJECT_ROOT, "dataset/right.png"),
            "--intrinsic_file", os.path.join(PROJECT_ROOT, "dataset/K.txt"),
            "--model_dir", os.path.join(PROJECT_ROOT, "Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth"),
        ],
    },
}

COMPARISON_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/comparison")


def build_cmd(key: str, repeat: int, warmup: int) -> list:
    cfg = CONFIG[key]
    cmd = [
        "conda", "run", "-n", cfg["env_name"], "--no-capture-output",
        "python", cfg["script"],
        "--out_dir", cfg["out_dir"],
        "--headless", "1",
        "--repeat", str(repeat),
        "--warmup", str(warmup),
        *cfg["extra_args"],
    ]
    return cmd


def clean_out_dir(key: str):
    """이전 실행의 잔여 산출물(metrics.json, depth_meter.npy, *.ply 등)을 지워서
    이번 실행이 실패했을 때 옛날 결과로 잘못 비교되는 것을 방지."""
    cfg = CONFIG[key]
    if os.path.exists(cfg["out_dir"]):
        shutil.rmtree(cfg["out_dir"])
    os.makedirs(cfg["out_dir"], exist_ok=True)


def run_model(key: str, repeat: int, warmup: int) -> float:
    cfg = CONFIG[key]
    clean_out_dir(key)
    cmd = build_cmd(key, repeat, warmup)
    print(f"\n[{key}] 실행 명령:\n  {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    wall_time = time.time() - t0
    if result.returncode != 0:
        print(f"[{key}] !! 실행 실패 (returncode={result.returncode})")
    return wall_time


def run_sequential(repeat: int, warmup: int) -> dict:
    wall_times = {}
    for key in ["fs", "ffs"]:
        wall_times[key] = run_model(key, repeat, warmup)
    return wall_times


def run_parallel(repeat: int, warmup: int) -> dict:
    """
    두 모델을 동시에 실행 (2개 터미널처럼). 단, 같은 GPU를 공유하므로
    서로 자원 경쟁이 생겨 순수 성능 측정치로는 부정확할 수 있습니다.
    (참고용 / '동시에 돌렸을 때 실사용 체감 속도' 확인용)
    """
    procs = {}
    t0s = {}
    for key in ["fs", "ffs"]:
        cfg = CONFIG[key]
        clean_out_dir(key)
        cmd = build_cmd(key, repeat, warmup)
        print(f"\n[{key}] 실행 명령 (병렬):\n  {' '.join(cmd)}\n")
        t0s[key] = time.time()
        procs[key] = subprocess.Popen(cmd)

    wall_times = {}
    for key, p in procs.items():
        p.wait()
        wall_times[key] = time.time() - t0s[key]
        if p.returncode != 0:
            print(f"[{key}] !! 실행 실패 (returncode={p.returncode})")
    return wall_times


def load_metrics(key: str) -> dict:
    path = os.path.join(CONFIG[key]["out_dir"], "metrics.json")
    if not os.path.exists(path):
        print(f"[경고] {path} 가 없습니다. 스크립트가 정상적으로 끝났는지 확인하세요.")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_depth(key: str):
    path = os.path.join(CONFIG[key]["out_dir"], "depth_meter.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def load_pointcloud(key: str):
    if o3d is None:
        return None
    path = os.path.join(CONFIG[key]["out_dir"], "cloud_denoise.ply")
    if not os.path.exists(path):
        path = os.path.join(CONFIG[key]["out_dir"], "cloud.ply")
    if not os.path.exists(path):
        return None
    return o3d.io.read_point_cloud(path)


def compare_depth(depth_fs: np.ndarray, depth_ffs: np.ndarray) -> dict:
    """두 depth map 간의 픽셀 단위 차이 (GT가 없으므로 '두 모델 간 일치도' 지표)"""
    if depth_fs is None or depth_ffs is None:
        return {}
    if depth_fs.shape != depth_ffs.shape:
        print(f"[경고] depth shape 불일치: fs={depth_fs.shape}, ffs={depth_ffs.shape} -> 비교 생략")
        return {}

    valid = np.isfinite(depth_fs) & np.isfinite(depth_ffs) & (depth_fs > 0) & (depth_ffs > 0)
    if valid.sum() == 0:
        return {}

    diff = np.abs(depth_fs[valid] - depth_ffs[valid])
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    bad_2cm = float((diff > 0.02).mean())
    bad_5cm = float((diff > 0.05).mean())
    return {
        "valid_pixel_ratio": float(valid.mean()),
        "depth_MAE_m": mae,
        "depth_RMSE_m": rmse,
        "bad_pixel_ratio_2cm": bad_2cm,
        "bad_pixel_ratio_5cm": bad_5cm,
    }


def compare_pointcloud(pcd_fs, pcd_ffs) -> dict:
    """ICP로 두 포인트클라우드를 정렬해 기하학적 일치도를 측정"""
    if o3d is None or pcd_fs is None or pcd_ffs is None:
        return {}
    if len(pcd_fs.points) == 0 or len(pcd_ffs.points) == 0:
        return {}

    voxel = 0.01  # 1cm 다운샘플 (연산 속도용)
    src = pcd_fs.voxel_down_sample(voxel)
    dst = pcd_ffs.voxel_down_sample(voxel)

    threshold = 0.05  # 5cm
    reg = o3d.pipelines.registration.registration_icp(
        src, dst, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return {
        "icp_fitness": float(reg.fitness),          # 겹치는 비율 (1에 가까울수록 형상 일치)
        "icp_inlier_rmse_m": float(reg.inlier_rmse), # 겹치는 부분의 평균 오차
    }


def print_table(rows: list):
    keys = sorted({k for r in rows for k in r if k != "model"})
    header = ["metric"] + [r.get("model", "?") for r in rows]
    print("\n" + " | ".join(f"{h:>28}" for h in header))
    print("-" * (30 * len(header)))
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k, "-")
            if isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v))
        print(" | ".join(f"{c:>28}" for c in [k] + vals))


def make_plots(metrics_fs: dict, metrics_ffs: dict, depth_cmp: dict, pc_cmp: dict):
    if plt is None:
        print("[경고] matplotlib이 없어 그래프 생성을 건너뜁니다. `pip install matplotlib`")
        return

    os.makedirs(COMPARISON_OUT_DIR, exist_ok=True)
    labels = ["FoundationStereo", "Fast-FoundationStereo"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    # 1) inference time
    times = [metrics_fs.get("inference_time_sec_mean", 0), metrics_ffs.get("inference_time_sec_mean", 0)]
    stds = [metrics_fs.get("inference_time_sec_std", 0), metrics_ffs.get("inference_time_sec_std", 0)]
    axes[0].bar(labels, times, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Inference Time (s)")
    axes[0].set_ylabel("seconds")

    # 2) GPU memory: 전체 VRAM 대비 %
    gpu_mem_pct = [metrics_fs.get("gpu_mem_percent_of_total", 0), metrics_ffs.get("gpu_mem_percent_of_total", 0)]
    axes[1].bar(labels, gpu_mem_pct, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("GPU Memory Usage (% of total VRAM)")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)

    # 3) GPU core utilization (nvidia-smi Volatile GPU-Util과 동일 개념)
    gpu_util = [metrics_fs.get("gpu_util_percent_mean", 0) or 0, metrics_ffs.get("gpu_util_percent_mean", 0) or 0]
    axes[2].bar(labels, gpu_util, color=["#4C72B0", "#DD8452"])
    axes[2].set_title("GPU Core Utilization (mean, %)")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(0, 100)

    # 4) CPU: 전체 코어 대비 %
    cpu_pct = [metrics_fs.get("cpu_percent_mean_of_total", 0) or 0, metrics_ffs.get("cpu_percent_mean_of_total", 0) or 0]
    axes[3].bar(labels, cpu_pct, color=["#4C72B0", "#DD8452"])
    axes[3].set_title("CPU Usage (% of all cores)")
    axes[3].set_ylabel("%")
    axes[3].set_ylim(0, 100)

    # 5) RAM: 전체 시스템 RAM 대비 %
    ram_pct = [metrics_fs.get("ram_percent_of_total", 0), metrics_ffs.get("ram_percent_of_total", 0)]
    axes[4].bar(labels, ram_pct, color=["#4C72B0", "#DD8452"])
    axes[4].set_title("RAM Usage (% of total system RAM)")
    axes[4].set_ylabel("%")
    axes[4].set_ylim(0, 100)

    # 6) point count
    pts = [metrics_fs.get("num_points_denoised", metrics_fs.get("num_points_raw", 0)),
           metrics_ffs.get("num_points_denoised", metrics_ffs.get("num_points_raw", 0))]
    axes[5].bar(labels, pts, color=["#4C72B0", "#DD8452"])
    axes[5].set_title("Point Cloud Count (denoised)")
    axes[5].set_ylabel("# points")

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    out_path = os.path.join(COMPARISON_OUT_DIR, "comparison_bars.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n비교 그래프 저장됨: {out_path}")

    # depth / pointcloud 일치도 별도 텍스트로 저장
    summary_path = os.path.join(COMPARISON_OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "fs": metrics_fs,
            "ffs": metrics_ffs,
            "depth_comparison": depth_cmp,
            "pointcloud_comparison": pc_cmp,
        }, f, indent=2)
    print(f"전체 결과(JSON) 저장됨: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true",
                         help="두 모델을 동시에 실행 (기본은 순차 실행 - 벤치마크 정확도 권장)")
    parser.add_argument("--repeat", type=int, default=5,
                         help="각 모델 forward 반복 측정 횟수 (평균/표준편차용)")
    parser.add_argument("--warmup", type=int, default=2,
                         help="측정 전 워밍업 forward 횟수")
    args = parser.parse_args()

    print("=" * 70)
    print("FoundationStereo vs Fast-FoundationStereo 정량 비교 시작")
    print("=" * 70)

    if args.parallel:
        wall_times = run_parallel(args.repeat, args.warmup)
    else:
        wall_times = run_sequential(args.repeat, args.warmup)

    metrics_fs = load_metrics("fs")
    metrics_ffs = load_metrics("ffs")

    metrics_fs.setdefault("model", "FoundationStereo")
    metrics_ffs.setdefault("model", "Fast-FoundationStereo")

    for key, wt in wall_times.items():
        target = metrics_fs if key == "fs" else metrics_ffs
        target["process_wall_time_sec"] = wt

    fs_ok = os.path.exists(os.path.join(CONFIG["fs"]["out_dir"], "metrics.json"))
    ffs_ok = os.path.exists(os.path.join(CONFIG["ffs"]["out_dir"], "metrics.json"))
    if not fs_ok:
        print("\n[fs] 실행이 실패한 것으로 보입니다 (metrics.json 없음). 위 에러 로그를 확인하세요.")
    if not ffs_ok:
        print("[ffs] 실행이 실패한 것으로 보입니다 (metrics.json 없음). 위 에러 로그를 확인하세요.")

    print_table([metrics_fs, metrics_ffs])

    if not (fs_ok and ffs_ok):
        print("\n두 모델 중 하나 이상이 실패했으므로, depth/point cloud 비교는 건너뜁니다.")
        return

    depth_fs = load_depth("fs")
    depth_ffs = load_depth("ffs")
    depth_cmp = compare_depth(depth_fs, depth_ffs)
    if depth_cmp:
        print("\n[Depth 비교 (fs vs ffs, GT 없이 상호 일치도)]")
        for k, v in depth_cmp.items():
            print(f"  {k}: {v}")

    pcd_fs = load_pointcloud("fs")
    pcd_ffs = load_pointcloud("ffs")
    pc_cmp = compare_pointcloud(pcd_fs, pcd_ffs)
    if pc_cmp:
        print("\n[Point Cloud 정합(ICP) 비교]")
        for k, v in pc_cmp.items():
            print(f"  {k}: {v}")

    make_plots(metrics_fs, metrics_ffs, depth_cmp, pc_cmp)


if __name__ == "__main__":
    main()