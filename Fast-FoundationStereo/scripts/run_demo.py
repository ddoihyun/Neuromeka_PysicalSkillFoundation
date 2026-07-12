# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# ---------------------------------------------------------------------------
# [PATCHED] added:
#   --headless : skip cv2 / open3d blocking visualization windows
#   timing + GPU peak-memory measurement
#   metrics.json output for quantitative comparison (used by main.py)
# ---------------------------------------------------------------------------

import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio, logging, yaml, time, json, threading, psutil
import numpy as np
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import cv2


class CpuSampler:
  """forward pass가 실행되는 동안 백그라운드 스레드에서 프로세스 CPU 사용률(%)을 주기적으로 샘플링."""
  def __init__(self, interval=0.05):
    self.proc = psutil.Process(os.getpid())
    self.interval = interval
    self.samples = []
    self._stop = threading.Event()
    self.proc.cpu_percent()  # 첫 호출은 항상 0.0을 반환하므로 미리 소모

  def _run(self):
    while not self._stop.is_set():
      self.samples.append(self.proc.cpu_percent(interval=self.interval))

  def start(self):
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    return self

  def stop(self):
    self._stop.set()
    self._thread.join()
    return self.samples


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth', type=str)
  parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default='/home/nrmk/opt/Fast-FoundationStereo/debug/stereo_output', type=str)
  parser.add_argument('--remove_invisible', default=1, type=int)
  parser.add_argument('--denoise_cloud', default=1, type=int)
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--scale', default=1, type=float)
  parser.add_argument('--hiera', default=0, type=int)
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
  parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
  parser.add_argument('--zfar', type=float, default=100, help="max depth to include in point cloud")
  parser.add_argument('--headless', type=int, default=0, help='1이면 cv2/open3d 시각화 창을 띄우지 않고 바로 종료 (벤치마크 자동화용)')
  parser.add_argument('--warmup', type=int, default=1, help='측정 전 워밍업 forward 횟수 (첫 실행 컴파일 시간 제외용)')
  parser.add_argument('--repeat', type=int, default=1, help='측정을 위해 forward를 반복할 횟수 (평균/표준편차 계산용)')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')

  with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
    cfg:dict = yaml.safe_load(ff)
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
  model.args.valid_iters = args.valid_iters
  model.args.max_disp = args.max_disp

  model.cuda().eval()

  scale = args.scale

  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  if len(img0.shape)==2:
    img0 = np.tile(img0[...,None], (1,1,3))
    img1 = np.tile(img1[...,None], (1,1,3))
  img0 = img0[...,:3]
  img1 = img1[...,:3]
  H,W = img0.shape[:2]

  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  img1_ori = img1.copy()
  logging.info(f"img0: {img0.shape}")

  img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
  img0_t, img1_t = padder.pad(img0_t, img1_t)

  logging.info(f"Start forward, 1st time run can be slow due to compilation")

  # ---------------- [PATCH] warmup (첫 실행 compile time 제외) ----------------
  for _ in range(args.warmup):
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
      if not args.hiera:
        _ = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
      else:
        _ = model.run_hierachical(img0_t, img1_t, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    torch.cuda.synchronize()

  # ---------------- [PATCH] timed forward passes (+ CPU 사용률 샘플링) ----------------
  torch.cuda.reset_peak_memory_stats()
  cpu_sampler = CpuSampler(interval=0.05).start()
  times = []
  for _ in range(args.repeat):
    t0 = time.time()
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
      if not args.hiera:
        disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
      else:
        disp = model.run_hierachical(img0_t, img1_t, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
  cpu_samples = cpu_sampler.stop()
  logging.info("forward done")

  infer_time_mean = float(np.mean(times))
  infer_time_std = float(np.std(times))
  gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
  cpu_percent_mean = float(np.mean(cpu_samples)) if cpu_samples else None
  cpu_percent_max = float(np.max(cpu_samples)) if cpu_samples else None
  ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2

  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W).clip(0, None)

  cmap = None
  min_val = None
  max_val = None
  vis = vis_disparity(disp, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
  vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)

  # ---------------- [PATCH] headless-guarded display ----------------
  if not args.headless:
    s = 1280/vis.shape[1]
    resized_vis = cv2.resize(vis, (int(vis.shape[1]*s), int(vis.shape[0]*s)))
    cv2.imshow('disp', resized_vis[:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  metrics = {
    'model': 'Fast-FoundationStereo',
    'image_size_hw': [int(H), int(W)],
    'valid_iters': int(args.valid_iters),
    'inference_time_sec_mean': infer_time_mean,
    'inference_time_sec_std': infer_time_std,
    'inference_time_sec_all': times,
    'gpu_peak_mem_mb': float(gpu_peak_mem_mb),
    'cpu_percent_mean': cpu_percent_mean,   # 100% = 코어 1개 풀사용. 400%면 4코어 풀사용 등 (htop 관례와 동일)
    'cpu_percent_max': cpu_percent_max,
    'ram_mb': float(ram_mb),
  }

  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.zfar)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
    logging.info(f"PCL saved to {args.out_dir}")
    metrics['num_points_raw'] = int(len(np.asarray(pcd.points)))

    if args.denoise_cloud:
      logging.info("[Optional step] denoise point cloud...")
      pcd = pcd.voxel_down_sample(voxel_size=0.001)
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      metrics['num_points_denoised'] = int(len(np.asarray(inlier_cloud.points)))
      metrics['outlier_removed_ratio'] = 1.0 - metrics['num_points_denoised'] / max(metrics['num_points_raw'], 1)
      pcd = inlier_cloud

    # ---------------- [PATCH] headless-guarded o3d viewer ----------------
    if not args.headless:
      logging.info("Visualizing point cloud. Press ESC to exit.")
      vis3d = o3d.visualization.Visualizer()
      vis3d.create_window()
      vis3d.add_geometry(pcd)
      vis3d.get_render_option().point_size = 1.0
      vis3d.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
      ctr = vis3d.get_view_control()
      ctr.set_front([0, 0, -1])
      id = np.asarray(pcd.points)[:,2].argmin()
      ctr.set_lookat(np.asarray(pcd.points)[id])
      ctr.set_up([0, -1, 0])
      vis3d.run()
      vis3d.destroy_window()

  # ---------------- [PATCH] save metrics.json ----------------
  with open(f'{args.out_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
  logging.info(f"Metrics saved to {args.out_dir}/metrics.json")