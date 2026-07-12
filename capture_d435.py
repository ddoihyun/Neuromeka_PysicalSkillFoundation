#!/usr/bin/env python3

"""
Capture stereo IR pair from Intel RealSense D435 for FoundationStereo.

Streams:
    IR1 -> LEFT
    IR2 -> RIGHT

Resolution:
    1280 x 720 @ 30 FPS

Output:
    left.png
    right.png
    K.txt

K.txt format:

fx 0 cx
0 fy cy
0 0 1

baseline(m)

Description:
- D435의 IR1과 IR2를 stereo camera pair로 사용한다.
- IR projector(emitter)는 옵션으로 ON/OFF 설정 가능하다.
- IR1 stream에서 camera intrinsic matrix K를 가져온다.
- IR1과 IR2의 extrinsic parameter를 이용하여 stereo baseline을 계산한다.
- 저장되는 IR 영상은 FoundationStereo 입력 형식에 맞게 grayscale image를 3-channel RGB 형태로 변환한다.
- K.txt에는 camera intrinsic parameter와 stereo baseline을 저장한다.
- 해상도, FPS, 저장 위치, emitter 설정은 command line argument로 변경 가능하다.

Depth calculation:

depth = fx * baseline / disparity

where:
    fx       : focal length
    baseline : distance between IR cameras
    disparity: pixel difference between left and right images
"""
import os
import argparse

import numpy as np
import cv2
import pyrealsense2 as rs


def get_camera_info(profile):

    device = profile.get_device()
    return device.get_info(rs.camera_info.name)


def get_intrinsics(profile):

    ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    intr = ir_profile.get_intrinsics()

    K = np.array(
        [
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ],
        dtype=np.float64
    )
    return K


def get_baseline(profile):

    left = profile.get_stream(rs.stream.infrared, 1)
    right = profile.get_stream(rs.stream.infrared, 2)

    extr = left.get_extrinsics_to(right)

    return abs(extr.translation[0])



def save_K(path, K, baseline):

    with open(path, "w") as f:
        f.write(" ".join([f"{x:.10f}" for x in K.reshape(-1)]) + "\n")
        f.write(f"{baseline:.10f}\n")


def save_image(path, img):

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, bgr)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--emitter", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    # Stereo IR stream
    # IR1 = LEFT
    # IR2 = RIGHT
    
    config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)

    print("Starting RealSense...")
    profile = pipeline.start(config)
    print("Camera:", get_camera_info(profile))

    # Emitter 설정

    sensor = profile.get_device().first_depth_sensor()

    if sensor.supports(rs.option.emitter_enabled):
        sensor.set_option(rs.option.emitter_enabled, args.emitter)

    # Calibration

    K = get_intrinsics(profile)
    baseline = get_baseline(profile)

    print("K=")
    print(K)
    print("Baseline(m):", baseline)


    # Warm-up

    print("Warmup...")

    for _ in range(30):
        pipeline.wait_for_frames()

    print("Press s : save")
    print("Press q : quit")


    try:

        while True:

            frames = pipeline.wait_for_frames()

            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)

            if not left_frame or not right_frame:
                continue

            left = np.asanyarray(left_frame.get_data())
            right = np.asanyarray(right_frame.get_data())

            display = np.hstack([left, right])

            cv2.imshow("IR Left | IR Right", display)

            key = cv2.waitKey(1)

            if key == 27:      # ESC
                break

            elif key != -1:    # Any other key
                print("Saving...")

                save_image(os.path.join(args.out_dir, "left.png"), left)
                save_image(os.path.join(args.out_dir, "right.png"), right)
                save_K(os.path.join(args.out_dir, "K.txt"), K, baseline)

                print("Saved")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()