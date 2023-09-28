"""Recording a sequence in colmap format
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import argparse


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=f"{root_dir}/test_data")
    parser.add_argument("--data_name", type=str, default="exp_0")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.data_folder, args.data_name), exist_ok=True)
    # clean the folder with verification
    print(f"Are you sure to clean the folder {os.path.join(args.data_folder, args.data_name)}? (y/n)")
    answer = input()
    if answer == "y":
        os.system(f"rm -rf {os.path.join(args.data_folder, args.data_name)}/*")
    # The recording scripts are from realsense SDK examples
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # Get the intrinsics of color
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    color_folder = os.path.join(args.data_folder, args.data_name, "color")
    depth_folder = os.path.join(args.data_folder, args.data_name, "depth")
    os.makedirs(color_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    counter = 0
    frame_jump = 100  # jump the first 100 frames because it is dark
    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            counter += 1
            if counter < frame_jump:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(os.path.join(color_folder, f"{(counter - frame_jump)}.png"), color_image)
            cv2.imwrite(os.path.join(depth_folder, f"{(counter - frame_jump)}.png"), depth_image)
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

    # Save the intrisic.txt (for bundle-tracking)
    with open(os.path.join(args.data_folder, args.data_name, "intrinsics.txt"), "w") as f:
        f.writelines([f"fx: {depth_intrinsics.fx}\n", f"fy: {depth_intrinsics.fy}\n", f"cx: {depth_intrinsics.ppx}\n", f"cy: {depth_intrinsics.ppy}\n"])
    # Save the cam_info.json (for other process)
    cam_info = {"intrinsic": [[depth_intrinsics.fx, 0, depth_intrinsics.ppx], [0, depth_intrinsics.fy, depth_intrinsics.ppy], [0, 0, 1]], "depth_scale": depth_scale}
    json.dump(cam_info, open(os.path.join(args.data_folder, args.data_name, "cam_info.json"), "w"))