import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import vil2.utils.mp_utils as mp_utils
import natsort
import tqdm
import argparse


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(root_dir, "models", "hand_landmarker.task")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=f"{root_dir}/test_data")
    parser.add_argument("--data_name", type=str, default="exp_0")
    args = parser.parse_args()
    
    color_folder = os.path.join(args.data_folder, args.data_name, "color")
    color_list = os.listdir(color_folder)
    color_list = natsort.natsorted(color_list)
    hand_folder = os.path.join(args.data_folder, args.data_name, "hand")
    os.makedirs(hand_folder, exist_ok=True)

    # Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)


    for color_image_file in tqdm.tqdm(color_list):
        full_color_image_file = os.path.join(args.data_folder, args.data_name, "color", color_image_file)
        image = mp.Image.create_from_file(full_color_image_file)

        # Detect hand landmarks from the input image.
        detection_result = detector.detect(image)

        # Process the classification result. In this case, visualize it.
        annotated_image = mp_utils.draw_landmarks_on_image(image.numpy_view(), detection_result)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(hand_folder, color_image_file), annotated_image)
