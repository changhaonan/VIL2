import os
import torch

from typing import Tuple, List

from base64 import b64encode
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import cv2
import pyrealsense2 as rs
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from tqdm import tqdm

import json

import supervision as sv
from torchvision.ops import box_convert

device = "cpu"
if torch.cuda.is_available():
  device = "cuda:0"

class GroundingDinoObjectDetection:
  def __init__(self, model_path = "weights/groundingdino_swint_ogc.path", model_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py", device = "cpu"):
    self.model = load_model(model_config_path, model_path).to(device)

  def annotate_image(self, image_path, text_prompt, box_thresh= 0.35, text_thresh = 0.25, annotated_image = False):
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=self.model,
        image=image,
        caption=text_prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )

    if annotated_image:
      annotated_image = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
      return boxes, logits, phrases, annotated_image
    else:
      return boxes, logits, phrases

  def transform_image(self, array) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(Image.fromarray(array).convert("RGB"), None)
    return image_transformed

  def annotate_image_array(self, image_source, text_prompt, box_thresh= 0.35, text_thresh = 0.25, annotated_image = False):
    image = self.transform_image(image_source)
    boxes, logits, phrases = predict(
        model=self.model,
        image=image,
        caption=text_prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )

    if annotated_image:
      annotated_image = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
      return boxes, logits, phrases, annotated_image
    else:
      return boxes, logits, phrases
    
class TextBasedObjectTracker:
  def __init__(self, tracker_path = 'co-tracker/checkpoints/cotracker_stride_4_wind_8.pth', detector_path = "weights/groundingdino_swint_ogc.path", detector_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py", device = "cpu"):
    self.device = device
    self.detector = GroundingDinoObjectDetection(model_path=detector_path, model_config_path=detector_config_path, device = device)
    self.tracker = CoTrackerPredictor(
        checkpoint=tracker_path
    ).to(self.device)

  def get_mask_from_frame(self, image, text, box_thresh = 0.35, text_thresh= 0.25):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _, _ = self.detector.annotate_image_array(image, text, box_thresh=box_thresh, text_thresh=text_thresh)
    h, w, _ = image.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    mask = np.zeros(image.shape[:2])
    for det in detections.xyxy.astype(np.int64):
      x1, y1, x2, y2 = det
      mask[y1:y2 , x1:x2] = 1
    return mask

  @staticmethod
  def read_bag_file(bag_file_path : str):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file_path, False)
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    color_frames = []
    depth_frames = []
    aligned_depth_frames = []

    try:
      while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_frame.keep()
        color_frame = aligned_frames.get_color_frame()
        color_frame.keep()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_frames.append(color_image)
        depth_frames.append(depth_image)
        aligned_depth_frames.append(aligned_depth_frame)
    except RuntimeError as e:
        if "Frame didn't arrive within" in str(e):
            print("Stream ended")
        else:
            raise e
    finally:
        pipeline.stop()
    return np.array(color_frames), np.array(depth_frames), aligned_depth_frames

  @staticmethod
  def save_points_to_json(points, output_file):
    json_data = {}
    for frame_number, frame_points in enumerate(points):
      frame_points_list = frame_points.tolist()
      json_data[frame_number] = frame_points_list
    json_string = json.dumps(json_data, indent=4)
    with open(output_file, 'w') as json_file:
        json_file.write(json_string)
  
  @staticmethod
  def read_points_from_json(input_file):
    with open(input_file, 'r') as json_file:
        json_data = json.load(json_file)
    num_frames = len(json_data)
    num_points = len(json_data[0]) if num_frames > 0 else 0
    points = np.zeros((num_frames, num_points, 2), dtype=int)
    for frame_number, frame_points_list in json_data.items():
        points[int(frame_number)] = np.array(frame_points_list)
    return points

  def load_bag_file(self, bag_file_path):
    self.color_frames, self.depth_frames, self.aligned_depth_frames = TextBasedObjectTracker.read_bag_file(bag_file_path)
    self.color_images_tensor = torch.from_numpy(self.color_frames).permute(0, 3, 1, 2)[None].float().to(device)

  def process_bag_file(self, text_prompt, box_thresh =0.35, text_thresh =0.25,grid_size=50, output_path="./output.json", save_video = False):
    mask = self.get_mask_from_frame(self.color_frames[0], text_prompt)
    pred_tracks, pred_visibility = self.tracker(self.color_images_tensor, grid_size=grid_size, segm_mask=torch.from_numpy(mask)[None, None])

    if save_video:
      vis = Visualizer(
          save_dir=os.path.dirname(output_path),
          pad_value=100,
          linewidth=2,
      )
      vis.visualize(
          video=self.color_images_tensor,
          tracks=pred_tracks,
          visibility=pred_visibility,
          filename= os.path.splitext(os.path.basename(output_path))[0]);

    video_world_points = []
    for i, aligned_depth_frame in enumerate(self.aligned_depth_frames):
      depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
      frame_points = pred_tracks[0][i]
      world_points = []
      for point in frame_points:
        r, c = [int(x) for x in point]
        depth = aligned_depth_frame.get_distance(r, c)
        world_points.append(rs.rs2_deproject_pixel_to_point(depth_intrin, [r, c], depth))
      video_world_points.append(world_points)
    TextBasedObjectTracker.save_points_to_json(np.array(video_world_points), output_path)


if __name__ == "__main__":
  torch.cuda.empty_cache()
  bag_file_path = "/path/to/bag/file.bag"

  # Download the files from here
  # GroudingDINO : https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  # cotracker : https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
  
  root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  groundingdino_path = os.path.join(root_dir, "GroundingDINO")
  cotracker_path = os.path.join(root_dir, "co-tracker")
  object_tracker = TextBasedObjectTracker(tracker_path=os.path.join(cotracker_path,'/checkpoints/cotracker_stride_4_wind_8.pth'),
    detector_path=os.path.join(groundingdino_path,"weights/groundingdino_swint_ogc.pth"), 
    detector_config_path=os.path.join(groundingdino_path,"groundingdino/config/GroundingDINO_SwinT_OGC.py"), 
    device=device)